from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.db import DEFAULT_DB_PATH, init_db
from src.engine.bankroll import update_bankroll
from src.engine.edge_calculator import payout_for_american_odds
from src.models.bet import BetDecision, BetResult


def _normalize_timestamp(value: datetime | None) -> datetime:
    resolved = value or datetime.now(timezone.utc)
    if resolved.tzinfo is None or resolved.tzinfo.utcoffset(resolved) is None:
        raise ValueError("settled_at must be timezone-aware")
    return resolved.astimezone(timezone.utc)


def _validate_score(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(value)


def _validate_innings_completed(value: float) -> float:
    resolved = float(value)
    if resolved < 0:
        raise ValueError("innings_completed must be non-negative")
    return resolved


def _profit_loss_for_result(decision: BetDecision, result: BetResult) -> float:
    if result is BetResult.WIN:
        return float(payout_for_american_odds(decision.odds_at_bet) * decision.kelly_stake)
    if result is BetResult.LOSS:
        return float(-decision.kelly_stake)
    return 0.0


def _load_pending_bets(connection: sqlite3.Connection, game_pk: int) -> list[BetDecision]:
    rows = connection.execute(
        """
        SELECT game_pk, market_type, side, edge_pct, kelly_stake, odds_at_bet, result
        FROM bets
        WHERE game_pk = ? AND result = ?
        ORDER BY id DESC
        """,
        (game_pk, BetResult.PENDING.value),
    ).fetchall()

    return [
        BetDecision(
            game_pk=row[0],
            market_type=row[1],
            side=row[2],
            model_probability=0.5,
            fair_probability=0.5,
            edge_pct=row[3],
            ev=0.0,
            is_positive_ev=row[3] >= 0.03,
            kelly_stake=row[4],
            odds_at_bet=row[5],
            result=BetResult(row[6]),
        )
        for row in rows
    ]


def settle_bet(
    decision: BetDecision,
    *,
    home_score: int | None,
    away_score: int | None,
    innings_completed: float,
    starter_scratched: bool = False,
) -> BetResult:
    """Return the F5 settlement result for a single bet decision."""

    resolved_home_score = _validate_score("home_score", home_score)
    resolved_away_score = _validate_score("away_score", away_score)
    resolved_innings_completed = _validate_innings_completed(innings_completed)

    if starter_scratched or resolved_innings_completed < 5:
        return BetResult.NO_ACTION

    if resolved_home_score is None or resolved_away_score is None:
        return BetResult.NO_ACTION

    if decision.market_type == "f5_ml":
        if resolved_home_score == resolved_away_score:
            return BetResult.PUSH

        winning_side = "home" if resolved_home_score > resolved_away_score else "away"
        return BetResult.WIN if decision.side == winning_side else BetResult.LOSS

    home_margin = resolved_home_score - resolved_away_score
    if decision.line_at_bet is not None:
        selected_margin = float(home_margin if decision.side == "home" else -home_margin)
        covered_margin = selected_margin + float(decision.line_at_bet)
        if covered_margin > 0:
            return BetResult.WIN
        if covered_margin < 0:
            return BetResult.LOSS
        return BetResult.PUSH

    if decision.side == "home":
        return BetResult.WIN if home_margin >= 2 else BetResult.LOSS

    return BetResult.WIN if home_margin <= 1 else BetResult.LOSS


def settle_game_bets(
    game_pk: int,
    *,
    home_score: int | None,
    away_score: int | None,
    innings_completed: float,
    starter_scratched: bool = False,
    db_path: str | Path = DEFAULT_DB_PATH,
    connection: sqlite3.Connection | None = None,
    starting_bankroll: float = 1000.0,
    settled_at: datetime | None = None,
    notes: str | None = None,
    commit: bool = True,
) -> list[BetDecision]:
    """Settle all pending bets for one game_pk and persist bankroll effects."""

    normalized_timestamp = _normalize_timestamp(settled_at)
    database_path = init_db(db_path) if connection is None else Path(db_path)

    owns_connection = connection is None
    resolved_connection = connection or sqlite3.connect(database_path)

    try:
        resolved_connection.execute("PRAGMA foreign_keys = ON")
        pending_bets = _load_pending_bets(resolved_connection, game_pk)
        settled_bets: list[BetDecision] = []

        for pending_bet in pending_bets:
            result = settle_bet(
                pending_bet,
                home_score=home_score,
                away_score=away_score,
                innings_completed=innings_completed,
                starter_scratched=starter_scratched,
            )
            profit_loss = _profit_loss_for_result(pending_bet, result)
            settled_bet = pending_bet.model_copy(
                update={
                    "result": result,
                    "settled_at": normalized_timestamp,
                    "profit_loss": profit_loss,
                }
            )
            update_bankroll(
                action="settle",
                decision=settled_bet,
                db_path=database_path,
                connection=resolved_connection,
                starting_bankroll=starting_bankroll,
                timestamp=normalized_timestamp,
                notes=notes,
                commit=False,
            )
            settled_bets.append(settled_bet)

        if commit:
            resolved_connection.commit()

        return settled_bets
    except Exception:
        if owns_connection:
            resolved_connection.rollback()
        raise
    finally:
        if owns_connection:
            resolved_connection.close()


__all__ = ["settle_bet", "settle_game_bets"]
