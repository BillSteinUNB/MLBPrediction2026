from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.clients.odds_client import american_to_implied
from src.db import DEFAULT_DB_PATH, init_db
from src.engine.bankroll import update_bankroll
from src.engine.edge_calculator import expected_value, payout_for_american_odds
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


def _is_first_five_market(market_type: str) -> bool:
    return market_type in {"f5_ml", "f5_rl", "f5_total"}


def _settle_moneyline(decision: BetDecision, *, home_score: int, away_score: int) -> BetResult:
    if home_score == away_score:
        return BetResult.PUSH

    winning_side = "home" if home_score > away_score else "away"
    return BetResult.WIN if decision.side == winning_side else BetResult.LOSS


def _settle_runline(decision: BetDecision, *, home_score: int, away_score: int) -> BetResult:
    home_margin = home_score - away_score
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


def _settle_total(decision: BetDecision, *, home_score: int, away_score: int) -> BetResult:
    if decision.line_at_bet is None:
        return BetResult.NO_ACTION

    total_runs = home_score + away_score
    line = float(decision.line_at_bet)
    if decision.side == "over":
        if total_runs > line:
            return BetResult.WIN
        if total_runs < line:
            return BetResult.LOSS
        return BetResult.PUSH

    if decision.side == "under":
        if total_runs < line:
            return BetResult.WIN
        if total_runs > line:
            return BetResult.LOSS
        return BetResult.PUSH

    return BetResult.NO_ACTION


def _profit_loss_for_result(decision: BetDecision, result: BetResult) -> float:
    if result is BetResult.WIN:
        return float(payout_for_american_odds(decision.odds_at_bet) * decision.kelly_stake)
    if result is BetResult.LOSS:
        return float(-decision.kelly_stake)
    return 0.0


def _load_pending_bets(connection: sqlite3.Connection, game_pk: int) -> list[BetDecision]:
    rows = connection.execute(
        """
        SELECT
            bets.id,
            bets.game_pk,
            bets.market_type,
            bets.side,
            bets.book_name,
            bets.source_model,
            bets.source_model_version,
            bets.model_probability,
            bets.fair_probability,
            bets.edge_pct,
            bets.ev,
            bets.kelly_stake,
            bets.odds_at_bet,
            bets.line_at_bet,
            bets.result,
            bet_performance.book_name,
            bet_performance.model_probability,
            bet_performance.market_probability
        FROM bets
        LEFT JOIN bet_performance ON bet_performance.bet_id = bets.id
        WHERE bets.game_pk = ? AND bets.result = ?
        ORDER BY bets.id DESC
        """,
        (game_pk, BetResult.PENDING.value),
    ).fetchall()

    decisions: list[BetDecision] = []
    for row in rows:
        edge_pct = float(row[9])
        odds_at_bet = int(row[12])
        implied_probability = float(american_to_implied(odds_at_bet))
        model_probability_raw = row[7]
        if model_probability_raw is None:
            model_probability_raw = row[16]
        model_probability = (
            float(model_probability_raw)
            if model_probability_raw is not None
            else min(0.99, max(0.01, implied_probability + edge_pct))
        )
        fair_probability_raw = row[8]
        if fair_probability_raw is None:
            fair_probability_raw = model_probability - edge_pct
        fair_probability = float(fair_probability_raw)
        ev_raw = row[10]
        ev = float(ev_raw) if ev_raw is not None else float(expected_value(model_probability, odds_at_bet))
        book_name = row[4]
        if book_name is None and row[15] not in (None, "", "manual"):
            book_name = row[15]
        decisions.append(
            BetDecision(
                game_pk=int(row[1]),
                market_type=str(row[2]),
                side=str(row[3]),
                book_name=None if book_name is None else str(book_name),
                source_model=None if row[5] is None else str(row[5]),
                source_model_version=None if row[6] is None else str(row[6]),
                model_probability=min(0.99, max(0.01, float(model_probability))),
                fair_probability=min(0.99, max(0.01, float(fair_probability))),
                edge_pct=edge_pct,
                ev=ev,
                is_positive_ev=edge_pct >= 0.03,
                kelly_stake=float(row[11]),
                odds_at_bet=odds_at_bet,
                line_at_bet=None if row[13] is None else float(row[13]),
                result=BetResult(row[14]),
            )
        )
    return decisions


def settle_bet(
    decision: BetDecision,
    *,
    home_score: int | None,
    away_score: int | None,
    innings_completed: float,
    starter_scratched: bool = False,
) -> BetResult:
    """Return the settlement result for a single bet decision."""

    resolved_home_score = _validate_score("home_score", home_score)
    resolved_away_score = _validate_score("away_score", away_score)
    resolved_innings_completed = _validate_innings_completed(innings_completed)

    if resolved_home_score is None or resolved_away_score is None:
        return BetResult.NO_ACTION

    if _is_first_five_market(decision.market_type):
        if starter_scratched or resolved_innings_completed < 5:
            return BetResult.NO_ACTION

    if decision.market_type == "f5_ml" or decision.market_type == "full_game_ml":
        return _settle_moneyline(decision, home_score=resolved_home_score, away_score=resolved_away_score)

    if decision.market_type == "f5_rl" or decision.market_type == "full_game_rl":
        return _settle_runline(decision, home_score=resolved_home_score, away_score=resolved_away_score)

    if decision.market_type == "f5_total" or decision.market_type == "full_game_total":
        return _settle_total(decision, home_score=resolved_home_score, away_score=resolved_away_score)

    if starter_scratched or resolved_innings_completed < 5:
        return BetResult.NO_ACTION

    return BetResult.NO_ACTION


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
