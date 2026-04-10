from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Sequence

from pydantic import TypeAdapter

from src.config import _load_settings_yaml
from src.db import DEFAULT_DB_PATH, init_db
from src.engine.edge_calculator import payout_for_american_odds
from src.models._base import AmericanOdds, BetSide, MarketType, ModelBase, NonNegativeFloat, Probability
from src.models.bet import BetDecision, BetResult
from src.ops.performance_tracker import record_bet_placement, record_bet_settlement


_SETTINGS_PAYLOAD = _load_settings_yaml()
DEFAULT_KELLY_FRACTION = float(_SETTINGS_PAYLOAD["thresholds"]["kelly_fraction"])
DEFAULT_MAX_DRAWDOWN = float(_SETTINGS_PAYLOAD["thresholds"]["max_drawdown"])
MAX_BET_FRACTION = 0.05

_AMERICAN_ODDS_ADAPTER = TypeAdapter(AmericanOdds)
_NON_NEGATIVE_ADAPTER = TypeAdapter(NonNegativeFloat)
_PROBABILITY_ADAPTER = TypeAdapter(Probability)


class KellyStakeResult(ModelBase):
    """Calculated Kelly sizing details for a single team exposure."""

    stake: NonNegativeFloat
    stake_fraction: NonNegativeFloat
    full_kelly_fraction: float
    drawdown_pct: Probability
    kill_switch_active: bool
    is_capped: bool
    game_pk: int | None = None
    selected_market_type: MarketType | None = None
    selected_side: BetSide | None = None
    selected_source_model: str | None = None
    suppressed_market_types: tuple[MarketType, ...] = ()


class BankrollSummary(ModelBase):
    """Current bankroll state and headline performance metrics."""

    current_bankroll: NonNegativeFloat
    peak_bankroll: NonNegativeFloat
    drawdown_pct: Probability
    total_bets: int
    win_rate: Probability
    roi: float
    kill_switch_active: bool


@dataclass(frozen=True)
class _KellyCandidate:
    game_pk: int | None
    market_type: str | None
    side: str | None
    source_model: str | None
    model_probability: float
    odds: int
    suppressed_market_types: tuple[str, ...] = ()


def _normalize_timestamp(value: datetime | None) -> datetime:
    resolved = value or datetime.now(timezone.utc)
    if resolved.tzinfo is None or resolved.tzinfo.utcoffset(resolved) is None:
        raise ValueError("timestamp must be timezone-aware")
    return resolved.astimezone(timezone.utc)


def _validate_fraction(name: str, value: float) -> float:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return float(value)


def _resolve_peak_bankroll(bankroll: float, peak_bankroll: float | None) -> float:
    if peak_bankroll is None:
        return bankroll

    return max(bankroll, _NON_NEGATIVE_ADAPTER.validate_python(peak_bankroll))


def _calculate_drawdown(bankroll: float, peak_bankroll: float) -> float:
    if peak_bankroll <= 0:
        return 0.0
    return float((peak_bankroll - bankroll) / peak_bankroll)


def _full_kelly_fraction(model_probability: float, odds: int) -> float:
    validated_probability = _PROBABILITY_ADAPTER.validate_python(model_probability)
    validated_odds = _AMERICAN_ODDS_ADAPTER.validate_python(odds)
    payout_multiple = payout_for_american_odds(validated_odds)
    full_kelly = ((payout_multiple * validated_probability) - (1.0 - validated_probability)) / payout_multiple
    return max(0.0, float(full_kelly))


def _select_candidate(
    *,
    model_probability: float | None,
    odds: int | None,
    decision: BetDecision | None,
    correlated_decisions: Sequence[BetDecision] | None,
) -> _KellyCandidate:
    if correlated_decisions is not None:
        if decision is not None or model_probability is not None or odds is not None:
            raise ValueError("correlated_decisions cannot be combined with other Kelly inputs")
        if not correlated_decisions:
            raise ValueError("correlated_decisions must not be empty")

        game_pks = {candidate.game_pk for candidate in correlated_decisions}
        sides = {candidate.side for candidate in correlated_decisions}
        if len(game_pks) != 1 or len(sides) != 1:
            raise ValueError("correlated_decisions must describe the same game and team side")

        ranked_candidates = sorted(
            correlated_decisions,
            key=lambda candidate: (
                _full_kelly_fraction(candidate.model_probability, candidate.odds_at_bet),
                candidate.edge_pct,
                candidate.ev,
            ),
            reverse=True,
        )
        selected = ranked_candidates[0]
        suppressed = tuple(
            candidate.market_type for candidate in correlated_decisions if candidate != selected
        )
        return _KellyCandidate(
            game_pk=selected.game_pk,
            market_type=selected.market_type,
            side=selected.side,
            source_model=selected.source_model,
            model_probability=selected.model_probability,
            odds=selected.odds_at_bet,
            suppressed_market_types=suppressed,
        )

    if decision is not None:
        if model_probability is not None or odds is not None:
            raise ValueError("decision cannot be combined with model_probability or odds")
        return _KellyCandidate(
            game_pk=decision.game_pk,
            market_type=decision.market_type,
            side=decision.side,
            source_model=decision.source_model,
            model_probability=decision.model_probability,
            odds=decision.odds_at_bet,
        )

    if model_probability is None or odds is None:
        raise ValueError("Provide decision, correlated_decisions, or model_probability with odds")

    return _KellyCandidate(
        game_pk=None,
        market_type=None,
        side=None,
        source_model=None,
        model_probability=float(model_probability),
        odds=int(odds),
    )


def calculate_kelly_stake(
    bankroll: float,
    *,
    model_probability: float | None = None,
    odds: int | None = None,
    decision: BetDecision | None = None,
    correlated_decisions: Sequence[BetDecision] | None = None,
    peak_bankroll: float | None = None,
    fraction: float = DEFAULT_KELLY_FRACTION,
    max_bet_fraction: float = MAX_BET_FRACTION,
    max_drawdown: float = DEFAULT_MAX_DRAWDOWN,
) -> KellyStakeResult:
    """Return quarter-Kelly sizing, enforcing drawdown and hard-cap constraints."""

    validated_bankroll = _NON_NEGATIVE_ADAPTER.validate_python(bankroll)
    resolved_fraction = _validate_fraction("fraction", fraction)
    resolved_max_bet_fraction = _validate_fraction("max_bet_fraction", max_bet_fraction)
    resolved_max_drawdown = _validate_fraction("max_drawdown", max_drawdown)
    resolved_peak_bankroll = _resolve_peak_bankroll(validated_bankroll, peak_bankroll)
    drawdown_pct = _calculate_drawdown(validated_bankroll, resolved_peak_bankroll)
    kill_switch_active = drawdown_pct >= resolved_max_drawdown
    selected_candidate = _select_candidate(
        model_probability=model_probability,
        odds=odds,
        decision=decision,
        correlated_decisions=correlated_decisions,
    )

    if kill_switch_active or validated_bankroll == 0:
        return KellyStakeResult(
            stake=0.0,
            stake_fraction=0.0,
            full_kelly_fraction=0.0,
            drawdown_pct=drawdown_pct,
            kill_switch_active=kill_switch_active,
            is_capped=False,
            game_pk=selected_candidate.game_pk,
            selected_market_type=selected_candidate.market_type,
            selected_side=selected_candidate.side,
            selected_source_model=selected_candidate.source_model,
            suppressed_market_types=selected_candidate.suppressed_market_types,
        )

    full_kelly_fraction = _full_kelly_fraction(
        selected_candidate.model_probability,
        selected_candidate.odds,
    )
    uncapped_fraction = max(0.0, full_kelly_fraction * resolved_fraction)
    capped_fraction = min(uncapped_fraction, resolved_max_bet_fraction)

    return KellyStakeResult(
        stake=float(validated_bankroll * capped_fraction),
        stake_fraction=float(capped_fraction),
        full_kelly_fraction=full_kelly_fraction,
        drawdown_pct=drawdown_pct,
        kill_switch_active=False,
        is_capped=uncapped_fraction > capped_fraction,
        game_pk=selected_candidate.game_pk,
        selected_market_type=selected_candidate.market_type,
        selected_side=selected_candidate.side,
        selected_source_model=selected_candidate.source_model,
        suppressed_market_types=selected_candidate.suppressed_market_types,
    )


def _get_current_balance(connection: sqlite3.Connection, starting_bankroll: float) -> float:
    row = connection.execute(
        "SELECT running_balance FROM bankroll_ledger ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return starting_bankroll
    return float(row[0])


def _is_kill_switch_active(
    connection: sqlite3.Connection,
    *,
    starting_bankroll: float,
    max_drawdown: float,
) -> bool:
    summary = _summarize_bankroll(
        connection,
        starting_bankroll=starting_bankroll,
        max_drawdown=max_drawdown,
    )
    return summary.kill_switch_active


def _insert_ledger_event(
    connection: sqlite3.Connection,
    *,
    event_type: str,
    amount: float,
    running_balance: float,
    timestamp: datetime,
    notes: str,
) -> None:
    connection.execute(
        """
        INSERT INTO bankroll_ledger (timestamp, event_type, amount, running_balance, notes)
        VALUES (?, ?, ?, ?, ?)
        """,
        (timestamp.isoformat(), event_type, amount, running_balance, notes),
    )


def _bet_notes(decision: BetDecision, action: str, extra_notes: str | None) -> str:
    notes = (
        f"{action}: game_pk={decision.game_pk}, market_type={decision.market_type}, "
        f"side={decision.side}, stake={decision.kelly_stake:.2f}"
    )
    if extra_notes:
        notes = f"{notes}; {extra_notes}"
    return notes


def _store_pending_bet(connection: sqlite3.Connection, decision: BetDecision) -> int:
    cursor = connection.execute(
        """
        INSERT INTO bets (
            game_pk,
            market_type,
            side,
            book_name,
            source_model,
            source_model_version,
            model_probability,
            fair_probability,
            edge_pct,
            ev,
            kelly_stake,
            odds_at_bet,
            line_at_bet,
            result
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            decision.game_pk,
            decision.market_type,
            decision.side,
            decision.book_name,
            decision.source_model,
            decision.source_model_version,
            decision.model_probability,
            decision.fair_probability,
            decision.edge_pct,
            decision.ev,
            decision.kelly_stake,
            decision.odds_at_bet,
            decision.line_at_bet,
            BetResult.PENDING.value,
        ),
    )
    return int(cursor.lastrowid)


def _settlement_delta(decision: BetDecision) -> tuple[float, float]:
    if decision.result == BetResult.PENDING:
        raise ValueError("Settlements require a final bet result")

    if decision.result == BetResult.WIN:
        profit_loss = payout_for_american_odds(decision.odds_at_bet) * decision.kelly_stake
        return decision.kelly_stake + profit_loss, profit_loss

    if decision.result == BetResult.LOSS:
        return 0.0, -decision.kelly_stake

    if decision.result in {BetResult.PUSH, BetResult.NO_ACTION}:
        return decision.kelly_stake, 0.0

    raise ValueError(f"Unsupported bet result: {decision.result}")


def _upsert_settled_bet(
    connection: sqlite3.Connection,
    *,
    decision: BetDecision,
    profit_loss: float,
) -> int:
    query = """
        SELECT id
        FROM bets
        WHERE game_pk = ? AND market_type = ? AND side = ? AND result = ?
    """
    params: list[object] = [
        decision.game_pk,
        decision.market_type,
        decision.side,
        BetResult.PENDING.value,
    ]
    if decision.book_name:
        query += " AND COALESCE(book_name, '') = ?"
        params.append(decision.book_name)
    query += " ORDER BY id DESC LIMIT 1"
    row = connection.execute(query, tuple(params)).fetchone()

    settled_at = decision.settled_at.isoformat() if decision.settled_at else None
    if row is None:
        cursor = connection.execute(
            """
            INSERT INTO bets (
                game_pk,
                market_type,
                side,
                book_name,
                source_model,
                source_model_version,
                model_probability,
                fair_probability,
                edge_pct,
                ev,
                kelly_stake,
                odds_at_bet,
                line_at_bet,
                result,
                settled_at,
                profit_loss
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision.game_pk,
                decision.market_type,
                decision.side,
                decision.book_name,
                decision.source_model,
                decision.source_model_version,
                decision.model_probability,
                decision.fair_probability,
                decision.edge_pct,
                decision.ev,
                decision.kelly_stake,
                decision.odds_at_bet,
                decision.line_at_bet,
                decision.result.value,
                settled_at,
                profit_loss,
            ),
        )
        return int(cursor.lastrowid)

    connection.execute(
        """
        UPDATE bets
        SET book_name = ?,
            source_model = ?,
            source_model_version = ?,
            model_probability = ?,
            fair_probability = ?,
            edge_pct = ?,
            ev = ?,
            kelly_stake = ?,
            odds_at_bet = ?,
            line_at_bet = ?,
            result = ?,
            settled_at = ?,
            profit_loss = ?
        WHERE id = ?
        """,
        (
            decision.book_name,
            decision.source_model,
            decision.source_model_version,
            decision.model_probability,
            decision.fair_probability,
            decision.edge_pct,
            decision.ev,
            decision.kelly_stake,
            decision.odds_at_bet,
            decision.line_at_bet,
            decision.result.value,
            settled_at,
            profit_loss,
            row[0],
        ),
    )
    return int(row[0])


def _maybe_log_kill_switch(
    connection: sqlite3.Connection,
    *,
    starting_bankroll: float,
    max_drawdown: float,
    timestamp: datetime,
) -> None:
    summary = _summarize_bankroll(
        connection,
        starting_bankroll=starting_bankroll,
        max_drawdown=max_drawdown,
    )
    if not summary.kill_switch_active:
        return

    existing = connection.execute(
        "SELECT COUNT(*) FROM bankroll_ledger WHERE event_type = 'kill_switch'"
    ).fetchone()
    if existing and existing[0] > 0:
        return

    drawdown_note = (
        f"Drawdown reached {summary.drawdown_pct:.2%}; threshold is {max_drawdown:.2%}"
    )
    _insert_ledger_event(
        connection,
        event_type="drawdown_alert",
        amount=0.0,
        running_balance=summary.current_bankroll,
        timestamp=timestamp,
        notes=drawdown_note,
    )
    _insert_ledger_event(
        connection,
        event_type="kill_switch",
        amount=0.0,
        running_balance=summary.current_bankroll,
        timestamp=timestamp,
        notes="Kill-switch activated after 30% drawdown threshold was reached",
    )


def _summarize_bankroll(
    connection: sqlite3.Connection,
    *,
    starting_bankroll: float,
    max_drawdown: float,
) -> BankrollSummary:
    current_bankroll = _get_current_balance(connection, starting_bankroll)
    peak_row = connection.execute("SELECT MAX(running_balance) FROM bankroll_ledger").fetchone()
    peak_bankroll = max(starting_bankroll, float(peak_row[0] or 0.0))
    drawdown_pct = _calculate_drawdown(current_bankroll, peak_bankroll)

    total_bets = int(connection.execute("SELECT COUNT(*) FROM bets").fetchone()[0])
    wins = int(connection.execute("SELECT COUNT(*) FROM bets WHERE result = 'WIN'").fetchone()[0])
    losses = int(connection.execute("SELECT COUNT(*) FROM bets WHERE result = 'LOSS'").fetchone()[0])
    graded_bets = wins + losses
    win_rate = (wins / graded_bets) if graded_bets else 0.0

    profit_row = connection.execute(
        "SELECT COALESCE(SUM(profit_loss), 0.0) FROM bets WHERE result != 'PENDING'"
    ).fetchone()
    staked_row = connection.execute(
        """
        SELECT COALESCE(SUM(kelly_stake), 0.0)
        FROM bets
        WHERE result NOT IN ('PENDING', 'NO_ACTION')
        """
    ).fetchone()
    total_profit = float(profit_row[0] or 0.0)
    total_staked = float(staked_row[0] or 0.0)
    roi = (total_profit / total_staked) if total_staked else 0.0

    kill_switch_count = connection.execute(
        "SELECT COUNT(*) FROM bankroll_ledger WHERE event_type = 'kill_switch'"
    ).fetchone()
    kill_switch_active = bool(drawdown_pct >= max_drawdown or (kill_switch_count and kill_switch_count[0] > 0))

    return BankrollSummary(
        current_bankroll=current_bankroll,
        peak_bankroll=peak_bankroll,
        drawdown_pct=drawdown_pct,
        total_bets=total_bets,
        win_rate=win_rate,
        roi=roi,
        kill_switch_active=kill_switch_active,
    )


def update_bankroll(
    *,
    action: Literal["place", "settle"],
    decision: BetDecision,
    db_path: str | Path = DEFAULT_DB_PATH,
    connection: sqlite3.Connection | None = None,
    starting_bankroll: float = 1000.0,
    timestamp: datetime | None = None,
    notes: str | None = None,
    max_drawdown: float = DEFAULT_MAX_DRAWDOWN,
    commit: bool = True,
) -> BankrollSummary:
    """Persist a bet placement or settlement and return updated bankroll metrics."""

    validated_starting_bankroll = _NON_NEGATIVE_ADAPTER.validate_python(starting_bankroll)
    resolved_max_drawdown = _validate_fraction("max_drawdown", max_drawdown)
    normalized_timestamp = _normalize_timestamp(timestamp)
    database_path = init_db(db_path) if connection is None else Path(db_path)

    owns_connection = connection is None
    resolved_connection = connection or sqlite3.connect(database_path)

    try:
        resolved_connection.execute("PRAGMA foreign_keys = ON")

        if action == "place" and _is_kill_switch_active(
            resolved_connection,
            starting_bankroll=validated_starting_bankroll,
            max_drawdown=resolved_max_drawdown,
        ):
            raise ValueError("Kill-switch is active; no new bets may be placed")

        current_bankroll = _get_current_balance(resolved_connection, validated_starting_bankroll)

        if action == "place":
            if decision.kelly_stake <= 0:
                raise ValueError("Placed bets must have a positive kelly_stake")

            bet_id = _store_pending_bet(resolved_connection, decision)
            updated_balance = max(0.0, current_bankroll - decision.kelly_stake)
            _insert_ledger_event(
                resolved_connection,
                event_type="bet_placed",
                amount=-decision.kelly_stake,
                running_balance=updated_balance,
                timestamp=normalized_timestamp,
                notes=_bet_notes(decision, "bet_placed", notes),
            )
            record_bet_placement(
                bet_id=bet_id,
                decision=decision,
                placed_at=normalized_timestamp,
                db_path=database_path,
                connection=resolved_connection,
                commit=False,
            )
        else:
            settlement_amount, profit_loss = _settlement_delta(decision)
            bet_id = _upsert_settled_bet(
                resolved_connection,
                decision=decision,
                profit_loss=profit_loss,
            )
            updated_balance = current_bankroll + settlement_amount
            _insert_ledger_event(
                resolved_connection,
                event_type="bet_settled",
                amount=settlement_amount,
                running_balance=updated_balance,
                timestamp=normalized_timestamp,
                notes=_bet_notes(decision, f"bet_settled:{decision.result.value}", notes),
            )
            record_bet_settlement(
                bet_id=bet_id,
                decision=decision,
                profit_loss=profit_loss,
                db_path=database_path,
                connection=resolved_connection,
                commit=False,
            )

        _maybe_log_kill_switch(
            resolved_connection,
            starting_bankroll=validated_starting_bankroll,
            max_drawdown=resolved_max_drawdown,
            timestamp=normalized_timestamp,
        )
        if commit:
            resolved_connection.commit()

        return _summarize_bankroll(
            resolved_connection,
            starting_bankroll=validated_starting_bankroll,
            max_drawdown=resolved_max_drawdown,
        )
    except Exception:
        if owns_connection:
            resolved_connection.rollback()
        raise
    finally:
        if owns_connection:
            resolved_connection.close()


def get_bankroll_summary(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    starting_bankroll: float = 1000.0,
    max_drawdown: float = DEFAULT_MAX_DRAWDOWN,
) -> BankrollSummary:
    """Return the persisted bankroll summary for notification and reporting flows."""

    validated_starting_bankroll = _NON_NEGATIVE_ADAPTER.validate_python(starting_bankroll)
    resolved_max_drawdown = _validate_fraction("max_drawdown", max_drawdown)
    database_path = init_db(db_path)

    with sqlite3.connect(database_path) as connection:
        return _summarize_bankroll(
            connection,
            starting_bankroll=validated_starting_bankroll,
            max_drawdown=resolved_max_drawdown,
        )


__all__ = [
    "BankrollSummary",
    "DEFAULT_KELLY_FRACTION",
    "DEFAULT_MAX_DRAWDOWN",
    "KellyStakeResult",
    "MAX_BET_FRACTION",
    "calculate_kelly_stake",
    "get_bankroll_summary",
    "update_bankroll",
]
