from __future__ import annotations

import csv
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Sequence

from src.clients.odds_client import american_to_implied
from src.db import DEFAULT_DB_PATH, init_db
from src.models._base import AmericanOdds, ModelBase, NonNegativeFloat, Probability, UtcDatetime
from src.models.bet import BetDecision, BetResult


ReportPeriod = Literal["daily", "weekly", "monthly"]
UTC = timezone.utc
_LOG_LOSS_EPSILON = 1e-15
_PERIOD_WINDOWS: dict[ReportPeriod, timedelta] = {
    "daily": timedelta(days=1),
    "weekly": timedelta(days=7),
    "monthly": timedelta(days=30),
}


class PerformanceBetRecord(ModelBase):
    """Tracked performance details for one persisted bet."""

    bet_id: int
    game_pk: int
    market_type: Literal["f5_ml", "f5_rl"]
    side: Literal["home", "away"]
    model_probability: Probability
    market_probability: Probability
    edge_pct: float
    odds_at_bet: AmericanOdds
    stake: NonNegativeFloat
    result: BetResult
    profit_loss: float | None = None
    closing_odds: AmericanOdds | None = None
    closing_probability: float | None = None
    clv: float | None = None
    placed_at: UtcDatetime
    settled_at: UtcDatetime | None = None


class PerformanceReport(ModelBase):
    """Aggregate betting metrics for a reporting window."""

    period: ReportPeriod
    period_start: UtcDatetime
    period_end: UtcDatetime
    total_bets: int
    graded_bets: int
    wins: int
    losses: int
    pushes: int
    no_actions: int
    pending: int
    win_rate: Probability
    roi: float
    total_profit_loss: float
    average_clv: float | None = None
    brier_score: float | None = None
    log_loss: float | None = None


def _normalize_timestamp(value: datetime | None, *, field_name: str) -> datetime:
    resolved = value or datetime.now(UTC)
    if resolved.tzinfo is None or resolved.tzinfo.utcoffset(resolved) is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return resolved.astimezone(UTC)


def _resolve_market_probability(decision: BetDecision) -> float:
    return float(american_to_implied(decision.odds_at_bet))


def _period_bounds(period: ReportPeriod, as_of: datetime | None) -> tuple[datetime, datetime]:
    resolved_end = _normalize_timestamp(as_of, field_name="as_of")
    return resolved_end - _PERIOD_WINDOWS[period], resolved_end


def _coerce_record(row: sqlite3.Row | Sequence[object]) -> PerformanceBetRecord:
    return PerformanceBetRecord(
        bet_id=int(row["bet_id"]),
        game_pk=int(row["game_pk"]),
        market_type=str(row["market_type"]),
        side=str(row["side"]),
        model_probability=float(row["model_probability"]),
        market_probability=float(row["market_probability"]),
        edge_pct=float(row["edge_pct"]),
        odds_at_bet=int(row["odds_at_bet"]),
        stake=float(row["stake"]),
        result=BetResult(str(row["result"])),
        profit_loss=None if row["profit_loss"] is None else float(row["profit_loss"]),
        closing_odds=None if row["closing_odds"] is None else int(row["closing_odds"]),
        closing_probability=(
            None if row["closing_probability"] is None else float(row["closing_probability"])
        ),
        clv=None if row["clv"] is None else float(row["clv"]),
        placed_at=datetime.fromisoformat(str(row["placed_at"])),
        settled_at=(None if row["settled_at"] is None else datetime.fromisoformat(str(row["settled_at"]))),
    )


def _upsert_base_record(
    connection: sqlite3.Connection,
    *,
    bet_id: int,
    decision: BetDecision,
    placed_at: datetime,
) -> None:
    existing = connection.execute(
        "SELECT id FROM bet_performance WHERE bet_id = ?",
        (bet_id,),
    ).fetchone()

    base_values = (
        bet_id,
        decision.game_pk,
        decision.market_type,
        decision.side,
        decision.model_probability,
        _resolve_market_probability(decision),
        decision.edge_pct,
        decision.odds_at_bet,
        decision.kelly_stake,
        decision.result.value,
        placed_at.isoformat(),
    )

    if existing is None:
        connection.execute(
            """
            INSERT INTO bet_performance (
                bet_id,
                game_pk,
                market_type,
                side,
                model_probability,
                market_probability,
                edge_pct,
                odds_at_bet,
                stake,
                result,
                placed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            base_values,
        )
        return

    connection.execute(
        """
        UPDATE bet_performance
        SET game_pk = ?,
            market_type = ?,
            side = ?,
            model_probability = ?,
            market_probability = ?,
            edge_pct = ?,
            odds_at_bet = ?,
            stake = ?,
            result = ?,
            placed_at = ?
        WHERE bet_id = ?
        """,
        (
            decision.game_pk,
            decision.market_type,
            decision.side,
            decision.model_probability,
            _resolve_market_probability(decision),
            decision.edge_pct,
            decision.odds_at_bet,
            decision.kelly_stake,
            decision.result.value,
            placed_at.isoformat(),
            bet_id,
        ),
    )


def _fetch_records(
    connection: sqlite3.Connection,
    *,
    period: ReportPeriod | None = None,
    as_of: datetime | None = None,
) -> list[PerformanceBetRecord]:
    if period is None:
        rows = connection.execute(
            "SELECT * FROM bet_performance ORDER BY placed_at ASC, bet_id ASC"
        ).fetchall()
        return [_coerce_record(row) for row in rows]

    period_start, period_end = _period_bounds(period, as_of)
    rows = connection.execute(
        """
        SELECT *
        FROM bet_performance
        WHERE placed_at >= ? AND placed_at <= ?
        ORDER BY placed_at ASC, bet_id ASC
        """,
        (period_start.isoformat(), period_end.isoformat()),
    ).fetchall()
    return [_coerce_record(row) for row in rows]


def _build_report(
    *,
    period: ReportPeriod,
    records: Sequence[PerformanceBetRecord],
    period_start: datetime,
    period_end: datetime,
) -> PerformanceReport:
    wins = sum(record.result == BetResult.WIN for record in records)
    losses = sum(record.result == BetResult.LOSS for record in records)
    pushes = sum(record.result == BetResult.PUSH for record in records)
    no_actions = sum(record.result == BetResult.NO_ACTION for record in records)
    pending = sum(record.result == BetResult.PENDING for record in records)
    graded_records = [record for record in records if record.result in {BetResult.WIN, BetResult.LOSS}]
    settled_records = [
        record for record in records if record.result not in {BetResult.PENDING, BetResult.NO_ACTION}
    ]

    win_rate = (wins / len(graded_records)) if graded_records else 0.0
    total_profit_loss = float(sum((record.profit_loss or 0.0) for record in settled_records))
    total_stake = float(sum(record.stake for record in settled_records))
    roi = (total_profit_loss / total_stake) if total_stake else 0.0

    clv_values = [record.clv for record in records if record.clv is not None]
    average_clv = (sum(clv_values) / len(clv_values)) if clv_values else None

    if graded_records:
        outcomes = [1.0 if record.result == BetResult.WIN else 0.0 for record in graded_records]
        probabilities = [record.model_probability for record in graded_records]
        brier_score = sum((probability - outcome) ** 2 for probability, outcome in zip(probabilities, outcomes)) / len(graded_records)
        log_loss = -sum(
            outcome * math.log(min(max(probability, _LOG_LOSS_EPSILON), 1.0 - _LOG_LOSS_EPSILON))
            + (1.0 - outcome)
            * math.log(min(max(1.0 - probability, _LOG_LOSS_EPSILON), 1.0 - _LOG_LOSS_EPSILON))
            for probability, outcome in zip(probabilities, outcomes)
        ) / len(graded_records)
    else:
        brier_score = None
        log_loss = None

    return PerformanceReport(
        period=period,
        period_start=period_start,
        period_end=period_end,
        total_bets=len(records),
        graded_bets=len(graded_records),
        wins=wins,
        losses=losses,
        pushes=pushes,
        no_actions=no_actions,
        pending=pending,
        win_rate=win_rate,
        roi=roi,
        total_profit_loss=total_profit_loss,
        average_clv=average_clv,
        brier_score=brier_score,
        log_loss=log_loss,
    )


def record_bet_placement(
    *,
    bet_id: int,
    decision: BetDecision,
    placed_at: datetime,
    db_path: str | Path = DEFAULT_DB_PATH,
    connection: sqlite3.Connection | None = None,
    commit: bool = True,
) -> None:
    """Store opening performance fields when a bet is placed."""

    normalized_placed_at = _normalize_timestamp(placed_at, field_name="placed_at")
    database_path = init_db(db_path) if connection is None else Path(db_path)
    owns_connection = connection is None
    resolved_connection = connection or sqlite3.connect(database_path)

    try:
        resolved_connection.execute("PRAGMA foreign_keys = ON")
        _upsert_base_record(
            resolved_connection,
            bet_id=bet_id,
            decision=decision,
            placed_at=normalized_placed_at,
        )
        if commit:
            resolved_connection.commit()
    except Exception:
        if owns_connection:
            resolved_connection.rollback()
        raise
    finally:
        if owns_connection:
            resolved_connection.close()


def record_bet_settlement(
    *,
    bet_id: int,
    decision: BetDecision,
    profit_loss: float,
    db_path: str | Path = DEFAULT_DB_PATH,
    connection: sqlite3.Connection | None = None,
    commit: bool = True,
) -> None:
    """Update tracked performance fields when a bet settles."""

    settled_at = _normalize_timestamp(decision.settled_at, field_name="settled_at")
    database_path = init_db(db_path) if connection is None else Path(db_path)
    owns_connection = connection is None
    resolved_connection = connection or sqlite3.connect(database_path)

    try:
        resolved_connection.execute("PRAGMA foreign_keys = ON")
        existing_placed_at = resolved_connection.execute(
            "SELECT placed_at FROM bet_performance WHERE bet_id = ?",
            (bet_id,),
        ).fetchone()
        placement_timestamp = (
            datetime.fromisoformat(str(existing_placed_at[0])) if existing_placed_at else settled_at
        )
        _upsert_base_record(
            resolved_connection,
            bet_id=bet_id,
            decision=decision,
            placed_at=placement_timestamp,
        )
        resolved_connection.execute(
            """
            UPDATE bet_performance
            SET result = ?,
                profit_loss = ?,
                settled_at = ?
            WHERE bet_id = ?
            """,
            (decision.result.value, profit_loss, settled_at.isoformat(), bet_id),
        )
        sync_closing_lines_from_snapshots(
            game_pk=decision.game_pk,
            market_type=decision.market_type,
            db_path=database_path,
            connection=resolved_connection,
            commit=False,
        )
        if commit:
            resolved_connection.commit()
    except Exception:
        if owns_connection:
            resolved_connection.rollback()
        raise
    finally:
        if owns_connection:
            resolved_connection.close()


def sync_closing_lines_from_snapshots(
    *,
    game_pk: int,
    market_type: Literal["f5_ml", "f5_rl"],
    db_path: str | Path = DEFAULT_DB_PATH,
    connection: sqlite3.Connection | None = None,
    commit: bool = True,
) -> int:
    """Apply the latest stored odds snapshot as the closing line for tracked bets."""

    database_path = init_db(db_path) if connection is None else Path(db_path)
    owns_connection = connection is None
    resolved_connection = connection or sqlite3.connect(database_path)
    updated_rows = 0

    try:
        resolved_connection.row_factory = sqlite3.Row
        resolved_connection.execute("PRAGMA foreign_keys = ON")
        closing_snapshot = resolved_connection.execute(
            """
            SELECT home_odds, away_odds
            FROM odds_snapshots
            WHERE game_pk = ? AND market_type = ?
            ORDER BY fetched_at DESC, id DESC
            LIMIT 1
            """,
            (game_pk, market_type),
        ).fetchone()
        if closing_snapshot is None:
            return 0

        performance_rows = resolved_connection.execute(
            """
            SELECT id, side, market_probability
            FROM bet_performance
            WHERE game_pk = ? AND market_type = ?
            """,
            (game_pk, market_type),
        ).fetchall()

        for row in performance_rows:
            closing_odds = int(closing_snapshot["home_odds"] if row["side"] == "home" else closing_snapshot["away_odds"])
            closing_probability = float(american_to_implied(closing_odds))
            clv = float(closing_probability - float(row["market_probability"]))
            resolved_connection.execute(
                """
                UPDATE bet_performance
                SET closing_odds = ?,
                    closing_probability = ?,
                    clv = ?
                WHERE id = ?
                """,
                (closing_odds, closing_probability, clv, int(row["id"])),
            )
            updated_rows += 1

        if commit:
            resolved_connection.commit()

        return updated_rows
    except Exception:
        if owns_connection:
            resolved_connection.rollback()
        raise
    finally:
        if owns_connection:
            resolved_connection.close()


def generate_performance_report(
    period: ReportPeriod,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    as_of: datetime | None = None,
) -> PerformanceReport:
    """Generate aggregate metrics for one rolling reporting window."""

    database_path = init_db(db_path)
    period_start, period_end = _period_bounds(period, as_of)

    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        records = _fetch_records(connection, period=period, as_of=period_end)

    return _build_report(
        period=period,
        records=records,
        period_start=period_start,
        period_end=period_end,
    )


def generate_periodic_reports(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    as_of: datetime | None = None,
) -> dict[ReportPeriod, PerformanceReport]:
    """Generate the daily, weekly, and monthly aggregate reports."""

    return {
        period: generate_performance_report(period, db_path=db_path, as_of=as_of)
        for period in ("daily", "weekly", "monthly")
    }


def export_performance_csv(
    *,
    output_path: str | Path,
    db_path: str | Path = DEFAULT_DB_PATH,
    period: ReportPeriod | None = None,
    as_of: datetime | None = None,
) -> Path:
    """Export tracked per-bet performance rows to CSV."""

    database_path = init_db(db_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        records = _fetch_records(connection, period=period, as_of=as_of)

    fieldnames = [
        "bet_id",
        "game_pk",
        "market_type",
        "side",
        "model_probability",
        "market_probability",
        "edge_pct",
        "odds_at_bet",
        "stake",
        "result",
        "profit_loss",
        "closing_odds",
        "closing_probability",
        "clv",
        "placed_at",
        "settled_at",
    ]

    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "bet_id": record.bet_id,
                    "game_pk": record.game_pk,
                    "market_type": record.market_type,
                    "side": record.side,
                    "model_probability": record.model_probability,
                    "market_probability": record.market_probability,
                    "edge_pct": record.edge_pct,
                    "odds_at_bet": record.odds_at_bet,
                    "stake": record.stake,
                    "result": record.result.value,
                    "profit_loss": record.profit_loss,
                    "closing_odds": record.closing_odds,
                    "closing_probability": record.closing_probability,
                    "clv": record.clv,
                    "placed_at": record.placed_at.isoformat(),
                    "settled_at": None if record.settled_at is None else record.settled_at.isoformat(),
                }
            )

    return destination


__all__ = [
    "PerformanceBetRecord",
    "PerformanceReport",
    "export_performance_csv",
    "generate_performance_report",
    "generate_periodic_reports",
    "record_bet_placement",
    "record_bet_settlement",
    "sync_closing_lines_from_snapshots",
]
