from __future__ import annotations

import argparse
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import UTC, date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.db import DEFAULT_DB_PATH, init_db
from src.engine.edge_calculator import payout_for_american_odds
from src.pipeline.daily import (
    DailyPipelineResult,
    GameProcessingResult,
    _default_schedule_fetcher,
    _upsert_games,
    run_daily_pipeline,
)


UTC_TZ = timezone.utc
_LOG_LOSS_EPSILON = 1e-15
DEFAULT_PLAY_OF_DAY_MIN_EDGE = 0.06


LIVE_SEASON_TRACKING_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS live_season_tracking (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    season INTEGER NOT NULL,
    pipeline_date TEXT NOT NULL,
    game_pk INTEGER NOT NULL,
    matchup TEXT NOT NULL,
    run_id TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    model_version TEXT,
    status TEXT NOT NULL CHECK (status IN ('pick', 'no_pick', 'error')),
    paper_fallback INTEGER NOT NULL DEFAULT 0 CHECK (paper_fallback IN (0, 1)),
    f5_ml_home_prob REAL,
    f5_ml_away_prob REAL,
    f5_rl_home_prob REAL,
    f5_rl_away_prob REAL,
    selected_market_type TEXT,
    selected_side TEXT,
    source_model TEXT,
    source_model_version TEXT,
    book_name TEXT,
    odds_at_bet INTEGER,
    line_at_bet REAL,
    fair_probability REAL,
    model_probability REAL,
    edge_pct REAL,
    ev REAL,
    kelly_stake REAL,
    is_play_of_day INTEGER NOT NULL DEFAULT 0 CHECK (is_play_of_day IN (0, 1)),
    play_of_day_score REAL,
    forced_market_type TEXT,
    forced_side TEXT,
    forced_source_model TEXT,
    forced_source_model_version TEXT,
    forced_book_name TEXT,
    forced_odds_at_bet INTEGER,
    forced_line_at_bet REAL,
    forced_fair_probability REAL,
    forced_model_probability REAL,
    forced_edge_pct REAL,
    forced_ev REAL,
    forced_kelly_stake REAL,
    no_pick_reason TEXT,
    error_message TEXT,
    input_status_json TEXT,
    actual_status TEXT,
    actual_f5_home_score INTEGER,
    actual_f5_away_score INTEGER,
    actual_final_home_score INTEGER,
    actual_final_away_score INTEGER,
    settled_result TEXT CHECK (
        settled_result IS NULL OR settled_result IN ('WIN', 'LOSS', 'PUSH', 'NO_ACTION')
    ),
    flat_profit_loss REAL,
    forced_settled_result TEXT CHECK (
        forced_settled_result IS NULL OR forced_settled_result IN ('WIN', 'LOSS', 'PUSH', 'NO_ACTION')
    ),
    forced_flat_profit_loss REAL,
    settled_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (season, game_pk)
)
"""


@dataclass(frozen=True, slots=True)
class LiveSeasonSummary:
    season: int
    tracked_games: int
    settled_games: int
    picks: int
    graded_picks: int
    wins: int
    losses: int
    pushes: int
    no_picks: int
    errors: int
    paper_fallback_picks: int
    flat_profit_units: float
    flat_roi: float | None
    play_of_day_count: int
    play_of_day_graded_picks: int
    play_of_day_wins: int
    play_of_day_losses: int
    play_of_day_pushes: int
    play_of_day_profit_units: float
    play_of_day_roi: float | None
    forced_picks: int
    forced_graded_picks: int
    forced_wins: int
    forced_losses: int
    forced_pushes: int
    forced_profit_units: float
    forced_roi: float | None
    f5_ml_accuracy: float | None
    f5_ml_brier: float | None
    f5_ml_log_loss: float | None
    f5_rl_accuracy: float | None
    f5_rl_brier: float | None
    f5_rl_log_loss: float | None
    latest_capture_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "season": self.season,
            "tracked_games": self.tracked_games,
            "settled_games": self.settled_games,
            "picks": self.picks,
            "graded_picks": self.graded_picks,
            "wins": self.wins,
            "losses": self.losses,
            "pushes": self.pushes,
            "no_picks": self.no_picks,
            "errors": self.errors,
            "paper_fallback_picks": self.paper_fallback_picks,
            "flat_profit_units": self.flat_profit_units,
            "flat_roi": self.flat_roi,
            "play_of_day_count": self.play_of_day_count,
            "play_of_day_graded_picks": self.play_of_day_graded_picks,
            "play_of_day_wins": self.play_of_day_wins,
            "play_of_day_losses": self.play_of_day_losses,
            "play_of_day_pushes": self.play_of_day_pushes,
            "play_of_day_profit_units": self.play_of_day_profit_units,
            "play_of_day_roi": self.play_of_day_roi,
            "forced_picks": self.forced_picks,
            "forced_graded_picks": self.forced_graded_picks,
            "forced_wins": self.forced_wins,
            "forced_losses": self.forced_losses,
            "forced_pushes": self.forced_pushes,
            "forced_profit_units": self.forced_profit_units,
            "forced_roi": self.forced_roi,
            "f5_ml_accuracy": self.f5_ml_accuracy,
            "f5_ml_brier": self.f5_ml_brier,
            "f5_ml_log_loss": self.f5_ml_log_loss,
            "f5_rl_accuracy": self.f5_rl_accuracy,
            "f5_rl_brier": self.f5_rl_brier,
            "f5_rl_log_loss": self.f5_rl_log_loss,
            "latest_capture_at": self.latest_capture_at,
        }


def _ensure_live_season_tracking_table(connection: sqlite3.Connection) -> None:
    connection.execute(LIVE_SEASON_TRACKING_TABLE_SQL)
    _ensure_column(
        connection,
        table_name="live_season_tracking",
        column_name="is_play_of_day",
        column_sql="INTEGER NOT NULL DEFAULT 0 CHECK (is_play_of_day IN (0, 1))",
    )
    _ensure_column(
        connection,
        table_name="live_season_tracking",
        column_name="play_of_day_score",
        column_sql="REAL",
    )
    _ensure_column(
        connection, table_name="live_season_tracking", column_name="source_model", column_sql="TEXT"
    )
    _ensure_column(
        connection,
        table_name="live_season_tracking",
        column_name="source_model_version",
        column_sql="TEXT",
    )
    _ensure_column(
        connection,
        table_name="live_season_tracking",
        column_name="forced_market_type",
        column_sql="TEXT",
    )
    _ensure_column(
        connection, table_name="live_season_tracking", column_name="forced_side", column_sql="TEXT"
    )
    _ensure_column(
        connection,
        table_name="live_season_tracking",
        column_name="forced_source_model",
        column_sql="TEXT",
    )
    _ensure_column(
        connection,
        table_name="live_season_tracking",
        column_name="forced_source_model_version",
        column_sql="TEXT",
    )
    _ensure_column(
        connection,
        table_name="live_season_tracking",
        column_name="forced_book_name",
        column_sql="TEXT",
    )
    _ensure_column(
        connection,
        table_name="live_season_tracking",
        column_name="forced_odds_at_bet",
        column_sql="INTEGER",
    )
    _ensure_column(
        connection,
        table_name="live_season_tracking",
        column_name="forced_line_at_bet",
        column_sql="REAL",
    )
    _ensure_column(
        connection,
        table_name="live_season_tracking",
        column_name="forced_fair_probability",
        column_sql="REAL",
    )
    _ensure_column(
        connection,
        table_name="live_season_tracking",
        column_name="forced_model_probability",
        column_sql="REAL",
    )
    _ensure_column(
        connection,
        table_name="live_season_tracking",
        column_name="forced_edge_pct",
        column_sql="REAL",
    )
    _ensure_column(
        connection, table_name="live_season_tracking", column_name="forced_ev", column_sql="REAL"
    )
    _ensure_column(
        connection,
        table_name="live_season_tracking",
        column_name="forced_kelly_stake",
        column_sql="REAL",
    )
    _ensure_column(
        connection,
        table_name="live_season_tracking",
        column_name="forced_settled_result",
        column_sql="TEXT",
    )
    _ensure_column(
        connection,
        table_name="live_season_tracking",
        column_name="forced_flat_profit_loss",
        column_sql="REAL",
    )
    _ensure_column(
        connection, table_name="live_season_tracking", column_name="narrative", column_sql="TEXT"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_live_season_tracking_date ON live_season_tracking (season, pipeline_date)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_live_season_tracking_status ON live_season_tracking (season, status)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_live_season_tracking_play_of_day ON live_season_tracking (season, is_play_of_day)"
    )


def _ensure_column(
    connection: sqlite3.Connection,
    *,
    table_name: str,
    column_name: str,
    column_sql: str,
) -> None:
    existing_columns = {
        str(row[1]) for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name not in existing_columns:
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}")


def _normalize_timestamp(value: datetime | None) -> datetime:
    resolved = value or datetime.now(UTC)
    if resolved.tzinfo is None or resolved.tzinfo.utcoffset(resolved) is None:
        raise ValueError("timestamp must be timezone-aware")
    return resolved.astimezone(UTC)


def _actual_outcomes(
    home_score: int | None, away_score: int | None
) -> tuple[int | None, int | None]:
    if home_score is None or away_score is None:
        return None, None
    margin = int(home_score) - int(away_score)
    return int(margin > 0), int(margin >= 2)


def _log_loss(probabilities: Sequence[float], outcomes: Sequence[int]) -> float | None:
    if not probabilities or not outcomes:
        return None
    total = 0.0
    for probability, outcome in zip(probabilities, outcomes):
        p = min(max(float(probability), _LOG_LOSS_EPSILON), 1.0 - _LOG_LOSS_EPSILON)
        total += outcome * math.log(p) + (1 - outcome) * math.log(1.0 - p)
    return float(-total / len(outcomes))


def _brier(probabilities: Sequence[float], outcomes: Sequence[int]) -> float | None:
    if not probabilities or not outcomes:
        return None
    return float(
        sum(
            (float(probability) - int(outcome)) ** 2
            for probability, outcome in zip(probabilities, outcomes)
        )
        / len(outcomes)
    )


def _accuracy(probabilities: Sequence[float], outcomes: Sequence[int]) -> float | None:
    if not probabilities or not outcomes:
        return None
    correct = sum(
        (float(probability) >= 0.5) == bool(outcome)
        for probability, outcome in zip(probabilities, outcomes)
    )
    return float(correct / len(outcomes))


def _settle_live_pick(
    *,
    market_type: str | None,
    side: str | None,
    line_at_bet: float | None,
    odds_at_bet: int | None,
    home_score: int | None,
    away_score: int | None,
) -> tuple[str | None, float | None]:
    if market_type is None or side is None or odds_at_bet is None:
        return None, None
    if home_score is None or away_score is None:
        return None, None

    if market_type == "f5_ml":
        if home_score == away_score:
            return "PUSH", 0.0
        winning_side = "home" if home_score > away_score else "away"
        result = "WIN" if side == winning_side else "LOSS"
        return result, float(
            payout_for_american_odds(int(odds_at_bet)) if result == "WIN" else -1.0
        )

    home_margin = int(home_score) - int(away_score)
    if line_at_bet is not None:
        selected_margin = float(home_margin if side == "home" else -home_margin)
        covered_margin = selected_margin + float(line_at_bet)
        if covered_margin > 0:
            return "WIN", float(payout_for_american_odds(int(odds_at_bet)))
        if covered_margin < 0:
            return "LOSS", -1.0
        return "PUSH", 0.0

    if side == "home":
        result = "WIN" if home_margin >= 2 else "LOSS"
    else:
        result = "WIN" if home_margin <= 1 else "LOSS"
    return result, float(payout_for_american_odds(int(odds_at_bet)) if result == "WIN" else -1.0)


def _play_of_day_score(decision: BetDecision | None) -> float | None:
    if decision is None:
        return None
    if float(decision.edge_pct) < DEFAULT_PLAY_OF_DAY_MIN_EDGE:
        return None
    return float(decision.edge_pct)


def _resolve_play_of_day_game_pks(games: Sequence[GameProcessingResult]) -> dict[int, float]:
    ranked: list[tuple[float, float, float, int]] = []
    for game in games:
        if game.status != "pick" or game.selected_decision is None:
            continue
        score = _play_of_day_score(game.selected_decision)
        if score is None:
            continue
        ranked.append(
            (
                score,
                float(game.selected_decision.edge_pct),
                float(game.selected_decision.ev),
                int(game.game_pk),
            )
        )
    if not ranked:
        return {}
    ranked.sort(reverse=True)
    score, _edge, _ev, game_pk = ranked[0]
    return {game_pk: score}


def _strategy_stats(
    rows: Sequence[sqlite3.Row],
    *,
    pick_result_column: str,
    profit_column: str,
) -> dict[str, Any]:
    graded_rows = [row for row in rows if row[pick_result_column] in {"WIN", "LOSS", "PUSH"}]
    wins = sum(row[pick_result_column] == "WIN" for row in graded_rows)
    losses = sum(row[pick_result_column] == "LOSS" for row in graded_rows)
    pushes = sum(row[pick_result_column] == "PUSH" for row in graded_rows)
    profit_units = float(sum(float(row[profit_column] or 0.0) for row in graded_rows))
    risked_bets = max(1, wins + losses + pushes) if graded_rows else 0
    roi = float(profit_units / risked_bets) if risked_bets else None
    return {
        "count": len(rows),
        "graded_picks": len(graded_rows),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "profit_units": profit_units,
        "roi": roi,
    }


def capture_daily_result(
    *,
    result: DailyPipelineResult,
    db_path: str | Path = DEFAULT_DB_PATH,
    captured_at: datetime | None = None,
) -> int:
    pipeline_day = date.fromisoformat(result.pipeline_date)
    database_path = init_db(db_path)
    captured_timestamp = _normalize_timestamp(captured_at)
    play_of_day_by_game_pk = _resolve_play_of_day_game_pks(result.games)

    with sqlite3.connect(database_path) as connection:
        _ensure_live_season_tracking_table(connection)
        connection.executemany(
            """
            INSERT INTO live_season_tracking (
                season,
                pipeline_date,
                game_pk,
                matchup,
                run_id,
                captured_at,
                model_version,
                status,
                paper_fallback,
                f5_ml_home_prob,
                f5_ml_away_prob,
                f5_rl_home_prob,
                f5_rl_away_prob,
                selected_market_type,
                selected_side,
                source_model,
                source_model_version,
                book_name,
                odds_at_bet,
                line_at_bet,
                fair_probability,
                model_probability,
                edge_pct,
                ev,
                kelly_stake,
                is_play_of_day,
                play_of_day_score,
                forced_market_type,
                forced_side,
                forced_source_model,
                forced_source_model_version,
                forced_book_name,
                forced_odds_at_bet,
                forced_line_at_bet,
                forced_fair_probability,
                forced_model_probability,
                forced_edge_pct,
                forced_ev,
                forced_kelly_stake,
                no_pick_reason,
                error_message,
                input_status_json,
                narrative,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(season, game_pk) DO UPDATE SET
                pipeline_date = excluded.pipeline_date,
                matchup = excluded.matchup,
                run_id = excluded.run_id,
                captured_at = excluded.captured_at,
                model_version = excluded.model_version,
                status = excluded.status,
                paper_fallback = excluded.paper_fallback,
                f5_ml_home_prob = excluded.f5_ml_home_prob,
                f5_ml_away_prob = excluded.f5_ml_away_prob,
                f5_rl_home_prob = excluded.f5_rl_home_prob,
                f5_rl_away_prob = excluded.f5_rl_away_prob,
                selected_market_type = excluded.selected_market_type,
                selected_side = excluded.selected_side,
                source_model = excluded.source_model,
                source_model_version = excluded.source_model_version,
                book_name = excluded.book_name,
                odds_at_bet = excluded.odds_at_bet,
                line_at_bet = excluded.line_at_bet,
                fair_probability = excluded.fair_probability,
                model_probability = excluded.model_probability,
                edge_pct = excluded.edge_pct,
                ev = excluded.ev,
                kelly_stake = excluded.kelly_stake,
                is_play_of_day = excluded.is_play_of_day,
                play_of_day_score = excluded.play_of_day_score,
                forced_market_type = excluded.forced_market_type,
                forced_side = excluded.forced_side,
                forced_source_model = excluded.forced_source_model,
                forced_source_model_version = excluded.forced_source_model_version,
                forced_book_name = excluded.forced_book_name,
                forced_odds_at_bet = excluded.forced_odds_at_bet,
                forced_line_at_bet = excluded.forced_line_at_bet,
                forced_fair_probability = excluded.forced_fair_probability,
                forced_model_probability = excluded.forced_model_probability,
                forced_edge_pct = excluded.forced_edge_pct,
                forced_ev = excluded.forced_ev,
                forced_kelly_stake = excluded.forced_kelly_stake,
                no_pick_reason = excluded.no_pick_reason,
                error_message = excluded.error_message,
                input_status_json = excluded.input_status_json,
                narrative = excluded.narrative,
                updated_at = excluded.updated_at
            """,
            [
                (
                    pipeline_day.year,
                    result.pipeline_date,
                    game.game_pk,
                    game.matchup,
                    result.run_id,
                    captured_timestamp.isoformat(),
                    game.prediction.model_version if game.prediction else result.model_version,
                    game.status,
                    int(game.paper_fallback),
                    game.prediction.f5_ml_home_prob if game.prediction else None,
                    game.prediction.f5_ml_away_prob if game.prediction else None,
                    game.prediction.f5_rl_home_prob if game.prediction else None,
                    game.prediction.f5_rl_away_prob if game.prediction else None,
                    game.selected_decision.market_type if game.selected_decision else None,
                    game.selected_decision.side if game.selected_decision else None,
                    game.selected_decision.source_model if game.selected_decision else None,
                    game.selected_decision.source_model_version if game.selected_decision else None,
                    game.selected_decision.book_name if game.selected_decision else None,
                    game.selected_decision.odds_at_bet if game.selected_decision else None,
                    game.selected_decision.line_at_bet if game.selected_decision else None,
                    game.selected_decision.fair_probability if game.selected_decision else None,
                    game.selected_decision.model_probability if game.selected_decision else None,
                    game.selected_decision.edge_pct if game.selected_decision else None,
                    game.selected_decision.ev if game.selected_decision else None,
                    game.selected_decision.kelly_stake if game.selected_decision else None,
                    int(game.game_pk in play_of_day_by_game_pk),
                    play_of_day_by_game_pk.get(game.game_pk),
                    game.forced_decision.market_type if game.forced_decision else None,
                    game.forced_decision.side if game.forced_decision else None,
                    game.forced_decision.source_model if game.forced_decision else None,
                    game.forced_decision.source_model_version if game.forced_decision else None,
                    game.forced_decision.book_name if game.forced_decision else None,
                    game.forced_decision.odds_at_bet if game.forced_decision else None,
                    game.forced_decision.line_at_bet if game.forced_decision else None,
                    game.forced_decision.fair_probability if game.forced_decision else None,
                    game.forced_decision.model_probability if game.forced_decision else None,
                    game.forced_decision.edge_pct if game.forced_decision else None,
                    game.forced_decision.ev if game.forced_decision else None,
                    game.forced_decision.kelly_stake if game.forced_decision else None,
                    game.no_pick_reason,
                    game.error_message,
                    json.dumps(game.input_status or {}, separators=(",", ":"), sort_keys=True),
                    game.narrative,
                    captured_timestamp.isoformat(),
                )
                for game in result.games
            ],
        )
        connection.commit()

    return len(result.games)


def capture_live_slate(
    *,
    target_date: str | date | datetime,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> DailyPipelineResult:
    result = run_daily_pipeline(
        target_date=target_date,
        mode="prod",
        dry_run=True,
        db_path=db_path,
    )
    capture_daily_result(result=result, db_path=db_path)
    return result


def settle_tracked_games(
    *,
    pipeline_date: str | date | None = None,
    season: int | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    settled_at: datetime | None = None,
) -> int:
    database_path = init_db(db_path)
    normalized_settled_at = _normalize_timestamp(settled_at)

    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        _ensure_live_season_tracking_table(connection)

        refresh_dates: list[str] = []
        if pipeline_date is not None:
            resolved_date = (
                pipeline_date.isoformat()
                if isinstance(pipeline_date, date) and not isinstance(pipeline_date, datetime)
                else str(pipeline_date)
            )
            refresh_dates = [resolved_date]
        else:
            date_rows = connection.execute(
                "SELECT DISTINCT pipeline_date FROM live_season_tracking WHERE season = ? ORDER BY pipeline_date",
                (int(season or 2026),),
            ).fetchall()
            refresh_dates = [str(row[0]) for row in date_rows]

        for tracked_date in refresh_dates:
            try:
                refreshed_schedule = _default_schedule_fetcher(
                    date.fromisoformat(tracked_date), "prod"
                )
                if not refreshed_schedule.empty:
                    _upsert_games(database_path, refreshed_schedule)
            except Exception:
                # Leave existing DB state intact if live schedule refresh fails.
                pass

        where_clauses: list[str] = []
        params: list[Any] = []
        if pipeline_date is not None:
            resolved_date = (
                pipeline_date.isoformat()
                if isinstance(pipeline_date, date) and not isinstance(pipeline_date, datetime)
                else str(pipeline_date)
            )
            where_clauses.append("pipeline_date = ?")
            params.append(resolved_date)
        if season is not None:
            where_clauses.append("season = ?")
            params.append(int(season))
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        rows = connection.execute(
            f"""
            SELECT tracker.id, tracker.game_pk, tracker.selected_market_type, tracker.selected_side,
                   tracker.line_at_bet, tracker.odds_at_bet,
                   tracker.forced_market_type, tracker.forced_side,
                   tracker.forced_line_at_bet, tracker.forced_odds_at_bet,
                   games.status AS actual_status, games.f5_home_score, games.f5_away_score,
                   games.final_home_score, games.final_away_score
            FROM live_season_tracking AS tracker
            LEFT JOIN games ON games.game_pk = tracker.game_pk
            {where_sql}
            """,
            params,
        ).fetchall()

        updated = 0
        for row in rows:
            actual_status = str(row["actual_status"]) if row["actual_status"] is not None else None
            home_score = row["f5_home_score"]
            away_score = row["f5_away_score"]

            settled_result: str | None = None
            flat_profit_loss: float | None = None
            forced_settled_result: str | None = None
            forced_flat_profit_loss: float | None = None
            if actual_status == "final":
                settled_result, flat_profit_loss = _settle_live_pick(
                    market_type=row["selected_market_type"],
                    side=row["selected_side"],
                    line_at_bet=row["line_at_bet"],
                    odds_at_bet=row["odds_at_bet"],
                    home_score=home_score,
                    away_score=away_score,
                )
                forced_settled_result, forced_flat_profit_loss = _settle_live_pick(
                    market_type=row["forced_market_type"],
                    side=row["forced_side"],
                    line_at_bet=row["forced_line_at_bet"],
                    odds_at_bet=row["forced_odds_at_bet"],
                    home_score=home_score,
                    away_score=away_score,
                )

            connection.execute(
                """
                UPDATE live_season_tracking
                SET actual_status = ?,
                    actual_f5_home_score = ?,
                    actual_f5_away_score = ?,
                    actual_final_home_score = ?,
                    actual_final_away_score = ?,
                    settled_result = ?,
                    flat_profit_loss = ?,
                    forced_settled_result = ?,
                    forced_flat_profit_loss = ?,
                    settled_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    actual_status,
                    home_score,
                    away_score,
                    row["final_home_score"],
                    row["final_away_score"],
                    settled_result,
                    flat_profit_loss,
                    forced_settled_result,
                    forced_flat_profit_loss,
                    normalized_settled_at.isoformat()
                    if settled_result is not None or forced_settled_result is not None
                    else None,
                    normalized_settled_at.isoformat(),
                    int(row["id"]),
                ),
            )
            updated += 1

        connection.commit()
        return updated


def build_live_season_summary(
    *,
    season: int = 2026,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> LiveSeasonSummary:
    database_path = init_db(db_path)
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        _ensure_live_season_tracking_table(connection)
        rows = connection.execute(
            "SELECT * FROM live_season_tracking WHERE season = ? ORDER BY pipeline_date, game_pk",
            (int(season),),
        ).fetchall()

    tracked_games = len(rows)
    settled_rows = [row for row in rows if row["actual_status"] == "final"]
    pick_rows = [
        row for row in rows if row["status"] == "pick" and row["selected_market_type"] is not None
    ]
    graded_pick_rows = [
        row for row in pick_rows if row["settled_result"] in {"WIN", "LOSS", "PUSH"}
    ]
    play_of_day_rows = [row for row in pick_rows if bool(row["is_play_of_day"])]
    forced_rows = [row for row in rows if row["forced_market_type"] is not None]

    ml_probs: list[float] = []
    ml_outcomes: list[int] = []
    rl_probs: list[float] = []
    rl_outcomes: list[int] = []
    for row in settled_rows:
        ml_outcome, rl_outcome = _actual_outcomes(
            row["actual_f5_home_score"], row["actual_f5_away_score"]
        )
        if ml_outcome is not None and row["f5_ml_home_prob"] is not None:
            ml_probs.append(float(row["f5_ml_home_prob"]))
            ml_outcomes.append(ml_outcome)
        if rl_outcome is not None and row["f5_rl_home_prob"] is not None:
            rl_probs.append(float(row["f5_rl_home_prob"]))
            rl_outcomes.append(rl_outcome)

    wins = sum(row["settled_result"] == "WIN" for row in graded_pick_rows)
    losses = sum(row["settled_result"] == "LOSS" for row in graded_pick_rows)
    pushes = sum(row["settled_result"] == "PUSH" for row in graded_pick_rows)
    flat_profit_units = float(
        sum(float(row["flat_profit_loss"] or 0.0) for row in graded_pick_rows)
    )
    risked_bets = max(1, wins + losses + pushes) if graded_pick_rows else 0
    flat_roi = float(flat_profit_units / risked_bets) if risked_bets else None
    play_of_day_stats = _strategy_stats(
        play_of_day_rows,
        pick_result_column="settled_result",
        profit_column="flat_profit_loss",
    )
    forced_stats = _strategy_stats(
        forced_rows,
        pick_result_column="forced_settled_result",
        profit_column="forced_flat_profit_loss",
    )

    latest_capture_at = max((str(row["captured_at"]) for row in rows), default=None)

    return LiveSeasonSummary(
        season=int(season),
        tracked_games=tracked_games,
        settled_games=len(settled_rows),
        picks=len(pick_rows),
        graded_picks=len(graded_pick_rows),
        wins=wins,
        losses=losses,
        pushes=pushes,
        no_picks=sum(row["status"] == "no_pick" for row in rows),
        errors=sum(row["status"] == "error" for row in rows),
        paper_fallback_picks=sum(bool(row["paper_fallback"]) for row in pick_rows),
        flat_profit_units=flat_profit_units,
        flat_roi=flat_roi,
        play_of_day_count=int(play_of_day_stats["count"]),
        play_of_day_graded_picks=int(play_of_day_stats["graded_picks"]),
        play_of_day_wins=int(play_of_day_stats["wins"]),
        play_of_day_losses=int(play_of_day_stats["losses"]),
        play_of_day_pushes=int(play_of_day_stats["pushes"]),
        play_of_day_profit_units=float(play_of_day_stats["profit_units"]),
        play_of_day_roi=play_of_day_stats["roi"],
        forced_picks=int(forced_stats["count"]),
        forced_graded_picks=int(forced_stats["graded_picks"]),
        forced_wins=int(forced_stats["wins"]),
        forced_losses=int(forced_stats["losses"]),
        forced_pushes=int(forced_stats["pushes"]),
        forced_profit_units=float(forced_stats["profit_units"]),
        forced_roi=forced_stats["roi"],
        f5_ml_accuracy=_accuracy(ml_probs, ml_outcomes),
        f5_ml_brier=_brier(ml_probs, ml_outcomes),
        f5_ml_log_loss=_log_loss(ml_probs, ml_outcomes),
        f5_rl_accuracy=_accuracy(rl_probs, rl_outcomes),
        f5_rl_brier=_brier(rl_probs, rl_outcomes),
        f5_rl_log_loss=_log_loss(rl_probs, rl_outcomes),
        latest_capture_at=latest_capture_at,
    )


def list_tracked_games(
    *,
    season: int = 2026,
    pipeline_date: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    database_path = init_db(db_path)
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        _ensure_live_season_tracking_table(connection)
        if pipeline_date is None:
            rows = connection.execute(
                "SELECT * FROM live_season_tracking WHERE season = ? ORDER BY pipeline_date, game_pk",
                (int(season),),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT * FROM live_season_tracking WHERE season = ? AND pipeline_date = ? ORDER BY game_pk",
                (int(season), str(pipeline_date)),
            ).fetchall()
    return [dict(row) for row in rows]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track live MLB season paper slates and outcomes.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture_parser = subparsers.add_parser(
        "capture", help="Run a dry slate and capture it as the official live-season snapshot."
    )
    capture_parser.add_argument(
        "--date", required=True, help="Target slate date in YYYY-MM-DD format."
    )
    capture_parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))

    settle_parser = subparsers.add_parser(
        "settle", help="Settle tracked live-season games using final scores already in the DB."
    )
    settle_parser.add_argument("--date", help="Optional YYYY-MM-DD slate date filter.")
    settle_parser.add_argument("--season", type=int, default=2026)
    settle_parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))

    report_parser = subparsers.add_parser(
        "report", help="Emit a 2026 live-season summary JSON report."
    )
    report_parser.add_argument("--season", type=int, default=2026)
    report_parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "capture":
        result = capture_live_slate(target_date=args.date, db_path=args.db_path)
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    if args.command == "settle":
        updated = settle_tracked_games(
            pipeline_date=args.date,
            season=args.season,
            db_path=args.db_path,
        )
        print(json.dumps({"updated_rows": updated}, indent=2))
        return 0

    if args.command == "report":
        summary = build_live_season_summary(season=args.season, db_path=args.db_path)
        print(json.dumps(summary.to_dict(), indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
