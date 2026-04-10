from __future__ import annotations

import argparse
import json
import math
import sqlite3
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

from src.db import DEFAULT_DB_PATH, init_db
from src.engine.edge_calculator import payout_for_american_odds
from src.model.data_builder import _normalize_game_status, _normalize_team_code
from src.models.bet import BetDecision
from src.models.weather import WeatherData
from src.clients.odds_client import fetch_mlb_full_game_odds_context
from src.pipeline.daily import (
    DailyPipelineResult,
    GameProcessingResult,
    FULL_GAME_ODDS_CACHE_TTL_MINUTES,
    PipelineDependencies,
    SCRAPER_ODDS_DB_PATH,
    _default_feature_frame_builder,
    _default_full_game_odds_context_fetcher,
    _default_schedule_fetcher,
    _load_cached_full_game_odds_context,
    load_cached_slate_response,
    _load_fresh_odds_from_db_for_date,
    _load_scraper_f5_odds_for_date,
    _load_scraper_full_game_odds_context,
    _upsert_games,
    run_daily_pipeline,
)


UTC_TZ = timezone.utc
_LOG_LOSS_EPSILON = 1e-15
DEFAULT_PLAY_OF_DAY_MIN_EDGE = 0.06
DEFAULT_LIVE_GAME_STATE_PATH = Path("OddsScraper") / "data" / "live_game_state.json"
TRACKING_SOURCE_LIVE = "live"
TRACKING_SOURCE_CACHE_BACKFILL = "cache_backfill"
TRACKING_SOURCE_RETROSPECTIVE = "retrospective_backfill"


class _TrackerNoopNotifier:
    def send_picks(self, **payload: object) -> dict[str, object]:
        return dict(payload)

    def send_no_picks(self, **payload: object) -> dict[str, object]:
        return dict(payload)

    def send_failure_alert(self, **payload: object) -> dict[str, object]:
        return dict(payload)

    def send_drawdown_alert(self, **payload: object) -> dict[str, object]:
        return dict(payload)


def _neutral_weather_fetcher(
    team_abbr: str,
    game_datetime: object,
    db_path: object | None = None,
) -> WeatherData:
    del team_abbr, game_datetime, db_path
    return WeatherData(
        temperature_f=70.0,
        humidity_pct=50.0,
        wind_speed_mph=0.0,
        wind_direction_deg=0.0,
        pressure_hpa=1013.25,
        air_density=1.225,
        wind_factor=0.0,
        precipitation_probability=None,
        precipitation_mm=0.0,
        cloud_cover_pct=None,
        is_dome_default=False,
        forecast_time=None,
        fetched_at=None,
    )


def _local_only_full_game_odds_fetcher(
    target_day: date,
    _mode: str,
    repo_db_path: str | Path,
) -> dict[int, dict[str, Any]]:
    scraper_context = _load_scraper_full_game_odds_context(
        target_date=target_day,
        scraper_db_path=SCRAPER_ODDS_DB_PATH,
        repo_db_path=repo_db_path,
    )
    if scraper_context:
        return scraper_context
    cached_context = _load_cached_full_game_odds_context(
        repo_db_path,
        target_day,
        max_age=timedelta(minutes=FULL_GAME_ODDS_CACHE_TTL_MINUTES),
    )
    if cached_context:
        return cached_context
    return _load_cached_full_game_odds_context(
        repo_db_path,
        target_day,
        max_age=None,
    )


def _live_capture_quality_reasons(game: GameProcessingResult) -> list[str]:
    input_status = dict(game.input_status or {})
    reasons: list[str] = []
    if not bool(input_status.get("full_game_odds_available")):
        reasons.append("full-game odds unavailable")
    if not bool(input_status.get("home_lineup_available")):
        reasons.append("home lineup unavailable")
    if not bool(input_status.get("away_lineup_available")):
        reasons.append("away lineup unavailable")
    if not bool(input_status.get("weather_available")):
        reasons.append("weather unavailable")
    if bool(game.paper_fallback):
        reasons.append("paper fallback disabled for live tracking")
    deduped: list[str] = []
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)
    return deduped


def _cached_live_feature_frame_builder(**kwargs: Any):
    kwargs = dict(kwargs)
    kwargs["refresh"] = False
    return _default_feature_frame_builder(**kwargs)


def _sanitize_live_capture_result(result: DailyPipelineResult) -> DailyPipelineResult:
    sanitized_games = list(result.games)
    for game in sanitized_games:
        if game.status != "pick" or game.selected_decision is None:
            continue
        rejection_reasons = _live_capture_quality_reasons(game)
        if not rejection_reasons:
            continue
        game.status = "no_pick"
        game.selected_decision = None
        game.paper_fallback = False
        existing_reason = str(game.no_pick_reason or "").strip()
        combined_reasons = rejection_reasons[:]
        if existing_reason:
            combined_reasons.append(existing_reason)
        deduped: list[str] = []
        for reason in combined_reasons:
            if reason and reason not in deduped:
                deduped.append(reason)
        game.no_pick_reason = "; ".join(deduped)

    pick_count = sum(game.status == "pick" for game in sanitized_games)
    no_pick_count = sum(game.status == "no_pick" for game in sanitized_games)
    error_count = sum(game.status == "error" for game in sanitized_games)
    if pick_count > 0:
        notification_type = "picks"
    elif sanitized_games and all(game.status == "error" for game in sanitized_games):
        notification_type = "failure_alert"
    else:
        notification_type = "no_picks"
    return replace(
        result,
        pick_count=pick_count,
        no_pick_count=no_pick_count,
        error_count=error_count,
        notification_type=notification_type,
    )


LIVE_SEASON_TRACKING_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS live_season_tracking (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    season INTEGER NOT NULL,
    pipeline_date TEXT NOT NULL,
    game_pk INTEGER NOT NULL,
    matchup TEXT NOT NULL,
    run_id TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    tracking_source TEXT NOT NULL DEFAULT 'live',
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


LIVE_GAME_STATE_SNAPSHOTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS live_game_state_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT,
    game_pk INTEGER,
    game_date TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team_score INTEGER,
    home_team_score INTEGER,
    game_status_text TEXT,
    status TEXT,
    inning INTEGER,
    outs INTEGER,
    is_final INTEGER NOT NULL DEFAULT 0 CHECK (is_final IN (0, 1)),
    fetched_at TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'mac_scraper',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""


LIVE_MANUAL_TRACKING_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS live_manual_tracking (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    season INTEGER NOT NULL,
    pipeline_date TEXT NOT NULL,
    game_pk INTEGER NOT NULL,
    matchup TEXT NOT NULL,
    bet_key TEXT NOT NULL,
    run_id TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    submitted_at TEXT NOT NULL,
    model_version TEXT,
    status TEXT NOT NULL DEFAULT 'pick' CHECK (status IN ('pick', 'no_pick', 'error')),
    paper_fallback INTEGER NOT NULL DEFAULT 0 CHECK (paper_fallback IN (0, 1)),
    selected_market_type TEXT NOT NULL,
    selected_side TEXT NOT NULL,
    source_model TEXT,
    source_model_version TEXT,
    book_name TEXT,
    odds_at_bet INTEGER NOT NULL,
    line_at_bet REAL,
    fair_probability REAL,
    model_probability REAL,
    edge_pct REAL,
    ev REAL,
    kelly_stake REAL,
    bet_units REAL NOT NULL,
    no_pick_reason TEXT,
    error_message TEXT,
    input_status_json TEXT,
    narrative TEXT,
    actual_status TEXT,
    actual_f5_home_score INTEGER,
    actual_f5_away_score INTEGER,
    actual_final_home_score INTEGER,
    actual_final_away_score INTEGER,
    settled_result TEXT CHECK (
        settled_result IS NULL OR settled_result IN ('WIN', 'LOSS', 'PUSH', 'NO_ACTION')
    ),
    flat_profit_loss REAL,
    settled_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (season, bet_key)
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
        column_name="tracking_source",
        column_sql="TEXT NOT NULL DEFAULT 'live'",
    )
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


def _ensure_live_game_state_table(connection: sqlite3.Connection) -> None:
    connection.execute(LIVE_GAME_STATE_SNAPSHOTS_TABLE_SQL)
    connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_live_game_state_unique
        ON live_game_state_snapshots (
            COALESCE(event_id, ''),
            game_date,
            away_team,
            home_team,
            fetched_at
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_live_game_state_date
        ON live_game_state_snapshots (game_date, fetched_at)
        """
    )


def _ensure_live_manual_tracking_table(connection: sqlite3.Connection) -> None:
    connection.execute(LIVE_MANUAL_TRACKING_TABLE_SQL)
    _ensure_column(
        connection,
        table_name="live_manual_tracking",
        column_name="bet_units",
        column_sql="REAL NOT NULL DEFAULT 1.0",
    )
    _ensure_column(
        connection,
        table_name="live_manual_tracking",
        column_name="submitted_at",
        column_sql="TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP",
    )
    _ensure_column(
        connection,
        table_name="live_manual_tracking",
        column_name="narrative",
        column_sql="TEXT",
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_live_manual_tracking_date ON live_manual_tracking (season, pipeline_date)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_live_manual_tracking_game ON live_manual_tracking (season, game_pk)"
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


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_bet_units(value: Any) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        resolved = 1.0
    clamped = min(5.0, max(0.5, resolved))
    return round(clamped * 2.0) / 2.0


def _line_key(line_at_bet: float | None) -> str:
    if line_at_bet is None:
        return "none"
    return f"{float(line_at_bet):.3f}"


def _manual_bet_key(
    *,
    pipeline_date: str,
    game_pk: int,
    market_type: str,
    side: str,
    line_at_bet: float | None,
) -> str:
    return ":".join(
        [
            str(pipeline_date),
            str(int(game_pk)),
            str(market_type),
            str(side),
            _line_key(line_at_bet),
        ]
    )


def _load_live_game_state_payload(path: str | Path) -> list[dict[str, Any]]:
    source_path = Path(path)
    if not source_path.exists():
        return []
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        games = payload.get("games")
        if isinstance(games, list):
            return [row for row in games if isinstance(row, dict)]
        return [row for row in payload.values() if isinstance(row, dict)]
    return []


def _resolve_live_game_state_game_pk(
    *,
    row: dict[str, Any],
    game_lookup: dict[tuple[str, str, str], int],
) -> int | None:
    raw_game_pk = row.get("game_pk")
    if raw_game_pk is not None:
        try:
            return int(raw_game_pk)
        except (TypeError, ValueError):
            pass
    game_date = str(row.get("game_date") or row.get("date") or "").strip()
    away_team = _normalize_team_code(row.get("away_team"))
    home_team = _normalize_team_code(row.get("home_team"))
    if not game_date or away_team is None or home_team is None:
        return None
    return game_lookup.get((game_date, away_team, home_team))


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

    if market_type.endswith("_ml"):
        if home_score == away_score:
            return "PUSH", 0.0
        winning_side = "home" if home_score > away_score else "away"
        result = "WIN" if side == winning_side else "LOSS"
        return result, float(
            payout_for_american_odds(int(odds_at_bet)) if result == "WIN" else -1.0
        )

    if market_type.endswith("_rl"):
        if line_at_bet is None:
            return None, None
        home_margin = int(home_score) - int(away_score)
        selected_margin = float(home_margin if side == "home" else -home_margin)
        covered_margin = selected_margin + float(line_at_bet)
        if covered_margin > 0:
            return "WIN", float(payout_for_american_odds(int(odds_at_bet)))
        if covered_margin < 0:
            return "LOSS", -1.0
        return "PUSH", 0.0

    if market_type.endswith("_total"):
        if line_at_bet is None:
            return None, None
        total_runs = int(home_score) + int(away_score)
        if side == "over":
            if total_runs > float(line_at_bet):
                return "WIN", float(payout_for_american_odds(int(odds_at_bet)))
            if total_runs < float(line_at_bet):
                return "LOSS", -1.0
            return "PUSH", 0.0
        if side == "under":
            if total_runs < float(line_at_bet):
                return "WIN", float(payout_for_american_odds(int(odds_at_bet)))
            if total_runs > float(line_at_bet):
                return "LOSS", -1.0
            return "PUSH", 0.0
        return None, None

    return None, None


def _score_pair_for_market(
    *,
    market_type: str | None,
    row: sqlite3.Row,
) -> tuple[int | None, int | None]:
    if market_type and str(market_type).startswith("f5_"):
        return row["f5_home_score"], row["f5_away_score"]
    return row["final_home_score"], row["final_away_score"]


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
    tracking_source: str = TRACKING_SOURCE_LIVE,
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
                tracking_source,
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
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(season, game_pk) DO UPDATE SET
                pipeline_date = excluded.pipeline_date,
                matchup = excluded.matchup,
                run_id = excluded.run_id,
                captured_at = excluded.captured_at,
                tracking_source = live_season_tracking.tracking_source,
                model_version = excluded.model_version,
                status = CASE
                    WHEN live_season_tracking.selected_market_type IS NOT NULL
                    THEN live_season_tracking.status
                    ELSE excluded.status
                END,
                paper_fallback = CASE
                    WHEN live_season_tracking.selected_market_type IS NOT NULL
                    THEN live_season_tracking.paper_fallback
                    ELSE excluded.paper_fallback
                END,
                f5_ml_home_prob = excluded.f5_ml_home_prob,
                f5_ml_away_prob = excluded.f5_ml_away_prob,
                f5_rl_home_prob = excluded.f5_rl_home_prob,
                f5_rl_away_prob = excluded.f5_rl_away_prob,
                selected_market_type = COALESCE(
                    live_season_tracking.selected_market_type,
                    excluded.selected_market_type
                ),
                selected_side = CASE
                    WHEN live_season_tracking.selected_market_type IS NOT NULL
                    THEN live_season_tracking.selected_side
                    ELSE excluded.selected_side
                END,
                source_model = CASE
                    WHEN live_season_tracking.selected_market_type IS NOT NULL
                    THEN live_season_tracking.source_model
                    ELSE excluded.source_model
                END,
                source_model_version = CASE
                    WHEN live_season_tracking.selected_market_type IS NOT NULL
                    THEN live_season_tracking.source_model_version
                    ELSE excluded.source_model_version
                END,
                book_name = CASE
                    WHEN live_season_tracking.selected_market_type IS NOT NULL
                    THEN live_season_tracking.book_name
                    ELSE excluded.book_name
                END,
                odds_at_bet = CASE
                    WHEN live_season_tracking.selected_market_type IS NOT NULL
                    THEN live_season_tracking.odds_at_bet
                    ELSE excluded.odds_at_bet
                END,
                line_at_bet = CASE
                    WHEN live_season_tracking.selected_market_type IS NOT NULL
                    THEN live_season_tracking.line_at_bet
                    ELSE excluded.line_at_bet
                END,
                fair_probability = CASE
                    WHEN live_season_tracking.selected_market_type IS NOT NULL
                    THEN live_season_tracking.fair_probability
                    ELSE excluded.fair_probability
                END,
                model_probability = CASE
                    WHEN live_season_tracking.selected_market_type IS NOT NULL
                    THEN live_season_tracking.model_probability
                    ELSE excluded.model_probability
                END,
                edge_pct = CASE
                    WHEN live_season_tracking.selected_market_type IS NOT NULL
                    THEN live_season_tracking.edge_pct
                    ELSE excluded.edge_pct
                END,
                ev = CASE
                    WHEN live_season_tracking.selected_market_type IS NOT NULL
                    THEN live_season_tracking.ev
                    ELSE excluded.ev
                END,
                kelly_stake = CASE
                    WHEN live_season_tracking.selected_market_type IS NOT NULL
                    THEN live_season_tracking.kelly_stake
                    ELSE excluded.kelly_stake
                END,
                is_play_of_day = CASE
                    WHEN live_season_tracking.selected_market_type IS NOT NULL
                    THEN live_season_tracking.is_play_of_day
                    ELSE excluded.is_play_of_day
                END,
                play_of_day_score = CASE
                    WHEN live_season_tracking.selected_market_type IS NOT NULL
                    THEN live_season_tracking.play_of_day_score
                    ELSE excluded.play_of_day_score
                END,
                forced_market_type = CASE
                    WHEN live_season_tracking.selected_market_type IS NOT NULL
                    THEN live_season_tracking.forced_market_type
                    ELSE excluded.forced_market_type
                END,
                forced_side = CASE
                    WHEN live_season_tracking.selected_market_type IS NOT NULL
                    THEN live_season_tracking.forced_side
                    ELSE excluded.forced_side
                END,
                forced_source_model = CASE
                    WHEN live_season_tracking.selected_market_type IS NOT NULL
                    THEN live_season_tracking.forced_source_model
                    ELSE excluded.forced_source_model
                END,
                forced_source_model_version = CASE
                    WHEN live_season_tracking.selected_market_type IS NOT NULL
                    THEN live_season_tracking.forced_source_model_version
                    ELSE excluded.forced_source_model_version
                END,
                forced_book_name = CASE
                    WHEN live_season_tracking.selected_market_type IS NOT NULL
                    THEN live_season_tracking.forced_book_name
                    ELSE excluded.forced_book_name
                END,
                forced_odds_at_bet = CASE
                    WHEN live_season_tracking.selected_market_type IS NOT NULL
                    THEN live_season_tracking.forced_odds_at_bet
                    ELSE excluded.forced_odds_at_bet
                END,
                forced_line_at_bet = CASE
                    WHEN live_season_tracking.selected_market_type IS NOT NULL
                    THEN live_season_tracking.forced_line_at_bet
                    ELSE excluded.forced_line_at_bet
                END,
                forced_fair_probability = CASE
                    WHEN live_season_tracking.selected_market_type IS NOT NULL
                    THEN live_season_tracking.forced_fair_probability
                    ELSE excluded.forced_fair_probability
                END,
                forced_model_probability = CASE
                    WHEN live_season_tracking.selected_market_type IS NOT NULL
                    THEN live_season_tracking.forced_model_probability
                    ELSE excluded.forced_model_probability
                END,
                forced_edge_pct = CASE
                    WHEN live_season_tracking.selected_market_type IS NOT NULL
                    THEN live_season_tracking.forced_edge_pct
                    ELSE excluded.forced_edge_pct
                END,
                forced_ev = CASE
                    WHEN live_season_tracking.selected_market_type IS NOT NULL
                    THEN live_season_tracking.forced_ev
                    ELSE excluded.forced_ev
                END,
                forced_kelly_stake = CASE
                    WHEN live_season_tracking.selected_market_type IS NOT NULL
                    THEN live_season_tracking.forced_kelly_stake
                    ELSE excluded.forced_kelly_stake
                END,
                no_pick_reason = CASE
                    WHEN live_season_tracking.selected_market_type IS NOT NULL
                    THEN live_season_tracking.no_pick_reason
                    ELSE excluded.no_pick_reason
                END,
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
                    str(tracking_source),
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


def capture_live_slate_fast(
    *,
    target_date: str | date | datetime,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> DailyPipelineResult:
    resolved_db_path = init_db(db_path)
    local_odds_fetcher = lambda target_day, _mode, repo_db_path: [  # noqa: E731
        *(
            _load_scraper_f5_odds_for_date(
                target_date=target_day,
                scraper_db_path=SCRAPER_ODDS_DB_PATH,
                repo_db_path=repo_db_path,
            )
        ),
        *(
            _load_fresh_odds_from_db_for_date(
                repo_db_path,
                target_day,
                max_age=timedelta(minutes=15),
            )
        ),
    ]
    dependencies = PipelineDependencies(
        notifier=_TrackerNoopNotifier(),
        odds_fetcher=local_odds_fetcher,
        full_game_odds_fetcher=_local_only_full_game_odds_fetcher,
        feature_frame_builder=_cached_live_feature_frame_builder,
    )
    raw_result = run_daily_pipeline(
        target_date=target_date,
        mode="prod",
        dry_run=True,
        db_path=resolved_db_path,
        dependencies=dependencies,
    )
    result = _sanitize_live_capture_result(raw_result)
    capture_daily_result(result=result, db_path=resolved_db_path)
    return result


def _persist_cached_backtest_slate_payload(
    *,
    pipeline_date: date,
    db_path: str | Path,
    payload: dict[str, Any],
) -> None:
    database_path = init_db(db_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            UPDATE cached_slate_responses
            SET payload_json = ?,
                refreshed_at = ?
            WHERE pipeline_date = ? AND mode = 'backtest' AND dry_run = 1
            """,
            (
                json.dumps(payload),
                datetime.now(UTC).isoformat(),
                pipeline_date.isoformat(),
            ),
        )
        connection.commit()


def _refresh_frozen_slate_market_context(
    *,
    target_day: date,
    db_path: str | Path,
) -> int:
    database_path = init_db(db_path)
    payload = load_cached_slate_response(
        pipeline_date=target_day.isoformat(),
        mode="backtest",
        dry_run=True,
        db_path=database_path,
    )
    if payload is None:
        return 0

    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        game_status_rows = connection.execute(
            """
            SELECT game_pk, status
            FROM games
            WHERE date = ?
            """,
            (target_day.isoformat(),),
        ).fetchall()
    game_status_by_pk = {
        int(row["game_pk"]): _normalize_game_status(row["status"])
        for row in game_status_rows
        if row["game_pk"] is not None
    }

    full_game_context = _default_full_game_odds_context_fetcher(target_day, "prod", database_path)
    missing_total_game_pks = {
        int(game.get("game_pk") or 0)
        for game in payload.get("games") or []
        if game_status_by_pk.get(int(game.get("game_pk") or 0), "scheduled") == "scheduled"
        if not (dict(game.get("input_status") or {}).get("full_game_total") is not None)
    }
    if missing_total_game_pks:
        start = datetime.combine(target_day, time.min, tzinfo=UTC)
        end = start + timedelta(days=1, hours=8)
        try:
            live_context = fetch_mlb_full_game_odds_context(
                db_path=database_path,
                commence_time_from=start,
                commence_time_to=end,
            )
        except Exception:
            live_context = {}
        for game_pk, context in live_context.items():
            existing = dict(full_game_context.get(game_pk) or {})
            merged = dict(existing)
            for key, value in context.items():
                if value is not None:
                    merged[key] = value
            full_game_context[int(game_pk)] = merged
    if not full_game_context:
        return 0

    updated_games = 0
    for game in payload.get("games") or []:
        game_pk = int(game.get("game_pk") or 0)
        if game_status_by_pk.get(game_pk, "scheduled") != "scheduled":
            continue
        context = full_game_context.get(game_pk)
        if not context:
            continue
        input_status = dict(game.get("input_status") or {})
        changed = False
        for key, value in context.items():
            if value is None:
                continue
            if input_status.get(key) != value:
                input_status[key] = value
                changed = True
        if changed:
            game["input_status"] = input_status
            updated_games += 1

    if updated_games > 0:
        _persist_cached_backtest_slate_payload(
            pipeline_date=target_day,
            db_path=database_path,
            payload=payload,
        )
    return updated_games


def capture_live_slate_once(
    *,
    target_date: str | date | datetime,
    db_path: str | Path = DEFAULT_DB_PATH,
    fast: bool = True,
) -> dict[str, Any]:
    if isinstance(target_date, datetime):
        target_day = target_date.date()
    elif isinstance(target_date, date):
        target_day = target_date
    else:
        target_day = date.fromisoformat(str(target_date))

    settled_rows = _settle_outstanding_tracking_dates(
        season=target_day.year,
        through_date=target_day,
        db_path=db_path,
    )

    existing_rows = list_tracked_games(
        season=target_day.year,
        pipeline_date=target_day.isoformat(),
        db_path=db_path,
    )
    if existing_rows:
        _clear_scheduled_tracking_rows_for_date(
            season=target_day.year,
            pipeline_date=target_day.isoformat(),
            db_path=db_path,
        )
        result = (
            capture_live_slate_fast(target_date=target_day, db_path=db_path)
            if fast
            else capture_live_slate(target_date=target_day, db_path=db_path)
        )
        refreshed_games = _refresh_frozen_slate_market_context(
            target_day=target_day,
            db_path=db_path,
        )
        tracked_rows = list_tracked_games(
            season=target_day.year,
            pipeline_date=target_day.isoformat(),
            db_path=db_path,
        )
        return {
            "pipeline_date": target_day.isoformat(),
            "season": target_day.year,
            "captured": False,
            "already_captured": True,
            "settled_rows": int(settled_rows),
            "tracked_games": len(tracked_rows),
            "run_id": result.run_id,
            "pick_count": sum(1 for row in tracked_rows if row.get("selected_market_type")),
            "notification_type": result.notification_type,
            "refreshed_games": int(refreshed_games),
        }

    result = (
        capture_live_slate_fast(target_date=target_day, db_path=db_path)
        if fast
        else capture_live_slate(target_date=target_day, db_path=db_path)
    )
    refreshed_games = _refresh_frozen_slate_market_context(
        target_day=target_day,
        db_path=db_path,
    )
    tracked_rows = list_tracked_games(
        season=target_day.year,
        pipeline_date=target_day.isoformat(),
        db_path=db_path,
    )
    return {
        "pipeline_date": target_day.isoformat(),
        "season": target_day.year,
        "captured": True,
        "already_captured": False,
        "settled_rows": int(settled_rows),
        "tracked_games": len(tracked_rows),
        "run_id": result.run_id,
        "pick_count": result.pick_count,
        "notification_type": result.notification_type,
        "refreshed_games": int(refreshed_games),
    }


def _clear_scheduled_tracking_rows_for_date(
    *,
    season: int,
    pipeline_date: str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    database_path = init_db(db_path)
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT tracker.game_pk, games.status AS game_status
            FROM live_season_tracking AS tracker
            LEFT JOIN games ON games.game_pk = tracker.game_pk
            WHERE tracker.season = ?
              AND tracker.pipeline_date = ?
            """,
            (int(season), str(pipeline_date)),
        ).fetchall()
        removable_game_pks = [
            int(row["game_pk"])
            for row in rows
            if row["game_pk"] is not None
            and _normalize_game_status(row["game_status"]) == "scheduled"
        ]
        if not removable_game_pks:
            return 0
        connection.executemany(
            """
            DELETE FROM live_season_tracking
            WHERE season = ? AND game_pk = ?
            """,
            [(int(season), int(game_pk)) for game_pk in removable_game_pks],
        )
        connection.commit()
    return len(removable_game_pks)


def _settle_outstanding_tracking_dates(
    *,
    season: int,
    through_date: date,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    database_path = init_db(db_path)
    cutoff = through_date.isoformat()
    unsettled_dates: set[str] = set()

    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        _ensure_live_season_tracking_table(connection)
        _ensure_live_manual_tracking_table(connection)

        machine_rows = connection.execute(
            """
            SELECT DISTINCT pipeline_date
            FROM live_season_tracking
            WHERE season = ?
              AND pipeline_date < ?
              AND (
                    (selected_market_type IS NOT NULL AND settled_result IS NULL)
                 OR (forced_market_type IS NOT NULL AND forced_settled_result IS NULL)
              )
            """,
            (int(season), cutoff),
        ).fetchall()
        unsettled_dates.update(
            str(row["pipeline_date"]) for row in machine_rows if row["pipeline_date"] is not None
        )

        manual_rows = connection.execute(
            """
            SELECT DISTINCT pipeline_date
            FROM live_manual_tracking
            WHERE season = ?
              AND pipeline_date < ?
              AND selected_market_type IS NOT NULL
              AND settled_result IS NULL
            """,
            (int(season), cutoff),
        ).fetchall()
        unsettled_dates.update(
            str(row["pipeline_date"]) for row in manual_rows if row["pipeline_date"] is not None
        )

    settled_rows = 0
    for pipeline_date in sorted(unsettled_dates):
        settled_rows += settle_tracked_games(
            pipeline_date=pipeline_date,
            season=season,
            db_path=database_path,
        )
        settled_rows += settle_manual_tracked_bets(
            pipeline_date=pipeline_date,
            season=season,
            db_path=database_path,
        )
    return int(settled_rows)


def sync_live_game_state(
    *,
    input_path: str | Path = DEFAULT_LIVE_GAME_STATE_PATH,
    db_path: str | Path = DEFAULT_DB_PATH,
    settle: bool = True,
) -> dict[str, Any]:
    database_path = init_db(db_path)
    rows = _load_live_game_state_payload(input_path)
    if not rows:
        return {
            "imported_rows": 0,
            "affected_dates": [],
            "updated_games": 0,
            "settled_rows": 0,
        }

    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        _ensure_live_season_tracking_table(connection)
        _ensure_live_game_state_table(connection)

        game_rows = connection.execute(
            "SELECT game_pk, date, away_team, home_team FROM games"
        ).fetchall()
        game_lookup = {
            (str(row["date"]), str(row["away_team"]), str(row["home_team"])): int(row["game_pk"])
            for row in game_rows
        }

        imported_rows: list[tuple[Any, ...]] = []
        latest_by_game_pk: dict[int, dict[str, Any]] = {}
        affected_dates: set[str] = set()

        for row in rows:
            game_date = str(row.get("game_date") or row.get("date") or "").strip()
            away_team = _normalize_team_code(row.get("away_team"))
            home_team = _normalize_team_code(row.get("home_team"))
            fetched_at_raw = row.get("fetched_at")
            if not game_date or away_team is None or home_team is None or not fetched_at_raw:
                continue
            fetched_at = _normalize_timestamp(datetime.fromisoformat(str(fetched_at_raw).replace("Z", "+00:00")))
            game_pk = _resolve_live_game_state_game_pk(row=row, game_lookup=game_lookup)
            normalized_status = _normalize_game_status(
                row.get("game_status_text") or row.get("status")
            )
            is_final = normalized_status == "final" or bool(row.get("is_final"))
            imported_rows.append(
                (
                    str(row.get("event_id")) if row.get("event_id") is not None else None,
                    int(game_pk) if game_pk is not None else None,
                    game_date,
                    away_team,
                    home_team,
                    _coerce_int(row.get("away_team_score")),
                    _coerce_int(row.get("home_team_score")),
                    str(row.get("game_status_text")) if row.get("game_status_text") is not None else None,
                    str(row.get("status")) if row.get("status") is not None else None,
                    _coerce_int(row.get("inning")),
                    _coerce_int(row.get("outs")),
                    int(is_final),
                    fetched_at.isoformat(),
                )
            )
            affected_dates.add(game_date)
            if game_pk is not None:
                latest = latest_by_game_pk.get(int(game_pk))
                if latest is None or str(latest["fetched_at"]) < fetched_at.isoformat():
                    latest_by_game_pk[int(game_pk)] = {
                        "game_pk": int(game_pk),
                        "status": normalized_status,
                        "home_score": _coerce_int(row.get("home_team_score")),
                        "away_score": _coerce_int(row.get("away_team_score")),
                        "fetched_at": fetched_at.isoformat(),
                    }

        connection.executemany(
            """
            INSERT OR IGNORE INTO live_game_state_snapshots (
                event_id,
                game_pk,
                game_date,
                away_team,
                home_team,
                away_team_score,
                home_team_score,
                game_status_text,
                status,
                inning,
                outs,
                is_final,
                fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            imported_rows,
        )

        updated_games = 0
        for latest in latest_by_game_pk.values():
            connection.execute(
                """
                UPDATE games
                SET status = ?,
                    final_home_score = COALESCE(?, final_home_score),
                    final_away_score = COALESCE(?, final_away_score)
                WHERE game_pk = ?
                """,
                (
                    latest["status"],
                    latest["home_score"],
                    latest["away_score"],
                    latest["game_pk"],
                ),
            )
            updated_games += int(connection.total_changes > 0)

        connection.commit()

    refreshed_dates: list[str] = []
    with sqlite3.connect(database_path) as connection:
        for tracked_date in sorted(affected_dates):
            try:
                refreshed_schedule = _default_schedule_fetcher(date.fromisoformat(tracked_date), "prod")
                if not refreshed_schedule.empty:
                    _upsert_games(database_path, refreshed_schedule)
                    refreshed_dates.append(tracked_date)
            except Exception:
                continue

    settled_rows = 0
    if settle:
        for tracked_date in sorted(affected_dates):
            settled_rows += settle_tracked_games(
                pipeline_date=tracked_date,
                db_path=database_path,
            )

    return {
        "imported_rows": len(imported_rows),
        "affected_dates": sorted(affected_dates),
        "refreshed_dates": refreshed_dates,
        "updated_games": len(latest_by_game_pk),
        "settled_rows": settled_rows,
    }


def settle_tracked_games(
    *,
    pipeline_date: str | date | None = None,
    season: int | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    settled_at: datetime | None = None,
    refresh_schedule: bool = True,
) -> int:
    database_path = init_db(db_path)
    normalized_settled_at = _normalize_timestamp(settled_at)

    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        _ensure_live_season_tracking_table(connection)
        _ensure_live_manual_tracking_table(connection)

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

        if refresh_schedule:
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
            settled_result: str | None = None
            flat_profit_loss: float | None = None
            forced_settled_result: str | None = None
            forced_flat_profit_loss: float | None = None
            if actual_status == "final":
                selected_home_score, selected_away_score = _score_pair_for_market(
                    market_type=row["selected_market_type"],
                    row=row,
                )
                settled_result, flat_profit_loss = _settle_live_pick(
                    market_type=row["selected_market_type"],
                    side=row["selected_side"],
                    line_at_bet=row["line_at_bet"],
                    odds_at_bet=row["odds_at_bet"],
                    home_score=selected_home_score,
                    away_score=selected_away_score,
                )
                forced_home_score, forced_away_score = _score_pair_for_market(
                    market_type=row["forced_market_type"],
                    row=row,
                )
                forced_settled_result, forced_flat_profit_loss = _settle_live_pick(
                    market_type=row["forced_market_type"],
                    side=row["forced_side"],
                    line_at_bet=row["forced_line_at_bet"],
                    odds_at_bet=row["forced_odds_at_bet"],
                    home_score=forced_home_score,
                    away_score=forced_away_score,
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
                    row["f5_home_score"],
                    row["f5_away_score"],
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


def submit_manual_tracked_bet(
    *,
    season: int,
    pipeline_date: str,
    game_pk: int,
    matchup: str,
    market_type: str,
    side: str,
    odds_at_bet: int,
    bet_units: float,
    db_path: str | Path = DEFAULT_DB_PATH,
    line_at_bet: float | None = None,
    fair_probability: float | None = None,
    model_probability: float | None = None,
    edge_pct: float | None = None,
    ev: float | None = None,
    kelly_stake: float | None = None,
    book_name: str | None = None,
    model_version: str | None = None,
    source_model: str | None = None,
    source_model_version: str | None = None,
    input_status: dict[str, Any] | None = None,
    narrative: str | None = None,
) -> dict[str, Any]:
    database_path = init_db(db_path)
    submitted_at = _normalize_timestamp(datetime.now(UTC))
    normalized_units = _normalize_bet_units(bet_units)
    manual_bet_key = _manual_bet_key(
        pipeline_date=str(pipeline_date),
        game_pk=int(game_pk),
        market_type=str(market_type),
        side=str(side),
        line_at_bet=None if line_at_bet is None else float(line_at_bet),
    )

    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        _ensure_live_manual_tracking_table(connection)
        _ensure_live_season_tracking_table(connection)

        game_row = connection.execute(
            "SELECT status FROM games WHERE game_pk = ?",
            (int(game_pk),),
        ).fetchone()
        game_status = (
            _normalize_game_status(game_row["status"]) if game_row and game_row["status"] else None
        )
        if game_status not in {None, "scheduled"}:
            raise ValueError("Manual tracking cannot be edited after the game has started.")

        if any(value is None for value in (fair_probability, model_probability, edge_pct, ev, kelly_stake)):
            source_row = connection.execute(
                """
                SELECT model_version,
                       source_model,
                       source_model_version,
                       book_name,
                       fair_probability,
                       model_probability,
                       edge_pct,
                       ev,
                       kelly_stake,
                       narrative
                FROM live_season_tracking
                WHERE season = ?
                  AND pipeline_date = ?
                  AND game_pk = ?
                  AND selected_market_type = ?
                  AND selected_side = ?
                  AND (
                        (line_at_bet IS NULL AND ? IS NULL)
                     OR line_at_bet = ?
                  )
                ORDER BY id DESC
                LIMIT 1
                """,
                (
                    int(season),
                    str(pipeline_date),
                    int(game_pk),
                    str(market_type),
                    str(side),
                    None if line_at_bet is None else float(line_at_bet),
                    None if line_at_bet is None else float(line_at_bet),
                ),
            ).fetchone()
            if source_row is not None:
                model_version = model_version or source_row["model_version"]
                source_model = source_model or source_row["source_model"]
                source_model_version = source_model_version or source_row["source_model_version"]
                book_name = book_name or source_row["book_name"]
                fair_probability = (
                    source_row["fair_probability"] if fair_probability is None else fair_probability
                )
                model_probability = (
                    source_row["model_probability"] if model_probability is None else model_probability
                )
                edge_pct = source_row["edge_pct"] if edge_pct is None else edge_pct
                ev = source_row["ev"] if ev is None else ev
                kelly_stake = source_row["kelly_stake"] if kelly_stake is None else kelly_stake
                narrative = narrative or source_row["narrative"]

        connection.execute(
            """
            INSERT INTO live_manual_tracking (
                season,
                pipeline_date,
                game_pk,
                matchup,
                bet_key,
                run_id,
                captured_at,
                submitted_at,
                model_version,
                status,
                paper_fallback,
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
                bet_units,
                input_status_json,
                narrative,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pick', 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(season, bet_key) DO UPDATE SET
                matchup = excluded.matchup,
                run_id = excluded.run_id,
                captured_at = excluded.captured_at,
                submitted_at = excluded.submitted_at,
                model_version = excluded.model_version,
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
                bet_units = excluded.bet_units,
                input_status_json = excluded.input_status_json,
                narrative = excluded.narrative,
                updated_at = excluded.updated_at
            """,
            (
                int(season),
                str(pipeline_date),
                int(game_pk),
                str(matchup),
                manual_bet_key,
                f"manual-{pipeline_date}",
                submitted_at.isoformat(),
                submitted_at.isoformat(),
                model_version,
                str(market_type),
                str(side),
                source_model,
                source_model_version,
                book_name,
                int(odds_at_bet),
                None if line_at_bet is None else float(line_at_bet),
                fair_probability,
                model_probability,
                edge_pct,
                ev,
                kelly_stake,
                normalized_units,
                json.dumps(input_status) if input_status is not None else None,
                narrative,
                submitted_at.isoformat(),
            ),
        )
        connection.commit()

        row = connection.execute(
            "SELECT * FROM live_manual_tracking WHERE season = ? AND bet_key = ?",
            (int(season), manual_bet_key),
        ).fetchone()
    return dict(row) if row is not None else {}


def delete_manual_tracked_bet(
    *,
    manual_bet_id: int,
    season: int | None = None,
    allow_locked_delete: bool = False,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> bool:
    database_path = init_db(db_path)
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        _ensure_live_manual_tracking_table(connection)

        where_sql = "id = ?"
        params: list[Any] = [int(manual_bet_id)]
        if season is not None:
            where_sql += " AND season = ?"
            params.append(int(season))

        row = connection.execute(
            f"""
            SELECT tracker.id, tracker.game_pk, games.status AS game_status
            FROM live_manual_tracking AS tracker
            LEFT JOIN games ON games.game_pk = tracker.game_pk
            WHERE {where_sql}
            LIMIT 1
            """,
            params,
        ).fetchone()
        if row is None:
            raise ValueError("Manual tracked bet was not found.")

        game_status = row["game_status"]
        if not allow_locked_delete and game_status not in {None, "scheduled"}:
            raise ValueError("Manual tracking cannot be edited after the game has started.")

        cursor = connection.execute(
            "DELETE FROM live_manual_tracking WHERE id = ?",
            (int(manual_bet_id),),
        )
        connection.commit()
        return cursor.rowcount > 0


def settle_manual_tracked_bets(
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
        _ensure_live_manual_tracking_table(connection)

        where_clauses: list[str] = []
        params: list[Any] = []
        if pipeline_date is not None:
            resolved_date = (
                pipeline_date.isoformat()
                if isinstance(pipeline_date, date) and not isinstance(pipeline_date, datetime)
                else str(pipeline_date)
            )
            where_clauses.append("tracker.pipeline_date = ?")
            params.append(resolved_date)
        if season is not None:
            where_clauses.append("tracker.season = ?")
            params.append(int(season))
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        rows = connection.execute(
            f"""
            SELECT tracker.id, tracker.selected_market_type, tracker.selected_side,
                   tracker.line_at_bet, tracker.odds_at_bet,
                   games.status AS actual_status, games.f5_home_score, games.f5_away_score,
                   games.final_home_score, games.final_away_score
            FROM live_manual_tracking AS tracker
            LEFT JOIN games ON games.game_pk = tracker.game_pk
            {where_sql}
            """,
            params,
        ).fetchall()

        updated = 0
        for row in rows:
            actual_status = str(row["actual_status"]) if row["actual_status"] is not None else None
            settled_result: str | None = None
            flat_profit_loss: float | None = None
            if actual_status == "final":
                home_score, away_score = _score_pair_for_market(
                    market_type=row["selected_market_type"],
                    row=row,
                )
                settled_result, flat_profit_loss = _settle_live_pick(
                    market_type=row["selected_market_type"],
                    side=row["selected_side"],
                    line_at_bet=row["line_at_bet"],
                    odds_at_bet=row["odds_at_bet"],
                    home_score=home_score,
                    away_score=away_score,
                )

            connection.execute(
                """
                UPDATE live_manual_tracking
                SET actual_status = ?,
                    actual_f5_home_score = ?,
                    actual_f5_away_score = ?,
                    actual_final_home_score = ?,
                    actual_final_away_score = ?,
                    settled_result = ?,
                    flat_profit_loss = ?,
                    settled_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    actual_status,
                    row["f5_home_score"],
                    row["f5_away_score"],
                    row["final_home_score"],
                    row["final_away_score"],
                    settled_result,
                    flat_profit_loss,
                    normalized_settled_at.isoformat() if settled_result is not None else None,
                    normalized_settled_at.isoformat(),
                    int(row["id"]),
                ),
            )
            updated += 1

        connection.commit()
        return updated


def list_manual_tracked_bets(
    *,
    season: int = 2026,
    pipeline_date: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    database_path = init_db(db_path)
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        _ensure_live_manual_tracking_table(connection)
        if pipeline_date is None:
            rows = connection.execute(
                "SELECT * FROM live_manual_tracking WHERE season = ? ORDER BY pipeline_date, game_pk, id",
                (int(season),),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT * FROM live_manual_tracking WHERE season = ? AND pipeline_date = ? ORDER BY game_pk, id",
                (int(season), str(pipeline_date)),
            ).fetchall()
    return [dict(row) for row in rows]


def build_manual_tracking_summary(
    *,
    season: int = 2026,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> LiveSeasonSummary:
    database_path = init_db(db_path)
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        _ensure_live_manual_tracking_table(connection)
        rows = connection.execute(
            "SELECT * FROM live_manual_tracking WHERE season = ? ORDER BY pipeline_date, game_pk, id",
            (int(season),),
        ).fetchall()

    tracked_games = len(rows)
    settled_rows = [row for row in rows if row["actual_status"] == "final"]
    pick_rows = [row for row in rows if row["status"] == "pick" and row["selected_market_type"]]
    graded_pick_rows = [
        row for row in pick_rows if row["settled_result"] in {"WIN", "LOSS", "PUSH"}
    ]
    wins = sum(row["settled_result"] == "WIN" for row in graded_pick_rows)
    losses = sum(row["settled_result"] == "LOSS" for row in graded_pick_rows)
    pushes = sum(row["settled_result"] == "PUSH" for row in graded_pick_rows)
    flat_profit_units = float(
        sum(float(row["flat_profit_loss"] or 0.0) * float(row["bet_units"] or 0.0) for row in graded_pick_rows)
    )
    risked_units = float(sum(float(row["bet_units"] or 0.0) for row in graded_pick_rows))
    flat_roi = float(flat_profit_units / risked_units) if risked_units > 0 else None
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
        no_picks=0,
        errors=0,
        paper_fallback_picks=0,
        flat_profit_units=flat_profit_units,
        flat_roi=flat_roi,
        play_of_day_count=0,
        play_of_day_graded_picks=0,
        play_of_day_wins=0,
        play_of_day_losses=0,
        play_of_day_pushes=0,
        play_of_day_profit_units=0.0,
        play_of_day_roi=None,
        forced_picks=0,
        forced_graded_picks=0,
        forced_wins=0,
        forced_losses=0,
        forced_pushes=0,
        forced_profit_units=0.0,
        forced_roi=None,
        f5_ml_accuracy=None,
        f5_ml_brier=None,
        f5_ml_log_loss=None,
        f5_rl_accuracy=None,
        f5_rl_brier=None,
        f5_rl_log_loss=None,
        latest_capture_at=latest_capture_at,
    )


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

    sync_parser = subparsers.add_parser(
        "sync-game-state",
        help="Import pulled live game-state snapshots and settle affected tracked games.",
    )
    sync_parser.add_argument(
        "--input-path",
        default=str(DEFAULT_LIVE_GAME_STATE_PATH),
        help="Path to live_game_state.json pulled from the Mac scraper host.",
    )
    sync_parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    sync_parser.add_argument(
        "--no-settle",
        action="store_true",
        help="Import game-state snapshots without running settlement.",
    )

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

    if args.command == "sync-game-state":
        summary = sync_live_game_state(
            input_path=args.input_path,
            db_path=args.db_path,
            settle=not args.no_settle,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "report":
        summary = build_live_season_summary(season=args.season, db_path=args.db_path)
        print(json.dumps(summary.to_dict(), indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
