from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import httpx
import pandas as pd

from src.clients.statcast_client import (
    TEAM_GAME_LOG_CODES,
    TEAM_LABEL_TO_CODE,
    fetch_batting_stats,
    fetch_catcher_framing,
    fetch_fielding_stats,
    fetch_pitcher_stats,
    fetch_team_game_logs,
)
from src.clients.chadwick_client import fetch_chadwick_register
from src.clients.retrosheet_client import fetch_retrosheet_starting_lineups
from src.clients.weather_client import _get_default_weather, fetch_game_weather
from src.config import _load_settings_yaml
from src.db import DEFAULT_DB_PATH, init_db, sqlite_connection
from src.features.adjustments.abs_adjustment import (
    DEFAULT_STRIKEOUT_RATE_DELTA,
    DEFAULT_WALK_RATE_DELTA,
    apply_abs_adjustments,
    is_abs_active,
)
from src.features.adjustments.park_factors import get_park_factors
from src.features.adjustments.weather import NEUTRAL_WEATHER_FACTOR, compute_weather_adjustment
from src.features.baselines import compute_baseline_features
from src.features.bullpen import (
    DEFAULT_AVG_REST_DAYS,
    DEFAULT_TOP_RELIEVER_COUNT,
    DEFAULT_XFIP,
    compute_bullpen_features,
    _fetch_season_bullpen_metrics,
)
from src.features.defense import DEFAULT_DEFENSIVE_EFFICIENCY, compute_defense_features
from src.features.offense import (
    LEAGUE_WOBA_BASELINE,
    LEAGUE_WRC_PLUS_BASELINE,
    compute_offensive_features,
)
from src.features.pitching import (
    DEFAULT_METRIC_BASELINES as DEFAULT_PITCHING_METRIC_BASELINES,
    compute_pitching_features,
    _fetch_season_start_metrics,
)
from src.models.lineup import Lineup
from src.models.weather import WeatherData


DEFAULT_OUTPUT_PATH = Path("data") / "training" / "training_data_2019_2025.parquet"
DEFAULT_WINDOWS: tuple[int, ...] = (7, 14, 30, 60)
DEFAULT_PYTHAGOREAN_WINDOWS: tuple[int, ...] = (30, 60)
DEFAULT_FULL_REGULAR_SEASONS_TARGET = 7
SHORTENED_SEASON_GAME_THRESHOLD = 2_000
SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
DEFAULT_FEATURE_BUILD_CHUNK_DAYS = 14

_SETTINGS = _load_settings_yaml()
_TEAM_CODES = set(_SETTINGS["teams"].keys())
_TEAM_CODE_ALIASES = {value: key for key, value in TEAM_GAME_LOG_CODES.items()}
_TEAM_CODE_ALIASES.update(
    {
        "AZ": "ARI",
        "ATH": "OAK",
        "ANA": "LAA",
        "CHA": "CWS",
        "CHN": "CHC",
        "CHW": "CWS",
        "KCR": "KC",
        "KCA": "KC",
        "LAN": "LAD",
        "NYA": "NYY",
        "NYN": "NYM",
        "SDP": "SD",
        "SDN": "SD",
        "SFG": "SF",
        "SFN": "SF",
        "SLN": "STL",
        "TBR": "TB",
        "TBA": "TB",
        "WAS": "WSH",
        "WSN": "WSH",
    }
)
_FINAL_GAME_STATES = {"final", "game over", "completed early"}
_OFFENSE_FEATURE_DEFAULTS = {
    "wrc_plus": LEAGUE_WRC_PLUS_BASELINE,
    "woba": LEAGUE_WOBA_BASELINE,
    "iso": 0.0,
    "babip": 0.0,
    "k_pct": 0.0,
    "bb_pct": 0.0,
}
_PITCHING_FEATURE_DEFAULTS = {
    **DEFAULT_PITCHING_METRIC_BASELINES,
    "is_opener": 0.0,
    "uses_team_composite": 0.0,
}
_DEFENSE_FEATURE_DEFAULTS = {
    "drs": 0.0,
    "oaa": 0.0,
    "defensive_efficiency": DEFAULT_DEFENSIVE_EFFICIENCY,
    "adjusted_framing": 0.0,
}
_BULLPEN_FEATURE_DEFAULTS = {
    "pitch_count": 0.0,
    "avg_rest_days_top5": DEFAULT_AVG_REST_DAYS,
    "ir_pct": 0.0,
    "xfip": DEFAULT_XFIP,
    "high_leverage_available_count": float(DEFAULT_TOP_RELIEVER_COUNT),
}
_WEATHER_FEATURE_DEFAULTS = {
    "weather_temp_factor": NEUTRAL_WEATHER_FACTOR,
    "weather_air_density_factor": NEUTRAL_WEATHER_FACTOR,
    "weather_humidity_factor": NEUTRAL_WEATHER_FACTOR,
    "weather_wind_factor": 0.0,
    "weather_rain_risk": NEUTRAL_WEATHER_FACTOR,
    "weather_composite": NEUTRAL_WEATHER_FACTOR,
    "weather_data_missing": 1.0,
}
_SCHEDULE_FEATURE_DEFAULTS = {
    "park_runs_factor": 1.0,
    "park_hr_factor": 1.0,
    "abs_active": 1.0,
    "abs_walk_rate_delta": DEFAULT_WALK_RATE_DELTA,
    "abs_strikeout_rate_delta": DEFAULT_STRIKEOUT_RATE_DELTA,
}


ScheduleFetcher = Callable[[int], pd.DataFrame]
SeasonStatsFetcher = Callable[..., pd.DataFrame]
TeamLogsFetcher = Callable[..., pd.DataFrame]
LineupFetcher = Callable[[str | date | datetime], Sequence[Lineup]]
StartMetricsFetcher = Callable[..., pd.DataFrame]
BullpenMetricsFetcher = Callable[..., pd.DataFrame]
WeatherFetcher = Callable[..., WeatherData | None]
LineupPlayerIdsByDate = Mapping[str | date | datetime, Mapping[tuple[int, str], Sequence[int]]]

logger = logging.getLogger(__name__)


def _resolve_feature_build_workers() -> int:
    configured = os.getenv("MLB_FEATURE_BUILD_WORKERS")
    if configured is not None:
        try:
            return max(1, int(configured))
        except ValueError:
            logger.warning("Ignoring invalid MLB_FEATURE_BUILD_WORKERS value: %s", configured)
    detected_cpu_count = os.cpu_count() or 1
    return max(1, detected_cpu_count - 1)


DEFAULT_FEATURE_BUILD_WORKERS = _resolve_feature_build_workers()


@dataclass(frozen=True, slots=True)
class TrainingDataBuildResult:
    dataframe: pd.DataFrame
    output_path: Path
    metadata_path: Path
    data_version_hash: str
    build_timestamp: datetime
    requested_years: tuple[int, ...]
    effective_years: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TrainingDataCompletenessSummary:
    row_count: int
    expected_row_range: tuple[int, int]
    row_count_in_expected_range: bool
    target_null_counts: dict[str, int]
    game_type_counts: dict[str, int]
    non_regular_game_types: dict[str, int]
    seasons: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FeatureBuildTimingSummary:
    module_seconds: dict[str, float]
    total_seconds: float
    processed_dates: int


@dataclass(frozen=True, slots=True)
class _FeatureChunkResult:
    feature_rows: pd.DataFrame
    timing_summary: FeatureBuildTimingSummary


def resolve_training_years(
    *,
    start_year: int,
    end_year: int,
    full_regular_seasons_target: int = DEFAULT_FULL_REGULAR_SEASONS_TARGET,
    season_row_counts: Mapping[int, int] | None = None,
    shortened_season_game_threshold: int = SHORTENED_SEASON_GAME_THRESHOLD,
) -> list[int]:
    """Resolve effective training years, replacing shortened seasons with earlier full years."""

    if end_year < start_year:
        raise ValueError("end_year must be greater than or equal to start_year")

    requested_years = list(range(start_year, end_year + 1))
    row_counts = dict(season_row_counts or {})

    effective_years = [
        year
        for year in requested_years
        if row_counts.get(year, shortened_season_game_threshold) >= shortened_season_game_threshold
    ]

    backfill_year = start_year - 1
    while len(effective_years) < full_regular_seasons_target:
        if backfill_year not in effective_years and row_counts.get(
            backfill_year,
            shortened_season_game_threshold,
        ) >= shortened_season_game_threshold:
            effective_years.insert(0, backfill_year)
        backfill_year -= 1

    return sorted(effective_years)


def build_training_dataset(
    *,
    start_year: int = 2019,
    end_year: int = 2025,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    full_regular_seasons_target: int = DEFAULT_FULL_REGULAR_SEASONS_TARGET,
    shortened_season_game_threshold: int = SHORTENED_SEASON_GAME_THRESHOLD,
    scheduled_start_before: str | date | datetime | None = None,
    refresh: bool = False,
    refresh_raw_data: bool = False,
    schedule_fetcher: ScheduleFetcher | None = None,
    batting_stats_fetcher: SeasonStatsFetcher = fetch_batting_stats,
    pitching_stats_fetcher: SeasonStatsFetcher = fetch_pitcher_stats,
    fielding_stats_fetcher: SeasonStatsFetcher = fetch_fielding_stats,
    framing_stats_fetcher: SeasonStatsFetcher = fetch_catcher_framing,
    team_logs_fetcher: TeamLogsFetcher = fetch_team_game_logs,
    lineup_fetcher: LineupFetcher | None = None,
    start_metrics_fetcher: StartMetricsFetcher | None = None,
    bullpen_metrics_fetcher: BullpenMetricsFetcher | None = None,
    weather_fetcher: WeatherFetcher | None = None,
    lineup_player_ids_by_date: LineupPlayerIdsByDate | None = None,
) -> TrainingDataBuildResult:
    """Build historical training data using the same feature modules as inference."""

    _ = pitching_stats_fetcher

    resolved_schedule_fetcher = schedule_fetcher or _fetch_regular_season_schedule
    requested_years = tuple(range(start_year, end_year + 1))
    logger.info(
        "[build] Starting training-data build for requested years %s-%s",
        start_year,
        end_year,
    )

    schedules_by_year: dict[int, pd.DataFrame] = {}
    for year in requested_years:
        schedules_by_year[year] = _prepare_schedule_frame(resolved_schedule_fetcher(year))

    effective_years = tuple(
        resolve_training_years(
            start_year=start_year,
            end_year=end_year,
            full_regular_seasons_target=full_regular_seasons_target,
            season_row_counts={year: len(frame) for year, frame in schedules_by_year.items()},
            shortened_season_game_threshold=shortened_season_game_threshold,
        )
    )

    for year in effective_years:
        if year not in schedules_by_year:
            schedules_by_year[year] = _prepare_schedule_frame(resolved_schedule_fetcher(year))

    schedule = pd.concat(
        [schedules_by_year[year] for year in effective_years],
        ignore_index=True,
    )
    schedule = schedule.sort_values(["scheduled_start", "game_pk"]).reset_index(drop=True)
    if scheduled_start_before is not None:
        schedule_cutoff = _normalize_scheduled_start_before(scheduled_start_before)
        schedule = schedule.loc[
            pd.to_datetime(schedule["scheduled_start"], utc=True) < schedule_cutoff
        ].reset_index(drop=True)
        logger.info(
            "[build] Applied scheduled_start_before cutoff < %s",
            schedule_cutoff.isoformat(),
        )

    logger.info(
        "[build] Using effective years %s with %s scheduled games",
        ", ".join(str(year) for year in effective_years),
        len(schedule),
    )

    build_timestamp = datetime.now(UTC)
    resolved_lineup_fetcher = lineup_fetcher or _empty_lineups_fetcher
    resolved_lineup_player_ids_by_date = (
        lineup_player_ids_by_date
        if lineup_player_ids_by_date is not None
        else _load_historical_lineup_player_ids_by_date(
            schedule,
            refresh=refresh_raw_data,
        )
    )
    resolved_weather_fetcher = weather_fetcher or fetch_game_weather
    feature_build_workers = _resolve_effective_feature_build_workers(
        refresh_raw_data=refresh_raw_data,
        batting_stats_fetcher=batting_stats_fetcher,
        fielding_stats_fetcher=fielding_stats_fetcher,
        framing_stats_fetcher=framing_stats_fetcher,
        team_logs_fetcher=team_logs_fetcher,
        lineup_fetcher=lineup_fetcher,
        start_metrics_fetcher=start_metrics_fetcher,
        bullpen_metrics_fetcher=bullpen_metrics_fetcher,
        lineup_player_ids_by_date=lineup_player_ids_by_date,
    )

    if feature_build_workers > 1:
        logger.info(
            "[build] Parallel feature generation enabled with %s workers",
            feature_build_workers,
        )
        chunk_result = _compute_feature_modules_parallel(
            schedule,
            refresh=refresh_raw_data,
            lineup_player_ids_by_date=resolved_lineup_player_ids_by_date,
            worker_count=feature_build_workers,
        )
        feature_frame = _feature_rows_to_frame(chunk_result.feature_rows)
        _log_feature_build_timing_summary(
            chunk_result.timing_summary,
            label=f"parallel/{feature_build_workers}w",
        )
    else:
        temp_fd, temp_db_name = tempfile.mkstemp(prefix="training_builder_", suffix=".db")
        os.close(temp_fd)
        working_db_path = Path(temp_db_name)
        try:
            working_db_path = init_db(working_db_path)
            _seed_games_table(working_db_path, schedule)
            timing_summary = _compute_feature_modules(
                schedule,
                database_path=working_db_path,
                refresh=refresh_raw_data,
                batting_stats_fetcher=batting_stats_fetcher,
                fielding_stats_fetcher=fielding_stats_fetcher,
                framing_stats_fetcher=framing_stats_fetcher,
                team_logs_fetcher=team_logs_fetcher,
                lineup_fetcher=resolved_lineup_fetcher,
                start_metrics_fetcher=start_metrics_fetcher,
                bullpen_metrics_fetcher=bullpen_metrics_fetcher,
                lineup_player_ids_by_date=resolved_lineup_player_ids_by_date,
            )
            feature_rows = _load_feature_rows(working_db_path)
            feature_frame = _feature_rows_to_frame(feature_rows)
            _log_feature_build_timing_summary(timing_summary, label="serial")
            logger.info("[build] Loaded %s feature rows from sqlite cache", len(feature_rows))
            dataset = _assemble_training_rows(
                schedule,
                feature_frame=feature_frame,
                database_path=working_db_path,
                weather_fetcher=resolved_weather_fetcher,
            )
        finally:
            try:
                working_db_path.unlink(missing_ok=True)
            except PermissionError:
                pass
    if feature_build_workers > 1:
        temp_fd, temp_db_name = tempfile.mkstemp(prefix="training_builder_weather_", suffix=".db")
        os.close(temp_fd)
        weather_db_path = Path(temp_db_name)
        try:
            weather_db_path = init_db(weather_db_path)
            logger.info("[build] Loaded %s feature rows from parallel cache", len(chunk_result.feature_rows))
            dataset = _assemble_training_rows(
                schedule,
                feature_frame=feature_frame,
                database_path=weather_db_path,
                weather_fetcher=resolved_weather_fetcher,
            )
        finally:
            try:
                weather_db_path.unlink(missing_ok=True)
            except PermissionError:
                pass

    assert_training_data_is_leakage_free(dataset)
    _assert_targets_present(dataset)

    data_version_hash = _compute_data_version_hash(dataset)
    dataset = dataset.copy()
    dataset["data_version_hash"] = data_version_hash
    dataset["build_timestamp"] = build_timestamp.isoformat()
    logger.info("[build] Assembled dataset with %s rows", len(dataset))

    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(resolved_output_path, index=False)

    metadata_path = resolved_output_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "requested_years": list(requested_years),
                "effective_years": list(effective_years),
                "shortened_seasons_skipped": [
                    year for year in requested_years if year not in effective_years
                ],
                "scheduled_start_before": (
                    _normalize_scheduled_start_before(scheduled_start_before).isoformat()
                    if scheduled_start_before is not None
                    else None
                ),
                "row_count": int(len(dataset)),
                "feature_column_count": int(len(_feature_columns(dataset))),
                "feature_build_workers": int(feature_build_workers),
                "data_version_hash": data_version_hash,
                "build_timestamp": build_timestamp.isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return TrainingDataBuildResult(
        dataframe=dataset,
        output_path=resolved_output_path,
        metadata_path=metadata_path,
        data_version_hash=data_version_hash,
        build_timestamp=build_timestamp,
        requested_years=requested_years,
        effective_years=effective_years,
    )


def build_live_feature_frame(
    *,
    target_date: str | date | datetime,
    schedule: pd.DataFrame,
    historical_games: pd.DataFrame,
    db_path: str | Path = DEFAULT_DB_PATH,
    lineups: Sequence[Lineup] = (),
    refresh: bool = False,
    batting_stats_fetcher: SeasonStatsFetcher = fetch_batting_stats,
    team_logs_fetcher: TeamLogsFetcher = fetch_team_game_logs,
    fielding_stats_fetcher: SeasonStatsFetcher = fetch_fielding_stats,
    framing_stats_fetcher: SeasonStatsFetcher = fetch_catcher_framing,
    start_metrics_fetcher: StartMetricsFetcher | None = None,
    bullpen_metrics_fetcher: BullpenMetricsFetcher | None = None,
    weather_fetcher: WeatherFetcher | None = None,
) -> pd.DataFrame:
    """Build fresh same-day inference features without relying on persisted feature rows."""

    target_day = _coerce_date(target_date)
    prepared_schedule = _prepare_schedule_frame(schedule, require_final_scores=False)
    prepared_schedule = prepared_schedule.loc[
        pd.to_datetime(prepared_schedule["scheduled_start"], utc=True).dt.date == target_day
    ].reset_index(drop=True)
    if prepared_schedule.empty:
        return pd.DataFrame(columns=["game_pk"])

    prepared_history = _prepare_schedule_frame(historical_games)
    prepared_history = prepared_history.loc[
        pd.to_datetime(prepared_history["scheduled_start"], utc=True).dt.date < target_day
    ].reset_index(drop=True)

    lineup_list = list(lineups)
    lineup_player_ids = _lineup_player_ids_by_game_team(lineup_list)

    temp_fd, temp_db_name = tempfile.mkstemp(prefix="live_inference_", suffix=".db")
    os.close(temp_fd)
    working_db_path = Path(temp_db_name)
    try:
        working_db_path = init_db(working_db_path)
        combined_schedule = pd.concat(
            [prepared_history, prepared_schedule],
            ignore_index=True,
        )
        if not combined_schedule.empty:
            _seed_games_table(working_db_path, combined_schedule)

        _compute_feature_modules(
            prepared_schedule,
            database_path=working_db_path,
            refresh=refresh,
            batting_stats_fetcher=batting_stats_fetcher,
            fielding_stats_fetcher=fielding_stats_fetcher,
            framing_stats_fetcher=framing_stats_fetcher,
            team_logs_fetcher=team_logs_fetcher,
            lineup_fetcher=_static_lineup_fetcher(target_day, lineup_list),
            start_metrics_fetcher=start_metrics_fetcher,
            bullpen_metrics_fetcher=bullpen_metrics_fetcher,
            lineup_player_ids_by_date={target_day.isoformat(): lineup_player_ids},
        )

        feature_frame = _feature_rows_to_frame(_load_feature_rows(working_db_path))
    finally:
        try:
            working_db_path.unlink(missing_ok=True)
        except PermissionError:
            pass

    return _assemble_inference_rows(
        prepared_schedule,
        feature_frame=feature_frame,
        weather_database_path=Path(db_path),
        weather_fetcher=weather_fetcher,
    )


def summarize_training_data_completeness(
    source: pd.DataFrame | str | Path,
    *,
    min_rows: int = 16_500,
    max_rows: int = 17_500,
) -> TrainingDataCompletenessSummary:
    """Summarize cached training-data completeness checks without rebuilding live data."""

    if min_rows > max_rows:
        raise ValueError("min_rows must be less than or equal to max_rows")

    dataframe = _coerce_training_data_source(source)
    required_columns = {"season", "game_type", "f5_ml_result", "f5_rl_result"}
    missing_columns = sorted(required_columns.difference(dataframe.columns))
    if missing_columns:
        raise ValueError(
            "Training data source is missing required columns: " + ", ".join(missing_columns)
        )

    row_count = int(len(dataframe))
    target_null_counts = {
        column: int(pd.to_numeric(dataframe[column], errors="coerce").isna().sum())
        for column in ("f5_ml_result", "f5_rl_result")
    }
    game_type_counts = {
        str(game_type): int(count)
        for game_type, count in dataframe["game_type"].fillna("<missing>").value_counts().items()
    }
    non_regular_game_types = {
        game_type: count for game_type, count in game_type_counts.items() if game_type != "R"
    }
    seasons = tuple(
        sorted(
            {
                int(season)
                for season in pd.to_numeric(dataframe["season"], errors="coerce").dropna().tolist()
            }
        )
    )

    return TrainingDataCompletenessSummary(
        row_count=row_count,
        expected_row_range=(min_rows, max_rows),
        row_count_in_expected_range=min_rows <= row_count <= max_rows,
        target_null_counts=target_null_counts,
        game_type_counts=game_type_counts,
        non_regular_game_types=non_regular_game_types,
        seasons=seasons,
    )


def assert_training_data_is_complete(
    source: pd.DataFrame | str | Path,
    *,
    min_rows: int = 16_500,
    max_rows: int = 17_500,
) -> TrainingDataCompletenessSummary:
    """Assert cached training data satisfies the VAL-DATA-006 completeness contract."""

    summary = summarize_training_data_completeness(
        source,
        min_rows=min_rows,
        max_rows=max_rows,
    )
    errors: list[str] = []
    if not summary.row_count_in_expected_range:
        errors.append(
            f"Row count {summary.row_count} outside expected range {min_rows}-{max_rows}"
        )

    for column, null_count in summary.target_null_counts.items():
        if null_count:
            errors.append(f"Target column {column} contains {null_count} NaN values")

    if summary.non_regular_game_types:
        errors.append(f"Found non-regular game types: {summary.non_regular_game_types}")

    if errors:
        raise AssertionError("; ".join(errors))

    return summary


def assert_training_data_is_leakage_free(dataframe: pd.DataFrame) -> None:
    """Assert that every training row has an as-of timestamp strictly before scheduled start."""

    if dataframe.empty:
        return

    as_of_timestamp = pd.to_datetime(dataframe["as_of_timestamp"], utc=True, errors="coerce")
    scheduled_start = pd.to_datetime(dataframe["scheduled_start"], utc=True, errors="coerce")
    violations = dataframe.loc[as_of_timestamp >= scheduled_start, ["game_pk", "scheduled_start"]]
    if not violations.empty:
        raise AssertionError(
            f"Anti-leakage assertion failed for {len(violations)} rows: "
            f"{violations['game_pk'].tolist()[:5]}"
        )


def _fetch_regular_season_schedule(year: int) -> pd.DataFrame:
    with httpx.Client(timeout=60.0) as client:
        response = client.get(
            SCHEDULE_URL,
            params={
                "sportId": 1,
                "startDate": f"{year}-01-01",
                "endDate": f"{year}-12-31",
                "gameType": "R",
                "hydrate": "linescore,probablePitcher,venue,team",
            },
        )
        response.raise_for_status()

    payload = response.json()
    rows: list[dict[str, Any]] = []
    for date_entry in payload.get("dates", []):
        for game in date_entry.get("games", []):
            row = _schedule_game_to_row(game)
            if row is not None:
                rows.append(row)

    return pd.DataFrame(rows)


def _schedule_game_to_row(game: Mapping[str, Any]) -> dict[str, Any] | None:
    game_type = str(game.get("gameType") or "").upper()
    if game_type != "R":
        return None

    detailed_state = str(game.get("status", {}).get("detailedState") or "")
    if detailed_state.lower() not in _FINAL_GAME_STATES:
        return None

    innings = game.get("linescore", {}).get("innings") or []
    if len(innings) < 5:
        return None

    home_team = _normalize_team_code(
        game.get("teams", {}).get("home", {}).get("team", {}).get("abbreviation")
        or game.get("teams", {}).get("home", {}).get("team", {}).get("name")
    )
    away_team = _normalize_team_code(
        game.get("teams", {}).get("away", {}).get("team", {}).get("abbreviation")
        or game.get("teams", {}).get("away", {}).get("team", {}).get("name")
    )
    if home_team is None or away_team is None:
        return None

    scheduled_start = pd.Timestamp(game.get("gameDate"), tz="UTC")
    venue = str(game.get("venue", {}).get("name") or game.get("venue", {}).get("locationName") or home_team)
    park_factors = get_park_factors(team_code=home_team, venue=venue)

    f5_home_score = sum(_inning_runs(inning, "home") for inning in innings[:5])
    f5_away_score = sum(_inning_runs(inning, "away") for inning in innings[:5])
    final_home_score = _coerce_int(game.get("teams", {}).get("home", {}).get("score"))
    final_away_score = _coerce_int(game.get("teams", {}).get("away", {}).get("score"))
    if final_home_score is None or final_away_score is None:
        return None

    return {
        "game_pk": int(game["gamePk"]),
        "season": int(scheduled_start.year),
        "game_date": scheduled_start.date().isoformat(),
        "scheduled_start": scheduled_start.isoformat(),
        "home_team": home_team,
        "away_team": away_team,
        "home_starter_id": _coerce_int(
            game.get("teams", {}).get("home", {}).get("probablePitcher", {}).get("id")
        ),
        "away_starter_id": _coerce_int(
            game.get("teams", {}).get("away", {}).get("probablePitcher", {}).get("id")
        ),
        "venue": venue,
        "is_dome": bool(_SETTINGS["stadiums"].get(home_team, {}).get("is_dome", False)),
        "is_abs_active": bool(is_abs_active(venue)),
        "park_runs_factor": float(park_factors.runs),
        "park_hr_factor": float(park_factors.hr),
        "game_type": game_type,
        "status": "final",
        "f5_home_score": int(f5_home_score),
        "f5_away_score": int(f5_away_score),
        "final_home_score": int(final_home_score),
        "final_away_score": int(final_away_score),
    }


def _prepare_schedule_frame(
    dataframe: pd.DataFrame,
    *,
    require_final_scores: bool = True,
) -> pd.DataFrame:
    if dataframe.empty:
        return pd.DataFrame(
            columns=[
                "game_pk",
                "season",
                "game_date",
                "scheduled_start",
                "home_team",
                "away_team",
                "home_starter_id",
                "away_starter_id",
                "venue",
                "is_dome",
                "is_abs_active",
                "park_runs_factor",
                "park_hr_factor",
                "game_type",
                "status",
                "f5_home_score",
                "f5_away_score",
                "final_home_score",
                "final_away_score",
            ]
        )

    schedule = dataframe.copy()
    schedule = schedule.loc[schedule.get("game_type", "R").astype(str).str.upper() == "R"].copy()
    schedule["scheduled_start"] = pd.to_datetime(schedule["scheduled_start"], utc=True, errors="coerce")
    schedule["game_date"] = pd.to_datetime(schedule.get("game_date", schedule["scheduled_start"]), errors="coerce")
    schedule["season"] = pd.to_numeric(
        schedule.get("season", schedule["scheduled_start"].dt.year),
        errors="coerce",
    ).astype("Int64")
    schedule["home_team"] = schedule["home_team"].map(_normalize_team_code)
    schedule["away_team"] = schedule["away_team"].map(_normalize_team_code)
    for starter_column in ("home_starter_id", "away_starter_id"):
        if starter_column not in schedule.columns:
            schedule[starter_column] = pd.Series(pd.NA, index=schedule.index, dtype="Int64")
        else:
            schedule[starter_column] = pd.to_numeric(
                schedule[starter_column],
                errors="coerce",
            ).astype("Int64")
    if "venue" not in schedule.columns:
        schedule["venue"] = schedule["home_team"].map(
            lambda team: _SETTINGS["stadiums"].get(team, {}).get("park_name", team)
        )
    if "is_dome" not in schedule.columns:
        schedule["is_dome"] = schedule["home_team"].map(
            lambda team: bool(_SETTINGS["stadiums"].get(team, {}).get("is_dome", False))
        )
    if "is_abs_active" not in schedule.columns:
        schedule["is_abs_active"] = schedule["venue"].map(lambda venue: bool(is_abs_active(venue)))
    if "park_runs_factor" not in schedule.columns:
        schedule["park_runs_factor"] = schedule.apply(
            lambda row: get_park_factors(
                team_code=row["home_team"],
                venue=row.get("venue"),
            ).runs,
            axis=1,
        )
    if "park_hr_factor" not in schedule.columns:
        schedule["park_hr_factor"] = schedule.apply(
            lambda row: get_park_factors(
                team_code=row["home_team"],
                venue=row.get("venue"),
            ).hr,
            axis=1,
        )
    if "status" not in schedule.columns:
        schedule["status"] = "final"
    schedule["status"] = schedule["status"].map(_normalize_game_status)
    for column in (
        "f5_home_score",
        "f5_away_score",
        "final_home_score",
        "final_away_score",
        "park_runs_factor",
        "park_hr_factor",
    ):
        if column not in schedule.columns:
            schedule[column] = pd.Series(pd.NA, index=schedule.index)
        schedule[column] = pd.to_numeric(schedule[column], errors="coerce")

    schedule["is_dome"] = schedule["is_dome"].astype(bool)
    schedule["is_abs_active"] = schedule["is_abs_active"].astype(bool)
    required_columns = [
        "game_pk",
        "season",
        "scheduled_start",
        "home_team",
        "away_team",
    ]
    if require_final_scores:
        required_columns.extend(
            [
                "f5_home_score",
                "f5_away_score",
                "final_home_score",
                "final_away_score",
            ]
        )
    schedule = schedule.dropna(subset=required_columns).copy()
    schedule["game_pk"] = schedule["game_pk"].astype(int)
    schedule["season"] = schedule["season"].astype(int)
    schedule["game_date"] = schedule["game_date"].dt.date.astype(str)
    return schedule.sort_values(["scheduled_start", "game_pk"]).reset_index(drop=True)


def _compute_feature_modules(
    schedule: pd.DataFrame,
    *,
    database_path: Path,
    refresh: bool,
    batting_stats_fetcher: SeasonStatsFetcher,
    fielding_stats_fetcher: SeasonStatsFetcher,
    framing_stats_fetcher: SeasonStatsFetcher,
    team_logs_fetcher: TeamLogsFetcher,
    lineup_fetcher: LineupFetcher,
    start_metrics_fetcher: StartMetricsFetcher | None,
    bullpen_metrics_fetcher: BullpenMetricsFetcher | None,
    lineup_player_ids_by_date: LineupPlayerIdsByDate | None,
) -> FeatureBuildTimingSummary:
    normalized_lineup_player_ids = _normalize_lineup_player_ids_by_date(lineup_player_ids_by_date)
    cached_team_logs_fetcher = _memoize_dataframe_fetcher(team_logs_fetcher)
    cached_batting_stats_fetcher = _memoize_dataframe_fetcher(batting_stats_fetcher)
    cached_fielding_stats_fetcher = _memoize_dataframe_fetcher(fielding_stats_fetcher)
    cached_framing_stats_fetcher = _memoize_dataframe_fetcher(framing_stats_fetcher)
    cached_lineup_fetcher = _memoize_lineup_fetcher(lineup_fetcher)
    cached_start_metrics_fetcher = _build_cached_start_metrics_fetcher(start_metrics_fetcher)
    cached_bullpen_metrics_fetcher = _build_cached_bullpen_metrics_fetcher(
        bullpen_metrics_fetcher,
        team_logs_fetcher=cached_team_logs_fetcher,
    )

    game_dates = sorted(schedule["game_date"].astype(str).unique().tolist())
    total_days = len(game_dates)
    module_seconds = {
        "prepare": 0.0,
        "offense": 0.0,
        "pitching": 0.0,
        "defense": 0.0,
        "bullpen": 0.0,
        "baselines": 0.0,
    }
    build_started_at = perf_counter()
    for index, game_date in enumerate(game_dates, start=1):
        day_started_at = perf_counter()
        day_seconds = {
            "prepare": 0.0,
            "offense": 0.0,
            "pitching": 0.0,
            "defense": 0.0,
            "bullpen": 0.0,
            "baselines": 0.0,
        }
        logger.info("[build %s/%s] Preparing %s", index, total_days, game_date)
        stage_started_at = perf_counter()
        day_lineups = list(cached_lineup_fetcher(game_date))
        roster_turnover_by_team = _build_roster_turnover_by_team(
            game_date=game_date,
            schedule=schedule,
            lineups=day_lineups,
            lineup_player_ids=normalized_lineup_player_ids.get(game_date, {}),
            batting_stats_fetcher=cached_batting_stats_fetcher,
            start_metrics_fetcher=cached_start_metrics_fetcher,
            database_path=database_path,
            refresh=refresh,
        )
        stage_elapsed = perf_counter() - stage_started_at
        module_seconds["prepare"] += stage_elapsed
        day_seconds["prepare"] = stage_elapsed
        logger.info("[build %s/%s] offense %s", index, total_days, game_date)
        stage_started_at = perf_counter()
        compute_offensive_features(
            game_date,
            db_path=database_path,
            windows=DEFAULT_WINDOWS,
            refresh=refresh,
            lineup_player_ids=normalized_lineup_player_ids.get(game_date),
            roster_turnover_by_team=roster_turnover_by_team or None,
            team_logs_fetcher=cached_team_logs_fetcher,
            batting_stats_fetcher=cached_batting_stats_fetcher,
        )
        stage_elapsed = perf_counter() - stage_started_at
        module_seconds["offense"] += stage_elapsed
        day_seconds["offense"] = stage_elapsed
        logger.info("[build %s/%s] pitching %s", index, total_days, game_date)
        stage_started_at = perf_counter()
        compute_pitching_features(
            game_date,
            db_path=database_path,
            windows=DEFAULT_WINDOWS,
            refresh=refresh,
            roster_turnover_by_team=roster_turnover_by_team or None,
            lineup_fetcher=cached_lineup_fetcher,
            start_metrics_fetcher=cached_start_metrics_fetcher,
        )
        stage_elapsed = perf_counter() - stage_started_at
        module_seconds["pitching"] += stage_elapsed
        day_seconds["pitching"] = stage_elapsed
        logger.info("[build %s/%s] defense %s", index, total_days, game_date)
        stage_started_at = perf_counter()
        compute_defense_features(
            game_date,
            db_path=database_path,
            refresh=refresh,
            roster_turnover_by_team=roster_turnover_by_team or None,
            fielding_fetcher=cached_fielding_stats_fetcher,
            framing_fetcher=cached_framing_stats_fetcher,
            team_logs_fetcher=cached_team_logs_fetcher,
        )
        stage_elapsed = perf_counter() - stage_started_at
        module_seconds["defense"] += stage_elapsed
        day_seconds["defense"] = stage_elapsed
        logger.info("[build %s/%s] bullpen %s", index, total_days, game_date)
        stage_started_at = perf_counter()
        compute_bullpen_features(
            game_date,
            db_path=database_path,
            refresh=refresh,
            bullpen_metrics_fetcher=cached_bullpen_metrics_fetcher,
            team_logs_fetcher=cached_team_logs_fetcher,
        )
        stage_elapsed = perf_counter() - stage_started_at
        module_seconds["bullpen"] += stage_elapsed
        day_seconds["bullpen"] = stage_elapsed
        logger.info("[build %s/%s] baselines %s", index, total_days, game_date)
        stage_started_at = perf_counter()
        compute_baseline_features(
            game_date,
            db_path=database_path,
            windows=DEFAULT_PYTHAGOREAN_WINDOWS,
        )
        stage_elapsed = perf_counter() - stage_started_at
        module_seconds["baselines"] += stage_elapsed
        day_seconds["baselines"] = stage_elapsed
        logger.info(
            "[build %s/%s] complete %s timings prepare=%.2fs offense=%.2fs pitching=%.2fs defense=%.2fs bullpen=%.2fs baselines=%.2fs total=%.2fs",
            index,
            total_days,
            game_date,
            day_seconds["prepare"],
            day_seconds["offense"],
            day_seconds["pitching"],
            day_seconds["defense"],
            day_seconds["bullpen"],
            day_seconds["baselines"],
            perf_counter() - day_started_at,
        )

    return FeatureBuildTimingSummary(
        module_seconds={name: float(seconds) for name, seconds in module_seconds.items()},
        total_seconds=float(perf_counter() - build_started_at),
        processed_dates=total_days,
    )


def _resolve_effective_feature_build_workers(
    *,
    refresh_raw_data: bool,
    batting_stats_fetcher: SeasonStatsFetcher,
    fielding_stats_fetcher: SeasonStatsFetcher,
    framing_stats_fetcher: SeasonStatsFetcher,
    team_logs_fetcher: TeamLogsFetcher,
    lineup_fetcher: LineupFetcher | None,
    start_metrics_fetcher: StartMetricsFetcher | None,
    bullpen_metrics_fetcher: BullpenMetricsFetcher | None,
    lineup_player_ids_by_date: LineupPlayerIdsByDate | None,
) -> int:
    if refresh_raw_data:
        return 1
    if lineup_fetcher is not None:
        return 1
    if start_metrics_fetcher is not None or bullpen_metrics_fetcher is not None:
        return 1
    if lineup_player_ids_by_date is not None:
        return 1
    if batting_stats_fetcher is not fetch_batting_stats:
        return 1
    if fielding_stats_fetcher is not fetch_fielding_stats:
        return 1
    if framing_stats_fetcher is not fetch_catcher_framing:
        return 1
    if team_logs_fetcher is not fetch_team_game_logs:
        return 1
    return DEFAULT_FEATURE_BUILD_WORKERS


def _compute_feature_modules_parallel(
    schedule: pd.DataFrame,
    *,
    refresh: bool,
    lineup_player_ids_by_date: LineupPlayerIdsByDate | None,
    worker_count: int,
) -> _FeatureChunkResult:
    wall_started_at = perf_counter()
    game_dates = sorted(schedule["game_date"].astype(str).unique().tolist())
    if len(game_dates) <= 1 or worker_count <= 1:
        temp_fd, temp_db_name = tempfile.mkstemp(prefix="training_builder_serial_", suffix=".db")
        os.close(temp_fd)
        working_db_path = Path(temp_db_name)
        try:
            working_db_path = init_db(working_db_path)
            _seed_games_table(working_db_path, schedule)
            timing_summary = _compute_feature_modules(
                schedule,
                database_path=working_db_path,
                refresh=refresh,
                batting_stats_fetcher=fetch_batting_stats,
                fielding_stats_fetcher=fetch_fielding_stats,
                framing_stats_fetcher=fetch_catcher_framing,
                team_logs_fetcher=fetch_team_game_logs,
                lineup_fetcher=_empty_lineups_fetcher,
                start_metrics_fetcher=None,
                bullpen_metrics_fetcher=None,
                lineup_player_ids_by_date=lineup_player_ids_by_date,
            )
            feature_rows = _load_feature_rows(working_db_path)
            return _FeatureChunkResult(feature_rows=feature_rows, timing_summary=timing_summary)
        finally:
            try:
                working_db_path.unlink(missing_ok=True)
            except PermissionError:
                pass

    chunked_dates = _chunk_values(game_dates, DEFAULT_FEATURE_BUILD_CHUNK_DAYS)
    requested_workers = min(worker_count, len(chunked_dates))
    results: list[_FeatureChunkResult] = []
    with ProcessPoolExecutor(max_workers=requested_workers) as executor:
        futures = [
            executor.submit(
                _build_feature_chunk,
                schedule,
                tuple(chunk_dates),
                refresh,
                _normalize_lineup_player_ids_by_date(lineup_player_ids_by_date),
            )
            for chunk_dates in chunked_dates
        ]
        for future in as_completed(futures):
            results.append(future.result())

    feature_rows = pd.concat([result.feature_rows for result in results], ignore_index=True)
    timing_summary = _combine_timing_summaries([result.timing_summary for result in results])
    timing_summary = FeatureBuildTimingSummary(
        module_seconds=timing_summary.module_seconds,
        total_seconds=float(perf_counter() - wall_started_at),
        processed_dates=timing_summary.processed_dates,
    )
    return _FeatureChunkResult(feature_rows=feature_rows, timing_summary=timing_summary)


def _build_feature_chunk(
    schedule: pd.DataFrame,
    chunk_dates: tuple[str, ...],
    refresh: bool,
    lineup_player_ids_by_date: dict[str, Mapping[tuple[int, str], Sequence[int]]],
) -> _FeatureChunkResult:
    chunk_schedule = schedule.loc[schedule["game_date"].astype(str).isin(chunk_dates)].copy()
    temp_fd, temp_db_name = tempfile.mkstemp(prefix="training_builder_chunk_", suffix=".db")
    os.close(temp_fd)
    working_db_path = Path(temp_db_name)
    try:
        working_db_path = init_db(working_db_path)
        _seed_games_table(working_db_path, schedule)
        timing_summary = _compute_feature_modules(
            chunk_schedule,
            database_path=working_db_path,
            refresh=refresh,
            batting_stats_fetcher=fetch_batting_stats,
            fielding_stats_fetcher=fetch_fielding_stats,
            framing_stats_fetcher=fetch_catcher_framing,
            team_logs_fetcher=fetch_team_game_logs,
            lineup_fetcher=_empty_lineups_fetcher,
            start_metrics_fetcher=None,
            bullpen_metrics_fetcher=None,
            lineup_player_ids_by_date=lineup_player_ids_by_date,
        )
        return _FeatureChunkResult(
            feature_rows=_load_feature_rows(working_db_path),
            timing_summary=timing_summary,
        )
    finally:
        try:
            working_db_path.unlink(missing_ok=True)
        except PermissionError:
            pass


def _chunk_values(values: Sequence[str], chunk_size: int) -> list[list[str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [list(values[index : index + chunk_size]) for index in range(0, len(values), chunk_size)]


def _combine_timing_summaries(
    summaries: Sequence[FeatureBuildTimingSummary],
) -> FeatureBuildTimingSummary:
    combined_module_seconds: dict[str, float] = {}
    total_seconds = 0.0
    processed_dates = 0
    for summary in summaries:
        total_seconds += float(summary.total_seconds)
        processed_dates += int(summary.processed_dates)
        for module_name, seconds in summary.module_seconds.items():
            combined_module_seconds[module_name] = combined_module_seconds.get(module_name, 0.0) + float(seconds)
    return FeatureBuildTimingSummary(
        module_seconds=combined_module_seconds,
        total_seconds=total_seconds,
        processed_dates=processed_dates,
    )


def _log_feature_build_timing_summary(summary: FeatureBuildTimingSummary, *, label: str) -> None:
    logger.info(
        "[build] Timing summary (%s) prepare=%.2fs offense=%.2fs pitching=%.2fs defense=%.2fs bullpen=%.2fs baselines=%.2fs total=%.2fs dates=%s",
        label,
        summary.module_seconds.get("prepare", 0.0),
        summary.module_seconds.get("offense", 0.0),
        summary.module_seconds.get("pitching", 0.0),
        summary.module_seconds.get("defense", 0.0),
        summary.module_seconds.get("bullpen", 0.0),
        summary.module_seconds.get("baselines", 0.0),
        summary.total_seconds,
        summary.processed_dates,
    )


def _build_roster_turnover_by_team(
    *,
    game_date: str,
    schedule: pd.DataFrame,
    lineups: Sequence[Lineup],
    lineup_player_ids: Mapping[tuple[int, str], Sequence[int]],
    batting_stats_fetcher: SeasonStatsFetcher,
    start_metrics_fetcher: StartMetricsFetcher,
    database_path: Path,
    refresh: bool,
) -> dict[str, float]:
    target_day = _coerce_date(game_date)
    day_schedule = schedule.loc[schedule["game_date"].astype(str) == target_day.isoformat()].copy()
    if day_schedule.empty:
        return {}

    current_rosters: dict[str, set[int]] = {}
    teams_with_lineup_history: set[str] = set()
    for game in day_schedule.to_dict(orient="records"):
        for team_key in ("home_team", "away_team"):
            current_rosters.setdefault(str(game[team_key]), set())

    for (game_pk, team), player_ids in lineup_player_ids.items():
        _ = game_pk
        if team not in current_rosters:
            continue
        normalized_player_ids = {
            player_id
            for player_id in (_coerce_int(player_id) for player_id in player_ids)
            if player_id is not None
        }
        if normalized_player_ids:
            teams_with_lineup_history.add(team)
            current_rosters[team].update(normalized_player_ids)

    lineup_lookup = {
        (int(lineup.game_pk), str(lineup.team)): lineup
        for lineup in lineups
    }
    for game in day_schedule.to_dict(orient="records"):
        game_pk = int(game["game_pk"])
        for team_key, starter_key in (("home_team", "home_starter_id"), ("away_team", "away_starter_id")):
            team = str(game[team_key])
            lineup = lineup_lookup.get((game_pk, team))
            starter_id = None
            if lineup is not None:
                lineup_player_set = {
                    player.player_id for player in lineup.players if player.player_id is not None
                }
                if lineup_player_set:
                    teams_with_lineup_history.add(team)
                    current_rosters.setdefault(team, set()).update(lineup_player_set)
                starter_id = lineup.starting_pitcher_id or lineup.projected_starting_pitcher_id

            if starter_id is None:
                starter_id = _coerce_int(game.get(starter_key))
            if starter_id is not None:
                current_rosters.setdefault(team, set()).add(int(starter_id))

    prior_rosters = _extract_batting_rosters_by_team(
        batting_stats_fetcher(target_day.year - 1, min_pa=0, refresh=refresh)
    )
    prior_pitcher_rosters = _extract_pitcher_rosters_by_team(
        start_metrics_fetcher(
            target_day.year - 1,
            db_path=database_path,
            end_date=None,
            refresh=refresh,
        )
    )
    for team, pitcher_ids in prior_pitcher_rosters.items():
        prior_rosters.setdefault(team, set()).update(pitcher_ids)

    turnover_by_team: dict[str, float] = {}
    for team, current_ids in current_rosters.items():
        if team not in teams_with_lineup_history or not current_ids:
            continue

        prior_ids = prior_rosters.get(team)
        if not prior_ids:
            continue

        retained_count = len(current_ids.intersection(prior_ids))
        turnover_by_team[team] = max(0.0, 1.0 - (retained_count / len(current_ids)))

    return turnover_by_team


def _extract_batting_rosters_by_team(dataframe: pd.DataFrame) -> dict[str, set[int]]:
    if dataframe.empty:
        return {}

    team_column = _first_existing_column(dataframe, ("Team", "team", "Tm"))
    player_id_column = _first_existing_column(
        dataframe,
        ("player_id", "playerid", "key_mlbam", "mlb_id", "ID", "id"),
    )
    if team_column is None or player_id_column is None:
        return {}

    rosters: dict[str, set[int]] = {}
    for team_value, player_id_value in dataframe[[team_column, player_id_column]].itertuples(index=False):
        team = _normalize_team_code(team_value)
        player_id = _coerce_int(player_id_value)
        if team is None or player_id is None:
            continue
        rosters.setdefault(team, set()).add(player_id)

    return rosters


def _extract_pitcher_rosters_by_team(dataframe: pd.DataFrame) -> dict[str, set[int]]:
    if dataframe.empty:
        return {}

    team_column = _first_existing_column(dataframe, ("team", "Team"))
    pitcher_id_column = _first_existing_column(
        dataframe,
        ("pitcher_id", "pitcher", "player_id", "ID", "id"),
    )
    if team_column is None or pitcher_id_column is None:
        return {}

    rosters: dict[str, set[int]] = {}
    for team_value, pitcher_id_value in dataframe[[team_column, pitcher_id_column]].itertuples(index=False):
        team = _normalize_team_code(team_value)
        pitcher_id = _coerce_int(pitcher_id_value)
        if team is None or pitcher_id is None:
            continue
        rosters.setdefault(team, set()).add(pitcher_id)

    return rosters


def _seed_games_table(database_path: Path, schedule: pd.DataFrame) -> None:
    rows = [
        (
            int(game["game_pk"]),
            pd.Timestamp(game["scheduled_start"]).isoformat(),
            str(game["home_team"]),
            str(game["away_team"]),
            _coerce_int(game.get("home_starter_id")),
            _coerce_int(game.get("away_starter_id")),
            str(game["venue"]),
            int(bool(game["is_dome"])),
            int(bool(game["is_abs_active"])),
            _coerce_int(game.get("f5_home_score")),
            _coerce_int(game.get("f5_away_score")),
            _coerce_int(game.get("final_home_score")),
            _coerce_int(game.get("final_away_score")),
            _normalize_game_status(game.get("status")),
        )
        for game in schedule.to_dict(orient="records")
    ]

    with sqlite_connection(database_path, builder_optimized=True) as connection:
        connection.executemany(
            """
            INSERT OR REPLACE INTO games (
                game_pk,
                date,
                home_team,
                away_team,
                home_starter_id,
                away_starter_id,
                venue,
                is_dome,
                is_abs_active,
                f5_home_score,
                f5_away_score,
                final_home_score,
                final_away_score,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()


def _load_feature_rows(database_path: Path) -> pd.DataFrame:
    with sqlite_connection(database_path, builder_optimized=True) as connection:
        feature_rows = pd.read_sql_query(
            """
            SELECT game_pk, feature_name, feature_value
            FROM features
            ORDER BY game_pk, feature_name
            """,
            connection,
        )

    return feature_rows


def _feature_rows_to_frame(feature_rows: pd.DataFrame) -> pd.DataFrame:
    if feature_rows.empty:
        return pd.DataFrame(columns=["game_pk"])

    feature_frame = feature_rows.pivot_table(
        index="game_pk",
        columns="feature_name",
        values="feature_value",
        aggfunc="last",
    ).reset_index()
    feature_frame.columns.name = None
    return feature_frame


def _assemble_training_rows(
    schedule: pd.DataFrame,
    *,
    feature_frame: pd.DataFrame,
    database_path: Path,
    weather_fetcher: WeatherFetcher | None,
) -> pd.DataFrame:
    feature_lookup = (
        feature_frame.set_index("game_pk")
        if not feature_frame.empty and "game_pk" in feature_frame.columns
        else pd.DataFrame(index=pd.Index([], name="game_pk"))
    )
    rows: list[dict[str, Any]] = []

    for game in schedule.to_dict(orient="records"):
        game_pk = int(game["game_pk"])
        game_start = pd.Timestamp(game["scheduled_start"])
        if game_start.tzinfo is None:
            game_start = game_start.tz_localize("UTC")
        else:
            game_start = game_start.tz_convert("UTC")
        as_of_timestamp = (game_start.normalize() - pd.Timedelta(days=1)).isoformat()

        row: dict[str, Any] = {
            "game_pk": game_pk,
            "season": int(game["season"]),
            "game_date": str(game["game_date"]),
            "scheduled_start": game_start.isoformat(),
            "as_of_timestamp": as_of_timestamp,
            "home_team": str(game["home_team"]),
            "away_team": str(game["away_team"]),
            "venue": str(game["venue"]),
            "game_type": str(game["game_type"]),
            "status": _normalize_game_status(game.get("status")),
        }
        row.update(_schedule_adjustment_features(game))
        row.update(
            _resolve_weather_features(
                game,
                database_path=database_path,
                weather_fetcher=weather_fetcher,
            )
        )

        if game_pk in feature_lookup.index:
            feature_values = feature_lookup.loc[game_pk]
            if isinstance(feature_values, pd.DataFrame):
                feature_values = feature_values.iloc[-1]
            for feature_name, feature_value in feature_values.items():
                if pd.notna(feature_value):
                    row[str(feature_name)] = float(feature_value)

        f5_home_score = int(game["f5_home_score"])
        f5_away_score = int(game["f5_away_score"])
        final_home_score = int(game["final_home_score"])
        final_away_score = int(game["final_away_score"])
        row.update(
            {
                "f5_home_score": f5_home_score,
                "f5_away_score": f5_away_score,
                "final_home_score": final_home_score,
                "final_away_score": final_away_score,
                "f5_margin": float(f5_home_score - f5_away_score),
                "final_margin": float(final_home_score - final_away_score),
                "f5_tied_after_5": int(f5_home_score == f5_away_score),
                "f5_ml_result": int(f5_home_score > f5_away_score),
                "f5_rl_result": int((f5_home_score - f5_away_score) >= 2),
            }
        )
        rows.append(row)

    dataset = pd.DataFrame(rows).sort_values(["scheduled_start", "game_pk"]).reset_index(drop=True)
    return _fill_missing_feature_values(dataset)


def _assemble_inference_rows(
    schedule: pd.DataFrame,
    *,
    feature_frame: pd.DataFrame,
    weather_database_path: Path,
    weather_fetcher: WeatherFetcher | None,
) -> pd.DataFrame:
    feature_lookup = (
        feature_frame.set_index("game_pk")
        if not feature_frame.empty and "game_pk" in feature_frame.columns
        else pd.DataFrame(index=pd.Index([], name="game_pk"))
    )
    rows: list[dict[str, Any]] = []

    for game in schedule.to_dict(orient="records"):
        game_pk = int(game["game_pk"])
        game_start = pd.Timestamp(game["scheduled_start"])
        if game_start.tzinfo is None:
            game_start = game_start.tz_localize("UTC")
        else:
            game_start = game_start.tz_convert("UTC")

        row: dict[str, Any] = {
            "game_pk": game_pk,
            "season": int(game["season"]),
            "game_date": str(game["game_date"]),
            "scheduled_start": game_start.isoformat(),
            "as_of_timestamp": (game_start.normalize() - pd.Timedelta(days=1)).isoformat(),
            "home_team": str(game["home_team"]),
            "away_team": str(game["away_team"]),
            "venue": str(game["venue"]),
            "game_type": str(game.get("game_type", "R")),
            "status": _normalize_game_status(game.get("status")),
        }
        row.update(_schedule_adjustment_features(game))
        row.update(
            _resolve_weather_features(
                game,
                database_path=weather_database_path,
                weather_fetcher=weather_fetcher,
            )
        )

        if game_pk in feature_lookup.index:
            feature_values = feature_lookup.loc[game_pk]
            if isinstance(feature_values, pd.DataFrame):
                feature_values = feature_values.iloc[-1]
            for feature_name, feature_value in feature_values.items():
                if pd.notna(feature_value):
                    row[str(feature_name)] = float(feature_value)

        rows.append(row)

    inference_frame = pd.DataFrame(rows).sort_values(["scheduled_start", "game_pk"]).reset_index(drop=True)
    return _fill_missing_feature_values(inference_frame)


def _schedule_adjustment_features(game: Mapping[str, Any]) -> dict[str, float]:
    abs_adjustment = apply_abs_adjustments(
        0.0,
        0.0,
        venue=str(game.get("venue")),
        abs_active=bool(game.get("is_abs_active", True)),
    )
    return {
        "park_runs_factor": float(game["park_runs_factor"]),
        "park_hr_factor": float(game["park_hr_factor"]),
        "abs_active": float(abs_adjustment.abs_active),
        "abs_walk_rate_delta": float(abs_adjustment.walk_rate_delta),
        "abs_strikeout_rate_delta": float(abs_adjustment.strikeout_rate_delta),
    }


def _resolve_weather_features(
    game: Mapping[str, Any],
    *,
    database_path: Path,
    weather_fetcher: WeatherFetcher | None,
) -> dict[str, float]:
    is_dome = bool(game.get("is_dome", False))
    weather_missing = 0.0

    if weather_fetcher is None:
        weather = _get_default_weather(is_dome=is_dome)
        weather_missing = 1.0
    else:
        try:
            weather = _call_weather_fetcher(
                weather_fetcher,
                team_abbr=str(game["home_team"]),
                game_datetime=game["scheduled_start"],
                database_path=database_path,
            )
        except Exception:
            weather = None
        if weather is None:
            weather = _get_default_weather(is_dome=is_dome)
            weather_missing = 1.0

    adjustment = compute_weather_adjustment(
        weather,
        team_code=str(game["home_team"]),
        venue=str(game["venue"]),
        is_dome=is_dome,
    )
    return {
        "weather_temp_factor": float(adjustment.temp_factor),
        "weather_air_density_factor": float(adjustment.air_density_factor),
        "weather_humidity_factor": float(adjustment.humidity_factor),
        "weather_wind_factor": float(adjustment.wind_factor),
        "weather_rain_risk": float(adjustment.rain_risk),
        "weather_composite": float(adjustment.weather_composite),
        "weather_data_missing": float(weather_missing),
    }


def _call_weather_fetcher(
    weather_fetcher: WeatherFetcher,
    *,
    team_abbr: str,
    game_datetime: str | datetime,
    database_path: Path,
) -> WeatherData | None:
    try:
        return weather_fetcher(team_abbr, game_datetime, db_path=database_path)
    except TypeError:
        return weather_fetcher(team_abbr, game_datetime)


def _fill_missing_feature_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe

    dataset = dataframe.copy()
    for column in _feature_columns(dataset):
        numeric_values = pd.to_numeric(dataset[column], errors="coerce")
        if not numeric_values.isna().any():
            dataset[column] = numeric_values
            continue

        fill_value = _default_feature_fill_value(column)
        dataset[column] = numeric_values.fillna(fill_value)

    return dataset


def _default_feature_fill_value(column: str) -> float:
    if column in _SCHEDULE_FEATURE_DEFAULTS:
        return float(_SCHEDULE_FEATURE_DEFAULTS[column])
    if column in _WEATHER_FEATURE_DEFAULTS:
        return float(_WEATHER_FEATURE_DEFAULTS[column])
    if "_starter_" in column:
        return _resolve_pattern_default(column, _PITCHING_FEATURE_DEFAULTS)
    if "_team_bullpen_" in column:
        return _resolve_pattern_default(column, _BULLPEN_FEATURE_DEFAULTS)
    if "_pythagorean_wp_" in column or "_log5_" in column:
        return 0.5
    if _matches_feature_pattern(column, _DEFENSE_FEATURE_DEFAULTS):
        return _resolve_pattern_default(column, _DEFENSE_FEATURE_DEFAULTS)
    if "_lineup_" in column or _matches_feature_pattern(column, _OFFENSE_FEATURE_DEFAULTS):
        return _resolve_pattern_default(column, _OFFENSE_FEATURE_DEFAULTS)
    return 0.0


def _matches_feature_pattern(column: str, defaults: Mapping[str, float]) -> bool:
    return any(f"_{metric}" in column for metric in defaults)


def _resolve_pattern_default(column: str, defaults: Mapping[str, float]) -> float:
    for metric, default_value in defaults.items():
        if f"_{metric}" in column:
            return float(default_value)
    return 0.0


def _normalize_lineup_player_ids_by_date(
    mapping: LineupPlayerIdsByDate | None,
) -> dict[str, Mapping[tuple[int, str], Sequence[int]]]:
    if not mapping:
        return {}

    normalized: dict[str, Mapping[tuple[int, str], Sequence[int]]] = {}
    for game_date, lineup_ids in mapping.items():
        normalized[_normalize_lookup_date(game_date)] = lineup_ids
    return normalized


def _load_historical_lineup_player_ids_by_date(
    schedule: pd.DataFrame,
    *,
    refresh: bool,
) -> dict[str, dict[tuple[int, str], list[int]]]:
    if schedule.empty:
        return {}

    retrosheet_lineups = fetch_retrosheet_starting_lineups(refresh=refresh)
    if retrosheet_lineups.empty:
        return {}

    normalized_lineups = _normalize_retrosheet_lineups(retrosheet_lineups, schedule)
    if normalized_lineups.empty:
        return {}

    retrosheet_player_ids = sorted(
        {
            str(player_id).strip().lower()
            for column in [f"start_l{index}" for index in range(1, 10)]
            if column in normalized_lineups.columns
            for player_id in normalized_lineups[column].dropna().tolist()
            if str(player_id).strip()
        }
    )
    if not retrosheet_player_ids:
        return {}

    register = fetch_chadwick_register(refresh=refresh)
    if register.empty or "key_retro" not in register.columns or "key_mlbam" not in register.columns:
        return {}

    retrosheet_to_mlbam: dict[str, int] = {}
    id_pairs = register[["key_retro", "key_mlbam"]].dropna().drop_duplicates()
    for retrosheet_id, mlbam_id in id_pairs.itertuples(index=False):
        normalized_retro_id = str(retrosheet_id).strip().lower()
        normalized_mlbam_id = _coerce_int(mlbam_id)
        if normalized_retro_id and normalized_mlbam_id is not None:
            retrosheet_to_mlbam.setdefault(normalized_retro_id, int(normalized_mlbam_id))

    if not retrosheet_to_mlbam:
        return {}

    lineup_mapping: dict[str, dict[tuple[int, str], list[int]]] = {}
    lineup_columns = [column for column in [f"start_l{index}" for index in range(1, 10)] if column in normalized_lineups.columns]

    for row in normalized_lineups.to_dict(orient="records"):
        game_pk = int(row["game_pk"])
        team = str(row["team"])
        game_date = str(row["game_date"])
        player_ids = [
            retrosheet_to_mlbam[normalized_retro_id]
            for column in lineup_columns
            if (normalized_retro_id := str(row.get(column, "")).strip().lower()) in retrosheet_to_mlbam
        ]
        if not player_ids:
            continue
        lineup_mapping.setdefault(game_date, {})[(game_pk, team)] = player_ids

    return lineup_mapping


def _normalize_retrosheet_lineups(
    retrosheet_lineups: pd.DataFrame,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    if retrosheet_lineups.empty or schedule.empty:
        return pd.DataFrame()

    lineups = retrosheet_lineups.copy()
    if "date" not in lineups.columns or "team" not in lineups.columns or "opp" not in lineups.columns:
        return pd.DataFrame()
    if "vishome" not in lineups.columns:
        return pd.DataFrame()

    lineups["game_date"] = pd.to_datetime(lineups["date"], format="%Y%m%d", errors="coerce").dt.date.astype(str)
    lineups["team"] = lineups["team"].map(_normalize_team_code)
    lineups["opp"] = lineups["opp"].map(_normalize_team_code)
    lineups["vishome"] = lineups["vishome"].astype(str).str.lower()
    lineups = lineups.dropna(subset=["game_date", "team", "opp"])
    lineups = lineups.loc[lineups["vishome"].isin(["h", "v"])].copy()

    schedule_side_rows: list[dict[str, Any]] = []
    for game in schedule.to_dict(orient="records"):
        schedule_side_rows.append(
            {
                "game_pk": int(game["game_pk"]),
                "game_date": str(game["game_date"]),
                "team": str(game["home_team"]),
                "opp": str(game["away_team"]),
                "vishome": "h",
                "scheduled_start": str(game["scheduled_start"]),
            }
        )
        schedule_side_rows.append(
            {
                "game_pk": int(game["game_pk"]),
                "game_date": str(game["game_date"]),
                "team": str(game["away_team"]),
                "opp": str(game["home_team"]),
                "vishome": "v",
                "scheduled_start": str(game["scheduled_start"]),
            }
        )

    schedule_lookup = pd.DataFrame(schedule_side_rows)
    schedule_lookup = schedule_lookup.sort_values(
        ["game_date", "team", "opp", "vishome", "scheduled_start", "game_pk"]
    ).reset_index(drop=True)
    schedule_lookup["matchup_index"] = schedule_lookup.groupby(
        ["game_date", "team", "opp", "vishome"]
    ).cumcount()

    sort_columns = ["game_date", "team", "opp", "vishome"]
    if "number" in lineups.columns:
        sort_columns.append("number")
    if "gid" in lineups.columns:
        sort_columns.append("gid")
    lineups = lineups.sort_values(sort_columns).reset_index(drop=True)
    lineups["matchup_index"] = lineups.groupby(["game_date", "team", "opp", "vishome"]).cumcount()

    merged = lineups.merge(
        schedule_lookup[["game_pk", "game_date", "team", "opp", "vishome", "matchup_index"]],
        on=["game_date", "team", "opp", "vishome", "matchup_index"],
        how="inner",
    )
    return merged


def _lineup_player_ids_by_game_team(lineups: Sequence[Lineup]) -> dict[tuple[int, str], list[int]]:
    lookup: dict[tuple[int, str], list[int]] = {}
    for lineup in lineups:
        player_ids = [player.player_id for player in lineup.players]
        lookup[(lineup.game_pk, lineup.team)] = player_ids
    return lookup


def _static_lineup_fetcher(target_day: date, lineups: Sequence[Lineup]) -> LineupFetcher:
    cached_lineups = list(lineups)

    def fetcher(game_date: str | date | datetime) -> list[Lineup]:
        if _normalize_lookup_date(game_date) != target_day.isoformat():
            return []
        return list(cached_lineups)

    return fetcher


def _normalize_lookup_date(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _first_existing_column(dataframe: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate
    return None


def _coerce_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _normalize_scheduled_start_before(value: str | date | datetime) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp


def _memoize_dataframe_fetcher(fetcher: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
    cache: dict[tuple[Any, ...], pd.DataFrame] = {}

    def wrapper(*args: Any, **kwargs: Any) -> pd.DataFrame:
        key = _cache_key(*args, **kwargs)
        if key not in cache:
            cache[key] = _coerce_dataframe(fetcher(*args, **kwargs))
        return cache[key].copy()

    return wrapper


def _memoize_lineup_fetcher(fetcher: LineupFetcher) -> LineupFetcher:
    cache: dict[tuple[Any, ...], list[Lineup]] = {}

    def wrapper(game_date: str | date | datetime) -> list[Lineup]:
        key = _cache_key(game_date)
        if key not in cache:
            cache[key] = list(fetcher(game_date))
        return list(cache[key])

    return wrapper


def _build_cached_start_metrics_fetcher(
    start_metrics_fetcher: StartMetricsFetcher | None,
) -> StartMetricsFetcher:
    full_season_cache: dict[tuple[Any, ...], pd.DataFrame] = {}

    def wrapper(
        season: int,
        *,
        db_path: str | Path,
        end_date: date | None,
        refresh: bool = False,
    ) -> pd.DataFrame:
        key = (season, _cache_token(db_path), bool(refresh))
        if key not in full_season_cache:
            if start_metrics_fetcher is None:
                full_season_cache[key] = _coerce_dataframe(
                    _fetch_season_start_metrics(
                        season,
                        db_path=db_path,
                        end_date=None,
                        refresh=refresh,
                    )
                )
            else:
                full_season_cache[key] = _coerce_dataframe(
                    start_metrics_fetcher(
                        season,
                        db_path=db_path,
                        end_date=None,
                        refresh=refresh,
                    )
                )
        return _filter_frame_to_end_date(full_season_cache[key], end_date)

    return wrapper


def _build_cached_bullpen_metrics_fetcher(
    bullpen_metrics_fetcher: BullpenMetricsFetcher | None,
    *,
    team_logs_fetcher: TeamLogsFetcher,
) -> BullpenMetricsFetcher:
    full_season_cache: dict[tuple[Any, ...], pd.DataFrame] = {}

    def wrapper(
        season: int,
        *,
        db_path: str | Path,
        end_date: date | None,
        refresh: bool = False,
    ) -> pd.DataFrame:
        key = (season, _cache_token(db_path), bool(refresh))
        if key not in full_season_cache:
            if bullpen_metrics_fetcher is None:
                full_season_cache[key] = _coerce_dataframe(
                    _fetch_season_bullpen_metrics(
                        season,
                        db_path=Path(db_path),
                        end_date=date(season, 12, 31),
                        refresh=refresh,
                        team_logs_fetcher=team_logs_fetcher,
                    )
                )
            else:
                full_season_cache[key] = _coerce_dataframe(
                    bullpen_metrics_fetcher(
                        season,
                        db_path=db_path,
                        end_date=None,
                        refresh=refresh,
                    )
                )
        return _filter_frame_to_end_date(full_season_cache[key], end_date)

    return wrapper


def _filter_frame_to_end_date(dataframe: pd.DataFrame, end_date: date | None) -> pd.DataFrame:
    if dataframe.empty or end_date is None:
        return dataframe.copy()

    date_column = "game_date" if "game_date" in dataframe.columns else "date" if "date" in dataframe.columns else None
    if date_column is None:
        return dataframe.copy()

    game_dates = pd.to_datetime(dataframe[date_column], errors="coerce")
    if getattr(game_dates.dt, "tz", None) is not None:
        game_dates = game_dates.dt.tz_localize(None)
    filtered = dataframe.loc[game_dates.dt.date <= end_date].copy()
    return filtered.reset_index(drop=True)


def _coerce_dataframe(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return pd.DataFrame()


def _coerce_training_data_source(source: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if isinstance(source, (str, Path)):
        return pd.read_parquet(Path(source))
    raise TypeError("source must be a pandas DataFrame or parquet path")


def _empty_lineups_fetcher(_game_date: str | date | datetime) -> list[Lineup]:
    return []


def _cache_key(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
    return (
        tuple(_cache_token(value) for value in args),
        tuple(sorted((key, _cache_token(value)) for key, value in kwargs.items())),
    )


def _cache_token(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def _assert_targets_present(dataframe: pd.DataFrame) -> None:
    if dataframe[["f5_ml_result", "f5_rl_result"]].isna().any().any():
        raise AssertionError("Target columns must not contain NaN values")


def _compute_data_version_hash(dataframe: pd.DataFrame) -> str:
    stable_columns = [
        column
        for column in dataframe.columns
        if column not in {"build_timestamp", "data_version_hash"}
    ]
    stable_payload = dataframe[stable_columns].sort_values(["scheduled_start", "game_pk"])
    encoded = stable_payload.to_json(orient="records", date_format="iso").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _feature_columns(dataframe: pd.DataFrame) -> list[str]:
    non_feature_columns = {
        "game_pk",
        "season",
        "game_date",
        "scheduled_start",
        "as_of_timestamp",
        "build_timestamp",
        "data_version_hash",
        "home_team",
        "away_team",
        "venue",
        "game_type",
        "status",
        "f5_home_score",
        "f5_away_score",
        "final_home_score",
        "final_away_score",
        "f5_margin",
        "final_margin",
        "f5_tied_after_5",
        "f5_ml_result",
        "f5_rl_result",
    }
    return [column for column in dataframe.columns if column not in non_feature_columns]


def _normalize_team_code(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip().upper()
    if not text or text in {"TOT", "- - -"}:
        return None
    if text in _TEAM_CODES:
        return text
    if text in _TEAM_CODE_ALIASES:
        return _TEAM_CODE_ALIASES[text]
    return TEAM_LABEL_TO_CODE.get(text)


def _normalize_game_status(value: Any) -> str:
    normalized = str(value or "final").strip().lower()
    if normalized in _FINAL_GAME_STATES:
        return "final"
    if normalized in {"scheduled", "preview", "pre-game"}:
        return "scheduled"
    if normalized in {"suspended"}:
        return "suspended"
    if normalized in {"postponed", "postponed due to rain"}:
        return "postponed"
    if normalized in {"cancelled", "canceled"}:
        return "cancelled"
    return "final"


def _inning_runs(inning: Mapping[str, Any], side: str) -> int:
    return int(inning.get(side, {}).get("runs") or 0)


def _coerce_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
