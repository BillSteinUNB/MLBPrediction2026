from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
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
from src.clients.weather_client import _get_default_weather
from src.config import _load_settings_yaml
from src.db import init_db
from src.features.adjustments.abs_adjustment import (
    apply_abs_adjustments,
    is_abs_active,
)
from src.features.adjustments.park_factors import get_park_factors
from src.features.adjustments.weather import compute_weather_adjustment
from src.features.baselines import compute_baseline_features
from src.features.bullpen import compute_bullpen_features, _fetch_season_bullpen_metrics
from src.features.defense import compute_defense_features
from src.features.offense import compute_offensive_features
from src.features.pitching import compute_pitching_features, _fetch_season_start_metrics
from src.models.lineup import Lineup
from src.models.weather import WeatherData


DEFAULT_OUTPUT_PATH = Path("data") / "training" / "training_data_2019_2025.parquet"
DEFAULT_WINDOWS: tuple[int, ...] = (7, 14, 30, 60)
DEFAULT_PYTHAGOREAN_WINDOWS: tuple[int, ...] = (30, 60)
DEFAULT_FULL_REGULAR_SEASONS_TARGET = 7
SHORTENED_SEASON_GAME_THRESHOLD = 2_000
SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"

_SETTINGS = _load_settings_yaml()
_TEAM_CODES = set(_SETTINGS["teams"].keys())
_TEAM_CODE_ALIASES = {value: key for key, value in TEAM_GAME_LOG_CODES.items()}
_TEAM_CODE_ALIASES.update(
    {
        "AZ": "ARI",
        "ATH": "OAK",
        "CHW": "CWS",
        "KCR": "KC",
        "SDP": "SD",
        "SFG": "SF",
        "TBR": "TB",
        "WSN": "WSH",
    }
)
_FINAL_GAME_STATES = {"final", "game over", "completed early"}


ScheduleFetcher = Callable[[int], pd.DataFrame]
SeasonStatsFetcher = Callable[..., pd.DataFrame]
TeamLogsFetcher = Callable[..., pd.DataFrame]
LineupFetcher = Callable[[str | date | datetime], Sequence[Lineup]]
StartMetricsFetcher = Callable[..., pd.DataFrame]
BullpenMetricsFetcher = Callable[..., pd.DataFrame]
WeatherFetcher = Callable[..., WeatherData | None]
LineupPlayerIdsByDate = Mapping[str | date | datetime, Mapping[tuple[int, str], Sequence[int]]]


@dataclass(frozen=True, slots=True)
class TrainingDataBuildResult:
    dataframe: pd.DataFrame
    output_path: Path
    metadata_path: Path
    data_version_hash: str
    build_timestamp: datetime
    requested_years: tuple[int, ...]
    effective_years: tuple[int, ...]


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
    refresh: bool = False,
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

    build_timestamp = datetime.now(UTC)
    resolved_lineup_fetcher = lineup_fetcher or _empty_lineups_fetcher

    temp_fd, temp_db_name = tempfile.mkstemp(prefix="training_builder_", suffix=".db")
    os.close(temp_fd)
    working_db_path = Path(temp_db_name)
    try:
        working_db_path = init_db(working_db_path)
        _seed_games_table(working_db_path, schedule)
        _compute_feature_modules(
            schedule,
            database_path=working_db_path,
            refresh=refresh,
            batting_stats_fetcher=batting_stats_fetcher,
            fielding_stats_fetcher=fielding_stats_fetcher,
            framing_stats_fetcher=framing_stats_fetcher,
            team_logs_fetcher=team_logs_fetcher,
            lineup_fetcher=resolved_lineup_fetcher,
            start_metrics_fetcher=start_metrics_fetcher,
            bullpen_metrics_fetcher=bullpen_metrics_fetcher,
            lineup_player_ids_by_date=lineup_player_ids_by_date,
        )
        feature_frame = _load_feature_frame(working_db_path)
        dataset = _assemble_training_rows(
            schedule,
            feature_frame=feature_frame,
            database_path=working_db_path,
            weather_fetcher=weather_fetcher,
        )
    finally:
        try:
            working_db_path.unlink(missing_ok=True)
        except PermissionError:
            pass

    assert_training_data_is_leakage_free(dataset)
    _assert_targets_present(dataset)

    data_version_hash = _compute_data_version_hash(dataset)
    dataset = dataset.copy()
    dataset["data_version_hash"] = data_version_hash
    dataset["build_timestamp"] = build_timestamp.isoformat()

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
                "row_count": int(len(dataset)),
                "feature_column_count": int(len(_feature_columns(dataset))),
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


def _prepare_schedule_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
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
        schedule[column] = pd.to_numeric(schedule[column], errors="coerce")

    schedule["is_dome"] = schedule["is_dome"].astype(bool)
    schedule["is_abs_active"] = schedule["is_abs_active"].astype(bool)
    schedule = schedule.dropna(
        subset=[
            "game_pk",
            "season",
            "scheduled_start",
            "home_team",
            "away_team",
            "f5_home_score",
            "f5_away_score",
            "final_home_score",
            "final_away_score",
        ]
    ).copy()
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
) -> None:
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

    for game_date in sorted(schedule["game_date"].astype(str).unique().tolist()):
        compute_offensive_features(
            game_date,
            db_path=database_path,
            windows=DEFAULT_WINDOWS,
            refresh=refresh,
            lineup_player_ids=normalized_lineup_player_ids.get(game_date),
            team_logs_fetcher=cached_team_logs_fetcher,
            batting_stats_fetcher=cached_batting_stats_fetcher,
        )
        compute_pitching_features(
            game_date,
            db_path=database_path,
            windows=DEFAULT_WINDOWS,
            refresh=refresh,
            lineup_fetcher=cached_lineup_fetcher,
            start_metrics_fetcher=cached_start_metrics_fetcher,
        )
        compute_defense_features(
            game_date,
            db_path=database_path,
            refresh=refresh,
            fielding_fetcher=cached_fielding_stats_fetcher,
            framing_fetcher=cached_framing_stats_fetcher,
            team_logs_fetcher=cached_team_logs_fetcher,
        )
        compute_bullpen_features(
            game_date,
            db_path=database_path,
            refresh=refresh,
            bullpen_metrics_fetcher=cached_bullpen_metrics_fetcher,
            team_logs_fetcher=cached_team_logs_fetcher,
        )
        compute_baseline_features(
            game_date,
            db_path=database_path,
            windows=DEFAULT_PYTHAGOREAN_WINDOWS,
        )


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
            int(game["f5_home_score"]),
            int(game["f5_away_score"]),
            int(game["final_home_score"]),
            int(game["final_away_score"]),
            _normalize_game_status(game.get("status")),
        )
        for game in schedule.to_dict(orient="records")
    ]

    with sqlite3.connect(database_path) as connection:
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


def _load_feature_frame(database_path: Path) -> pd.DataFrame:
    with sqlite3.connect(database_path) as connection:
        feature_rows = pd.read_sql_query(
            """
            SELECT game_pk, feature_name, feature_value
            FROM features
            ORDER BY game_pk, feature_name
            """,
            connection,
        )

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

        non_null_values = numeric_values.dropna()
        fill_value = float(non_null_values.mean()) if not non_null_values.empty else 0.0
        dataset[column] = numeric_values.fillna(fill_value)

    return dataset


def _normalize_lineup_player_ids_by_date(
    mapping: LineupPlayerIdsByDate | None,
) -> dict[str, Mapping[tuple[int, str], Sequence[int]]]:
    if not mapping:
        return {}

    normalized: dict[str, Mapping[tuple[int, str], Sequence[int]]] = {}
    for game_date, lineup_ids in mapping.items():
        normalized[_normalize_lookup_date(game_date)] = lineup_ids
    return normalized


def _normalize_lookup_date(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


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
