from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import threading
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from random import uniform
from typing import Mapping

import httpx
import pandas as pd
from pydantic import ValidationError

from src.clients.weather_client import (
    HTTP_TIMEOUT,
    OPEN_METEO_ARCHIVE_BASE_URL,
    WeatherClientError,
    _calculate_air_density,
    _calculate_wind_factor,
    _build_weather_data_from_open_meteo,
    _ensure_weather_cache_table,
    _fetch_from_open_meteo_historical,
    _find_closest_historical_hour,
    _get_default_weather,
    _load_stadium,
    _normalize_datetime,
)
from src.db import DEFAULT_DB_PATH, sqlite_connection
from src.models.weather import WeatherData

try:
    from meteostat import stations as meteostat_stations
    from meteostat.api.config import config as meteostat_config
    from meteostat.api.hourly import hourly as meteostat_hourly
    from meteostat.api.point import Point as MeteostatPoint
except ImportError:  # pragma: no cover - optional runtime dependency
    meteostat_stations = None
    meteostat_config = None
    meteostat_hourly = None
    MeteostatPoint = None


logger = logging.getLogger(__name__)
TRAINING_PARQUET_PATH = "data/training/training_data_2018_2025.parquet"
DEFAULT_WORKERS = min(4, max(2, os.cpu_count() or 4))
DEFAULT_DELAY_MS = 100.0
DEFAULT_PROGRESS_EVERY = 250
DEFAULT_BATCH_SIZE = 250
DEFAULT_MAX_RETRIES = 6
DEFAULT_RATE_LIMIT_BACKOFF_MS = 2_000.0
METEOSTAT_CACHE_DIR = Path(".meteostat-cache")
NEUTRAL_HUMIDITY_PCT = 50.0


@dataclass(frozen=True, slots=True)
class GameRequest:
    index: int
    team_abbr: str
    game_datetime: datetime
    game_datetime_iso: str
    game_date: str
    latitude: float
    longitude: float
    elevation_m: float | None
    stadium_cf_orientation_deg: float
    is_dome: bool


@dataclass(slots=True)
class GroupFetchResult:
    games: list[GameRequest]
    fetched: list[tuple[GameRequest, WeatherData]]
    defaulted: list[tuple[GameRequest, str]]


class RequestPacer:
    """Globally space request start times while still allowing concurrent in-flight calls."""

    def __init__(self, delay_seconds: float) -> None:
        self._delay_seconds = max(float(delay_seconds), 0.0)
        self._lock = threading.Lock()
        self._next_allowed_at = 0.0

    def wait(self) -> None:
        if self._delay_seconds <= 0:
            return

        while True:
            with self._lock:
                now = time.monotonic()
                if now >= self._next_allowed_at:
                    self._next_allowed_at = now + self._delay_seconds
                    return
                sleep_for = self._next_allowed_at - now

            if sleep_for > 0:
                time.sleep(sleep_for)

    def penalize(self, delay_seconds: float) -> None:
        penalty = max(float(delay_seconds), 0.0)
        if penalty <= 0:
            return

        with self._lock:
            now = time.monotonic()
            self._next_allowed_at = max(self._next_allowed_at, now + penalty)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill historical weather for training games.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fetched without calling the weather APIs.",
    )
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        help="Inclusive UTC start date filter in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        help="Inclusive UTC end date filter in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel Open-Meteo fetch workers. Default: {DEFAULT_WORKERS}.",
    )
    parser.add_argument(
        "--delay-ms",
        type=float,
        default=DEFAULT_DELAY_MS,
        help=f"Minimum delay between request starts in milliseconds. Default: {DEFAULT_DELAY_MS:g}.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help=f"Print aggregate progress every N processed games. Default: {DEFAULT_PROGRESS_EVERY}.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"SQLite batch size for cache writes. Default: {DEFAULT_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-game fetch/default details instead of periodic aggregate progress.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Retry attempts for temporary API failures like 429. Default: {DEFAULT_MAX_RETRIES}.",
    )
    return parser.parse_args()


def _degrees_to_compass(degrees: float) -> str:
    directions = (
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    )
    normalized_degrees = float(degrees) % 360.0
    index = int((normalized_degrees + 11.25) // 22.5) % len(directions)
    return directions[index]


def _load_training_games(
    *,
    start_date: date | None,
    end_date: date | None,
) -> list[GameRequest]:
    dataframe = pd.read_parquet(
        TRAINING_PARQUET_PATH,
        columns=["home_team", "scheduled_start"],
    ).drop_duplicates(subset=["home_team", "scheduled_start"])
    dataframe["scheduled_start_dt"] = pd.to_datetime(dataframe["scheduled_start"], utc=True)

    if start_date is not None:
        dataframe = dataframe[dataframe["scheduled_start_dt"].dt.date >= start_date]
    if end_date is not None:
        dataframe = dataframe[dataframe["scheduled_start_dt"].dt.date <= end_date]

    dataframe = dataframe.sort_values(["scheduled_start_dt", "home_team"]).reset_index(drop=True)

    game_requests: list[GameRequest] = []
    for index, row in enumerate(dataframe.itertuples(index=False), start=1):
        team_abbr = str(row.home_team).upper()
        game_datetime = _normalize_datetime(row.scheduled_start_dt)
        stadium = _load_stadium(team_abbr)
        game_requests.append(
            GameRequest(
                index=index,
                team_abbr=team_abbr,
                game_datetime=game_datetime,
                game_datetime_iso=game_datetime.isoformat(),
                game_date=game_datetime.date().isoformat(),
                latitude=float(stadium["latitude"]),
                longitude=float(stadium["longitude"]),
                elevation_m=(
                    float(stadium["altitude_ft"]) * 0.3048
                    if stadium.get("altitude_ft") is not None
                    else None
                ),
                stadium_cf_orientation_deg=float(stadium["center_field_orientation_deg"]),
                is_dome=bool(stadium["is_dome"]),
            )
        )

    return game_requests


def _load_cached_keys(
    db_path: str | Path,
    *,
    start_datetime: datetime | None,
    end_datetime: datetime | None,
) -> set[tuple[str, str]]:
    database_path = Path(db_path)
    if not database_path.exists():
        return set()

    database_path = _ensure_weather_cache_table(database_path)
    query = "SELECT team_abbr, game_datetime FROM weather_cache"
    params: list[str] = []
    clauses: list[str] = []

    if start_datetime is not None:
        clauses.append("game_datetime >= ?")
        params.append(start_datetime.isoformat())
    if end_datetime is not None:
        clauses.append("game_datetime < ?")
        params.append(end_datetime.isoformat())
    if clauses:
        query = f"{query} WHERE {' AND '.join(clauses)}"

    with sqlite_connection(database_path) as connection:
        rows = connection.execute(query, params).fetchall()

    return {(str(team_abbr).upper(), str(game_datetime)) for team_abbr, game_datetime in rows}


def _weather_cache_row(
    game: GameRequest,
    weather: WeatherData,
) -> tuple[str, str, float, float, float, float, float, float, float, float | None, float, float | None, int, str, str]:
    forecast_time = weather.forecast_time or game.game_datetime
    fetched_at = weather.fetched_at or datetime.now(timezone.utc)
    return (
        game.team_abbr,
        game.game_datetime_iso,
        weather.temperature_f,
        weather.humidity_pct,
        weather.wind_speed_mph,
        weather.wind_direction_deg,
        weather.pressure_hpa,
        weather.air_density,
        weather.wind_factor,
        weather.precipitation_probability,
        weather.precipitation_mm,
        weather.cloud_cover_pct,
        int(weather.is_dome_default),
        forecast_time.isoformat(),
        fetched_at.isoformat(),
    )


def _bulk_cache_weather(
    connection: sqlite3.Connection,
    entries: list[tuple[GameRequest, WeatherData]],
) -> None:
    if not entries:
        return

    rows = [_weather_cache_row(game, weather) for game, weather in entries]
    connection.executemany(
        """
        INSERT OR REPLACE INTO weather_cache (
            team_abbr,
            game_datetime,
            temperature_f,
            humidity_pct,
            wind_speed_mph,
            wind_direction_deg,
            pressure_hpa,
            air_density,
            wind_factor,
            precipitation_probability,
            precipitation_mm,
            cloud_cover_pct,
            is_dome_default,
            forecast_time,
            fetched_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    connection.commit()


def _render_weather_details(weather: WeatherData) -> str:
    wind_direction = _degrees_to_compass(weather.wind_direction_deg)
    return (
        f"({weather.temperature_f:.0f}F, {weather.wind_speed_mph:.0f}mph "
        f"{wind_direction} -> wind_factor={weather.wind_factor:+.1f})"
    )


def _print_game_status(
    game: GameRequest,
    total_games: int,
    *,
    status: str,
    details: str | None = None,
) -> None:
    suffix = f" {details}" if details else ""
    print(
        f"Game {game.index}/{total_games} - {game.team_abbr} {game.game_date} -> "
        f"{status}{suffix}"
    )


def _print_progress(
    *,
    processed_games: int,
    total_games: int,
    backfilled_count: int,
    cached_count: int,
    defaulted_count: int,
    completed_groups: int,
    total_groups: int,
) -> None:
    print(
        f"Progress {processed_games}/{total_games} | "
        f"Fetched: {backfilled_count} | Cached: {cached_count} | "
        f"Defaulted: {defaulted_count} | Groups: {completed_groups}/{total_groups}"
    )


def _build_meteostat_hourly_payload(dataframe: pd.DataFrame) -> Mapping[str, object]:
    if dataframe.empty:
        raise WeatherClientError("Meteostat returned no hourly data")

    frame = dataframe.copy()
    if "time" in frame.columns:
        frame["time"] = pd.to_datetime(frame["time"], utc=True)
        frame = frame.set_index("time")
    elif not isinstance(frame.index, pd.DatetimeIndex):
        raise WeatherClientError("Meteostat payload missing datetime index")

    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")

    return {
        "time": [timestamp.isoformat() for timestamp in frame.index],
        "temperature_2m": [
            (((float(value) * 9.0) / 5.0) + 32.0) if not pd.isna(value) else None
            for value in frame.get("temp", pd.Series(index=frame.index, dtype=float))
        ],
        "wind_speed_10m": [
            (float(value) * 0.621371192237334) if not pd.isna(value) else None
            for value in frame.get("wspd", pd.Series(index=frame.index, dtype=float))
        ],
        "wind_direction_10m": [
            float(value) if not pd.isna(value) else 0.0
            for value in frame.get("wdir", pd.Series(index=frame.index, dtype=float))
        ],
        "precipitation": [
            float(value) if not pd.isna(value) else 0.0
            for value in frame.get("prcp", pd.Series(index=frame.index, dtype=float))
        ],
        "relative_humidity_2m": [
            float(value) if not pd.isna(value) else None
            for value in frame.get("rhum", pd.Series(index=frame.index, dtype=float))
        ],
        "cloud_cover": [
            min(max(float(value) * 12.5, 0.0), 100.0) if not pd.isna(value) else None
            for value in frame.get("cldc", pd.Series(index=frame.index, dtype=float))
        ],
        "surface_pressure": [
            float(value) if not pd.isna(value) else None
            for value in frame.get("pres", pd.Series(index=frame.index, dtype=float))
        ],
    }


def _build_weather_data_from_meteostat(
    hourly_entry: Mapping[str, object],
    *,
    stadium_cf_orientation_deg: float,
    retrieved_at: datetime,
) -> WeatherData:
    temperature_f = hourly_entry.get("temperature_2m")
    if temperature_f is None:
        raise WeatherClientError("Meteostat hour missing temperature")
    pressure_hpa = hourly_entry.get("surface_pressure")
    if pressure_hpa is None:
        raise WeatherClientError("Meteostat hour missing pressure")

    humidity_raw = hourly_entry.get("relative_humidity_2m")
    humidity_pct = (
        float(humidity_raw)
        if humidity_raw is not None and not pd.isna(humidity_raw)
        else NEUTRAL_HUMIDITY_PCT
    )
    wind_speed_mph = float(hourly_entry.get("wind_speed_10m") or 0.0)
    wind_direction_deg = float(hourly_entry.get("wind_direction_10m") or 0.0)
    precipitation_mm = float(hourly_entry.get("precipitation") or 0.0)
    cloud_cover_raw = hourly_entry.get("cloud_cover")
    cloud_cover_pct = (
        float(cloud_cover_raw)
        if cloud_cover_raw is not None and not pd.isna(cloud_cover_raw)
        else None
    )
    forecast_time = _normalize_datetime(str(hourly_entry["time"]))
    temperature_k = ((float(temperature_f) - 32.0) * 5.0 / 9.0) + 273.15
    pressure_pa = float(pressure_hpa) * 100.0

    return WeatherData(
        temperature_f=float(temperature_f),
        humidity_pct=humidity_pct,
        wind_speed_mph=wind_speed_mph,
        wind_direction_deg=wind_direction_deg,
        pressure_hpa=float(pressure_hpa),
        air_density=_calculate_air_density(temperature_k, pressure_pa, humidity_pct),
        wind_factor=_calculate_wind_factor(
            wind_speed_mph,
            wind_direction_deg,
            stadium_cf_orientation_deg,
        ),
        precipitation_probability=None,
        precipitation_mm=precipitation_mm,
        cloud_cover_pct=cloud_cover_pct,
        is_dome_default=False,
        forecast_time=forecast_time,
        fetched_at=retrieved_at,
    )


def _fetch_group_weather_from_meteostat(
    games: list[GameRequest],
) -> GroupFetchResult:
    if meteostat_hourly is None or MeteostatPoint is None or meteostat_stations is None:
        return GroupFetchResult(
            games=games,
            fetched=[],
            defaulted=[(game, "Meteostat fallback unavailable: package not installed") for game in games],
        )

    first_game = games[0]
    try:
        METEOSTAT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if meteostat_config is not None:
            cache_dir = str(METEOSTAT_CACHE_DIR.resolve())
            meteostat_config.cache_directory = cache_dir
            meteostat_config.stations_db_file = str(METEOSTAT_CACHE_DIR.resolve() / "stations.db")
        point = MeteostatPoint(
            first_game.latitude,
            first_game.longitude,
            int(round(first_game.elevation_m)) if first_game.elevation_m is not None else None,
        )
        nearby_stations = meteostat_stations.nearby(point, radius=50_000, limit=10)
        if nearby_stations.empty:
            raise WeatherClientError("Meteostat found no nearby stations within 50 km")
        start = datetime.combine(
            first_game.game_datetime.date(),
            datetime.min.time(),
            tzinfo=timezone.utc,
        )
        end = start + timedelta(days=1) - timedelta(hours=1)
        dataframe = None
        for station_id in nearby_stations.index.tolist():
            series = meteostat_hourly(station_id, start, end, timezone="UTC")
            dataframe = series.fetch()
            if dataframe is not None and not dataframe.empty:
                break
        if dataframe is None or dataframe.empty:
            raise WeatherClientError("Meteostat returned no hourly data from nearby stations")
        hourly_payload = _build_meteostat_hourly_payload(dataframe)
        retrieved_at = datetime.now(timezone.utc)
    except Exception as exc:  # pragma: no cover - external library boundary
        return GroupFetchResult(
            games=games,
            fetched=[],
            defaulted=[(game, f"Meteostat fallback failed: {exc}") for game in games],
        )

    fetched: list[tuple[GameRequest, WeatherData]] = []
    defaulted: list[tuple[GameRequest, str]] = []
    for game in games:
        try:
            matched_hour = _find_closest_historical_hour(hourly_payload, game.game_datetime)
            if matched_hour is None:
                raise WeatherClientError("Meteostat returned no hour within 3 hours")
            weather = _build_weather_data_from_meteostat(
                matched_hour,
                stadium_cf_orientation_deg=game.stadium_cf_orientation_deg,
                retrieved_at=retrieved_at,
            )
            fetched.append((game, weather))
        except (
            ValidationError,
            ValueError,
            TypeError,
            KeyError,
            WeatherClientError,
        ) as exc:
            defaulted.append((game, str(exc)))

    return GroupFetchResult(games=games, fetched=fetched, defaulted=defaulted)


def _fetch_group_weather_from_open_meteo(
    games: list[GameRequest],
    *,
    client: httpx.Client,
    pacer: RequestPacer,
    max_retries: int,
) -> GroupFetchResult:
    first_game = games[0]

    hourly_payload: Mapping[str, object] | None = None
    retrieved_at: datetime | None = None
    last_error: str | None = None

    for attempt in range(max_retries + 1):
        try:
            pacer.wait()
            payload = _fetch_from_open_meteo_historical(
                latitude=first_game.latitude,
                longitude=first_game.longitude,
                date_str=first_game.game_date,
                client=client,
            )
            hourly_payload = payload.get("hourly")
            if not isinstance(hourly_payload, Mapping):
                raise WeatherClientError("Historical weather payload missing hourly data")
            retrieved_at = datetime.now(timezone.utc)
            break
        except httpx.HTTPStatusError as exc:
            response = exc.response
            status_code = response.status_code
            last_error = str(exc)
            is_retryable = status_code == 429 or 500 <= status_code < 600
            if not is_retryable or attempt >= max_retries:
                break

            retry_after_header = response.headers.get("Retry-After")
            retry_after_seconds = None
            if retry_after_header is not None:
                try:
                    retry_after_seconds = float(retry_after_header)
                except ValueError:
                    retry_after_seconds = None

            backoff_seconds = (
                retry_after_seconds
                if retry_after_seconds is not None
                else (DEFAULT_RATE_LIMIT_BACKOFF_MS / 1000.0) * (2**attempt)
            )
            backoff_seconds += uniform(0.05, 0.25)
            pacer.penalize(backoff_seconds)
            logger.warning(
                "Rate limited fetching %s %s; retrying in %.2fs (attempt %s/%s)",
                first_game.team_abbr,
                first_game.game_date,
                backoff_seconds,
                attempt + 1,
                max_retries,
            )
            time.sleep(backoff_seconds)
        except (
            httpx.HTTPError,
            ValidationError,
            ValueError,
            TypeError,
            KeyError,
            WeatherClientError,
        ) as exc:
            last_error = str(exc)
            break

    if hourly_payload is None or retrieved_at is None:
        error_message = last_error or "unknown Open-Meteo historical weather fetch error"
        return GroupFetchResult(
            games=games,
            fetched=[],
            defaulted=[(game, error_message) for game in games],
        )

    fetched: list[tuple[GameRequest, WeatherData]] = []
    defaulted: list[tuple[GameRequest, str]] = []
    for game in games:
        try:
            matched_hour = _find_closest_historical_hour(hourly_payload, game.game_datetime)
            if matched_hour is None:
                raise WeatherClientError("No historical weather within 3 hours")

            weather = _build_weather_data_from_open_meteo(
                matched_hour,
                stadium_cf_orientation_deg=game.stadium_cf_orientation_deg,
                retrieved_at=retrieved_at,
            )
            fetched.append((game, weather))
        except (
            ValidationError,
            ValueError,
            TypeError,
            KeyError,
            WeatherClientError,
        ) as exc:
            defaulted.append((game, str(exc)))

    return GroupFetchResult(games=games, fetched=fetched, defaulted=defaulted)


def _fetch_group_weather(
    games: list[GameRequest],
    *,
    client: httpx.Client,
    pacer: RequestPacer,
    max_retries: int,
) -> GroupFetchResult:
    first_game = games[0]
    meteostat_result = _fetch_group_weather_from_meteostat(games)
    if meteostat_result.fetched and not meteostat_result.defaulted:
        return meteostat_result

    remaining_games = [game for game, _reason in meteostat_result.defaulted]
    if not remaining_games:
        return meteostat_result

    open_meteo_result = _fetch_group_weather_from_open_meteo(
        remaining_games,
        client=client,
        pacer=pacer,
        max_retries=max_retries,
    )
    if open_meteo_result.fetched:
        logger.info(
            "Used Open-Meteo fallback for %s %s after Meteostat miss",
            first_game.team_abbr,
            first_game.game_date,
        )

    merged_defaulted: list[tuple[GameRequest, str]] = []
    open_meteo_fetched_keys = {
        (game.team_abbr, game.game_datetime_iso)
        for game, _weather in open_meteo_result.fetched
    }
    open_meteo_defaulted_by_key = {
        (game.team_abbr, game.game_datetime_iso): reason
        for game, reason in open_meteo_result.defaulted
    }
    for game, meteostat_reason in meteostat_result.defaulted:
        cache_key = (game.team_abbr, game.game_datetime_iso)
        if cache_key in open_meteo_fetched_keys:
            continue
        combined_reason = meteostat_reason
        open_meteo_reason = open_meteo_defaulted_by_key.get(cache_key)
        if open_meteo_reason is not None:
            combined_reason = f"{meteostat_reason} | {open_meteo_reason}"
        merged_defaulted.append((game, combined_reason))

    return GroupFetchResult(
        games=games,
        fetched=[*meteostat_result.fetched, *open_meteo_result.fetched],
        defaulted=merged_defaulted,
    )


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("meteostat").setLevel(logging.WARNING)

    workers = max(1, int(args.workers))
    delay_seconds = max(float(args.delay_ms), 0.0) / 1000.0
    progress_every = max(int(args.progress_every), 1)
    batch_size = max(int(args.batch_size), 1)
    max_retries = max(int(args.max_retries), 0)

    game_requests = _load_training_games(
        start_date=args.start_date,
        end_date=args.end_date,
    )
    total_games = len(game_requests)
    if total_games == 0:
        print("No games matched the requested filters.")
        return 0

    start_datetime = game_requests[0].game_datetime if game_requests else None
    end_datetime = (
        game_requests[-1].game_datetime + timedelta(seconds=1) if game_requests else None
    )
    cached_keys = _load_cached_keys(
        DEFAULT_DB_PATH,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )

    cached_count = 0
    backfilled_count = 0
    defaulted_count = 0
    processed_games = 0
    last_logged_processed = 0

    uncached_dome_games: list[GameRequest] = []
    fetch_groups: dict[tuple[str, str], list[GameRequest]] = defaultdict(list)

    for game in game_requests:
        cache_key = (game.team_abbr, game.game_datetime_iso)
        if cache_key in cached_keys:
            cached_count += 1
            processed_games += 1
            continue

        if game.is_dome:
            uncached_dome_games.append(game)
            continue

        fetch_groups[(game.team_abbr, game.game_date)].append(game)

    total_groups = len(fetch_groups)
    fetchable_games = sum(len(games) for games in fetch_groups.values())

    if args.dry_run:
        print(
            f"Dry run: would fetch {fetchable_games} games across {total_groups} unique team-date calls | "
            f"Cached: {cached_count} games | Defaulted (domes/errors): {len(uncached_dome_games)} games"
        )
        if args.verbose:
            for game in uncached_dome_games:
                _print_game_status(game, total_games, status="WOULD_DEFAULT_DOME")
            for games in sorted(fetch_groups.values(), key=lambda group: group[0].index):
                for game in games:
                    _print_game_status(game, total_games, status="WOULD_FETCH")
        return 0

    database_path = _ensure_weather_cache_table(DEFAULT_DB_PATH)
    with sqlite_connection(database_path) as connection:
        if uncached_dome_games:
            dome_fetched_at = datetime.now(timezone.utc)
            dome_entries = [
                (
                    game,
                    _get_default_weather(is_dome=True).model_copy(
                        update={
                            "forecast_time": game.game_datetime,
                            "fetched_at": dome_fetched_at,
                        }
                    ),
                )
                for game in uncached_dome_games
            ]
            _bulk_cache_weather(connection, dome_entries)
            defaulted_count += len(dome_entries)
            processed_games += len(dome_entries)
            if args.verbose:
                for game, _weather in dome_entries:
                    _print_game_status(game, total_games, status="DEFAULTED (DOME)")

        if not args.verbose and processed_games > 0:
            _print_progress(
                processed_games=processed_games,
                total_games=total_games,
                backfilled_count=backfilled_count,
                cached_count=cached_count,
                defaulted_count=defaulted_count,
                completed_groups=0,
                total_groups=total_groups,
            )
            last_logged_processed = processed_games

        ordered_groups = sorted(fetch_groups.values(), key=lambda group: group[0].index)
        pending_writes: list[tuple[GameRequest, WeatherData]] = []
        completed_groups = 0
        pacer = RequestPacer(delay_seconds)

        with httpx.Client(base_url=OPEN_METEO_ARCHIVE_BASE_URL, timeout=HTTP_TIMEOUT) as http_client:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="weather-backfill") as executor:
                future_to_games: dict[Future[GroupFetchResult], list[GameRequest]] = {
                    executor.submit(
                        _fetch_group_weather,
                        games,
                        client=http_client,
                        pacer=pacer,
                        max_retries=max_retries,
                    ): games
                    for games in ordered_groups
                }

                for future in as_completed(future_to_games):
                    games = future_to_games[future]
                    completed_groups += 1

                    try:
                        result = future.result()
                    except Exception as exc:  # pragma: no cover - defensive fallback
                        logger.exception(
                            "Unexpected failure backfilling group %s %s: %s",
                            games[0].team_abbr,
                            games[0].game_date,
                            exc,
                        )
                        result = GroupFetchResult(
                            games=games,
                            fetched=[],
                            defaulted=[(game, f"unexpected error: {exc}") for game in games],
                        )

                    backfilled_count += len(result.fetched)
                    defaulted_count += len(result.defaulted)
                    processed_games += len(result.games)
                    pending_writes.extend(result.fetched)

                    if len(pending_writes) >= batch_size:
                        _bulk_cache_weather(connection, pending_writes)
                        pending_writes.clear()

                    if args.verbose:
                        for game, weather in result.fetched:
                            _print_game_status(
                                game,
                                total_games,
                                status="FETCHED",
                                details=_render_weather_details(weather),
                            )
                        for game, reason in result.defaulted:
                            _print_game_status(
                                game,
                                total_games,
                                status="DEFAULTED (ERROR)",
                                details=f"({reason})",
                            )
                    elif processed_games - last_logged_processed >= progress_every or processed_games == total_games:
                        _print_progress(
                            processed_games=processed_games,
                            total_games=total_games,
                            backfilled_count=backfilled_count,
                            cached_count=cached_count,
                            defaulted_count=defaulted_count,
                            completed_groups=completed_groups,
                            total_groups=total_groups,
                        )
                        last_logged_processed = processed_games

        if pending_writes:
            _bulk_cache_weather(connection, pending_writes)

    print(
        f"Backfilled: {backfilled_count} games | Cached: {cached_count} games | "
        f"Defaulted (domes/errors): {defaulted_count} games | "
        f"Unique fetches: {total_groups} | Workers: {workers} | Delay: {args.delay_ms:g}ms | "
        f"Retries: {max_retries}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
