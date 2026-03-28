from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from math import cos, exp, radians
from pathlib import Path
from typing import Any, Mapping

import httpx
from dotenv import load_dotenv
from pydantic import ValidationError

from src.config import DEFAULT_ENV_FILE, _load_settings_yaml
from src.db import DEFAULT_DB_PATH, init_db
from src.models.weather import WeatherData


logger = logging.getLogger(__name__)

OPENWEATHER_API_BASE_URL = "https://api.openweathermap.org"
OPENWEATHER_FORECAST_PATH = "/data/2.5/forecast"
OPEN_METEO_ARCHIVE_BASE_URL = "https://archive-api.open-meteo.com"
OPEN_METEO_ARCHIVE_PATH = "/v1/archive"
HTTP_TIMEOUT = 30.0
WEATHER_CACHE_HOURS = 6
OPENWEATHER_FORECAST_LOOKAHEAD = timedelta(days=5)
HISTORICAL_WEATHER_MATCH_HOURS = 3
DEFAULT_AIR_DENSITY = 1.225
R_DRY_AIR = 287.05
R_WATER_VAPOR = 461.495
OPEN_METEO_HOURLY_FIELDS = (
    "temperature_2m,"
    "wind_speed_10m,"
    "wind_direction_10m,"
    "precipitation,"
    "relative_humidity_2m,"
    "cloud_cover,"
    "surface_pressure"
)
_OPEN_METEO_HISTORICAL_RESPONSE_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
_ENSURED_WEATHER_CACHE_PATHS: set[Path] = set()
_WEATHER_MEMORY_CACHE: dict[tuple[str, str], WeatherData] = {}
_STADIUMS_CACHE: dict[str, Mapping[str, Any]] | None = None


class WeatherClientError(RuntimeError):
    """Base exception for weather client failures."""


def _normalize_datetime(value: datetime | str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))

    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


def _iso_datetime(value: datetime | str) -> str:
    return _normalize_datetime(value).isoformat()


def _resolve_api_key(api_key: str | None = None) -> str:
    """Resolve OpenWeatherMap API key from a parameter or `.env`."""

    if api_key:
        return api_key

    load_dotenv(DEFAULT_ENV_FILE)
    resolved_api_key = os.getenv("OPENWEATHER_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "OpenWeatherMap API key not found. Set OPENWEATHER_API_KEY or pass api_key."
        )

    return resolved_api_key


def _load_stadium(team_abbr: str) -> Mapping[str, Any]:
    global _STADIUMS_CACHE

    if _STADIUMS_CACHE is None:
        _STADIUMS_CACHE = {
            str(team_code).upper(): stadium_payload
            for team_code, stadium_payload in _load_settings_yaml()["stadiums"].items()
        }

    stadiums = _STADIUMS_CACHE
    normalized_team = team_abbr.upper()
    if normalized_team not in stadiums:
        raise ValueError(f"Team {normalized_team} not found in stadium configuration")
    return stadiums[normalized_team]


def _ensure_weather_cache_table(db_path: str | Path) -> Path:
    """Create the SQLite weather cache table if it does not exist."""

    database_path = Path(db_path)
    cache_key = database_path.resolve()
    if cache_key in _ENSURED_WEATHER_CACHE_PATHS and database_path.exists():
        return database_path

    database_path = init_db(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS weather_cache (
                team_abbr TEXT NOT NULL,
                game_datetime TEXT NOT NULL,
                temperature_f REAL NOT NULL,
                humidity_pct REAL NOT NULL,
                wind_speed_mph REAL NOT NULL,
                wind_direction_deg REAL NOT NULL,
                pressure_hpa REAL NOT NULL,
                air_density REAL NOT NULL,
                wind_factor REAL NOT NULL,
                precipitation_probability REAL,
                precipitation_mm REAL NOT NULL DEFAULT 0.0,
                cloud_cover_pct REAL,
                is_dome_default INTEGER NOT NULL CHECK (is_dome_default IN (0, 1)),
                forecast_time TEXT,
                fetched_at TEXT NOT NULL,
                PRIMARY KEY (team_abbr, game_datetime)
            )
            """
        )
        existing_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(weather_cache)").fetchall()
        }
        if "forecast_time" not in existing_columns:
            connection.execute("ALTER TABLE weather_cache ADD COLUMN forecast_time TEXT")
        if "precipitation_probability" not in existing_columns:
            connection.execute("ALTER TABLE weather_cache ADD COLUMN precipitation_probability REAL")
        if "precipitation_mm" not in existing_columns:
            connection.execute(
                "ALTER TABLE weather_cache ADD COLUMN precipitation_mm REAL NOT NULL DEFAULT 0.0"
            )
        if "cloud_cover_pct" not in existing_columns:
            connection.execute("ALTER TABLE weather_cache ADD COLUMN cloud_cover_pct REAL")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_weather_cache_fetched_at ON weather_cache (fetched_at)"
        )
        connection.commit()
    _ENSURED_WEATHER_CACHE_PATHS.add(cache_key)
    return database_path


def _kelvin_to_fahrenheit(kelvin: float) -> float:
    """Convert temperature from Kelvin to Fahrenheit."""

    celsius = kelvin - 273.15
    return (celsius * 9 / 5) + 32


def _mps_to_mph(mps: float) -> float:
    """Convert wind speed from meters per second to miles per hour."""

    return mps * 2.2369362920544


def _pa_to_hpa(pa: float) -> float:
    """Convert atmospheric pressure from Pascals to hectopascals."""

    return pa / 100


def _calculate_air_density(temperature_k: float, pressure_pa: float, humidity_pct: float) -> float:
    """Calculate humid-air density in kg/m³ using the ideal gas law."""

    if temperature_k <= 0 or pressure_pa <= 0 or not 0 <= humidity_pct <= 100:
        return DEFAULT_AIR_DENSITY

    celsius = temperature_k - 273.15
    saturation_vapor_pressure = 610.94 * exp((17.625 * celsius) / (celsius + 243.04))
    vapor_pressure = (humidity_pct / 100.0) * saturation_vapor_pressure
    dry_air_pressure = max(pressure_pa - vapor_pressure, 0.0)
    density = (dry_air_pressure / (R_DRY_AIR * temperature_k)) + (
        vapor_pressure / (R_WATER_VAPOR * temperature_k)
    )

    return density if density > 0 else DEFAULT_AIR_DENSITY


def _calculate_wind_factor(
    wind_speed_mph: float,
    wind_direction_deg: float,
    stadium_cf_orientation_deg: float,
) -> float:
    """Project wind onto the home-plate-to-center-field axis."""

    if wind_speed_mph <= 0:
        return 0.0

    angle_diff = (wind_direction_deg - stadium_cf_orientation_deg) % 360
    wind_component = -wind_speed_mph * cos(radians(angle_diff))
    return 0.0 if abs(wind_component) < 1e-9 else wind_component


def _get_default_weather(is_dome: bool = False) -> WeatherData:
    """Return neutral/default weather values."""

    return WeatherData(
        temperature_f=70.0,
        humidity_pct=50.0,
        wind_speed_mph=0.0,
        wind_direction_deg=0.0,
        pressure_hpa=1013.25,
        air_density=DEFAULT_AIR_DENSITY,
        wind_factor=0.0,
        precipitation_probability=None,
        precipitation_mm=0.0,
        cloud_cover_pct=None,
        is_dome_default=is_dome,
        forecast_time=None,
        fetched_at=None,
    )


def _get_cached_weather(
    db_path: str | Path,
    team_abbr: str,
    game_datetime: datetime | str,
    *,
    cache_hours: int = WEATHER_CACHE_HOURS,
) -> WeatherData | None:
    """Return a fresh cached weather row when one exists."""

    database_path = Path(db_path)
    if not database_path.exists():
        return None

    database_path = _ensure_weather_cache_table(database_path)

    normalized_team = team_abbr.upper()
    normalized_game_time_dt = _normalize_datetime(game_datetime)
    normalized_game_time = normalized_game_time_dt.isoformat()
    cache_key = (normalized_team, normalized_game_time)

    current_time = datetime.now(timezone.utc)
    is_historical = normalized_game_time_dt < current_time
    memory_cached_weather = _WEATHER_MEMORY_CACHE.get(cache_key)
    if memory_cached_weather is not None:
        memory_fetched_at = memory_cached_weather.fetched_at
        if memory_cached_weather.forecast_time is not None and memory_fetched_at is not None:
            if is_historical or memory_fetched_at >= current_time - timedelta(hours=cache_hours):
                return memory_cached_weather
        _WEATHER_MEMORY_CACHE.pop(cache_key, None)

    try:
        with sqlite3.connect(database_path) as connection:
            row = connection.execute(
                """
                SELECT
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
                FROM weather_cache
                WHERE team_abbr = ? AND game_datetime = ?
                """,
                (normalized_team, normalized_game_time),
            ).fetchone()
    except sqlite3.Error:
        return None

    if row is None:
        return None

    forecast_time_raw = row[11]
    if forecast_time_raw is None:
        return None

    fetched_at = _normalize_datetime(row[12])
    if not is_historical and fetched_at < current_time - timedelta(hours=cache_hours):
        _WEATHER_MEMORY_CACHE.pop(cache_key, None)
        return None

    weather = WeatherData(
        temperature_f=row[0],
        humidity_pct=row[1],
        wind_speed_mph=row[2],
        wind_direction_deg=row[3],
        pressure_hpa=row[4],
        air_density=row[5],
        wind_factor=row[6],
        precipitation_probability=row[7],
        precipitation_mm=row[8],
        cloud_cover_pct=row[9],
        is_dome_default=bool(row[10]),
        forecast_time=_normalize_datetime(forecast_time_raw),
        fetched_at=fetched_at,
    )
    _WEATHER_MEMORY_CACHE[cache_key] = weather
    return weather


def _cache_weather(
    db_path: str | Path,
    team_abbr: str,
    game_datetime: datetime | str,
    weather: WeatherData,
) -> None:
    """Store weather data in the SQLite cache."""

    database_path = _ensure_weather_cache_table(db_path)
    forecast_time = weather.forecast_time or _normalize_datetime(game_datetime)
    fetched_at = weather.fetched_at or datetime.now(timezone.utc)

    with sqlite3.connect(database_path) as connection:
        connection.execute(
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
            (
                team_abbr.upper(),
                _iso_datetime(game_datetime),
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
            ),
        )
        connection.commit()
    _WEATHER_MEMORY_CACHE[(team_abbr.upper(), _iso_datetime(game_datetime))] = weather.model_copy(
        update={
            "forecast_time": forecast_time,
            "fetched_at": fetched_at,
        }
    )


def _find_closest_forecast(
    forecasts: list[Mapping[str, Any]],
    target_time: datetime | str,
    *,
    max_hours: int = WEATHER_CACHE_HOURS,
) -> Mapping[str, Any] | None:
    """Return the closest forecast entry inside the allowed time window."""

    normalized_target_time = _normalize_datetime(target_time)
    max_seconds = max_hours * 3600
    best_match: Mapping[str, Any] | None = None
    best_diff: float | None = None

    for forecast in forecasts:
        timestamp = forecast.get("dt")
        if not isinstance(timestamp, (int, float)):
            continue

        forecast_time = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        diff_seconds = abs((forecast_time - normalized_target_time).total_seconds())
        if diff_seconds > max_seconds:
            continue

        if best_diff is None or diff_seconds < best_diff:
            best_match = forecast
            best_diff = diff_seconds

    return best_match


def _fetch_from_api(
    *,
    latitude: float,
    longitude: float,
    api_key: str,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Fetch the OpenWeatherMap five-day forecast payload."""

    client_context = (
        nullcontext(client)
        if client is not None
        else httpx.Client(base_url=OPENWEATHER_API_BASE_URL, timeout=HTTP_TIMEOUT)
    )

    with client_context as http_client:
        response = http_client.get(
            OPENWEATHER_FORECAST_PATH,
            params={"lat": latitude, "lon": longitude, "appid": api_key},
        )
        response.raise_for_status()
        payload = response.json()

    if not isinstance(payload, dict):
        raise WeatherClientError("Unexpected weather API response payload")

    return payload


def _fetch_from_open_meteo_historical(
    *,
    latitude: float,
    longitude: float,
    date_str: str,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Fetch a day of historical hourly weather from Open-Meteo."""

    client_context = (
        nullcontext(client)
        if client is not None
        else httpx.Client(base_url=OPEN_METEO_ARCHIVE_BASE_URL, timeout=HTTP_TIMEOUT)
    )

    with client_context as http_client:
        response = http_client.get(
            OPEN_METEO_ARCHIVE_PATH,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "start_date": date_str,
                "end_date": date_str,
                "hourly": OPEN_METEO_HOURLY_FIELDS,
                "wind_speed_unit": "mph",
                "temperature_unit": "fahrenheit",
                "precipitation_unit": "mm",
                "timezone": "UTC",
            },
        )
        response.raise_for_status()
        payload = response.json()

    if not isinstance(payload, dict):
        raise WeatherClientError("Unexpected Open-Meteo historical response payload")

    return payload


def _find_closest_historical_hour(
    hourly_payload: Mapping[str, Any],
    target_time: datetime | str,
    *,
    max_hours: int = HISTORICAL_WEATHER_MATCH_HOURS,
) -> Mapping[str, Any] | None:
    """Return the closest hourly archive entry, preferring times at or before first pitch."""

    time_values = hourly_payload.get("time")
    if not isinstance(time_values, list):
        raise WeatherClientError("Historical weather payload missing hourly time data")

    metric_names = (
        "temperature_2m",
        "wind_speed_10m",
        "wind_direction_10m",
        "precipitation",
        "relative_humidity_2m",
        "cloud_cover",
        "surface_pressure",
    )
    metric_values: dict[str, list[Any]] = {}
    for metric_name in metric_names:
        metric_payload = hourly_payload.get(metric_name)
        if not isinstance(metric_payload, list):
            raise WeatherClientError(
                f"Historical weather payload missing hourly field: {metric_name}"
            )
        metric_values[metric_name] = metric_payload

    normalized_target_time = _normalize_datetime(target_time)
    max_seconds = max_hours * 3600
    best_before: Mapping[str, Any] | None = None
    best_before_diff: float | None = None
    best_after: Mapping[str, Any] | None = None
    best_after_diff: float | None = None

    for index, timestamp_raw in enumerate(time_values):
        if not isinstance(timestamp_raw, str):
            continue

        hourly_time = _normalize_datetime(timestamp_raw)
        diff_seconds = (normalized_target_time - hourly_time).total_seconds()
        abs_diff_seconds = abs(diff_seconds)
        if abs_diff_seconds > max_seconds:
            continue

        hourly_entry: dict[str, Any] = {"time": timestamp_raw}
        for metric_name, metric_payload in metric_values.items():
            if index >= len(metric_payload):
                raise WeatherClientError(
                    f"Historical weather payload truncated for hourly field: {metric_name}"
                )
            hourly_entry[metric_name] = metric_payload[index]

        if diff_seconds >= 0:
            if best_before_diff is None or diff_seconds < best_before_diff:
                best_before = hourly_entry
                best_before_diff = diff_seconds
            continue

        if best_after_diff is None or abs_diff_seconds < best_after_diff:
            best_after = hourly_entry
            best_after_diff = abs_diff_seconds

    return best_before or best_after


def _build_weather_data(
    forecast: Mapping[str, Any],
    *,
    stadium_cf_orientation_deg: float,
    retrieved_at: datetime | str,
) -> WeatherData:
    """Build validated weather data from a forecast object."""

    main = forecast.get("main")
    if not isinstance(main, Mapping):
        raise WeatherClientError("Forecast payload missing main weather data")

    wind = forecast.get("wind")
    if not isinstance(wind, Mapping):
        wind = {}

    temperature_k = float(main["temp"])
    humidity_pct = float(main["humidity"])
    pressure_hpa = float(main["pressure"])
    wind_speed_mph = _mps_to_mph(float(wind.get("speed", 0.0)))
    wind_direction_deg = float(wind.get("deg", 0.0))
    forecast_time = datetime.fromtimestamp(float(forecast["dt"]), tz=timezone.utc)
    precipitation_probability_raw = forecast.get("pop")
    precipitation_probability = (
        max(0.0, min(1.0, float(precipitation_probability_raw)))
        if precipitation_probability_raw is not None
        else None
    )
    precipitation_mm = 0.0
    for key in ("rain", "snow"):
        payload = forecast.get(key)
        if isinstance(payload, Mapping):
            precipitation_mm += float(payload.get("3h", 0.0) or 0.0)
    clouds_payload = forecast.get("clouds", {})
    cloud_cover_pct = None
    if isinstance(clouds_payload, Mapping) and clouds_payload.get("all") is not None:
        cloud_cover_pct = float(clouds_payload["all"])

    return WeatherData(
        temperature_f=_kelvin_to_fahrenheit(temperature_k),
        humidity_pct=humidity_pct,
        wind_speed_mph=wind_speed_mph,
        wind_direction_deg=wind_direction_deg,
        pressure_hpa=pressure_hpa,
        air_density=_calculate_air_density(temperature_k, pressure_hpa * 100.0, humidity_pct),
        wind_factor=_calculate_wind_factor(
            wind_speed_mph,
            wind_direction_deg,
            stadium_cf_orientation_deg,
        ),
        precipitation_probability=precipitation_probability,
        precipitation_mm=precipitation_mm,
        cloud_cover_pct=cloud_cover_pct,
        is_dome_default=False,
        forecast_time=forecast_time,
        fetched_at=_normalize_datetime(retrieved_at),
    )


def _build_weather_data_from_open_meteo(
    hourly_entry: Mapping[str, Any],
    *,
    stadium_cf_orientation_deg: float,
    retrieved_at: datetime | str,
) -> WeatherData:
    """Build validated weather data from an Open-Meteo hourly archive entry."""

    temperature_f = float(hourly_entry["temperature_2m"])
    humidity_pct = float(hourly_entry["relative_humidity_2m"])
    pressure_hpa = float(hourly_entry["surface_pressure"])
    wind_speed_mph = float(hourly_entry["wind_speed_10m"])
    wind_direction_deg = float(hourly_entry["wind_direction_10m"])
    precipitation_mm = float(hourly_entry.get("precipitation", 0.0) or 0.0)
    cloud_cover_raw = hourly_entry.get("cloud_cover")
    cloud_cover_pct = float(cloud_cover_raw) if cloud_cover_raw is not None else None
    forecast_time = _normalize_datetime(hourly_entry["time"])
    temperature_k = ((temperature_f - 32.0) * 5.0 / 9.0) + 273.15
    pressure_pa = pressure_hpa * 100.0

    return WeatherData(
        temperature_f=temperature_f,
        humidity_pct=humidity_pct,
        wind_speed_mph=wind_speed_mph,
        wind_direction_deg=wind_direction_deg,
        pressure_hpa=pressure_hpa,
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
        fetched_at=_normalize_datetime(retrieved_at),
    )


def _fetch_historical_and_cache(
    *,
    team_abbr: str,
    game_datetime: datetime | str,
    stadium: Mapping[str, Any],
    db_path: str | Path,
) -> WeatherData:
    """Fetch historical weather from Open-Meteo, cache it, and degrade gracefully."""

    normalized_team = team_abbr.upper()
    normalized_game_time = _normalize_datetime(game_datetime)
    cached_weather = _get_cached_weather(db_path, normalized_team, normalized_game_time)
    if cached_weather is not None:
        return cached_weather

    date_str = normalized_game_time.date().isoformat()
    day_cache_key = (normalized_team, date_str)

    try:
        payload = _OPEN_METEO_HISTORICAL_RESPONSE_CACHE.get(day_cache_key)
        if payload is None:
            payload = _fetch_from_open_meteo_historical(
                latitude=float(stadium["latitude"]),
                longitude=float(stadium["longitude"]),
                date_str=date_str,
            )
            _OPEN_METEO_HISTORICAL_RESPONSE_CACHE[day_cache_key] = payload

        hourly_payload = payload.get("hourly")
        if not isinstance(hourly_payload, Mapping):
            raise WeatherClientError("Historical weather payload missing hourly data")

        matched_hour = _find_closest_historical_hour(hourly_payload, normalized_game_time)
        if matched_hour is None:
            raise WeatherClientError(
                f"No historical weather within {HISTORICAL_WEATHER_MATCH_HOURS} hours"
            )

        weather = _build_weather_data_from_open_meteo(
            matched_hour,
            stadium_cf_orientation_deg=float(stadium["center_field_orientation_deg"]),
            retrieved_at=datetime.now(timezone.utc),
        )
        _cache_weather(db_path, normalized_team, normalized_game_time, weather)
        return weather
    except (httpx.HTTPError, ValidationError, ValueError, TypeError, KeyError, WeatherClientError) as exc:
        logger.warning(
            "Failed to fetch historical weather for %s at %s: %s",
            normalized_team,
            normalized_game_time.isoformat(),
            exc,
        )
        return _get_default_weather(is_dome=False)


def _can_fetch_forecast_for_game_time(game_datetime: datetime | str) -> bool:
    normalized_game_time = _normalize_datetime(game_datetime)
    current_time = datetime.now(timezone.utc)
    return current_time <= normalized_game_time <= (current_time + OPENWEATHER_FORECAST_LOOKAHEAD)


def fetch_game_weather(
    team_abbr: str,
    game_datetime: datetime | str,
    *,
    api_key: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    client: httpx.Client | None = None,
) -> WeatherData:
    """Fetch, cache, and return game-time weather for a stadium."""

    stadium = _load_stadium(team_abbr)
    normalized_game_time = _normalize_datetime(game_datetime)

    if bool(stadium["is_dome"]):
        return _get_default_weather(is_dome=True)

    cached_weather = _get_cached_weather(db_path, team_abbr, normalized_game_time)
    if cached_weather is not None:
        return cached_weather

    if not _can_fetch_forecast_for_game_time(normalized_game_time):
        return _fetch_historical_and_cache(
            team_abbr=team_abbr,
            game_datetime=normalized_game_time,
            stadium=stadium,
            db_path=db_path,
        )

    resolved_api_key = _resolve_api_key(api_key)

    try:
        payload = _fetch_from_api(
            latitude=float(stadium["latitude"]),
            longitude=float(stadium["longitude"]),
            api_key=resolved_api_key,
            client=client,
        )
        retrieved_at = datetime.now(timezone.utc)
        forecasts = payload.get("list")
        if not isinstance(forecasts, list):
            raise WeatherClientError("Forecast payload missing list data")

        closest_forecast = _find_closest_forecast(forecasts, normalized_game_time)
        if closest_forecast is None:
            logger.debug(
                "No forecast within %s hours for %s at %s",
                WEATHER_CACHE_HOURS,
                team_abbr.upper(),
                normalized_game_time.isoformat(),
            )
            return _get_default_weather(is_dome=False)

        weather = _build_weather_data(
            closest_forecast,
            stadium_cf_orientation_deg=float(stadium["center_field_orientation_deg"]),
            retrieved_at=retrieved_at,
        )
    except (httpx.HTTPError, ValidationError, ValueError, TypeError, KeyError, WeatherClientError) as exc:
        logger.warning(
            "Failed to fetch weather for %s at %s: %s",
            team_abbr.upper(),
            normalized_game_time.isoformat(),
            exc,
        )
        return _get_default_weather(is_dome=False)

    _cache_weather(db_path, team_abbr, normalized_game_time, weather)
    return weather


def fetch_game_weather_local_only(
    team_abbr: str,
    game_datetime: datetime | str,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> WeatherData:
    """Return cached weather when available, otherwise a neutral default without network access."""

    stadium = _load_stadium(team_abbr)
    normalized_game_time = _normalize_datetime(game_datetime)
    cached_weather = _get_cached_weather(db_path, team_abbr, normalized_game_time)
    if cached_weather is not None:
        return cached_weather
    return _get_default_weather(is_dome=bool(stadium["is_dome"]))
