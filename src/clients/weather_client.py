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
HTTP_TIMEOUT = 30.0
WEATHER_CACHE_HOURS = 6
DEFAULT_AIR_DENSITY = 1.225
R_DRY_AIR = 287.05
R_WATER_VAPOR = 461.495


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
    stadiums = _load_settings_yaml()["stadiums"]
    normalized_team = team_abbr.upper()
    if normalized_team not in stadiums:
        raise ValueError(f"Team {normalized_team} not found in stadium configuration")
    return stadiums[normalized_team]


def _ensure_weather_cache_table(db_path: str | Path) -> Path:
    """Create the SQLite weather cache table if it does not exist."""

    database_path = init_db(db_path)
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
                is_dome_default INTEGER NOT NULL CHECK (is_dome_default IN (0, 1)),
                fetched_at TEXT NOT NULL,
                PRIMARY KEY (team_abbr, game_datetime)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_weather_cache_fetched_at ON weather_cache (fetched_at)"
        )
        connection.commit()
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
        is_dome_default=is_dome,
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

    normalized_team = team_abbr.upper()
    normalized_game_time = _iso_datetime(game_datetime)

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
                    is_dome_default,
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

    fetched_at = _normalize_datetime(row[8])
    if fetched_at < datetime.now(timezone.utc) - timedelta(hours=cache_hours):
        return None

    return WeatherData(
        temperature_f=row[0],
        humidity_pct=row[1],
        wind_speed_mph=row[2],
        wind_direction_deg=row[3],
        pressure_hpa=row[4],
        air_density=row[5],
        wind_factor=row[6],
        is_dome_default=bool(row[7]),
        fetched_at=fetched_at,
    )


def _cache_weather(
    db_path: str | Path,
    team_abbr: str,
    game_datetime: datetime | str,
    weather: WeatherData,
) -> None:
    """Store weather data in the SQLite cache."""

    database_path = _ensure_weather_cache_table(db_path)
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
                is_dome_default,
                fetched_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                int(weather.is_dome_default),
                fetched_at.isoformat(),
            ),
        )
        connection.commit()


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


def _build_weather_data(
    forecast: Mapping[str, Any],
    *,
    stadium_cf_orientation_deg: float,
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
    fetched_at = datetime.fromtimestamp(float(forecast["dt"]), tz=timezone.utc)

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
        is_dome_default=False,
        fetched_at=fetched_at,
    )


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

    resolved_api_key = _resolve_api_key(api_key)

    try:
        payload = _fetch_from_api(
            latitude=float(stadium["latitude"]),
            longitude=float(stadium["longitude"]),
            api_key=resolved_api_key,
            client=client,
        )
        forecasts = payload.get("list")
        if not isinstance(forecasts, list):
            raise WeatherClientError("Forecast payload missing list data")

        closest_forecast = _find_closest_forecast(forecasts, normalized_game_time)
        if closest_forecast is None:
            logger.warning(
                "No forecast within %s hours for %s at %s",
                WEATHER_CACHE_HOURS,
                team_abbr.upper(),
                normalized_game_time.isoformat(),
            )
            return _get_default_weather(is_dome=False)

        weather = _build_weather_data(
            closest_forecast,
            stadium_cf_orientation_deg=float(stadium["center_field_orientation_deg"]),
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
