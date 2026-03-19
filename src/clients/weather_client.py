import logging
from datetime import datetime, timedelta, timezone
from math import cos, radians
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import httpx
from sqlalchemy import create_engine, Column, String, Float, DateTime, Boolean
from sqlalchemy.orm import declarative_base, Session

from src.config import Settings
from src.models.weather import WeatherData

logger = logging.getLogger(__name__)
OPENWEATHER_API_BASE_URL = "https://api.openweathermap.org/data/2.5"
WEATHER_CACHE_HOURS = 6
DEFAULT_DB_PATH = Path("data/cache.db")
R_DRY = 287.05
R_VAPOR = 461.5

Base = declarative_base()


class WeatherCache(Base):
    __tablename__ = "weather_cache"
    team_abbr = Column(String, primary_key=True)
    game_datetime = Column(DateTime, primary_key=True)
    temperature_f = Column(Float)
    humidity_pct = Column(Float)
    wind_speed_mph = Column(Float)
    wind_direction_deg = Column(Float)
    pressure_hpa = Column(Float)
    air_density = Column(Float)
    wind_factor = Column(Float)
    is_dome_default = Column(Boolean)
    fetched_at = Column(DateTime)


def _resolve_api_key(api_key: Optional[str] = None) -> str:
    if api_key:
        return api_key
    api_key = Settings().openweathermap_api_key
    if not api_key:
        raise ValueError("OpenWeatherMap API key not found")
    return api_key


def _ensure_weather_cache_table(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)


def _kelvin_to_fahrenheit(kelvin: float) -> float:
    celsius = kelvin - 273.15
    return celsius * 9 / 5 + 32


def _mps_to_mph(mps: float) -> float:
    return mps * 2.237


def _pa_to_hpa(pa: float) -> float:
    return pa / 100


def _calculate_air_density(temperature_k: float, pressure_pa: float, humidity_pct: float) -> float:
    try:
        if temperature_k <= 0 or pressure_pa <= 0 or humidity_pct < 0:
            return 1.225
        alpha = 17.27 if temperature_k >= 273.15 else 21.87
        beta = 237.7 if temperature_k >= 273.15 else 265.5
        celsius = temperature_k - 273.15
        ln_term = alpha * celsius / (beta + celsius)
        saturation_vp = 610.5 * (2.71828**ln_term)
        vapor_pressure = (humidity_pct / 100) * saturation_vp
        dry_pressure = pressure_pa - vapor_pressure
        rho = (dry_pressure / (R_DRY * temperature_k)) + (
            vapor_pressure / (R_VAPOR * temperature_k)
        )
        return max(rho, 0.4)
    except:
        return 1.225


def _calculate_wind_factor(
    wind_speed_mph: float, wind_direction_deg: float, stadium_cf_orientation_deg: float
) -> float:
    try:
        angle_diff = wind_direction_deg - stadium_cf_orientation_deg
        wind_component = -wind_speed_mph * cos(radians(angle_diff))
        return wind_component
    except:
        return 0.0


def _get_default_weather(is_dome: bool = False) -> WeatherData:
    return WeatherData(
        temperature_f=70.0,
        humidity_pct=50.0,
        wind_speed_mph=0.0,
        wind_direction_deg=0.0,
        pressure_hpa=1013.25,
        air_density=1.225,
        wind_factor=0.0,
        is_dome_default=is_dome,
        fetched_at=None,
    )


def _get_cached_weather(
    db_path: Path, team_abbr: str, game_datetime: datetime
) -> Optional[WeatherData]:
    try:
        if not db_path.exists():
            return None
        engine = create_engine(f"sqlite:///{db_path}")
        session = Session(engine)
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=WEATHER_CACHE_HOURS)
        records = (
            session.query(WeatherCache)
            .filter(
                WeatherCache.team_abbr == team_abbr,
                WeatherCache.game_datetime == game_datetime,
                WeatherCache.fetched_at >= cutoff_time,
            )
            .all()
        )
        session.close()
        if not records:
            return None
        record = records[0]
        fetched_at = record.fetched_at
        if fetched_at and fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        return WeatherData(
            temperature_f=record.temperature_f,
            humidity_pct=record.humidity_pct,
            wind_speed_mph=record.wind_speed_mph,
            wind_direction_deg=record.wind_direction_deg,
            pressure_hpa=record.pressure_hpa,
            air_density=record.air_density,
            wind_factor=record.wind_factor,
            is_dome_default=record.is_dome_default,
            fetched_at=fetched_at,
        )
    except Exception as e:
        logger.warning(f"Failed to retrieve cached weather: {e}")
        return None


def _cache_weather(
    db_path: Path, team_abbr: str, game_datetime: datetime, weather: WeatherData
) -> None:
    try:
        _ensure_weather_cache_table(db_path)
        engine = create_engine(f"sqlite:///{db_path}")
        session = Session(engine)
        session.query(WeatherCache).filter(
            WeatherCache.team_abbr == team_abbr, WeatherCache.game_datetime == game_datetime
        ).delete()
        cache_record = WeatherCache(
            team_abbr=team_abbr,
            game_datetime=game_datetime,
            temperature_f=weather.temperature_f,
            humidity_pct=weather.humidity_pct,
            wind_speed_mph=weather.wind_speed_mph,
            wind_direction_deg=weather.wind_direction_deg,
            pressure_hpa=weather.pressure_hpa,
            air_density=weather.air_density,
            wind_factor=weather.wind_factor,
            is_dome_default=weather.is_dome_default,
            fetched_at=datetime.now(timezone.utc),
        )
        session.add(cache_record)
        session.commit()
        session.close()
    except Exception as e:
        logger.warning(f"Failed to cache weather: {e}")


def _find_closest_forecast(
    forecasts: list[dict], target_time: datetime, max_hours: int = 3
) -> Optional[dict]:
    try:
        target_timestamp = target_time.timestamp()
        max_diff = max_hours * 3600
        closest = None
        min_diff = max_diff + 1
        for forecast in forecasts:
            forecast_timestamp = forecast["dt"]
            diff = abs(forecast_timestamp - target_timestamp)
            if diff < min_diff and diff <= max_diff:
                min_diff = diff
                closest = forecast
        return closest
    except:
        return None


def _fetch_from_api(
    latitude: float, longitude: float, api_key: str, client: Optional[httpx.Client] = None
) -> Optional[dict]:
    try:
        params = {"lat": latitude, "lon": longitude, "appid": api_key, "units": "metric"}
        url = f"{OPENWEATHER_API_BASE_URL}/forecast?{urlencode(params)}"
        should_close = False
        if client is None:
            client = httpx.Client(timeout=10.0)
            should_close = True
        try:
            response = client.get(url)
            response.raise_for_status()
            return response.json()
        finally:
            if should_close:
                client.close()
    except Exception as e:
        logger.error(f"Failed to fetch weather from API: {e}")
        return None


def fetch_game_weather(
    team_abbr: str,
    game_datetime: datetime | str,
    api_key: Optional[str] = None,
    db_path: Path = DEFAULT_DB_PATH,
    client: Optional[httpx.Client] = None,
) -> WeatherData:
    if isinstance(game_datetime, str):
        game_datetime = datetime.fromisoformat(game_datetime.replace(" ", "T"))
    if game_datetime.tzinfo is None:
        game_datetime = game_datetime.replace(tzinfo=timezone.utc)
    settings = Settings()
    if team_abbr not in settings.stadiums:
        raise ValueError(f"Team {team_abbr} not found")
    stadium = settings.stadiums[team_abbr]
    if stadium.get("is_dome", False):
        return _get_default_weather(is_dome=True)
    cached = _get_cached_weather(db_path, team_abbr, game_datetime)
    if cached:
        return cached
    try:
        api_key = _resolve_api_key(api_key)
        response = _fetch_from_api(
            latitude=stadium["latitude"],
            longitude=stadium["longitude"],
            api_key=api_key,
            client=client,
        )
        if not response or "list" not in response:
            return _get_default_weather(is_dome=False)
        forecast = _find_closest_forecast(response["list"], game_datetime)
        if not forecast:
            return _get_default_weather(is_dome=False)
        temp_c = forecast["main"]["temp"]
        temp_k = temp_c + 273.15
        humidity = forecast["main"]["humidity"]
        pressure_pa = forecast["main"]["pressure"] * 100
        wind_speed_mps = forecast["wind"]["speed"]
        wind_direction = forecast["wind"].get("deg", 0)
        temp_f = _kelvin_to_fahrenheit(temp_k)
        wind_speed_mph = _mps_to_mph(wind_speed_mps)
        pressure_hpa = _pa_to_hpa(pressure_pa)
        air_density = _calculate_air_density(temp_k, pressure_pa, humidity)
        wind_factor = _calculate_wind_factor(
            wind_speed_mph, wind_direction, stadium.get("center_field_orientation_deg", 0)
        )
        weather = WeatherData(
            temperature_f=temp_f,
            humidity_pct=humidity,
            wind_speed_mph=wind_speed_mph,
            wind_direction_deg=wind_direction,
            pressure_hpa=pressure_hpa,
            air_density=air_density,
            wind_factor=wind_factor,
            is_dome_default=False,
            fetched_at=datetime.now(timezone.utc),
        )
        _cache_weather(db_path, team_abbr, game_datetime, weather)
        return weather
    except Exception as e:
        logger.error(f"Error fetching weather for {team_abbr}: {e}")
        return _get_default_weather(is_dome=False)
