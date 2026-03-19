from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from math import cos, radians

from src.config import _load_settings_yaml
from src.models.weather import WeatherData


NEUTRAL_WEATHER_FACTOR = 1.0
NEUTRAL_TEMPERATURE_F = 70.0
NEUTRAL_HUMIDITY_PCT = 50.0
STANDARD_AIR_DENSITY = 1.225
TEMPERATURE_FACTOR_PER_DEGREE_F = 0.0025
AIR_DENSITY_FACTOR_PER_KG_M3 = 1.2
HUMIDITY_FACTOR_PER_PERCENT = 0.0005
WIND_COMPOSITE_FACTOR_PER_MPH = 0.01
MAX_RAIN_PENALTY = 0.15
HUMIDITY_RAIN_PROXY_START = 70.0
HUMIDITY_RAIN_PROXY_SPAN = 30.0
HUMIDITY_RAIN_PROXY_MAX_PENALTY = 0.05
MIN_FACTOR = 0.85
MAX_FACTOR = 1.15

_SETTINGS_PAYLOAD = _load_settings_yaml()


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""

    normalized = unicodedata.normalize("NFKD", str(value))
    without_marks = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return " ".join(without_marks.casefold().split())


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(float(value), maximum))


@dataclass(frozen=True, slots=True)
class WeatherContext:
    """Resolved stadium weather metadata for a team or venue."""

    team_code: str | None
    park_name: str
    is_dome: bool
    center_field_orientation_deg: float


@dataclass(frozen=True, slots=True)
class WeatherAdjustment:
    """Weather-derived adjustment factors for one game context."""

    temp_factor: float
    air_density_factor: float
    humidity_factor: float
    wind_factor: float
    rain_risk: float
    weather_composite: float
    is_dome: bool


def load_weather_contexts() -> dict[str, WeatherContext]:
    """Load configured stadium weather context for all MLB teams."""

    stadiums = _SETTINGS_PAYLOAD.get("stadiums", {})
    contexts: dict[str, WeatherContext] = {}

    for team_code, stadium_payload in stadiums.items():
        if not isinstance(stadium_payload, dict):
            continue

        normalized_team_code = str(team_code).upper()
        contexts[normalized_team_code] = WeatherContext(
            team_code=normalized_team_code,
            park_name=str(stadium_payload.get("park_name") or normalized_team_code),
            is_dome=bool(stadium_payload.get("is_dome", False)),
            center_field_orientation_deg=float(
                stadium_payload.get("center_field_orientation_deg", 0.0)
            )
            % 360,
        )

    return contexts


_WEATHER_CONTEXTS_BY_TEAM = load_weather_contexts()
_WEATHER_CONTEXTS_BY_VENUE = {
    _normalize_text(context.park_name): context for context in _WEATHER_CONTEXTS_BY_TEAM.values()
}


def get_weather_context(
    *,
    team_code: str | None = None,
    venue: str | None = None,
) -> WeatherContext:
    """Resolve weather context by team code or venue name."""

    if team_code:
        normalized_team_code = team_code.strip().upper()
        resolved = _WEATHER_CONTEXTS_BY_TEAM.get(normalized_team_code)
        if resolved is not None:
            return resolved

    if venue:
        normalized_venue = _normalize_text(venue)
        if normalized_venue:
            resolved = _WEATHER_CONTEXTS_BY_VENUE.get(normalized_venue)
            if resolved is not None:
                return resolved

            for known_venue, known_context in _WEATHER_CONTEXTS_BY_VENUE.items():
                if known_venue in normalized_venue or normalized_venue in known_venue:
                    return known_context

    return WeatherContext(
        team_code=team_code.strip().upper() if team_code else None,
        park_name=(venue or team_code or "Unknown Park").strip() or "Unknown Park",
        is_dome=False,
        center_field_orientation_deg=0.0,
    )


def calculate_temp_factor(temperature_f: float) -> float:
    """Scale temperature around a neutral 70°F baseline."""

    factor = NEUTRAL_WEATHER_FACTOR + (
        (float(temperature_f) - NEUTRAL_TEMPERATURE_F) * TEMPERATURE_FACTOR_PER_DEGREE_F
    )
    return _clamp(factor, MIN_FACTOR, MAX_FACTOR)


def calculate_air_density_factor(air_density: float) -> float:
    """Reward lower-density air and penalize denser air."""

    factor = NEUTRAL_WEATHER_FACTOR + (
        (STANDARD_AIR_DENSITY - float(air_density)) * AIR_DENSITY_FACTOR_PER_KG_M3
    )
    return _clamp(factor, MIN_FACTOR, MAX_FACTOR)


def calculate_humidity_factor(humidity_pct: float) -> float:
    """Apply a modest humidity adjustment around a 50% baseline."""

    factor = NEUTRAL_WEATHER_FACTOR + (
        (float(humidity_pct) - NEUTRAL_HUMIDITY_PCT) * HUMIDITY_FACTOR_PER_PERCENT
    )
    return _clamp(factor, 0.95, 1.05)


def calculate_wind_factor(
    wind_speed_mph: float,
    wind_direction_deg: float,
    stadium_cf_orientation_deg: float,
) -> float:
    """Project source-direction wind onto the home-plate-to-center-field axis.

    OpenWeather reports meteorological source direction, so a value 180° off the
    center-field bearing means the wind is blowing out toward center field and
    should be positive.
    """

    resolved_wind_speed = max(float(wind_speed_mph), 0.0)
    if resolved_wind_speed <= 0:
        return 0.0

    angle_diff = (float(wind_direction_deg) - float(stadium_cf_orientation_deg)) % 360
    projected_component = -resolved_wind_speed * cos(radians(angle_diff))
    return 0.0 if abs(projected_component) < 1e-9 else projected_component


def calculate_rain_risk(
    *,
    humidity_pct: float,
    precipitation_probability: float | None = None,
) -> float:
    """Return a neutral-to-penalty factor for rain disruption risk."""

    if precipitation_probability is not None:
        resolved_probability = _clamp(float(precipitation_probability), 0.0, 1.0)
        return _clamp(
            NEUTRAL_WEATHER_FACTOR - (resolved_probability * MAX_RAIN_PENALTY),
            MIN_FACTOR,
            NEUTRAL_WEATHER_FACTOR,
        )

    humidity_excess = max(float(humidity_pct) - HUMIDITY_RAIN_PROXY_START, 0.0)
    proxy_penalty = (humidity_excess / HUMIDITY_RAIN_PROXY_SPAN) * HUMIDITY_RAIN_PROXY_MAX_PENALTY
    return _clamp(
        NEUTRAL_WEATHER_FACTOR - proxy_penalty,
        NEUTRAL_WEATHER_FACTOR - HUMIDITY_RAIN_PROXY_MAX_PENALTY,
        NEUTRAL_WEATHER_FACTOR,
    )


def _neutral_weather_adjustment() -> WeatherAdjustment:
    return WeatherAdjustment(
        temp_factor=NEUTRAL_WEATHER_FACTOR,
        air_density_factor=NEUTRAL_WEATHER_FACTOR,
        humidity_factor=NEUTRAL_WEATHER_FACTOR,
        wind_factor=NEUTRAL_WEATHER_FACTOR,
        rain_risk=NEUTRAL_WEATHER_FACTOR,
        weather_composite=NEUTRAL_WEATHER_FACTOR,
        is_dome=True,
    )


def _wind_multiplier(wind_factor: float) -> float:
    multiplier = NEUTRAL_WEATHER_FACTOR + (float(wind_factor) * WIND_COMPOSITE_FACTOR_PER_MPH)
    return _clamp(multiplier, MIN_FACTOR, MAX_FACTOR)


def compute_weather_adjustment(
    weather: WeatherData,
    *,
    team_code: str | None = None,
    venue: str | None = None,
    is_dome: bool | None = None,
    center_field_orientation_deg: float | None = None,
    precipitation_probability: float | None = None,
) -> WeatherAdjustment:
    """Compute weather adjustment factors and a composite score for a game."""

    context = get_weather_context(team_code=team_code, venue=venue)
    resolved_is_dome = bool(weather.is_dome_default) if is_dome is None else bool(is_dome)
    resolved_is_dome = resolved_is_dome or context.is_dome
    if resolved_is_dome:
        return _neutral_weather_adjustment()

    resolved_orientation = (
        float(center_field_orientation_deg)
        if center_field_orientation_deg is not None
        else context.center_field_orientation_deg
    )

    temp_factor = calculate_temp_factor(weather.temperature_f)
    air_density_factor = calculate_air_density_factor(weather.air_density)
    humidity_factor = calculate_humidity_factor(weather.humidity_pct)
    wind_factor = calculate_wind_factor(
        weather.wind_speed_mph,
        weather.wind_direction_deg,
        resolved_orientation,
    )
    rain_risk = calculate_rain_risk(
        humidity_pct=weather.humidity_pct,
        precipitation_probability=precipitation_probability,
    )
    weather_composite = (
        temp_factor
        * air_density_factor
        * humidity_factor
        * _wind_multiplier(wind_factor)
        * rain_risk
    )

    return WeatherAdjustment(
        temp_factor=temp_factor,
        air_density_factor=air_density_factor,
        humidity_factor=humidity_factor,
        wind_factor=wind_factor,
        rain_risk=rain_risk,
        weather_composite=weather_composite,
        is_dome=False,
    )
