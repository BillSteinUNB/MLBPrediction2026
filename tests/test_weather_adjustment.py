from __future__ import annotations

import pytest

from src.clients.weather_client import _calculate_air_density
from src.features.adjustments.weather import (
    NEUTRAL_WEATHER_FACTOR,
    WeatherAdjustment,
    calculate_air_density_factor,
    calculate_rain_risk,
    calculate_wind_factor,
    compute_weather_adjustment,
)
from src.models.weather import WeatherData


def _weather(
    *,
    temperature_f: float = 70.0,
    humidity_pct: float = 50.0,
    wind_speed_mph: float = 0.0,
    wind_direction_deg: float = 0.0,
    pressure_hpa: float = 1013.25,
    air_density: float = 1.225,
    is_dome_default: bool = False,
) -> WeatherData:
    return WeatherData(
        temperature_f=temperature_f,
        humidity_pct=humidity_pct,
        wind_speed_mph=wind_speed_mph,
        wind_direction_deg=wind_direction_deg,
        pressure_hpa=pressure_hpa,
        air_density=air_density,
        wind_factor=0.0,
        is_dome_default=is_dome_default,
        forecast_time=None,
        fetched_at=None,
    )


def test_compute_weather_adjustment_returns_neutral_values_for_domed_stadiums() -> None:
    result = compute_weather_adjustment(
        _weather(
            temperature_f=92.0,
            humidity_pct=88.0,
            wind_speed_mph=18.0,
            wind_direction_deg=180.0,
            air_density=1.17,
            is_dome_default=True,
        ),
        team_code="TB",
        precipitation_probability=0.9,
    )

    assert result == WeatherAdjustment(
        temp_factor=NEUTRAL_WEATHER_FACTOR,
        air_density_factor=NEUTRAL_WEATHER_FACTOR,
        humidity_factor=NEUTRAL_WEATHER_FACTOR,
        wind_factor=NEUTRAL_WEATHER_FACTOR,
        rain_risk=NEUTRAL_WEATHER_FACTOR,
        weather_composite=NEUTRAL_WEATHER_FACTOR,
        is_dome=True,
    )


def test_compute_weather_adjustment_calculates_all_open_air_factors() -> None:
    result = compute_weather_adjustment(
        _weather(
            temperature_f=90.0,
            humidity_pct=70.0,
            wind_speed_mph=12.0,
            wind_direction_deg=35.0,
            air_density=1.18,
        ),
        team_code="NYY",
        precipitation_probability=0.2,
    )

    assert result.is_dome is False
    assert result.temp_factor == pytest.approx(1.05)
    assert result.air_density_factor == pytest.approx(1.054)
    assert result.humidity_factor == pytest.approx(1.01)
    assert result.wind_factor == pytest.approx(12.0)
    assert result.rain_risk == pytest.approx(0.97)
    assert result.weather_composite == pytest.approx(1.2143420688, rel=1e-6)


def test_wind_factor_uses_cosine_projection_for_blowing_out_in_and_crosswind() -> None:
    assert calculate_wind_factor(10.0, 0.0, 0.0) == pytest.approx(10.0)
    assert calculate_wind_factor(10.0, 180.0, 0.0) == pytest.approx(-10.0)
    assert calculate_wind_factor(10.0, 90.0, 0.0) == pytest.approx(0.0, abs=1e-9)


def test_air_density_factor_rewards_less_dense_more_humid_air() -> None:
    dry_density = _calculate_air_density(295.15, 101325.0, 0.0)
    humid_density = _calculate_air_density(295.15, 101325.0, 80.0)

    assert humid_density < dry_density
    assert calculate_air_density_factor(humid_density) > calculate_air_density_factor(dry_density)


def test_rain_risk_uses_precip_probability_or_humidity_proxy() -> None:
    assert calculate_rain_risk(humidity_pct=85.0, precipitation_probability=0.8) == pytest.approx(0.88)
    assert calculate_rain_risk(humidity_pct=95.0, precipitation_probability=None) == pytest.approx(0.9583333333)
    assert calculate_rain_risk(humidity_pct=50.0, precipitation_probability=None) == pytest.approx(1.0)
