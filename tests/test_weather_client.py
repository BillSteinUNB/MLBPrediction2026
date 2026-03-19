from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from src.clients.weather_client import (
    _cache_weather,
    _calculate_air_density,
    _calculate_wind_factor,
    _find_closest_forecast,
    _get_cached_weather,
    _get_default_weather,
    _kelvin_to_fahrenheit,
    _mps_to_mph,
    _pa_to_hpa,
    _resolve_api_key,
    fetch_game_weather,
)
from src.models.weather import WeatherData


UTC = timezone.utc
GAME_TIME = datetime(2026, 7, 4, 19, 0, tzinfo=UTC)


def _stadium_settings() -> dict[str, object]:
    return {
        "stadiums": {
            "NYY": {
                "latitude": 40.8296,
                "longitude": -73.9262,
                "is_dome": False,
                "center_field_orientation_deg": 0,
            },
            "TB": {
                "latitude": 27.7682,
                "longitude": -82.6534,
                "is_dome": True,
                "center_field_orientation_deg": 45,
            },
            "LAD": {
                "latitude": 34.0739,
                "longitude": -118.24,
                "is_dome": False,
                "center_field_orientation_deg": 40,
            },
        }
    }


def test_unit_conversions() -> None:
    assert _kelvin_to_fahrenheit(273.15) == pytest.approx(32.0, abs=0.1)
    assert _mps_to_mph(10.0) == pytest.approx(22.37, abs=0.1)
    assert _pa_to_hpa(101325) == pytest.approx(1013.25, abs=0.01)


def test_air_density_matches_standard_conditions_and_humidity_effect() -> None:
    dry_density = _calculate_air_density(288.15, 101325, 0)
    humid_density = _calculate_air_density(288.15, 101325, 60)

    assert dry_density == pytest.approx(1.225, abs=0.01)
    assert humid_density < dry_density
    assert _calculate_air_density(0, 101325, 50) == pytest.approx(1.225)


def test_wind_factor_sign_matches_blowing_out_and_in() -> None:
    assert _calculate_wind_factor(10.0, 180, 0) > 0
    assert _calculate_wind_factor(10.0, 0, 0) < 0
    assert _calculate_wind_factor(10.0, 90, 0) == pytest.approx(0.0, abs=0.1)


def test_find_closest_forecast_respects_time_window() -> None:
    forecasts = [
        {"dt": int((GAME_TIME - timedelta(hours=1)).timestamp()), "main": {}},
        {"dt": int(GAME_TIME.timestamp()), "main": {}},
        {"dt": int((GAME_TIME + timedelta(hours=7)).timestamp()), "main": {}},
    ]

    match = _find_closest_forecast(forecasts, GAME_TIME, max_hours=6)

    assert match is not None
    assert match["dt"] == int(GAME_TIME.timestamp())


def test_get_default_weather_sets_neutral_values() -> None:
    weather = _get_default_weather(is_dome=True)

    assert weather == WeatherData(
        temperature_f=70.0,
        humidity_pct=50.0,
        wind_speed_mph=0.0,
        wind_direction_deg=0.0,
        pressure_hpa=1013.25,
        air_density=1.225,
        wind_factor=0.0,
        is_dome_default=True,
        fetched_at=None,
    )


def test_cache_round_trip_and_expiry(tmp_path: Path) -> None:
    db_path = tmp_path / "weather.db"
    fresh_weather = WeatherData(
        temperature_f=72.5,
        humidity_pct=55.0,
        wind_speed_mph=8.0,
        wind_direction_deg=180.0,
        pressure_hpa=1012.0,
        air_density=1.21,
        wind_factor=5.5,
        is_dome_default=False,
        fetched_at=datetime.now(UTC),
    )

    _cache_weather(db_path, "NYY", GAME_TIME, fresh_weather)
    cached_weather = _get_cached_weather(db_path, "NYY", GAME_TIME)

    assert cached_weather is not None
    assert cached_weather.temperature_f == pytest.approx(72.5)
    assert cached_weather.wind_factor == pytest.approx(5.5)

    stale_weather = fresh_weather.model_copy(
        update={"fetched_at": datetime.now(UTC) - timedelta(hours=8)}
    )
    _cache_weather(db_path, "NYY", GAME_TIME, stale_weather)

    assert _get_cached_weather(db_path, "NYY", GAME_TIME) is None


def test_resolve_api_key_prefers_parameter_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.clients.weather_client.load_dotenv", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("OPENWEATHER_API_KEY", "env-key")

    assert _resolve_api_key("param-key") == "param-key"
    assert _resolve_api_key() == "env-key"


def test_resolve_api_key_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.clients.weather_client.load_dotenv", lambda *_args, **_kwargs: None)
    monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OpenWeatherMap API key not found"):
        _resolve_api_key()


def test_fetch_game_weather_skips_dome_without_api_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.clients.weather_client._load_settings_yaml", _stadium_settings)

    calls = {"count": 0}

    def _unexpected_fetch(**_kwargs: object) -> dict[str, object]:
        calls["count"] += 1
        raise AssertionError("dome lookup should not hit weather API")

    monkeypatch.setattr("src.clients.weather_client._fetch_from_api", _unexpected_fetch)

    weather = fetch_game_weather("TB", GAME_TIME)

    assert weather.is_dome_default is True
    assert weather.wind_factor == 0.0
    assert calls["count"] == 0


def test_fetch_game_weather_fetches_open_air_and_then_uses_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "weather.db"
    monkeypatch.setattr("src.clients.weather_client._load_settings_yaml", _stadium_settings)

    payload = {
        "list": [
            {
                "dt": int(GAME_TIME.timestamp()),
                "main": {"temp": 298.15, "humidity": 60, "pressure": 1015},
                "wind": {"speed": 4.4704, "deg": 180},
            }
        ]
    }

    monkeypatch.setattr(
        "src.clients.weather_client._fetch_from_api",
        lambda **_kwargs: payload,
    )

    fetched_weather = fetch_game_weather(
        "NYY",
        GAME_TIME,
        api_key="test-key",
        db_path=db_path,
    )

    assert fetched_weather.temperature_f == pytest.approx(77.0, abs=0.1)
    assert fetched_weather.humidity_pct == 60
    assert fetched_weather.wind_speed_mph == pytest.approx(10.0, abs=0.1)
    assert fetched_weather.wind_factor > 0
    assert fetched_weather.is_dome_default is False

    monkeypatch.setattr(
        "src.clients.weather_client._fetch_from_api",
        lambda **_kwargs: pytest.fail("cached weather should avoid a second API call"),
    )

    cached_weather = fetch_game_weather("NYY", GAME_TIME, db_path=db_path)

    assert cached_weather == fetched_weather


def test_fetch_game_weather_returns_default_on_api_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("src.clients.weather_client._load_settings_yaml", _stadium_settings)
    monkeypatch.setattr(
        "src.clients.weather_client._fetch_from_api",
        lambda **_kwargs: (_ for _ in ()).throw(httpx.ConnectError("boom")),
    )

    weather = fetch_game_weather("LAD", GAME_TIME, api_key="test-key", db_path=tmp_path / "x.db")

    assert weather == _get_default_weather(is_dome=False)


def test_fetch_game_weather_invalid_team_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.clients.weather_client._load_settings_yaml", _stadium_settings)

    with pytest.raises(ValueError, match="Team XYZ not found"):
        fetch_game_weather("XYZ", GAME_TIME, api_key="test-key")


def test_weather_data_validates_ranges() -> None:
    WeatherData(
        temperature_f=72.0,
        humidity_pct=55.0,
        wind_speed_mph=8.0,
        wind_direction_deg=370.0,
        pressure_hpa=1013.0,
        air_density=1.21,
        wind_factor=5.5,
        is_dome_default=False,
        fetched_at=datetime.now(UTC),
    )

    with pytest.raises(ValidationError):
        WeatherData(
            temperature_f=72.0,
            humidity_pct=-10.0,
            wind_speed_mph=8.0,
            wind_direction_deg=180.0,
            pressure_hpa=1013.0,
            air_density=1.21,
            wind_factor=5.5,
            is_dome_default=False,
        )
