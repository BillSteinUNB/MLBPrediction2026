from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

import src.clients.weather_client as weather_client
from src.clients.weather_client import (
    _OPEN_METEO_HISTORICAL_RESPONSE_CACHE,
    _WEATHER_MEMORY_CACHE,
    _cache_weather,
    _build_weather_data_from_open_meteo,
    _can_fetch_forecast_for_game_time,
    _calculate_air_density,
    _calculate_wind_factor,
    _find_closest_forecast,
    _find_closest_historical_hour,
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


@pytest.fixture(autouse=True)
def clear_historical_response_cache() -> None:
    _OPEN_METEO_HISTORICAL_RESPONSE_CACHE.clear()
    _WEATHER_MEMORY_CACHE.clear()
    weather_client._STADIUMS_CACHE = None
    yield
    _OPEN_METEO_HISTORICAL_RESPONSE_CACHE.clear()
    _WEATHER_MEMORY_CACHE.clear()
    weather_client._STADIUMS_CACHE = None


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


def test_find_closest_historical_hour_prefers_same_or_previous_hour() -> None:
    hourly_payload = {
        "time": [
            "2021-07-15T18:00",
            "2021-07-15T20:00",
        ],
        "temperature_2m": [80.0, 77.0],
        "wind_speed_10m": [12.0, 8.0],
        "wind_direction_10m": [225.0, 180.0],
        "precipitation": [0.0, 0.4],
        "relative_humidity_2m": [55.0, 60.0],
        "cloud_cover": [25.0, 65.0],
        "surface_pressure": [1012.0, 1011.0],
    }

    match = _find_closest_historical_hour(
        hourly_payload,
        datetime(2021, 7, 15, 19, 5, tzinfo=UTC),
    )

    assert match is not None
    assert match["time"] == "2021-07-15T18:00"


def test_build_weather_data_from_open_meteo_uses_requested_units() -> None:
    retrieved_at = datetime.now(UTC)

    weather = _build_weather_data_from_open_meteo(
        {
            "time": "2020-08-20T19:00",
            "temperature_2m": 88.0,
            "wind_speed_10m": 14.0,
            "wind_direction_10m": 225.0,
            "precipitation": 0.2,
            "relative_humidity_2m": 30.0,
            "cloud_cover": 18.0,
            "surface_pressure": 840.0,
        },
        stadium_cf_orientation_deg=45.0,
        retrieved_at=retrieved_at,
    )

    assert weather.temperature_f == pytest.approx(88.0)
    assert weather.wind_speed_mph == pytest.approx(14.0)
    assert weather.pressure_hpa == pytest.approx(840.0)
    assert weather.air_density < 1.1
    assert weather.wind_factor > 0
    assert weather.precipitation_probability is None
    assert weather.precipitation_mm == pytest.approx(0.2)
    assert weather.cloud_cover_pct == pytest.approx(18.0)
    assert weather.forecast_time == datetime(2020, 8, 20, 19, 0, tzinfo=UTC)
    assert weather.fetched_at == retrieved_at


def test_can_fetch_forecast_for_game_time_only_allows_openweather_future_window() -> None:
    now = datetime.now(UTC)

    assert _can_fetch_forecast_for_game_time(now + timedelta(hours=1)) is True
    assert _can_fetch_forecast_for_game_time(now - timedelta(minutes=1)) is False
    assert _can_fetch_forecast_for_game_time(now + timedelta(days=6)) is False


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
        forecast_time=None,
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
        precipitation_probability=0.35,
        precipitation_mm=1.2,
        cloud_cover_pct=82.0,
        is_dome_default=False,
        forecast_time=GAME_TIME,
        fetched_at=datetime.now(UTC),
    )

    _cache_weather(db_path, "NYY", GAME_TIME, fresh_weather)
    cached_weather = _get_cached_weather(db_path, "NYY", GAME_TIME)

    assert cached_weather is not None
    assert cached_weather.temperature_f == pytest.approx(72.5)
    assert cached_weather.wind_factor == pytest.approx(5.5)
    assert cached_weather.precipitation_probability == pytest.approx(0.35)
    assert cached_weather.precipitation_mm == pytest.approx(1.2)
    assert cached_weather.cloud_cover_pct == pytest.approx(82.0)
    assert cached_weather.forecast_time == GAME_TIME

    stale_weather = fresh_weather.model_copy(
        update={"fetched_at": datetime.now(UTC) - timedelta(hours=8)}
    )
    _cache_weather(db_path, "NYY", GAME_TIME, stale_weather)

    assert _get_cached_weather(db_path, "NYY", GAME_TIME) is None

    historical_game_time = datetime.now(UTC) - timedelta(days=30)
    historical_stale_weather = fresh_weather.model_copy(
        update={
            "forecast_time": historical_game_time,
            "fetched_at": datetime.now(UTC) - timedelta(hours=24),
        }
    )
    _cache_weather(db_path, "LAD", historical_game_time, historical_stale_weather)

    cached_historical_weather = _get_cached_weather(db_path, "LAD", historical_game_time)

    assert cached_historical_weather is not None
    assert cached_historical_weather.temperature_f == pytest.approx(72.5)


def test_fetch_game_weather_uses_retrieval_time_for_cache_ttl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "weather.db"
    monkeypatch.setattr("src.clients.weather_client._load_settings_yaml", _stadium_settings)
    monkeypatch.setattr(
        "src.clients.weather_client._can_fetch_forecast_for_game_time",
        lambda _value: True,
    )

    forecast_time = GAME_TIME + timedelta(hours=3)
    payload = {
        "list": [
            {
                "dt": int(forecast_time.timestamp()),
                "main": {"temp": 298.15, "humidity": 60, "pressure": 1015},
                "wind": {"speed": 4.4704, "deg": 180},
            }
        ]
    }

    monkeypatch.setattr(
        "src.clients.weather_client._fetch_from_api",
        lambda **_kwargs: payload,
    )

    retrieved_before = datetime.now(UTC)
    weather = fetch_game_weather("NYY", GAME_TIME, api_key="test-key", db_path=db_path)
    retrieved_after = datetime.now(UTC)

    assert weather.forecast_time == forecast_time
    assert weather.fetched_at is not None
    assert retrieved_before <= weather.fetched_at <= retrieved_after
    assert weather.fetched_at != weather.forecast_time

    cached_weather = _get_cached_weather(db_path, "NYY", GAME_TIME)

    assert cached_weather == weather


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


def test_fetch_game_weather_uses_open_meteo_for_historical_open_air_games(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("src.clients.weather_client._load_settings_yaml", _stadium_settings)

    forecast_calls = {"count": 0}
    historical_calls = {"count": 0}

    def _unexpected_fetch(**_kwargs: object) -> dict[str, object]:
        forecast_calls["count"] += 1
        raise AssertionError("historical weather lookup should not hit forecast API")

    monkeypatch.setattr("src.clients.weather_client._fetch_from_api", _unexpected_fetch)
    monkeypatch.setattr(
        "src.clients.weather_client._fetch_from_open_meteo_historical",
        lambda **_kwargs: historical_calls.__setitem__("count", historical_calls["count"] + 1)
        or {
            "hourly": {
                "time": ["2026-02-24T19:00"],
                "temperature_2m": [61.0],
                "wind_speed_10m": [11.0],
                "wind_direction_10m": [180.0],
                "precipitation": [0.0],
                "relative_humidity_2m": [47.0],
                "cloud_cover": [35.0],
                "surface_pressure": [1016.0],
            }
        },
    )

    historical_game_time = datetime(2026, 2, 24, 19, 10, tzinfo=UTC)
    weather = fetch_game_weather("NYY", historical_game_time, db_path=tmp_path / "weather.db")

    assert weather.temperature_f == pytest.approx(61.0)
    assert weather.wind_speed_mph == pytest.approx(11.0)
    assert weather.forecast_time == datetime(2026, 2, 24, 19, 0, tzinfo=UTC)
    assert forecast_calls["count"] == 0
    assert historical_calls["count"] == 1


def test_fetch_game_weather_reuses_open_meteo_day_payload_for_same_team_and_date(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "weather.db"
    monkeypatch.setattr("src.clients.weather_client._load_settings_yaml", _stadium_settings)

    calls = {"count": 0}

    def _historical_payload(**_kwargs: object) -> dict[str, object]:
        calls["count"] += 1
        return {
            "hourly": {
                "time": ["2026-02-24T18:00", "2026-02-24T21:00"],
                "temperature_2m": [58.0, 54.0],
                "wind_speed_10m": [9.0, 6.0],
                "wind_direction_10m": [180.0, 90.0],
                "precipitation": [0.0, 0.1],
                "relative_humidity_2m": [52.0, 63.0],
                "cloud_cover": [20.0, 75.0],
                "surface_pressure": [1015.0, 1013.0],
            }
        }

    monkeypatch.setattr(
        "src.clients.weather_client._fetch_from_open_meteo_historical",
        _historical_payload,
    )

    first_weather = fetch_game_weather("NYY", datetime(2026, 2, 24, 18, 5, tzinfo=UTC), db_path=db_path)
    second_weather = fetch_game_weather("NYY", datetime(2026, 2, 24, 21, 5, tzinfo=UTC), db_path=db_path)

    assert calls["count"] == 1
    assert first_weather.temperature_f == pytest.approx(58.0)
    assert second_weather.temperature_f == pytest.approx(54.0)


def test_fetch_game_weather_fetches_open_air_and_then_uses_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "weather.db"
    monkeypatch.setattr("src.clients.weather_client._load_settings_yaml", _stadium_settings)
    monkeypatch.setattr(
        "src.clients.weather_client._can_fetch_forecast_for_game_time",
        lambda _value: True,
    )

    payload = {
        "list": [
            {
                "dt": int(GAME_TIME.timestamp()),
                "main": {"temp": 298.15, "humidity": 60, "pressure": 1015},
                "wind": {"speed": 4.4704, "deg": 180},
                "pop": 0.42,
                "rain": {"3h": 1.8},
                "clouds": {"all": 74},
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
    assert fetched_weather.precipitation_probability == pytest.approx(0.42)
    assert fetched_weather.precipitation_mm == pytest.approx(1.8)
    assert fetched_weather.cloud_cover_pct == pytest.approx(74.0)
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


def test_fetch_game_weather_logs_forecast_miss_at_debug(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr("src.clients.weather_client._load_settings_yaml", _stadium_settings)
    monkeypatch.setattr(
        "src.clients.weather_client._fetch_from_api",
        lambda **_kwargs: {"list": []},
    )

    with caplog.at_level("WARNING"):
        weather = fetch_game_weather("LAD", GAME_TIME, api_key="test-key", db_path=tmp_path / "x.db")

    assert weather == _get_default_weather(is_dome=False)
    assert "No forecast within" not in caplog.text


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
        forecast_time=datetime.now(UTC),
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
