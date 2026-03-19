import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

from src.clients.weather_client import (
    _kelvin_to_fahrenheit,
    _mps_to_mph,
    _pa_to_hpa,
    _calculate_air_density,
    _calculate_wind_factor,
    _get_default_weather,
    _find_closest_forecast,
    _fetch_from_api,
    fetch_game_weather,
    _get_cached_weather,
    _cache_weather,
    _resolve_api_key,
)
from src.models.weather import WeatherData


class TestUnitConversions:
    """Test unit conversion functions."""

    def test_kelvin_to_fahrenheit_freezing(self):
        """Test conversion at freezing point."""
        assert _kelvin_to_fahrenheit(273.15) == pytest.approx(32.0, abs=0.1)

    def test_kelvin_to_fahrenheit_room_temp(self):
        """Test conversion at room temperature."""
        assert _kelvin_to_fahrenheit(293.15) == pytest.approx(68.0, abs=0.1)

    def test_mps_to_mph_conversion(self):
        """Test m/s to mph conversion."""
        assert _mps_to_mph(10.0) == pytest.approx(22.37, abs=0.1)

    def test_pa_to_hpa_conversion(self):
        """Test Pa to hPa conversion."""
        assert _pa_to_hpa(101325) == pytest.approx(1013.25, abs=0.1)


class TestAirDensity:
    """Test air density calculation."""

    def test_air_density_standard_conditions(self):
        """Test air density at standard sea-level conditions."""
        temp_k = 288.15
        pressure_pa = 101325
        humidity = 0
        density = _calculate_air_density(temp_k, pressure_pa, humidity)
        assert density == pytest.approx(1.225, abs=0.01)

    def test_air_density_with_humidity(self):
        """Test that humidity reduces air density."""
        temp_k = 288.15
        pressure_pa = 101325
        density_dry = _calculate_air_density(temp_k, pressure_pa, 0)
        density_humid = _calculate_air_density(temp_k, pressure_pa, 50)
        assert density_humid < density_dry

    def test_air_density_high_temperature(self):
        """Test air density decreases with temperature."""
        pressure_pa = 101325
        density_cool = _calculate_air_density(288.15, pressure_pa, 0)
        density_hot = _calculate_air_density(308.15, pressure_pa, 0)
        assert density_hot < density_cool

    def test_air_density_invalid_inputs(self):
        """Test air density returns nominal value for invalid inputs."""
        assert _calculate_air_density(0, 101325, 50) == 1.225
        assert _calculate_air_density(288.15, 0, 50) == 1.225
        assert _calculate_air_density(288.15, 101325, -10) == 1.225


class TestWindFactor:
    """Test wind factor calculation."""

    def test_wind_factor_wind_blowing_out(self):
        """Test positive wind factor when wind blows toward outfield."""
        wind_factor = _calculate_wind_factor(10.0, 180, 0)
        assert wind_factor > 0

    def test_wind_factor_wind_blowing_in(self):
        """Test negative wind factor when wind blows toward infield."""
        wind_factor = _calculate_wind_factor(10.0, 0, 0)
        assert wind_factor < 0

    def test_wind_factor_no_wind(self):
        """Test zero wind factor with no wind."""
        wind_factor = _calculate_wind_factor(0.0, 90, 0)
        assert wind_factor == 0.0

    def test_wind_factor_perpendicular(self):
        """Test wind factor is zero when wind perpendicular to stadium orientation."""
        wind_factor = _calculate_wind_factor(10.0, 90, 0)
        assert wind_factor == pytest.approx(0.0, abs=0.1)


class TestForecastFinding:
    """Test closest forecast matching."""

    def test_find_closest_forecast_exact_match(self):
        """Test finding exact forecast match."""
        target_time = datetime(2026, 7, 4, 19, 0, 0, tzinfo=timezone.utc)
        target_timestamp = target_time.timestamp()

        forecasts = [
            {"dt": target_timestamp - 3600, "temp": 70},
            {"dt": target_timestamp, "temp": 72},
            {"dt": target_timestamp + 3600, "temp": 71},
        ]

        result = _find_closest_forecast(forecasts, target_time)
        assert result is not None
        assert result["dt"] == target_timestamp
        assert result["temp"] == 72

    def test_find_closest_forecast_within_window(self):
        """Test finding closest forecast within max_hours window."""
        target_time = datetime(2026, 7, 4, 19, 0, 0, tzinfo=timezone.utc)
        target_timestamp = target_time.timestamp()

        forecasts = [
            {"dt": target_timestamp + 1800, "temp": 72},
            {"dt": target_timestamp + 7200, "temp": 71},
        ]

        result = _find_closest_forecast(forecasts, target_time, max_hours=3)
        assert result is not None
        assert result["temp"] == 72

    def test_find_closest_forecast_outside_window(self):
        """Test returning None when no forecast within window."""
        target_time = datetime(2026, 7, 4, 19, 0, 0, tzinfo=timezone.utc)
        target_timestamp = target_time.timestamp()

        forecasts = [
            {"dt": target_timestamp + 14400, "temp": 71},
        ]

        result = _find_closest_forecast(forecasts, target_time, max_hours=3)
        assert result is None


class TestDefaultWeather:
    """Test default weather generation."""

    def test_default_weather_open_air(self):
        """Test default weather for open-air stadium."""
        weather = _get_default_weather(is_dome=False)
        assert weather.temperature_f == 70.0
        assert weather.humidity_pct == 50.0
        assert weather.wind_speed_mph == 0.0
        assert weather.wind_factor == 0.0
        assert weather.is_dome_default is False

    def test_default_weather_dome(self):
        """Test default weather for domed stadium."""
        weather = _get_default_weather(is_dome=True)
        assert weather.is_dome_default is True
        assert weather.wind_factor == 0.0

    def test_default_weather_is_weatherdata(self):
        """Test default weather returns valid WeatherData."""
        weather = _get_default_weather()
        assert isinstance(weather, WeatherData)


class TestWeatherCache:
    """Test weather caching functionality."""

    def test_cache_and_retrieve_weather(self):
        """Test caching and retrieving weather data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_cache.db"

            team_abbr = "NYY"
            game_time = datetime(2026, 7, 4, 19, 0, 0, tzinfo=timezone.utc)

            weather = WeatherData(
                temperature_f=72.5,
                humidity_pct=55.0,
                wind_speed_mph=8.0,
                wind_direction_deg=180.0,
                pressure_hpa=1012.0,
                air_density=1.21,
                wind_factor=5.5,
                is_dome_default=False,
                fetched_at=datetime.now(timezone.utc),
            )

            _cache_weather(db_path, team_abbr, game_time, weather)
            cached = _get_cached_weather(db_path, team_abbr, game_time)

            assert cached is not None
            assert cached.temperature_f == pytest.approx(72.5, abs=0.1)
            assert cached.humidity_pct == 55.0
            assert cached.wind_factor == pytest.approx(5.5, abs=0.1)

    def test_cache_returns_none_for_missing_game(self):
        """Test that cache returns None for non-existent game."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_cache.db"

            team_abbr = "LAD"
            game_time = datetime(2026, 7, 5, 19, 0, 0, tzinfo=timezone.utc)

            cached = _get_cached_weather(db_path, team_abbr, game_time)
            assert cached is None

    def test_cache_expiry(self):
        """Test that old cache entries are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_cache.db"

            team_abbr = "BOS"
            game_time = datetime(2026, 7, 4, 19, 0, 0, tzinfo=timezone.utc)

            old_time = datetime.now(timezone.utc) - timedelta(hours=8)
            weather = WeatherData(
                temperature_f=72.0,
                humidity_pct=50.0,
                wind_speed_mph=5.0,
                wind_direction_deg=0.0,
                pressure_hpa=1013.0,
                air_density=1.225,
                wind_factor=0.0,
                is_dome_default=False,
                fetched_at=old_time,
            )

            _cache_weather(db_path, team_abbr, game_time, weather)
            cached = _get_cached_weather(db_path, team_abbr, game_time)
            assert cached is None


class TestAPIKeyResolution:
    """Test API key resolution."""

    def test_resolve_api_key_from_parameter(self):
        """Test that provided API key is used."""
        api_key = _resolve_api_key(api_key="test-key-123")
        assert api_key == "test-key-123"

    def test_resolve_api_key_from_env(self):
        """Test that env var is used when parameter not provided."""
        with patch("src.clients.weather_client.Settings") as mock_settings:
            mock_settings.return_value.openweathermap_api_key = "env-key-456"
            api_key = _resolve_api_key()
            assert api_key == "env-key-456"

    def test_resolve_api_key_missing_raises(self):
        """Test that ValueError raised when API key not available."""
        with patch("src.clients.weather_client.Settings") as mock_settings:
            mock_settings.return_value.openweathermap_api_key = None
            with pytest.raises(ValueError, match="OpenWeatherMap API key not found"):
                _resolve_api_key(api_key=None)


class TestFetchGameWeather:
    """Test main fetch_game_weather function."""

    def test_fetch_domed_stadium_returns_default(self):
        """Test that domed stadiums return defaults immediately."""
        with patch("src.clients.weather_client.Settings") as mock_settings:
            mock_settings.return_value.stadiums = {
                "MIA": {
                    "is_dome": True,
                    "latitude": 25.7282,
                    "longitude": -80.2294,
                    "center_field_orientation_deg": 0,
                }
            }

            weather = fetch_game_weather("MIA", "2026-07-04 19:00:00")

            assert weather.is_dome_default is True
            assert weather.wind_factor == 0.0

    def test_fetch_open_air_stadium_with_api_mock(self):
        """Test fetching weather for open-air stadium with mocked API."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_cache.db"

            game_time = datetime(2026, 7, 4, 19, 0, 0, tzinfo=timezone.utc)

            mock_response = {
                "list": [
                    {
                        "dt": int(game_time.timestamp()),
                        "main": {"temp": 25.0, "humidity": 60, "pressure": 1013.25},
                        "wind": {"speed": 4.0, "deg": 180},
                    }
                ]
            }

            with patch("src.clients.weather_client.Settings") as mock_settings:
                mock_settings.return_value.stadiums = {
                    "NYY": {
                        "is_dome": False,
                        "latitude": 40.8296,
                        "longitude": -73.9262,
                        "center_field_orientation_deg": 0,
                    }
                }
                mock_settings.return_value.openweathermap_api_key = "test-key"

                with patch("src.clients.weather_client._fetch_from_api") as mock_fetch:
                    mock_fetch.return_value = mock_response

                    weather = fetch_game_weather("NYY", game_time, db_path=db_path)

                    assert weather.temperature_f == pytest.approx(77.0, abs=0.5)
                    assert weather.humidity_pct == 60
                    assert weather.wind_speed_mph == pytest.approx(8.95, abs=0.1)
                    assert weather.is_dome_default is False

    def test_fetch_invalid_team_raises(self):
        """Test that invalid team abbreviation raises ValueError."""
        with patch("src.clients.weather_client.Settings") as mock_settings:
            mock_settings.return_value.stadiums = {"NYY": {}}

            with pytest.raises(ValueError, match="Team XYZ not found"):
                fetch_game_weather("XYZ", "2026-07-04 19:00:00")

    def test_fetch_api_failure_returns_default(self):
        """Test that API failure gracefully returns defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_cache.db"

            with patch("src.clients.weather_client.Settings") as mock_settings:
                mock_settings.return_value.stadiums = {
                    "LAD": {
                        "is_dome": False,
                        "latitude": 34.0742,
                        "longitude": -118.2437,
                        "center_field_orientation_deg": 90,
                    }
                }
                mock_settings.return_value.openweathermap_api_key = "test-key"

                with patch("src.clients.weather_client._fetch_from_api") as mock_fetch:
                    mock_fetch.return_value = None

                    weather = fetch_game_weather("LAD", "2026-07-04 19:00:00", db_path=db_path)

                    assert weather.temperature_f == 70.0
                    assert weather.wind_factor == 0.0
                    assert weather.is_dome_default is False


class TestWeatherDataModel:
    """Test WeatherData Pydantic model."""

    def test_weather_data_valid(self):
        """Test creating valid WeatherData."""
        weather = WeatherData(
            temperature_f=72.0,
            humidity_pct=55.0,
            wind_speed_mph=8.0,
            wind_direction_deg=180.0,
            pressure_hpa=1013.0,
            air_density=1.21,
            wind_factor=5.5,
            is_dome_default=False,
            fetched_at=datetime.now(timezone.utc),
        )
        assert weather.temperature_f == 72.0

    def test_weather_data_rejects_negative_values(self):
        """Test that WeatherData rejects negative humidity."""
        from pydantic import ValidationError

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
