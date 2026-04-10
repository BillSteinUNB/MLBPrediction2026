from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.config import Settings


REPO_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = REPO_ROOT / "config" / "settings.yaml"


def load_yaml() -> dict[str, object]:
    return yaml.safe_load(SETTINGS_PATH.read_text(encoding="utf-8"))


def test_settings_yaml_contains_all_teams_and_required_constants() -> None:
    payload = load_yaml()

    assert len(payload["teams"]) == 30
    assert len(payload["stadiums"]) == 30
    assert sorted(payload["rolling_windows"]) == [7, 14, 30, 60]
    assert {
        "Mexico City",
        "Field of Dreams",
        "Little League Classic",
    }.issubset(set(payload["abs_exceptions"]))


def test_sutter_health_park_uses_2025_mlb_factors() -> None:
    payload = load_yaml()
    oakland = payload["stadiums"]["OAK"]

    assert oakland["park_name"] == "Sutter Health Park"
    assert oakland["park_factors"]["runs"] == pytest.approx(1.25)
    assert oakland["park_factors"]["hr"] == pytest.approx(1.30)


def test_settings_loads_env_and_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ODDS_API_KEY", "odds-test-key")
    monkeypatch.setenv("OPENWEATHER_API_KEY", "weather-test-key")
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/123/abc")

    settings = Settings()

    assert len(settings.teams) == 30
    assert settings.stadiums["OAK"]["park_name"] == "Sutter Health Park"
    assert settings.thresholds.edge_min == pytest.approx(0.03)
    assert settings.rolling_windows == [7, 14, 30, 60]


def test_settings_raises_on_missing_required_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
