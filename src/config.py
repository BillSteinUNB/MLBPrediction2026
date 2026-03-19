from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import AnyHttpUrl, BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
DEFAULT_SETTINGS_FILE = REPO_ROOT / "config" / "settings.yaml"


def _load_settings_yaml() -> dict[str, Any]:
    if not DEFAULT_SETTINGS_FILE.exists():
        raise FileNotFoundError(f"Settings YAML not found: {DEFAULT_SETTINGS_FILE}")

    payload = yaml.safe_load(DEFAULT_SETTINGS_FILE.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config/settings.yaml must contain a top-level mapping")

    return payload


class ConfigSection(BaseModel):
    """Base model that supports dictionary-style access for config sections."""

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class TeamConfig(ConfigSection):
    full_name: str
    city: str
    nickname: str
    league: str
    division: str


class ParkFactors(ConfigSection):
    runs: float = Field(gt=0)
    hr: float = Field(gt=0)


class StadiumConfig(ConfigSection):
    park_name: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    altitude_ft: int
    is_dome: bool
    roof_type: str
    center_field_orientation_deg: int = Field(ge=0, lt=360)
    park_factors: ParkFactors


class Thresholds(ConfigSection):
    edge_min: float = Field(gt=0, lt=1)
    kelly_fraction: float = Field(gt=0, le=1)
    max_drawdown: float = Field(gt=0, le=1)
    min_games_rolling: int = Field(gt=0)


class AbsAdjustments(ConfigSection):
    walk_rate_delta: float
    strikeout_rate_delta: float


class Settings(BaseSettings):
    """Load required environment variables and static YAML configuration."""

    model_config = SettingsConfigDict(
        env_file=DEFAULT_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    odds_api_key: SecretStr = Field(validation_alias="ODDS_API_KEY")
    openweather_api_key: SecretStr = Field(validation_alias="OPENWEATHER_API_KEY")
    discord_webhook_url: AnyHttpUrl = Field(validation_alias="DISCORD_WEBHOOK_URL")

    teams: dict[str, TeamConfig]
    stadiums: dict[str, StadiumConfig]
    abs_exceptions: list[str]
    thresholds: Thresholds
    rolling_windows: list[int]
    pythagorean_exponent: float = Field(gt=0)
    abs_retention_factor: float = Field(gt=0, le=1)
    abs_adjustments: AbsAdjustments

    @model_validator(mode="before")
    @classmethod
    def merge_yaml_defaults(cls, data: Any) -> Any:
        merged = dict(data) if isinstance(data, dict) else {}

        for key, value in _load_settings_yaml().items():
            merged.setdefault(key, value)

        return merged

    @model_validator(mode="after")
    def validate_static_config(self) -> Settings:
        team_codes = set(self.teams)
        stadium_codes = set(self.stadiums)

        if len(team_codes) != 30:
            raise ValueError("settings.yaml must define exactly 30 teams")

        if len(stadium_codes) != 30:
            raise ValueError("settings.yaml must define exactly 30 stadiums")

        if team_codes != stadium_codes:
            raise ValueError("teams and stadiums must use the same 30 team codes")

        if sorted(self.rolling_windows) != [7, 14, 30, 60]:
            raise ValueError("rolling_windows must contain 7, 14, 30, and 60")

        required_exceptions = {"Mexico City", "Field of Dreams", "Little League Classic"}
        if not required_exceptions.issubset(self.abs_exceptions):
            raise ValueError("abs_exceptions must include Mexico City, Field of Dreams, and Little League Classic")

        if self.thresholds.min_games_rolling != min(self.rolling_windows):
            raise ValueError("thresholds.min_games_rolling must match the smallest rolling window")

        return self


def get_settings() -> Settings:
    """Return validated application settings."""

    return Settings()
