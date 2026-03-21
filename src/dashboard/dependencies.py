from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class DashboardSettings(BaseSettings):
    """Dashboard-specific settings for data directories."""

    models_dir: Path = Path("data") / "models"
    experiments_dir: Path = Path("data") / "experiments"

    model_config = SettingsConfigDict(env_prefix="DASHBOARD_", extra="ignore")


_settings: DashboardSettings | None = None


def get_settings() -> DashboardSettings:
    """Get singleton dashboard settings instance."""
    global _settings
    if _settings is None:
        _settings = DashboardSettings()
    return _settings


def get_models_dir() -> Path:
    """Dependency: Get models directory path."""
    return get_settings().models_dir


def get_experiments_dir() -> Path:
    """Dependency: Get experiments directory path."""
    return get_settings().experiments_dir
