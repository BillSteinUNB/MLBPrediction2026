from __future__ import annotations

from pydantic import Field, field_validator

from src.models._base import ModelBase, UtcDatetime


class WeatherData(ModelBase):
    """Weather snapshot for a scheduled game."""

    temperature_f: float
    humidity_pct: float = Field(ge=0, le=100)
    wind_speed_mph: float = Field(ge=0)
    wind_direction_deg: float = Field(ge=0, lt=360)
    pressure_hpa: float = Field(gt=0)
    air_density: float = Field(gt=0)
    wind_factor: float
    is_dome_default: bool = False
    forecast_time: UtcDatetime | None = None
    fetched_at: UtcDatetime | None = None

    @field_validator("wind_direction_deg", mode="before")
    @classmethod
    def normalize_wind_direction(cls, value: float | int | None) -> float:
        if value is None:
            return 0.0
        return float(value) % 360
