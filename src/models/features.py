from __future__ import annotations

from pydantic import Field

from src.models._base import ModelBase, UtcDatetime


class GameFeatures(ModelBase):
    """A single anti-leakage-safe feature observation for one game."""

    game_pk: int
    feature_name: str = Field(min_length=1)
    feature_value: float
    window_size: int | None = Field(default=None, ge=1)
    as_of_timestamp: UtcDatetime
