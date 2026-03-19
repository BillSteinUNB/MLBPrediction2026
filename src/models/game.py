from __future__ import annotations

from typing import Literal

from pydantic import Field

from src.models._base import ModelBase, UtcDatetime


GameStatus = Literal["scheduled", "final", "suspended", "postponed", "cancelled"]


class Game(ModelBase):
    """Canonical game metadata and score state."""

    game_pk: int
    scheduled_start: UtcDatetime
    home_team: str = Field(min_length=2, max_length=3)
    away_team: str = Field(min_length=2, max_length=3)
    home_starter_id: int | None = None
    away_starter_id: int | None = None
    venue: str = Field(min_length=1)
    is_dome: bool = False
    is_abs_active: bool = True
    status: GameStatus = "scheduled"
    f5_home_score: int | None = Field(default=None, ge=0)
    f5_away_score: int | None = Field(default=None, ge=0)
    final_home_score: int | None = Field(default=None, ge=0)
    final_away_score: int | None = Field(default=None, ge=0)
