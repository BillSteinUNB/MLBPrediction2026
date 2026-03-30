from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field


def _ensure_utc_datetime(value: datetime) -> datetime:
    """Require timezone-aware datetimes and normalize them to UTC."""

    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError("datetime value must be timezone-aware")

    return value.astimezone(timezone.utc)


def _validate_american_odds(value: int) -> int:
    """Validate standard American odds values."""

    if value == 0 or abs(value) < 100:
        raise ValueError("American odds must be <= -100 or >= 100")

    return value


class ModelBase(BaseModel):
    """Shared base model for all repository Pydantic models."""

    model_config = ConfigDict(extra="forbid")


Probability = Annotated[float, Field(ge=0, le=1)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
UtcDatetime = Annotated[datetime, AfterValidator(_ensure_utc_datetime)]
AmericanOdds = Annotated[int, AfterValidator(_validate_american_odds)]
MarketType = Literal[
    "full_game_ml",
    "full_game_rl",
    "full_game_total",
    "full_game_team_total_home",
    "full_game_team_total_away",
    "f5_ml",
    "f5_rl",
    "f5_total",
]
BetSide = Literal["home", "away", "over", "under"]
