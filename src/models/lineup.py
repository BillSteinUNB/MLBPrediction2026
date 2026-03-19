from __future__ import annotations

from pydantic import Field, model_validator

from src.models._base import ModelBase, UtcDatetime


class LineupPlayer(ModelBase):
    """Single batting-order entry for a team's lineup."""

    batting_order: int = Field(ge=1, le=9)
    player_id: int
    player_name: str
    position: str | None = None


class Lineup(ModelBase):
    """Projected or confirmed lineup for one team in one game."""

    game_pk: int
    team: str = Field(min_length=2, max_length=3)
    source: str
    confirmed: bool = False
    as_of_timestamp: UtcDatetime
    starting_pitcher_id: int | None = None
    projected_starting_pitcher_id: int | None = None
    starter_avg_innings_pitched: float | None = Field(default=None, ge=0)
    is_opener: bool = False
    is_bullpen_game: bool = False
    players: list[LineupPlayer] = Field(default_factory=list, max_length=9)

    @model_validator(mode="after")
    def validate_players(self) -> Lineup:
        """Ensure batting-order slots are unique when players are present."""

        batting_orders = [player.batting_order for player in self.players]
        if len(batting_orders) != len(set(batting_orders)):
            raise ValueError("lineup batting_order values must be unique")

        return self
