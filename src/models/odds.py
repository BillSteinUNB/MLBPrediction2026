from __future__ import annotations

from src.models._base import AmericanOdds, MarketType, ModelBase, UtcDatetime


class OddsSnapshot(ModelBase):
    """Raw sportsbook odds captured for a single market snapshot."""

    game_pk: int
    book_name: str
    market_type: MarketType
    home_odds: AmericanOdds
    away_odds: AmericanOdds
    fetched_at: UtcDatetime
    is_frozen: bool = False
