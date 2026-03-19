from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from src.models._base import (
    AmericanOdds,
    BetSide,
    MarketType,
    ModelBase,
    NonNegativeFloat,
    Probability,
    UtcDatetime,
)


class BetResult(StrEnum):
    """Supported bet settlement outcomes."""

    WIN = "WIN"
    LOSS = "LOSS"
    PUSH = "PUSH"
    NO_ACTION = "NO_ACTION"
    PENDING = "PENDING"


class BetDecision(ModelBase):
    """Decision-engine record for whether and how to place a bet."""

    game_pk: int
    market_type: MarketType
    side: BetSide
    model_probability: Probability
    fair_probability: Probability
    edge_pct: float
    ev: float
    is_positive_ev: bool
    kelly_stake: NonNegativeFloat = 0.0
    odds_at_bet: AmericanOdds
    result: BetResult = BetResult.PENDING
    settled_at: UtcDatetime | None = None
    profit_loss: float | None = None
