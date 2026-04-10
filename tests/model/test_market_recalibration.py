from __future__ import annotations

from src.model.market_recalibration import shrink_probability_toward_market


def test_shrink_probability_toward_market_reduces_plus_money_high_edge_more_aggressively() -> None:
    adjusted = shrink_probability_toward_market(
        model_probability=0.62,
        fair_probability=0.48,
        odds=140,
        base_multiplier=0.8,
        plus_money_multiplier=0.5,
        high_edge_threshold=0.10,
        high_edge_multiplier=0.5,
    )

    assert adjusted < 0.62
    assert adjusted > 0.48


def test_shrink_probability_toward_market_is_identity_at_full_multipliers() -> None:
    adjusted = shrink_probability_toward_market(
        model_probability=0.58,
        fair_probability=0.51,
        odds=-120,
        base_multiplier=1.0,
        plus_money_multiplier=1.0,
        high_edge_threshold=0.10,
        high_edge_multiplier=1.0,
    )

    assert adjusted == 0.58

