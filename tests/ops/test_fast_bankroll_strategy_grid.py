from __future__ import annotations

import pandas as pd

from src.ops.fast_bankroll_strategy_grid import build_default_variants
from src.ops.fast_bankroll_strategy_grid import _filter_base_bets
from src.ops.fast_bankroll_strategy_grid import size_bet_units


def test_build_default_variants_creates_edge_window_grid() -> None:
    variants = build_default_variants(
        standard_edge_pcts=(0.03, 0.05),
        odds_windows=((-150, 120), (-300, 200)),
    )

    assert len(variants) == 4
    assert variants[0].name == "e30_-150_120"


def test_size_bet_units_is_nonlinear_and_capped() -> None:
    assert size_bet_units(edge_pct=0.03, standard_edge_pct=0.03, american_odds=-110) == 1.0
    assert size_bet_units(edge_pct=0.06, standard_edge_pct=0.03, american_odds=-110) == 5.0
    assert size_bet_units(edge_pct=0.02, standard_edge_pct=0.03, american_odds=-260) >= 0.5
    assert size_bet_units(edge_pct=0.40, standard_edge_pct=0.03, american_odds=100) == 5.0


def test_filter_base_bets_applies_market_and_coinflip_rl_filters() -> None:
    base_bets = pd.DataFrame(
        [
            {"market_type": "full_game_ml", "odds_at_bet": -120},
            {"market_type": "full_game_rl", "odds_at_bet": -110},
            {"market_type": "full_game_rl", "odds_at_bet": -300},
            {"market_type": "f5_ml", "odds_at_bet": 100},
        ]
    )

    filtered = _filter_base_bets(
        base_bets=base_bets,
        include_markets=("full_game_ml", "full_game_rl", "full_game_total"),
        rl_coinflip_min_odds=-140,
        rl_coinflip_max_odds=105,
    )

    assert list(filtered["market_type"]) == ["full_game_ml", "full_game_rl"]
    assert list(filtered["odds_at_bet"]) == [-120, -110]
