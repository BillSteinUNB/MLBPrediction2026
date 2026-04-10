from __future__ import annotations

import pandas as pd

from src.ops.market_strategy_sweep import (
    StrategyParams,
    _run_nested_walk_forward_sweep,
    _simulate_strategy,
)


def _params(*, strategy_mode: str) -> StrategyParams:
    return StrategyParams(
        strategy_mode=strategy_mode,  # type: ignore[arg-type]
        rl_source="direct",
        ml_market_base_multiplier=1.0,
        ml_market_plus_money_multiplier=1.0,
        ml_market_high_edge_threshold=0.10,
        ml_market_high_edge_multiplier=1.0,
        ml_edge_threshold=0.01,
        rl_edge_threshold=0.01,
        rl_selection_penalty=0.0,
        selector_score_mode="edge",
        staking_mode="flat",
        flat_bet_size_units=1.0,
        min_bet_size_units=0.25,
        max_bet_size_units=2.0,
        edge_scale_cap=0.05,
        max_bet_odds=150,
        min_bet_odds=-175,
        play_of_day_min_edge=0.01,
    )


def _evaluation_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    game_pk = 1000
    for season, ml_home_win, rl_home_cover, rl_home_odds in (
        (2021, 1, 0, 150),
        (2021, 1, 0, 150),
        (2022, 0, 1, 150),
        (2022, 0, 1, 150),
        (2023, 1, 1, 150),
        (2023, 1, 1, 150),
    ):
        rows.append(
            {
                "game_pk": game_pk,
                "season": season,
                "game_date": f"{season}-04-{(game_pk % 20) + 1:02d}",
                "scheduled_start": f"{season}-04-{(game_pk % 20) + 1:02d}T19:05:00+00:00",
                "home_team": "AAA",
                "away_team": "BBB",
                "f5_ml_result": ml_home_win,
                "f5_tied_after_5": 0,
                "ml_home_probability": 0.60,
                "ml_away_probability": 0.40,
                "ml_home_odds": 110,
                "ml_away_odds": -120,
                "home_cover_at_posted_line": rl_home_cover,
                "away_cover_at_posted_line": 1 - rl_home_cover,
                "push_at_posted_line": 0,
                "rl_home_odds": rl_home_odds,
                "rl_away_odds": -165,
                "rl_home_point": -0.5,
                "rl_away_point": 0.5,
                "rl_direct_home_probability": 0.60,
                "rl_margin_home_probability": 0.58,
                "rl_blend_home_probability": 0.59,
            }
        )
        game_pk += 1
    return pd.DataFrame(rows)


def test_simulate_strategy_respects_starting_bankroll_state() -> None:
    frame = _evaluation_frame().loc[lambda current: current["season"] == 2021].copy()

    metrics = _simulate_strategy(
        evaluation_frame=frame,
        params=_params(strategy_mode="ml_only"),
        starting_bankroll=250.0,
        starting_peak_bankroll=300.0,
    )

    assert metrics.starting_bankroll_units == 250.0
    assert metrics.peak_bankroll_units >= 300.0
    assert metrics.ending_bankroll_units > 250.0


def test_nested_walk_forward_selects_on_prior_seasons_and_carries_bankroll() -> None:
    leaderboard, payload = _run_nested_walk_forward_sweep(
        evaluation_frame=_evaluation_frame(),
        trial_count=0,
        allowed_modes=("ml_only", "rl_only"),
        include_baselines=True,
        baseline_params=[_params(strategy_mode="ml_only"), _params(strategy_mode="rl_only")],
        min_training_seasons=1,
    )

    assert leaderboard["test_season"].tolist() == [2022, 2023]
    assert leaderboard.loc[0, "strategy_mode"] == "ml_only"
    assert leaderboard.loc[1, "strategy_mode"] == "rl_only"
    assert leaderboard.loc[1, "test_starting_bankroll_units"] == leaderboard.loc[0, "test_ending_bankroll_units"]
    assert payload["aggregate_test_metrics"]["outer_window_count"] == 2
