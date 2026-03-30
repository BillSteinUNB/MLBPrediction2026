from __future__ import annotations

import numpy as np
import pytest

from src.model.mcmc_engine import (
    BASE_OUT_STATE_COUNT,
    apply_event_to_state,
    decode_state_index,
    distribution_quantiles,
    expected_runs_for_game,
    expected_runs_per_half_inning,
    normalize_event_probabilities,
    simulate_away_game_distribution,
    simulate_half_inning,
    state_index,
    summarize_distribution_shape,
)


def test_state_index_and_decode_cover_24_base_out_states() -> None:
    seen = {
        state_index(outs=outs, bases=bases)
        for outs in range(3)
        for bases in range(8)
    }

    assert len(seen) == BASE_OUT_STATE_COUNT
    assert decode_state_index(state_index(outs=2, bases=7)) == (2, 7)


def test_state_transitions_cover_walk_single_and_home_run() -> None:
    next_outs, next_bases, runs, inning_over = apply_event_to_state(outs=1, bases=7, event="walk_hbp")
    assert (next_outs, next_bases, runs, inning_over) == (1, 7, 1, False)

    next_outs, next_bases, runs, inning_over = apply_event_to_state(outs=0, bases=6, event="single")
    assert (next_outs, next_bases, runs, inning_over) == (0, 5, 1, False)

    next_outs, next_bases, runs, inning_over = apply_event_to_state(outs=2, bases=5, event="home_run")
    assert (next_outs, next_bases, runs, inning_over) == (2, 0, 3, False)


def test_half_inning_with_only_outs_terminates_in_three_plate_appearances() -> None:
    rng = np.random.default_rng(7)
    inning = simulate_half_inning(
        normalize_event_probabilities({"out": 1.0}),
        simulations=8,
        rng=rng,
    )

    np.testing.assert_array_equal(inning.runs, np.zeros(8, dtype=int))
    assert inning.mean_plate_appearances == 3.0
    assert inning.event_counts["out"] == 24
    assert inning.truncated_half_innings == 0


def test_expected_runs_helpers_increase_with_more_positive_event_mass() -> None:
    conservative = normalize_event_probabilities(
        {"out": 0.73, "walk_hbp": 0.07, "single": 0.12, "double": 0.04, "triple": 0.01, "home_run": 0.03}
    )
    aggressive = normalize_event_probabilities(
        {"out": 0.63, "walk_hbp": 0.09, "single": 0.16, "double": 0.06, "triple": 0.01, "home_run": 0.05}
    )

    assert expected_runs_per_half_inning(aggressive) > expected_runs_per_half_inning(conservative)
    assert expected_runs_for_game(starter_profile=aggressive, bullpen_profile=conservative, starter_innings=5) > 0.0


def test_game_distribution_normalizes_and_is_reproducible_with_fixed_seed() -> None:
    starter_profile = normalize_event_probabilities(
        {
            "out": 0.69,
            "walk_hbp": 0.08,
            "single": 0.14,
            "double": 0.05,
            "triple": 0.01,
            "home_run": 0.03,
        }
    )
    bullpen_profile = normalize_event_probabilities(
        {
            "out": 0.67,
            "walk_hbp": 0.09,
            "single": 0.15,
            "double": 0.05,
            "triple": 0.01,
            "home_run": 0.03,
        }
    )

    first = simulate_away_game_distribution(
        starter_profile=starter_profile,
        bullpen_profile=bullpen_profile,
        simulations=512,
        seed=20260328,
    )
    second = simulate_away_game_distribution(
        starter_profile=starter_profile,
        bullpen_profile=bullpen_profile,
        simulations=512,
        seed=20260328,
    )

    np.testing.assert_allclose(first.pmf.sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(first.pmf, second.pmf, atol=1e-12)
    np.testing.assert_array_equal(first.simulated_runs, second.simulated_runs)
    assert first.quantiles["p50"] >= first.quantiles["p25"]
    assert first.shape_summary["entropy"] > 0.0
    assert first.diagnostics["truncated_half_innings"] == 0


def test_distribution_quantiles_and_shape_summary_cover_discrete_run_distribution() -> None:
    support = np.array([0, 1, 2, 3], dtype=int)
    probabilities = np.array([0.10, 0.20, 0.30, 0.40], dtype=float)

    quantiles = distribution_quantiles(support, probabilities, quantiles=(0.25, 0.50, 0.75))
    shape = summarize_distribution_shape(support, probabilities, quantiles=quantiles)

    assert quantiles == {"p25": 1.0, "p50": 2.0, "p75": 3.0}
    assert shape["variance"] == pytest.approx(1.0)
    assert shape["stddev"] == pytest.approx(1.0)
    assert shape["iqr"] == pytest.approx(2.0)
    assert shape["entropy"] == pytest.approx(1.2798542258)
