from __future__ import annotations

import pytest

from src.model.score_pricing import (
    moneyline_probability,
    moneyline_probabilities,
    projected_margin,
    projected_total,
    spread_outcome_probabilities,
    spread_cover_probability,
    spread_cover_probabilities,
    totals_outcome_probabilities,
    totals_over_probability,
    totals_probabilities,
    totals_under_probability,
)


def test_projected_margin_and_total_are_direct_sums() -> None:
    assert projected_margin(home_runs_mean=4.3, away_runs_mean=3.1) == pytest.approx(1.2)
    assert projected_total(home_runs_mean=4.3, away_runs_mean=3.1) == pytest.approx(7.4)


def test_moneyline_probabilities_sum_to_one_and_move_with_margin() -> None:
    home_probability, away_probability = moneyline_probabilities(
        home_runs_mean=4.4,
        away_runs_mean=3.6,
        home_runs_std=1.8,
        away_runs_std=1.7,
    )
    assert home_probability is not None
    assert away_probability is not None
    assert home_probability + away_probability == pytest.approx(1.0)
    assert home_probability > 0.5

    lower_home_probability, _ = moneyline_probabilities(
        home_runs_mean=4.0,
        away_runs_mean=3.6,
        home_runs_std=1.8,
        away_runs_std=1.7,
    )
    assert lower_home_probability is not None
    assert lower_home_probability < home_probability


def test_single_value_moneyline_wrapper_matches_tuple_result() -> None:
    home_probability = moneyline_probability(
        home_runs_mean=5.0,
        away_runs_mean=3.5,
        home_runs_std=1.4,
        away_runs_std=1.6,
    )
    tuple_home_probability, _ = moneyline_probabilities(
        home_runs_mean=5.0,
        away_runs_mean=3.5,
        home_runs_std=1.4,
        away_runs_std=1.6,
    )
    assert home_probability == pytest.approx(tuple_home_probability)


def test_spread_cover_probabilities_sum_to_one_and_move_with_point() -> None:
    home_probability, away_probability = spread_cover_probabilities(
        home_runs_mean=4.6,
        away_runs_mean=3.9,
        home_runs_std=1.9,
        away_runs_std=1.8,
        home_point=-1.5,
    )
    assert home_probability is not None
    assert away_probability is not None
    assert home_probability + away_probability == pytest.approx(1.0)

    looser_home_probability = spread_cover_probability(
        home_runs_mean=4.6,
        away_runs_mean=3.9,
        home_runs_std=1.9,
        away_runs_std=1.8,
        home_point=-2.5,
    )
    assert looser_home_probability is not None
    assert looser_home_probability < home_probability


def test_spread_outcome_probabilities_include_push_mass_on_whole_number_lines() -> None:
    home_probability, away_probability, push_probability = spread_outcome_probabilities(
        home_runs_mean=4.6,
        away_runs_mean=3.9,
        home_runs_std=1.9,
        away_runs_std=1.8,
        home_point=-1.0,
    )

    assert home_probability is not None
    assert away_probability is not None
    assert push_probability is not None
    assert push_probability > 0.0
    assert home_probability + away_probability + push_probability == pytest.approx(1.0)


def test_totals_probabilities_sum_to_one_and_move_with_total_line() -> None:
    over_probability, under_probability = totals_probabilities(
        home_runs_mean=4.2,
        away_runs_mean=3.7,
        home_runs_std=1.8,
        away_runs_std=1.8,
        total_point=7.5,
    )
    assert over_probability is not None
    assert under_probability is not None
    assert over_probability + under_probability == pytest.approx(1.0)

    higher_total_over_probability = totals_over_probability(
        home_runs_mean=4.2,
        away_runs_mean=3.7,
        home_runs_std=1.8,
        away_runs_std=1.8,
        total_point=8.5,
    )
    assert higher_total_over_probability is not None
    assert higher_total_over_probability < over_probability


def test_totals_outcome_probabilities_include_push_mass_on_whole_number_lines() -> None:
    over_probability, under_probability, push_probability = totals_outcome_probabilities(
        home_runs_mean=4.2,
        away_runs_mean=3.7,
        home_runs_std=1.8,
        away_runs_std=1.8,
        total_point=8.0,
    )

    assert over_probability is not None
    assert under_probability is not None
    assert push_probability is not None
    assert push_probability > 0.0
    assert over_probability + under_probability + push_probability == pytest.approx(1.0)


def test_whole_number_totals_are_not_priced_like_adjacent_half_run_lines() -> None:
    over_ten, under_ten, push_ten = totals_outcome_probabilities(
        home_runs_mean=5.2,
        away_runs_mean=4.1,
        home_runs_std=3.13,
        away_runs_std=3.36,
        total_point=10.0,
    )
    over_ten_half, under_ten_half, push_ten_half = totals_outcome_probabilities(
        home_runs_mean=5.2,
        away_runs_mean=4.1,
        home_runs_std=3.13,
        away_runs_std=3.36,
        total_point=10.5,
    )

    assert push_ten is not None and push_ten > 0.0
    assert push_ten_half == pytest.approx(0.0)
    assert under_ten != pytest.approx(under_ten_half)
    assert over_ten == pytest.approx(over_ten_half)


def test_totals_under_wrapper_matches_tuple_result() -> None:
    under_probability = totals_under_probability(
        home_runs_mean=3.8,
        away_runs_mean=3.4,
        home_runs_std=1.6,
        away_runs_std=1.5,
        total_point=8.0,
    )
    tuple_over_probability, tuple_under_probability = totals_probabilities(
        home_runs_mean=3.8,
        away_runs_mean=3.4,
        home_runs_std=1.6,
        away_runs_std=1.5,
        total_point=8.0,
    )
    assert tuple_over_probability is not None
    assert under_probability == pytest.approx(tuple_under_probability)
