from __future__ import annotations

import pytest

from src.features.marcel_blend import calculate_marcel_blend, get_regression_weight


def test_get_regression_weight_uses_metric_type_defaults_and_overrides() -> None:
    assert get_regression_weight("offense") == 30.0
    assert get_regression_weight("pitching") == 15.0
    assert get_regression_weight("defense") == 50.0
    assert get_regression_weight("offense", regression_weights={"offense": 24.0}) == 24.0


def test_calculate_marcel_blend_converges_toward_current_value_as_games_increase() -> None:
    current_value = 0.360
    prior_value = 0.300

    ten_games = calculate_marcel_blend(
        current_value,
        games_played=10,
        metric_type="offense",
        prior_value=prior_value,
    )
    sixty_games = calculate_marcel_blend(
        current_value,
        games_played=60,
        metric_type="offense",
        prior_value=prior_value,
    )
    full_season = calculate_marcel_blend(
        current_value,
        games_played=162,
        metric_type="offense",
        prior_value=prior_value,
    )

    assert (
        prior_value
        < ten_games.blended_value
        < sixty_games.blended_value
        < full_season.blended_value
        < current_value
    )
    assert abs(current_value - full_season.blended_value) < abs(current_value - sixty_games.blended_value)
    assert abs(current_value - sixty_games.blended_value) < abs(current_value - ten_games.blended_value)


def test_calculate_marcel_blend_uses_league_average_for_first_year_players() -> None:
    result = calculate_marcel_blend(
        120.0,
        games_played=12,
        metric_type="offense",
        prior_value=None,
        league_average=100.0,
        is_first_year=True,
    )

    assert result.used_league_average
    assert result.resolved_prior_value == 100.0
    assert result.blended_value == pytest.approx((120.0 * 12 + 100.0 * 30) / 42)


def test_calculate_marcel_blend_halves_prior_weight_when_roster_turnover_exceeds_threshold() -> None:
    stable_roster = calculate_marcel_blend(
        4.20,
        games_played=10,
        metric_type="pitching",
        prior_value=3.50,
        roster_turnover_pct=0.40,
    )
    turnover_adjusted = calculate_marcel_blend(
        4.20,
        games_played=10,
        metric_type="pitching",
        prior_value=3.50,
        roster_turnover_pct=0.41,
    )

    assert stable_roster.prior_weight == 15.0
    assert not stable_roster.turnover_adjusted
    assert turnover_adjusted.prior_weight == 7.5
    assert turnover_adjusted.turnover_adjusted
    assert abs(4.20 - turnover_adjusted.blended_value) < abs(4.20 - stable_roster.blended_value)
