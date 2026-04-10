from __future__ import annotations

import pytest

from src.config import _load_settings_yaml
from src.features.adjustments.abs_adjustment import (
    apply_abs_adjustments,
    build_abs_challenge_proxy_context,
    estimate_framing_retention_proxy,
    is_abs_active,
    is_abs_exception_venue,
)
from src.features.adjustments.park_factors import (
    LEAGUE_AVERAGE_PARK_FACTOR,
    adjust_for_park,
    get_park_factors,
    load_park_factors,
)


def test_load_park_factors_returns_all_stadiums_and_sutter_health_park_values() -> None:
    park_factors = load_park_factors()

    assert len(park_factors) == 30
    assert park_factors["OAK"].park_name == "Sutter Health Park"
    assert park_factors["OAK"].runs == pytest.approx(1.25)
    assert park_factors["OAK"].hr == pytest.approx(1.30)


def test_adjust_for_park_scales_metric_and_uses_league_average_for_unknown_venue() -> None:
    coors = get_park_factors(team_code="COL")
    unknown = get_park_factors(venue="Temporary Exhibition Park")

    assert adjust_for_park(4.0, coors.runs) == pytest.approx(5.08)
    assert unknown.runs == pytest.approx(LEAGUE_AVERAGE_PARK_FACTOR)
    assert unknown.hr == pytest.approx(LEAGUE_AVERAGE_PARK_FACTOR)
    assert adjust_for_park(4.0, unknown.runs) == pytest.approx(4.0)


def test_apply_abs_adjustments_uses_configured_walk_and_strikeout_deltas() -> None:
    settings = _load_settings_yaml()
    result = apply_abs_adjustments(
        walk_rate=0.08,
        strikeout_rate=0.25,
        venue="Yankee Stadium",
    )

    assert result.abs_active is True
    assert result.walk_rate_delta == pytest.approx(settings["abs_adjustments"]["walk_rate_delta"])
    assert result.strikeout_rate_delta == pytest.approx(
        settings["abs_adjustments"]["strikeout_rate_delta"]
    )
    assert result.adjusted_walk_rate == pytest.approx(0.0832)
    assert result.adjusted_strikeout_rate == pytest.approx(0.2425)


def test_abs_exception_venues_disable_adjustments_and_match_case_insensitively() -> None:
    mexico_city_result = apply_abs_adjustments(
        walk_rate=0.08,
        strikeout_rate=0.25,
        venue="Alfredo Harp Helu Stadium - Mexico City Series",
    )

    assert is_abs_exception_venue("FIELD OF DREAMS presented by GEICO") is True
    assert is_abs_active("Little League Classic at Bowman Field") is False
    assert mexico_city_result.abs_active is False
    assert mexico_city_result.adjusted_walk_rate == pytest.approx(0.08)
    assert mexico_city_result.adjusted_strikeout_rate == pytest.approx(0.25)


def test_abs_challenge_proxy_context_models_proxy_pressure_without_event_logs() -> None:
    aggressive = build_abs_challenge_proxy_context(
        abs_active=True,
        lineup_walk_rate=9.6,
        lineup_strikeout_rate=19.8,
        lineup_quality=0.334,
        abs_walk_rate_delta=0.04,
        abs_strikeout_rate_delta=-0.03,
        umpire_zone_suppression=0.8,
        umpire_zone_volatility=1.1,
        umpire_abs_active_share=0.72,
        framing_retention_proxy=0.80,
        framing_stability=0.84,
        framing_zone_support=0.65,
        run_environment_anchor=1.09,
        market_anchor_confidence=0.85,
    )
    passive = build_abs_challenge_proxy_context(
        abs_active=True,
        lineup_walk_rate=7.6,
        lineup_strikeout_rate=24.8,
        lineup_quality=0.302,
        abs_walk_rate_delta=0.04,
        abs_strikeout_rate_delta=-0.03,
        umpire_zone_suppression=0.2,
        umpire_zone_volatility=0.3,
        umpire_abs_active_share=0.30,
        framing_retention_proxy=0.68,
        framing_stability=0.60,
        framing_zone_support=0.10,
        run_environment_anchor=0.97,
        market_anchor_confidence=0.20,
    )
    inactive = build_abs_challenge_proxy_context(
        abs_active=False,
        lineup_walk_rate=9.6,
        lineup_strikeout_rate=19.8,
        lineup_quality=0.334,
    )

    assert aggressive.challenge_opportunity_proxy > passive.challenge_opportunity_proxy
    assert aggressive.challenge_pressure_proxy > passive.challenge_pressure_proxy
    assert aggressive.challenge_conservation_proxy > passive.challenge_conservation_proxy
    assert aggressive.leverage_framing_retention_proxy > passive.leverage_framing_retention_proxy
    assert aggressive.umpire_zone_suppression_proxy > passive.umpire_zone_suppression_proxy
    assert inactive.challenge_pressure_proxy == pytest.approx(0.0)
    assert inactive.leverage_framing_retention_proxy == pytest.approx(1.0)


def test_estimate_framing_retention_proxy_uses_adjusted_ratio_and_neutralizes_non_abs_games() -> None:
    assert estimate_framing_retention_proxy(4.0, adjusted_framing_runs=3.0, abs_active=True) == pytest.approx(0.75)
    assert estimate_framing_retention_proxy(0.0, adjusted_framing_runs=0.0, abs_active=True) == pytest.approx(0.75)
    assert estimate_framing_retention_proxy(4.0, adjusted_framing_runs=4.0, abs_active=False) == pytest.approx(1.0)
