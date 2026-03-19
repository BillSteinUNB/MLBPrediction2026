from __future__ import annotations

import pytest

from src.config import _load_settings_yaml
from src.features.adjustments.abs_adjustment import (
    apply_abs_adjustments,
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
