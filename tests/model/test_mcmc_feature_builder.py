from __future__ import annotations

import pytest

from src.model.mcmc_feature_builder import build_mcmc_feature_bundle


def test_build_mcmc_feature_bundle_uses_abs_regime_proxies_in_called_strike_environment() -> None:
    aggressive_abs = build_mcmc_feature_bundle(
        {
            "abs_challenge_opportunity_proxy": 0.72,
            "abs_expected_challenge_pressure_proxy": 0.78,
            "abs_challenge_conservation_proxy": 0.70,
            "abs_leverage_framing_retention_proxy": 0.88,
            "abs_umpire_zone_suppression_proxy": 0.62,
        },
        target_mean_runs=4.6,
    )
    depleted_abs = build_mcmc_feature_bundle(
        {
            "abs_challenge_opportunity_proxy": 0.35,
            "abs_expected_challenge_pressure_proxy": 0.88,
            "abs_challenge_conservation_proxy": 0.20,
            "abs_leverage_framing_retention_proxy": 0.48,
            "abs_umpire_zone_suppression_proxy": 0.05,
        },
        target_mean_runs=4.6,
    )

    assert (
        aggressive_abs.called_strike_environment_factor
        > depleted_abs.called_strike_environment_factor
    )
    assert aggressive_abs.post_anchor_implied_mean_runs == pytest.approx(aggressive_abs.target_mean_runs, abs=1e-6)
    assert aggressive_abs.mean_anchor_applied is True
    assert aggressive_abs.fallback_applied is False
    assert aggressive_abs.raw_feature_snapshot["abs_challenge_opportunity_proxy"] == 0.72
    assert aggressive_abs.raw_feature_snapshot["abs_leverage_framing_retention_proxy"] == 0.88


def test_build_mcmc_feature_bundle_reanchors_extreme_environment_rows() -> None:
    extreme = build_mcmc_feature_bundle(
        {
            "park_runs_factor": 1.35,
            "park_hr_factor": 1.42,
            "weather_composite": 1.24,
            "weather_air_density_factor": 1.16,
            "market_run_environment_anchor": 1.28,
            "market_anchor_confidence": 1.0,
            "away_lineup_iso_7g": 0.240,
            "away_lineup_barrel_pct_7g": 12.0,
            "away_team_runs_scored_7g": 6.8,
        },
        target_mean_runs=5.8,
    )

    assert extreme.mean_anchor_applied is True
    assert abs(extreme.post_anchor_implied_mean_runs - 5.8) <= 0.5
    assert extreme.starter_profile.out >= 0.56
    assert extreme.starter_profile.home_run <= 0.055
