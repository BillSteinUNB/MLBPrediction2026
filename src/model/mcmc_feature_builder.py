from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.model.mcmc_engine import (
    DEFAULT_GAME_INNINGS,
    DEFAULT_STARTER_INNINGS,
    EventProbabilityProfile,
    expected_runs_for_game,
    normalize_event_probabilities,
)


LEAGUE_BASE_EVENT_RATES = {
    "out": 0.685,
    "walk_hbp": 0.085,
    "single": 0.145,
    "double": 0.045,
    "triple": 0.005,
    "home_run": 0.035,
}

LEAGUE_REFERENCE_VALUES = {
    "away_runs_mean": 4.481242,
    "away_lineup_woba_7g": 0.314909,
    "away_lineup_woba_30g": 0.314587,
    "away_lineup_xwoba_7g": 0.318681,
    "away_team_woba_vs_opposing_hand": 0.314339,
    "away_team_log5_30g": 0.499814,
    "away_team_runs_scored_7g": 4.517924,
    "away_lineup_bb_pct_7g": 8.19335,
    "away_lineup_bb_pct_30g": 8.206567,
    "away_lineup_k_pct_7g": 22.846972,
    "away_lineup_iso_7g": 0.161253,
    "away_lineup_barrel_pct_7g": 7.176048,
    "home_starter_k_pct_30s": 22.253942,
    "home_starter_k_pct_7s": 22.208927,
    "home_starter_bb_pct_30s": 7.948325,
    "home_starter_bb_pct_7s": 7.88563,
    "home_starter_xfip_30s": 4.693491,
    "home_starter_xera_30s": 3.437384,
    "home_starter_siera_30s": 4.774365,
    "home_team_bullpen_xfip": 4.130242,
    "park_runs_factor": 1.015674,
    "park_hr_factor": 1.02368,
    "weather_air_density_factor": 1.055647,
    "weather_temp_factor": 1.010925,
    "weather_composite": 1.08417,
    "plate_umpire_total_runs_avg_30g": 9.005371,
    "home_team_defensive_efficiency_30g": 0.713792,
    "home_team_oaa_30g": -0.051083,
    "away_matchup_ttop_exposure_index": 0.50,
    "away_matchup_ttop_penalty_index": 0.75,
    "home_starter_fatigue_escape_hatch": 1.0,
    "away_matchup_power_archetype_mismatch": 0.0,
    "away_matchup_contact_archetype_mismatch": 0.0,
    "away_matchup_platoon_archetype_edge": 0.0,
    "away_matchup_pitch_mix_surprise_gap": 0.0,
    "home_team_range_defense_index": 0.0,
    "away_bip_hit_probability_proxy": 0.235,
    "away_bip_extra_base_probability_proxy": 0.155,
    "away_bip_advancement_probability_proxy": 0.32,
    "away_weather_handedness_wind_gap": 0.0,
    "away_weather_power_wind_interaction": 0.0,
    "away_weather_air_density_power_interaction": 0.0,
    "plate_umpire_zone_suppression_index": 0.0,
    "plate_umpire_zone_volatility_index": 0.0,
    "home_team_framing_zone_support_index": 0.0,
    "home_team_framing_stability_index": 0.75,
    "abs_challenge_opportunity_proxy": 0.50,
    "abs_expected_challenge_pressure_proxy": 0.50,
    "abs_challenge_conservation_proxy": 0.50,
    "abs_leverage_framing_retention_proxy": 0.75,
    "abs_umpire_zone_suppression_proxy": 0.0,
    "market_implied_full_game_away_runs": 4.4,
    "market_run_environment_anchor": 1.0,
    "market_anchor_confidence": 0.0,
}

PROFILE_EVENT_BOUNDS = {
    "out": (0.56, 0.78),
    "walk_hbp": (0.045, 0.125),
    "single": (0.09, 0.24),
    "double": (0.02, 0.09),
    "triple": (0.001, 0.012),
    "home_run": (0.015, 0.055),
}

PRE_ANCHOR_MEAN_ABS_TOLERANCE = 0.75
PRE_ANCHOR_MEAN_REL_TOLERANCE = 0.12
POST_ANCHOR_MEAN_ABS_TOLERANCE = 0.50
POST_ANCHOR_MEAN_REL_TOLERANCE = 0.08


@dataclass(frozen=True, slots=True)
class MCMCFeatureBundle:
    target_mean_runs: float
    starter_profile: EventProbabilityProfile
    bullpen_profile: EventProbabilityProfile
    run_environment_factor: float
    offense_factor: float
    power_factor: float
    starter_pitching_factor: float
    bullpen_pitching_factor: float
    called_strike_environment_factor: float
    pre_anchor_implied_mean_runs: float
    post_anchor_implied_mean_runs: float
    mean_anchor_applied: bool
    fallback_applied: bool
    fallback_reason: str | None
    profile_out_of_bounds: bool
    raw_feature_snapshot: dict[str, float]
    diagnostics: dict[str, object]


def build_mcmc_feature_bundle(
    row: Mapping[str, object] | pd.Series,
    *,
    target_mean_runs: float,
    starter_innings: int = DEFAULT_STARTER_INNINGS,
    innings: int = DEFAULT_GAME_INNINGS,
) -> MCMCFeatureBundle:
    snapshot = {
        "target_mean_runs": float(target_mean_runs),
        "away_lineup_woba_7g": _coerce_feature(row, "away_lineup_woba_7g"),
        "away_lineup_woba_30g": _coerce_feature(row, "away_lineup_woba_30g"),
        "away_lineup_xwoba_7g": _coerce_feature(row, "away_lineup_xwoba_7g"),
        "away_team_woba_vs_opposing_hand": _coerce_feature(row, "away_team_woba_vs_opposing_hand"),
        "away_team_log5_30g": _coerce_feature(row, "away_team_log5_30g"),
        "away_team_runs_scored_7g": _coerce_feature(row, "away_team_runs_scored_7g"),
        "away_lineup_bb_pct_7g": _coerce_feature(row, "away_lineup_bb_pct_7g"),
        "away_lineup_bb_pct_30g": _coerce_feature(row, "away_lineup_bb_pct_30g"),
        "away_lineup_k_pct_7g": _coerce_feature(row, "away_lineup_k_pct_7g"),
        "away_lineup_iso_7g": _coerce_feature(row, "away_lineup_iso_7g"),
        "away_lineup_barrel_pct_7g": _coerce_feature(row, "away_lineup_barrel_pct_7g"),
        "home_starter_k_pct_30s": _coerce_feature(row, "home_starter_k_pct_30s"),
        "home_starter_k_pct_7s": _coerce_feature(row, "home_starter_k_pct_7s"),
        "home_starter_bb_pct_30s": _coerce_feature(row, "home_starter_bb_pct_30s"),
        "home_starter_bb_pct_7s": _coerce_feature(row, "home_starter_bb_pct_7s"),
        "home_starter_xfip_30s": _coerce_feature(row, "home_starter_xfip_30s"),
        "home_starter_xera_30s": _coerce_feature(row, "home_starter_xera_30s"),
        "home_starter_siera_30s": _coerce_feature(row, "home_starter_siera_30s"),
        "home_team_bullpen_xfip": _coerce_feature(row, "home_team_bullpen_xfip"),
        "park_runs_factor": _coerce_feature(row, "park_runs_factor"),
        "park_hr_factor": _coerce_feature(row, "park_hr_factor"),
        "weather_air_density_factor": _coerce_feature(row, "weather_air_density_factor"),
        "weather_temp_factor": _coerce_feature(row, "weather_temp_factor"),
        "weather_composite": _coerce_feature(row, "weather_composite"),
        "plate_umpire_total_runs_avg_30g": _coerce_feature(row, "plate_umpire_total_runs_avg_30g"),
        "home_team_defensive_efficiency_30g": _coerce_feature(row, "home_team_defensive_efficiency_30g"),
        "home_team_oaa_30g": _coerce_feature(row, "home_team_oaa_30g"),
        "home_team_adjusted_framing_30g": _coerce_feature(row, "home_team_adjusted_framing_30g", default=0.0),
        "away_matchup_ttop_exposure_index": _coerce_feature(row, "away_matchup_ttop_exposure_index"),
        "away_matchup_ttop_penalty_index": _coerce_feature(row, "away_matchup_ttop_penalty_index"),
        "home_starter_fatigue_escape_hatch": _coerce_feature(row, "home_starter_fatigue_escape_hatch"),
        "away_matchup_power_archetype_mismatch": _coerce_feature(row, "away_matchup_power_archetype_mismatch"),
        "away_matchup_contact_archetype_mismatch": _coerce_feature(row, "away_matchup_contact_archetype_mismatch"),
        "away_matchup_platoon_archetype_edge": _coerce_feature(row, "away_matchup_platoon_archetype_edge"),
        "away_matchup_pitch_mix_surprise_gap": _coerce_feature(row, "away_matchup_pitch_mix_surprise_gap"),
        "home_team_range_defense_index": _coerce_feature(row, "home_team_range_defense_index"),
        "away_bip_hit_probability_proxy": _coerce_feature(row, "away_bip_hit_probability_proxy"),
        "away_bip_extra_base_probability_proxy": _coerce_feature(row, "away_bip_extra_base_probability_proxy"),
        "away_bip_advancement_probability_proxy": _coerce_feature(row, "away_bip_advancement_probability_proxy"),
        "away_weather_handedness_wind_gap": _coerce_feature(row, "away_weather_handedness_wind_gap"),
        "away_weather_power_wind_interaction": _coerce_feature(row, "away_weather_power_wind_interaction"),
        "away_weather_air_density_power_interaction": _coerce_feature(row, "away_weather_air_density_power_interaction"),
        "plate_umpire_zone_suppression_index": _coerce_feature(row, "plate_umpire_zone_suppression_index"),
        "plate_umpire_zone_volatility_index": _coerce_feature(row, "plate_umpire_zone_volatility_index"),
        "home_team_framing_zone_support_index": _coerce_feature(row, "home_team_framing_zone_support_index"),
        "home_team_framing_stability_index": _coerce_feature(row, "home_team_framing_stability_index"),
        "abs_challenge_opportunity_proxy": _coerce_feature(row, "abs_challenge_opportunity_proxy"),
        "abs_expected_challenge_pressure_proxy": _coerce_feature(row, "abs_expected_challenge_pressure_proxy"),
        "abs_challenge_conservation_proxy": _coerce_feature(row, "abs_challenge_conservation_proxy"),
        "abs_leverage_framing_retention_proxy": _coerce_feature(row, "abs_leverage_framing_retention_proxy"),
        "abs_umpire_zone_suppression_proxy": _coerce_feature(row, "abs_umpire_zone_suppression_proxy"),
        "market_implied_full_game_away_runs": _coerce_feature(row, "market_implied_full_game_away_runs"),
        "market_run_environment_anchor": _coerce_feature(row, "market_run_environment_anchor"),
        "market_anchor_confidence": _coerce_feature(row, "market_anchor_confidence"),
    }

    target_run_factor = _clip(snapshot["target_mean_runs"] / LEAGUE_REFERENCE_VALUES["away_runs_mean"], 0.70, 1.35)
    market_target_factor = _clip(
        snapshot["market_implied_full_game_away_runs"] / LEAGUE_REFERENCE_VALUES["market_implied_full_game_away_runs"],
        0.70,
        1.35,
    )
    offense_factor = _compress_ratio(
        _weighted_mean(
            (
                snapshot["away_lineup_woba_7g"] / LEAGUE_REFERENCE_VALUES["away_lineup_woba_7g"],
                snapshot["away_lineup_woba_30g"] / LEAGUE_REFERENCE_VALUES["away_lineup_woba_30g"],
                snapshot["away_lineup_xwoba_7g"] / LEAGUE_REFERENCE_VALUES["away_lineup_xwoba_7g"],
                snapshot["away_team_woba_vs_opposing_hand"]
                / LEAGUE_REFERENCE_VALUES["away_team_woba_vs_opposing_hand"],
                snapshot["away_team_log5_30g"] / LEAGUE_REFERENCE_VALUES["away_team_log5_30g"],
                snapshot["away_team_runs_scored_7g"] / LEAGUE_REFERENCE_VALUES["away_team_runs_scored_7g"],
                snapshot["away_bip_hit_probability_proxy"] / LEAGUE_REFERENCE_VALUES["away_bip_hit_probability_proxy"],
                snapshot["away_bip_advancement_probability_proxy"]
                / LEAGUE_REFERENCE_VALUES["away_bip_advancement_probability_proxy"],
                target_run_factor,
                1.0 + ((market_target_factor - 1.0) * snapshot["market_anchor_confidence"] * 0.50),
            ),
            weights=(0.13, 0.07, 0.14, 0.11, 0.11, 0.09, 0.09, 0.08, 0.09, 0.09),
        ),
        lower=0.78,
        upper=1.28,
        strength=0.62,
    )
    walk_factor = _compress_ratio(
        _weighted_mean(
            (
                snapshot["away_lineup_bb_pct_7g"] / LEAGUE_REFERENCE_VALUES["away_lineup_bb_pct_7g"],
                snapshot["away_lineup_bb_pct_30g"] / LEAGUE_REFERENCE_VALUES["away_lineup_bb_pct_30g"],
                target_run_factor,
            ),
            weights=(0.45, 0.30, 0.25),
        ),
        lower=0.75,
        upper=1.30,
        strength=0.68,
    )
    strikeout_pressure = _compress_ratio(
        _weighted_mean(
            (
                snapshot["home_starter_k_pct_30s"] / LEAGUE_REFERENCE_VALUES["home_starter_k_pct_30s"],
                snapshot["home_starter_k_pct_7s"] / LEAGUE_REFERENCE_VALUES["home_starter_k_pct_7s"],
                snapshot["away_lineup_k_pct_7g"] / LEAGUE_REFERENCE_VALUES["away_lineup_k_pct_7g"],
            ),
            weights=(0.38, 0.32, 0.30),
        ),
        lower=0.82,
        upper=1.25,
        strength=0.66,
    )
    power_factor = _compress_ratio(
        _weighted_mean(
            (
                snapshot["away_lineup_iso_7g"] / LEAGUE_REFERENCE_VALUES["away_lineup_iso_7g"],
                snapshot["away_lineup_barrel_pct_7g"] / LEAGUE_REFERENCE_VALUES["away_lineup_barrel_pct_7g"],
                target_run_factor,
                snapshot["park_hr_factor"] / LEAGUE_REFERENCE_VALUES["park_hr_factor"],
                1.0 + (snapshot["away_matchup_power_archetype_mismatch"] * 0.08),
                1.0 + (
                    snapshot["away_weather_power_wind_interaction"]
                    / max(1e-6, abs(LEAGUE_REFERENCE_VALUES["away_weather_power_wind_interaction"]) + 1.0)
                ),
            ),
            weights=(0.26, 0.20, 0.16, 0.15, 0.13, 0.10),
        ),
        lower=0.80,
        upper=1.50,
        strength=0.55,
    )
    run_environment_factor = _compress_ratio(
        _weighted_mean(
            (
                snapshot["park_runs_factor"] / LEAGUE_REFERENCE_VALUES["park_runs_factor"],
                snapshot["weather_air_density_factor"] / LEAGUE_REFERENCE_VALUES["weather_air_density_factor"],
                snapshot["weather_temp_factor"] / LEAGUE_REFERENCE_VALUES["weather_temp_factor"],
                snapshot["weather_composite"] / LEAGUE_REFERENCE_VALUES["weather_composite"],
                snapshot["plate_umpire_total_runs_avg_30g"]
                / LEAGUE_REFERENCE_VALUES["plate_umpire_total_runs_avg_30g"],
                1.0 + (snapshot["away_weather_handedness_wind_gap"] * 0.02),
                1.0 + (snapshot["away_weather_air_density_power_interaction"] * 0.25),
                1.0 + ((snapshot["market_run_environment_anchor"] - 1.0) * snapshot["market_anchor_confidence"] * 0.55),
            ),
            weights=(0.20, 0.14, 0.08, 0.18, 0.14, 0.09, 0.07, 0.10),
        ),
        lower=0.88,
        upper=1.12,
        strength=0.58,
    )
    defense_factor = _compress_ratio(
        _weighted_mean(
            (
                snapshot["home_team_defensive_efficiency_30g"]
                / LEAGUE_REFERENCE_VALUES["home_team_defensive_efficiency_30g"],
                1.0 + (
                    (snapshot["home_team_oaa_30g"] - LEAGUE_REFERENCE_VALUES["home_team_oaa_30g"])
                    / 2.5
                ),
                1.0 + (snapshot["home_team_adjusted_framing_30g"] / 10.0),
                1.0 + (snapshot["home_team_range_defense_index"] * 0.10),
                1.0 + (snapshot["home_team_framing_zone_support_index"] * 0.06),
                1.0 + (
                    ((snapshot["abs_leverage_framing_retention_proxy"] - 0.75) / 0.25) * 0.08
                ),
            ),
            weights=(0.32, 0.22, 0.12, 0.16, 0.10, 0.08),
        ),
        lower=0.90,
        upper=1.12,
        strength=0.65,
    )
    called_strike_environment_factor = _compress_ratio(
        _weighted_mean(
            (
                1.0 + (snapshot["plate_umpire_zone_suppression_index"] * 0.06),
                1.0 + (snapshot["home_team_framing_zone_support_index"] * 0.05),
                snapshot["home_team_framing_stability_index"],
                1.0 - (snapshot["plate_umpire_zone_volatility_index"] * 0.04),
                1.0 + (snapshot["abs_umpire_zone_suppression_proxy"] * 0.05),
                1.0 + (
                    ((snapshot["abs_leverage_framing_retention_proxy"] - 0.75) / 0.25) * 0.10
                ),
                1.0 + ((snapshot["abs_challenge_conservation_proxy"] - 0.50) * 0.14),
                1.0 - ((snapshot["abs_expected_challenge_pressure_proxy"] - 0.50) * 0.18),
            ),
            weights=(0.22, 0.18, 0.15, 0.11, 0.12, 0.10, 0.07, 0.05),
        ),
        lower=0.88,
        upper=1.12,
        strength=0.62,
    )
    starter_pitching_factor = _compress_ratio(
        _weighted_mean(
            (
                snapshot["home_starter_k_pct_30s"] / LEAGUE_REFERENCE_VALUES["home_starter_k_pct_30s"],
                snapshot["home_starter_k_pct_7s"] / LEAGUE_REFERENCE_VALUES["home_starter_k_pct_7s"],
                LEAGUE_REFERENCE_VALUES["home_starter_xfip_30s"] / max(snapshot["home_starter_xfip_30s"], 1e-6),
                LEAGUE_REFERENCE_VALUES["home_starter_xera_30s"] / max(snapshot["home_starter_xera_30s"], 1e-6),
                LEAGUE_REFERENCE_VALUES["home_starter_siera_30s"] / max(snapshot["home_starter_siera_30s"], 1e-6),
                defense_factor,
                snapshot["home_starter_fatigue_escape_hatch"],
                1.0 - ((snapshot["away_matchup_ttop_penalty_index"] - 0.75) * 0.25),
                1.0 - (snapshot["away_matchup_contact_archetype_mismatch"] * 0.06),
            ),
            weights=(0.14, 0.11, 0.19, 0.17, 0.10, 0.09, 0.08, 0.07, 0.05),
        ),
        lower=0.82,
        upper=1.22,
        strength=0.62,
    )
    bullpen_pitching_factor = _compress_ratio(
        _weighted_mean(
            (
                LEAGUE_REFERENCE_VALUES["home_team_bullpen_xfip"] / max(snapshot["home_team_bullpen_xfip"], 1e-6),
                defense_factor,
                strikeout_pressure,
                called_strike_environment_factor,
            ),
            weights=(0.52, 0.18, 0.16, 0.14),
        ),
        lower=0.84,
        upper=1.18,
        strength=0.64,
    )
    walk_suppression = _compress_ratio(
        _weighted_mean(
            (
                LEAGUE_REFERENCE_VALUES["home_starter_bb_pct_30s"] / max(snapshot["home_starter_bb_pct_30s"], 1e-6),
                LEAGUE_REFERENCE_VALUES["home_starter_bb_pct_7s"] / max(snapshot["home_starter_bb_pct_7s"], 1e-6),
                called_strike_environment_factor,
                1.0 + (snapshot["abs_umpire_zone_suppression_proxy"] * 0.04),
                1.0 + ((snapshot["abs_challenge_conservation_proxy"] - 0.50) * 0.08),
                1.0 - ((snapshot["abs_expected_challenge_pressure_proxy"] - 0.50) * 0.10),
            ),
            weights=(0.34, 0.26, 0.16, 0.10, 0.08, 0.06),
        ),
        lower=0.82,
        upper=1.20,
        strength=0.66,
    )

    extreme_environment_signal = (
        max(0.0, (snapshot["park_runs_factor"] / LEAGUE_REFERENCE_VALUES["park_runs_factor"]) - 1.0)
        + max(0.0, (snapshot["weather_composite"] / LEAGUE_REFERENCE_VALUES["weather_composite"]) - 1.0)
        + max(
            0.0,
            (snapshot["market_run_environment_anchor"] - 1.0) * max(snapshot["market_anchor_confidence"], 0.0),
        )
    )
    if extreme_environment_signal > 0.08:
        compression = _clip(1.0 - ((extreme_environment_signal - 0.08) * 0.60), 0.78, 1.0)
        run_environment_factor = _shrink_toward_one(run_environment_factor, compression)
        power_factor = _shrink_toward_one(power_factor, compression)
        offense_factor = _shrink_toward_one(offense_factor, _clip(compression + 0.08, 0.82, 1.0))

    starter_profile = _build_event_profile(
        offense_factor=offense_factor,
        power_factor=power_factor,
        target_run_factor=target_run_factor,
        walk_factor=walk_factor,
        run_environment_factor=run_environment_factor,
        pitching_factor=starter_pitching_factor,
        walk_suppression=walk_suppression,
        called_strike_environment_factor=called_strike_environment_factor,
    )
    bullpen_profile = _build_event_profile(
        offense_factor=offense_factor,
        power_factor=power_factor,
        target_run_factor=target_run_factor,
        walk_factor=walk_factor,
        run_environment_factor=run_environment_factor,
        pitching_factor=bullpen_pitching_factor,
        walk_suppression=_clip((walk_suppression * 0.97) + 0.03, 0.65, 1.45),
        called_strike_environment_factor=_clip((called_strike_environment_factor * 0.98) + 0.02, 0.82, 1.18),
    )

    pre_anchor_implied_mean_runs = expected_runs_for_game(
        starter_profile=starter_profile,
        bullpen_profile=bullpen_profile,
        innings=innings,
        starter_innings=starter_innings,
    )
    profile_out_of_bounds = _profile_out_of_bounds(starter_profile) or _profile_out_of_bounds(bullpen_profile)
    anchored_starter_profile, anchored_bullpen_profile, post_anchor_implied_mean_runs = _anchor_profile_pair_to_target_mean(
        starter_profile=starter_profile,
        bullpen_profile=bullpen_profile,
        target_mean_runs=float(snapshot["target_mean_runs"]),
        innings=innings,
        starter_innings=starter_innings,
    )
    mean_anchor_applied = not np.isclose(
        pre_anchor_implied_mean_runs,
        post_anchor_implied_mean_runs,
        atol=1e-6,
    )

    fallback_reason: str | None = None
    if profile_out_of_bounds:
        fallback_reason = "profile_out_of_bounds"
    elif _mean_drift_exceeds_guardrail(
        target_mean_runs=float(snapshot["target_mean_runs"]),
        implied_mean_runs=post_anchor_implied_mean_runs,
        absolute_tolerance=POST_ANCHOR_MEAN_ABS_TOLERANCE,
        relative_tolerance=POST_ANCHOR_MEAN_REL_TOLERANCE,
    ):
        fallback_reason = "post_anchor_mean_drift"

    fallback_applied = fallback_reason is not None
    if fallback_applied:
        anchored_starter_profile, anchored_bullpen_profile, post_anchor_implied_mean_runs = _build_fallback_profile_pair(
            target_mean_runs=float(snapshot["target_mean_runs"]),
            innings=innings,
            starter_innings=starter_innings,
            offense_factor=offense_factor,
            power_factor=power_factor,
            run_environment_factor=run_environment_factor,
            starter_pitching_factor=starter_pitching_factor,
            bullpen_pitching_factor=bullpen_pitching_factor,
            walk_factor=walk_factor,
            walk_suppression=walk_suppression,
            called_strike_environment_factor=called_strike_environment_factor,
        )

    return MCMCFeatureBundle(
        target_mean_runs=float(snapshot["target_mean_runs"]),
        starter_profile=anchored_starter_profile,
        bullpen_profile=anchored_bullpen_profile,
        run_environment_factor=float(run_environment_factor),
        offense_factor=float(offense_factor),
        power_factor=float(power_factor),
        starter_pitching_factor=float(starter_pitching_factor),
        bullpen_pitching_factor=float(bullpen_pitching_factor),
        called_strike_environment_factor=float(called_strike_environment_factor),
        pre_anchor_implied_mean_runs=float(pre_anchor_implied_mean_runs),
        post_anchor_implied_mean_runs=float(post_anchor_implied_mean_runs),
        mean_anchor_applied=bool(mean_anchor_applied),
        fallback_applied=bool(fallback_applied),
        fallback_reason=fallback_reason,
        profile_out_of_bounds=bool(profile_out_of_bounds),
        raw_feature_snapshot=snapshot,
        diagnostics={
            "innings": int(max(1, int(innings))),
            "starter_innings": int(min(max(0, int(starter_innings)), max(1, int(innings)))),
            "pre_anchor_implied_mean_runs": float(pre_anchor_implied_mean_runs),
            "post_anchor_implied_mean_runs": float(post_anchor_implied_mean_runs),
            "mean_anchor_applied": bool(mean_anchor_applied),
            "fallback_applied": bool(fallback_applied),
            "fallback_reason": fallback_reason,
            "profile_out_of_bounds": bool(profile_out_of_bounds),
            "pre_anchor_mean_guardrail_failed": _mean_drift_exceeds_guardrail(
                target_mean_runs=float(snapshot["target_mean_runs"]),
                implied_mean_runs=pre_anchor_implied_mean_runs,
                absolute_tolerance=PRE_ANCHOR_MEAN_ABS_TOLERANCE,
                relative_tolerance=PRE_ANCHOR_MEAN_REL_TOLERANCE,
            ),
            "post_anchor_mean_guardrail_failed": _mean_drift_exceeds_guardrail(
                target_mean_runs=float(snapshot["target_mean_runs"]),
                implied_mean_runs=post_anchor_implied_mean_runs,
                absolute_tolerance=POST_ANCHOR_MEAN_ABS_TOLERANCE,
                relative_tolerance=POST_ANCHOR_MEAN_REL_TOLERANCE,
            ),
            "starter_profile_pre_anchor": starter_profile.as_dict(),
            "bullpen_profile_pre_anchor": bullpen_profile.as_dict(),
            "starter_profile_final": anchored_starter_profile.as_dict(),
            "bullpen_profile_final": anchored_bullpen_profile.as_dict(),
            "extreme_environment_signal": float(extreme_environment_signal),
        },
    )


def build_feature_bundles_for_frame(
    dataframe: pd.DataFrame,
    *,
    target_mean_predictions: Sequence[float] | np.ndarray,
    starter_innings: int = DEFAULT_STARTER_INNINGS,
    innings: int = DEFAULT_GAME_INNINGS,
) -> list[MCMCFeatureBundle]:
    means = np.asarray(target_mean_predictions, dtype=float)
    if len(means) != len(dataframe):
        raise ValueError("target_mean_predictions length must match dataframe length")
    return [
        build_mcmc_feature_bundle(
            row,
            target_mean_runs=float(mean_prediction),
            starter_innings=starter_innings,
            innings=innings,
        )
        for (_, row), mean_prediction in zip(dataframe.iterrows(), means, strict=True)
    ]


def _build_event_profile(
    *,
    offense_factor: float,
    power_factor: float,
    target_run_factor: float,
    walk_factor: float,
    run_environment_factor: float,
    pitching_factor: float,
    walk_suppression: float,
    called_strike_environment_factor: float,
) -> EventProbabilityProfile:
    raw = {
        "walk_hbp": LEAGUE_BASE_EVENT_RATES["walk_hbp"]
        * _clip(
            (walk_factor * (target_run_factor**0.30) * (run_environment_factor**0.10))
            / (walk_suppression * called_strike_environment_factor),
            0.75,
            1.35,
        ),
        "single": LEAGUE_BASE_EVENT_RATES["single"]
        * _clip(
            (offense_factor * (target_run_factor**0.55) * run_environment_factor)
            / pitching_factor,
            0.70,
            1.45,
        ),
        "double": LEAGUE_BASE_EVENT_RATES["double"]
        * _clip(
            (((0.55 * offense_factor) + (0.45 * power_factor)) * (target_run_factor**0.60) * run_environment_factor)
            / pitching_factor,
            0.65,
            1.40,
        ),
        "triple": LEAGUE_BASE_EVENT_RATES["triple"]
        * _clip(((0.70 * offense_factor) + (0.30 * run_environment_factor)) * (target_run_factor**0.25), 0.55, 1.10),
        "home_run": LEAGUE_BASE_EVENT_RATES["home_run"]
        * _clip((power_factor * (target_run_factor**0.72) * run_environment_factor) / pitching_factor, 0.65, 1.55),
        "out": LEAGUE_BASE_EVENT_RATES["out"]
        * _clip(
            (pitching_factor * called_strike_environment_factor)
            / ((target_run_factor**0.72) * (run_environment_factor**0.25)),
            0.72,
            1.30,
        ),
    }
    return _normalize_profile_with_bounds(raw)


def _build_fallback_profile_pair(
    *,
    target_mean_runs: float,
    innings: int,
    starter_innings: int,
    offense_factor: float,
    power_factor: float,
    run_environment_factor: float,
    starter_pitching_factor: float,
    bullpen_pitching_factor: float,
    walk_factor: float,
    walk_suppression: float,
    called_strike_environment_factor: float,
) -> tuple[EventProbabilityProfile, EventProbabilityProfile, float]:
    fallback_starter = _build_event_profile(
        offense_factor=_shrink_toward_one(offense_factor, 0.35),
        power_factor=_shrink_toward_one(power_factor, 0.25),
        target_run_factor=_clip(target_mean_runs / LEAGUE_REFERENCE_VALUES["away_runs_mean"], 0.78, 1.25),
        walk_factor=_shrink_toward_one(walk_factor, 0.35),
        run_environment_factor=_shrink_toward_one(run_environment_factor, 0.30),
        pitching_factor=_shrink_toward_one(starter_pitching_factor, 0.40),
        walk_suppression=_shrink_toward_one(walk_suppression, 0.40),
        called_strike_environment_factor=_shrink_toward_one(called_strike_environment_factor, 0.40),
    )
    fallback_bullpen = _build_event_profile(
        offense_factor=_shrink_toward_one(offense_factor, 0.35),
        power_factor=_shrink_toward_one(power_factor, 0.25),
        target_run_factor=_clip(target_mean_runs / LEAGUE_REFERENCE_VALUES["away_runs_mean"], 0.78, 1.25),
        walk_factor=_shrink_toward_one(walk_factor, 0.35),
        run_environment_factor=_shrink_toward_one(run_environment_factor, 0.30),
        pitching_factor=_shrink_toward_one(bullpen_pitching_factor, 0.40),
        walk_suppression=_shrink_toward_one(walk_suppression, 0.40),
        called_strike_environment_factor=_shrink_toward_one(called_strike_environment_factor, 0.40),
    )
    return _anchor_profile_pair_to_target_mean(
        starter_profile=fallback_starter,
        bullpen_profile=fallback_bullpen,
        target_mean_runs=target_mean_runs,
        innings=innings,
        starter_innings=starter_innings,
    )


def _anchor_profile_pair_to_target_mean(
    *,
    starter_profile: EventProbabilityProfile,
    bullpen_profile: EventProbabilityProfile,
    target_mean_runs: float,
    innings: int,
    starter_innings: int,
) -> tuple[EventProbabilityProfile, EventProbabilityProfile, float]:
    resolved_target_mean = max(float(target_mean_runs), 0.25)
    raw_implied_mean = expected_runs_for_game(
        starter_profile=starter_profile,
        bullpen_profile=bullpen_profile,
        innings=innings,
        starter_innings=starter_innings,
    )
    if not _mean_drift_exceeds_guardrail(
        target_mean_runs=resolved_target_mean,
        implied_mean_runs=raw_implied_mean,
        absolute_tolerance=POST_ANCHOR_MEAN_ABS_TOLERANCE,
        relative_tolerance=POST_ANCHOR_MEAN_REL_TOLERANCE,
    ):
        return starter_profile, bullpen_profile, float(raw_implied_mean)

    low, high = 0.45, 2.40
    low_mean = _implied_game_mean_for_positive_mass_scale(
        starter_profile=starter_profile,
        bullpen_profile=bullpen_profile,
        positive_mass_scale=low,
        innings=innings,
        starter_innings=starter_innings,
    )
    high_mean = _implied_game_mean_for_positive_mass_scale(
        starter_profile=starter_profile,
        bullpen_profile=bullpen_profile,
        positive_mass_scale=high,
        innings=innings,
        starter_innings=starter_innings,
    )

    if resolved_target_mean <= low_mean:
        anchored_starter = _scale_profile_positive_mass(starter_profile, low)
        anchored_bullpen = _scale_profile_positive_mass(bullpen_profile, low)
    elif resolved_target_mean >= high_mean:
        anchored_starter = _scale_profile_positive_mass(starter_profile, high)
        anchored_bullpen = _scale_profile_positive_mass(bullpen_profile, high)
    else:
        for _ in range(28):
            midpoint = (low + high) / 2.0
            midpoint_mean = _implied_game_mean_for_positive_mass_scale(
                starter_profile=starter_profile,
                bullpen_profile=bullpen_profile,
                positive_mass_scale=midpoint,
                innings=innings,
                starter_innings=starter_innings,
            )
            if midpoint_mean < resolved_target_mean:
                low = midpoint
            else:
                high = midpoint
        positive_mass_scale = (low + high) / 2.0
        anchored_starter = _scale_profile_positive_mass(starter_profile, positive_mass_scale)
        anchored_bullpen = _scale_profile_positive_mass(bullpen_profile, positive_mass_scale)

    anchored_mean = expected_runs_for_game(
        starter_profile=anchored_starter,
        bullpen_profile=anchored_bullpen,
        innings=innings,
        starter_innings=starter_innings,
    )
    return anchored_starter, anchored_bullpen, float(anchored_mean)


def _implied_game_mean_for_positive_mass_scale(
    *,
    starter_profile: EventProbabilityProfile,
    bullpen_profile: EventProbabilityProfile,
    positive_mass_scale: float,
    innings: int,
    starter_innings: int,
) -> float:
    scaled_starter = _scale_profile_positive_mass(starter_profile, positive_mass_scale)
    scaled_bullpen = _scale_profile_positive_mass(bullpen_profile, positive_mass_scale)
    return float(
        expected_runs_for_game(
            starter_profile=scaled_starter,
            bullpen_profile=scaled_bullpen,
            innings=innings,
            starter_innings=starter_innings,
        )
    )


def _scale_profile_positive_mass(
    profile: EventProbabilityProfile,
    positive_mass_scale: float,
) -> EventProbabilityProfile:
    resolved_scale = float(max(positive_mass_scale, 0.1))
    raw = profile.as_dict()
    scaled = {
        "out": raw["out"],
        "walk_hbp": raw["walk_hbp"] * resolved_scale,
        "single": raw["single"] * resolved_scale,
        "double": raw["double"] * resolved_scale,
        "triple": raw["triple"] * resolved_scale,
        "home_run": raw["home_run"] * resolved_scale,
    }
    return _normalize_profile_with_bounds(scaled)


def _normalize_profile_with_bounds(raw_probabilities: Mapping[str, float]) -> EventProbabilityProfile:
    clipped = {
        event_name: _clip(raw_probabilities.get(event_name, LEAGUE_BASE_EVENT_RATES[event_name]), lower, upper)
        for event_name, (lower, upper) in PROFILE_EVENT_BOUNDS.items()
    }
    return normalize_event_probabilities(clipped)


def _profile_out_of_bounds(profile: EventProbabilityProfile) -> bool:
    for event_name, value in profile.as_dict().items():
        lower, upper = PROFILE_EVENT_BOUNDS[event_name]
        if float(value) < float(lower) - 1e-9 or float(value) > float(upper) + 1e-9:
            return True
    return False


def _mean_drift_exceeds_guardrail(
    *,
    target_mean_runs: float,
    implied_mean_runs: float,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> bool:
    drift = abs(float(implied_mean_runs) - float(target_mean_runs))
    relative_drift = drift / max(abs(float(target_mean_runs)), 1e-6)
    return bool(
        drift > float(absolute_tolerance)
        or relative_drift > float(relative_tolerance)
    )


def _coerce_feature(
    row: Mapping[str, object] | pd.Series,
    column: str,
    *,
    default: float | None = None,
) -> float:
    if column in row:
        value = row[column]
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(numeric):
            return float(numeric)
    if default is not None:
        return float(default)
    return float(LEAGUE_REFERENCE_VALUES.get(column, 0.0))


def _weighted_mean(values: Sequence[float], *, weights: Sequence[float]) -> float:
    resolved_values = np.asarray(values, dtype=float)
    resolved_weights = np.asarray(weights, dtype=float)
    if resolved_values.shape != resolved_weights.shape:
        raise ValueError("values and weights must have matching shapes")
    return float(np.average(resolved_values, weights=resolved_weights))


def _clip(value: float, lower: float, upper: float) -> float:
    return float(np.clip(float(value), float(lower), float(upper)))


def _compress_ratio(value: float, *, lower: float, upper: float, strength: float) -> float:
    resolved = _clip(value, lower, upper)
    return float(np.exp(np.log(resolved) * float(strength)))


def _shrink_toward_one(value: float, weight: float) -> float:
    resolved_weight = _clip(weight, 0.0, 1.0)
    return float(1.0 + ((float(value) - 1.0) * resolved_weight))


__all__ = [
    "LEAGUE_BASE_EVENT_RATES",
    "LEAGUE_REFERENCE_VALUES",
    "MCMCFeatureBundle",
    "build_feature_bundles_for_frame",
    "build_mcmc_feature_bundle",
]
