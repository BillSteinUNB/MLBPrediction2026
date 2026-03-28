from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.model.mcmc_engine import EventProbabilityProfile, normalize_event_probabilities


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
}


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
    raw_feature_snapshot: dict[str, float]


def build_mcmc_feature_bundle(
    row: Mapping[str, object] | pd.Series,
    *,
    target_mean_runs: float,
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
    }

    target_run_factor = _clip(snapshot["target_mean_runs"] / LEAGUE_REFERENCE_VALUES["away_runs_mean"], 0.60, 1.55)
    offense_factor = _clip(
        _weighted_mean(
            (
                snapshot["away_lineup_woba_7g"] / LEAGUE_REFERENCE_VALUES["away_lineup_woba_7g"],
                snapshot["away_lineup_woba_30g"] / LEAGUE_REFERENCE_VALUES["away_lineup_woba_30g"],
                snapshot["away_lineup_xwoba_7g"] / LEAGUE_REFERENCE_VALUES["away_lineup_xwoba_7g"],
                snapshot["away_team_woba_vs_opposing_hand"]
                / LEAGUE_REFERENCE_VALUES["away_team_woba_vs_opposing_hand"],
                snapshot["away_team_log5_30g"] / LEAGUE_REFERENCE_VALUES["away_team_log5_30g"],
                snapshot["away_team_runs_scored_7g"] / LEAGUE_REFERENCE_VALUES["away_team_runs_scored_7g"],
                target_run_factor,
            ),
            weights=(0.18, 0.10, 0.18, 0.14, 0.14, 0.12, 0.14),
        ),
        0.70,
        1.45,
    )
    walk_factor = _clip(
        _weighted_mean(
            (
                snapshot["away_lineup_bb_pct_7g"] / LEAGUE_REFERENCE_VALUES["away_lineup_bb_pct_7g"],
                snapshot["away_lineup_bb_pct_30g"] / LEAGUE_REFERENCE_VALUES["away_lineup_bb_pct_30g"],
                target_run_factor,
            ),
            weights=(0.45, 0.30, 0.25),
        ),
        0.60,
        1.55,
    )
    strikeout_pressure = _clip(
        _weighted_mean(
            (
                snapshot["home_starter_k_pct_30s"] / LEAGUE_REFERENCE_VALUES["home_starter_k_pct_30s"],
                snapshot["home_starter_k_pct_7s"] / LEAGUE_REFERENCE_VALUES["home_starter_k_pct_7s"],
                snapshot["away_lineup_k_pct_7g"] / LEAGUE_REFERENCE_VALUES["away_lineup_k_pct_7g"],
            ),
            weights=(0.38, 0.32, 0.30),
        ),
        0.70,
        1.55,
    )
    power_factor = _clip(
        _weighted_mean(
            (
                snapshot["away_lineup_iso_7g"] / LEAGUE_REFERENCE_VALUES["away_lineup_iso_7g"],
                snapshot["away_lineup_barrel_pct_7g"] / LEAGUE_REFERENCE_VALUES["away_lineup_barrel_pct_7g"],
                target_run_factor,
                snapshot["park_hr_factor"] / LEAGUE_REFERENCE_VALUES["park_hr_factor"],
            ),
            weights=(0.35, 0.25, 0.20, 0.20),
        ),
        0.55,
        1.90,
    )
    run_environment_factor = _clip(
        _weighted_mean(
            (
                snapshot["park_runs_factor"] / LEAGUE_REFERENCE_VALUES["park_runs_factor"],
                snapshot["weather_air_density_factor"] / LEAGUE_REFERENCE_VALUES["weather_air_density_factor"],
                snapshot["weather_temp_factor"] / LEAGUE_REFERENCE_VALUES["weather_temp_factor"],
                snapshot["weather_composite"] / LEAGUE_REFERENCE_VALUES["weather_composite"],
                snapshot["plate_umpire_total_runs_avg_30g"]
                / LEAGUE_REFERENCE_VALUES["plate_umpire_total_runs_avg_30g"],
            ),
            weights=(0.28, 0.17, 0.10, 0.25, 0.20),
        ),
        0.82,
        1.22,
    )
    defense_factor = _clip(
        _weighted_mean(
            (
                snapshot["home_team_defensive_efficiency_30g"]
                / LEAGUE_REFERENCE_VALUES["home_team_defensive_efficiency_30g"],
                1.0 + (
                    (snapshot["home_team_oaa_30g"] - LEAGUE_REFERENCE_VALUES["home_team_oaa_30g"])
                    / 2.5
                ),
                1.0 + (snapshot["home_team_adjusted_framing_30g"] / 10.0),
            ),
            weights=(0.50, 0.35, 0.15),
        ),
        0.85,
        1.20,
    )
    starter_pitching_factor = _clip(
        _weighted_mean(
            (
                snapshot["home_starter_k_pct_30s"] / LEAGUE_REFERENCE_VALUES["home_starter_k_pct_30s"],
                snapshot["home_starter_k_pct_7s"] / LEAGUE_REFERENCE_VALUES["home_starter_k_pct_7s"],
                LEAGUE_REFERENCE_VALUES["home_starter_xfip_30s"] / max(snapshot["home_starter_xfip_30s"], 1e-6),
                LEAGUE_REFERENCE_VALUES["home_starter_xera_30s"] / max(snapshot["home_starter_xera_30s"], 1e-6),
                LEAGUE_REFERENCE_VALUES["home_starter_siera_30s"] / max(snapshot["home_starter_siera_30s"], 1e-6),
                defense_factor,
            ),
            weights=(0.18, 0.14, 0.24, 0.22, 0.12, 0.10),
        ),
        0.68,
        1.45,
    )
    bullpen_pitching_factor = _clip(
        _weighted_mean(
            (
                LEAGUE_REFERENCE_VALUES["home_team_bullpen_xfip"] / max(snapshot["home_team_bullpen_xfip"], 1e-6),
                defense_factor,
                strikeout_pressure,
            ),
            weights=(0.60, 0.20, 0.20),
        ),
        0.68,
        1.35,
    )
    walk_suppression = _clip(
        _weighted_mean(
            (
                LEAGUE_REFERENCE_VALUES["home_starter_bb_pct_30s"] / max(snapshot["home_starter_bb_pct_30s"], 1e-6),
                LEAGUE_REFERENCE_VALUES["home_starter_bb_pct_7s"] / max(snapshot["home_starter_bb_pct_7s"], 1e-6),
            ),
            weights=(0.55, 0.45),
        ),
        0.65,
        1.50,
    )

    starter_profile = _build_event_profile(
        offense_factor=offense_factor,
        power_factor=power_factor,
        target_run_factor=target_run_factor,
        walk_factor=walk_factor,
        run_environment_factor=run_environment_factor,
        pitching_factor=starter_pitching_factor,
        walk_suppression=walk_suppression,
    )
    bullpen_profile = _build_event_profile(
        offense_factor=offense_factor,
        power_factor=power_factor,
        target_run_factor=target_run_factor,
        walk_factor=walk_factor,
        run_environment_factor=run_environment_factor,
        pitching_factor=bullpen_pitching_factor,
        walk_suppression=_clip((walk_suppression * 0.97) + 0.03, 0.65, 1.45),
    )

    return MCMCFeatureBundle(
        target_mean_runs=float(snapshot["target_mean_runs"]),
        starter_profile=starter_profile,
        bullpen_profile=bullpen_profile,
        run_environment_factor=float(run_environment_factor),
        offense_factor=float(offense_factor),
        power_factor=float(power_factor),
        starter_pitching_factor=float(starter_pitching_factor),
        bullpen_pitching_factor=float(bullpen_pitching_factor),
        raw_feature_snapshot=snapshot,
    )


def build_feature_bundles_for_frame(
    dataframe: pd.DataFrame,
    *,
    target_mean_predictions: Sequence[float] | np.ndarray,
) -> list[MCMCFeatureBundle]:
    means = np.asarray(target_mean_predictions, dtype=float)
    if len(means) != len(dataframe):
        raise ValueError("target_mean_predictions length must match dataframe length")
    return [
        build_mcmc_feature_bundle(row, target_mean_runs=float(mean_prediction))
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
) -> EventProbabilityProfile:
    raw = {
        "walk_hbp": LEAGUE_BASE_EVENT_RATES["walk_hbp"]
        * _clip((walk_factor * (target_run_factor**0.30) * (run_environment_factor**0.10)) / walk_suppression, 0.55, 1.80),
        "single": LEAGUE_BASE_EVENT_RATES["single"]
        * _clip((offense_factor * (target_run_factor**0.55) * run_environment_factor) / pitching_factor, 0.45, 2.05),
        "double": LEAGUE_BASE_EVENT_RATES["double"]
        * _clip((((0.55 * offense_factor) + (0.45 * power_factor)) * (target_run_factor**0.60) * run_environment_factor) / pitching_factor, 0.40, 2.10),
        "triple": LEAGUE_BASE_EVENT_RATES["triple"]
        * _clip(((0.70 * offense_factor) + (0.30 * run_environment_factor)) * (target_run_factor**0.25), 0.30, 1.60),
        "home_run": LEAGUE_BASE_EVENT_RATES["home_run"]
        * _clip((power_factor * (target_run_factor**0.72) * run_environment_factor) / pitching_factor, 0.30, 2.35),
        "out": LEAGUE_BASE_EVENT_RATES["out"]
        * _clip(pitching_factor / ((target_run_factor**0.72) * (run_environment_factor**0.25)), 0.45, 1.80),
    }
    return normalize_event_probabilities(raw)


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


__all__ = [
    "LEAGUE_BASE_EVENT_RATES",
    "LEAGUE_REFERENCE_VALUES",
    "MCMCFeatureBundle",
    "build_feature_bundles_for_frame",
    "build_mcmc_feature_bundle",
]
