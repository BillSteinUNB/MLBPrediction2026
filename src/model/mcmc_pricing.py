from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class AwayRunDistributionSummary:
    away_run_pmf: list[dict[str, float | int]]
    expected_away_runs: float
    shutout_probability: float
    tail_probabilities: dict[str, float]
    central_intervals: dict[str, dict[str, float | int]]


def normalize_probability_vector(probabilities: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(probabilities, dtype=float)
    if array.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")
    clipped = np.clip(array, 0.0, None)
    total = float(clipped.sum())
    if total <= 0.0:
        raise ValueError("probabilities must sum to a positive value")
    return clipped / total


def summarize_away_run_distribution(
    *,
    support: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> AwayRunDistributionSummary:
    resolved_support = np.asarray(list(support), dtype=int)
    normalized = normalize_probability_vector(probabilities)
    if len(resolved_support) != len(normalized):
        raise ValueError("support and probabilities must have matching lengths")

    cdf = np.cumsum(normalized)
    expected_runs = float(np.dot(resolved_support, normalized))
    tail_probabilities = {
        "p_ge_1": tail_probability(resolved_support, normalized, threshold=1),
        "p_ge_3": tail_probability(resolved_support, normalized, threshold=3),
        "p_ge_5": tail_probability(resolved_support, normalized, threshold=5),
        "p_ge_10": tail_probability(resolved_support, normalized, threshold=10),
    }
    return AwayRunDistributionSummary(
        away_run_pmf=[
            {"runs": int(run_value), "probability": float(probability)}
            for run_value, probability in zip(resolved_support.tolist(), normalized.tolist(), strict=True)
        ],
        expected_away_runs=expected_runs,
        shutout_probability=float(normalized[0]) if len(normalized) > 0 else 0.0,
        tail_probabilities=tail_probabilities,
        central_intervals={
            "central_50": central_interval(resolved_support, cdf, nominal_coverage=0.50),
            "central_80": central_interval(resolved_support, cdf, nominal_coverage=0.80),
            "central_95": central_interval(resolved_support, cdf, nominal_coverage=0.95),
        },
    )


def tail_probability(
    support: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    threshold: int,
) -> float:
    resolved_support = np.asarray(list(support), dtype=int)
    normalized = normalize_probability_vector(probabilities)
    return float(np.sum(normalized[resolved_support >= int(threshold)]))


def central_interval(
    support: Sequence[int] | np.ndarray,
    cdf_or_probabilities: Sequence[float] | np.ndarray,
    *,
    nominal_coverage: float,
) -> dict[str, float | int]:
    resolved_support = np.asarray(list(support), dtype=int)
    values = np.asarray(cdf_or_probabilities, dtype=float)
    cdf = (
        values
        if np.all(np.diff(values) >= -1e-12) and values[-1] <= 1.0000001
        else np.cumsum(normalize_probability_vector(values))
    )
    lower_quantile = (1.0 - float(nominal_coverage)) / 2.0
    upper_quantile = 1.0 - lower_quantile
    lower_index = int(np.argmax(cdf >= lower_quantile))
    upper_index = int(np.argmax(cdf >= upper_quantile))
    return {
        "nominal_coverage": float(nominal_coverage),
        "lower_bound": int(resolved_support[lower_index]),
        "upper_bound": int(resolved_support[upper_index]),
        "width": int(resolved_support[upper_index] - resolved_support[lower_index]),
    }


def build_distribution_comparison(
    *,
    challenger_label: str,
    challenger_mean_metrics: Mapping[str, Any],
    challenger_distribution_metrics: Mapping[str, Any],
    baseline_label: str,
    baseline_mean_metrics: Mapping[str, Any],
    baseline_distribution_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    challenger_bias = float(challenger_mean_metrics["predicted_mean"] - challenger_mean_metrics["actual_mean"])
    baseline_bias = float(baseline_mean_metrics["predicted_mean"] - baseline_mean_metrics["actual_mean"])
    challenger_tail_errors = [
        float(item["absolute_error"])
        for item in challenger_distribution_metrics["tail_calibration"].values()
    ]
    baseline_tail_errors = [
        float(item["absolute_error"])
        for item in baseline_distribution_metrics["tail_calibration"].values()
    ]
    rmse_regression_pct = (
        ((float(challenger_mean_metrics["rmse"]) - float(baseline_mean_metrics["rmse"])) / float(baseline_mean_metrics["rmse"]))
        * 100.0
        if float(baseline_mean_metrics["rmse"]) > 0.0
        else 0.0
    )
    bias_regression = abs(challenger_bias) - abs(baseline_bias)
    tail_error_change = max(challenger_tail_errors) - max(baseline_tail_errors)
    guardrails = {
        "rmse_within_2pct": rmse_regression_pct <= 2.0,
        "mean_bias_within_0_15_runs": abs(challenger_bias) <= 0.15,
        "tail_calibration_stable": tail_error_change <= 0.05,
    }
    return {
        "challenger_label": str(challenger_label),
        "baseline_label": str(baseline_label),
        "challenger_mean_metrics": dict(challenger_mean_metrics),
        "baseline_mean_metrics": dict(baseline_mean_metrics),
        "challenger_distribution_metrics": dict(challenger_distribution_metrics),
        "baseline_distribution_metrics": dict(baseline_distribution_metrics),
        "deltas": {
            "mean_crps": float(challenger_distribution_metrics["mean_crps"])
            - float(baseline_distribution_metrics["mean_crps"]),
            "mean_negative_log_score": float(challenger_distribution_metrics["mean_negative_log_score"])
            - float(baseline_distribution_metrics["mean_negative_log_score"]),
            "rmse": float(challenger_mean_metrics["rmse"]) - float(baseline_mean_metrics["rmse"]),
            "rmse_regression_pct": rmse_regression_pct,
            "mean_bias": challenger_bias - baseline_bias,
            "mean_bias_abs_regression": bias_regression,
            "zero_calibration_abs_error": float(challenger_distribution_metrics["zero_calibration"]["p_0"]["absolute_error"])
            - float(baseline_distribution_metrics["zero_calibration"]["p_0"]["absolute_error"]),
            "max_tail_abs_error": tail_error_change,
        },
        "improvement_flags": {
            "beats_baseline_on_crps": float(challenger_distribution_metrics["mean_crps"])
            < float(baseline_distribution_metrics["mean_crps"]),
            "beats_baseline_on_negative_log_score": float(challenger_distribution_metrics["mean_negative_log_score"])
            < float(baseline_distribution_metrics["mean_negative_log_score"]),
            "improves_zero_calibration": float(challenger_distribution_metrics["zero_calibration"]["p_0"]["absolute_error"])
            < float(baseline_distribution_metrics["zero_calibration"]["p_0"]["absolute_error"]),
            "improves_max_tail_calibration": max(challenger_tail_errors) < max(baseline_tail_errors),
        },
        "guardrails": guardrails,
        "catastrophic_regression": not all(guardrails.values()),
    }


def flatten_mcmc_report_row(
    *,
    model_version: str,
    metadata_path: Path,
    mean_metrics: Mapping[str, Any],
    distribution_metrics: Mapping[str, Any],
    control_comparison: Mapping[str, Any],
    stage3_comparison: Mapping[str, Any] | None,
) -> dict[str, Any]:
    row = {
        "model_version": model_version,
        "artifact_path": str(metadata_path),
        "family": "markov_monte_carlo",
        "rmse": mean_metrics["rmse"],
        "mean_bias": float(mean_metrics["predicted_mean"] - mean_metrics["actual_mean"]),
        "mean_crps": distribution_metrics["mean_crps"],
        "mean_negative_log_score": distribution_metrics["mean_negative_log_score"],
        "zero_abs_error": distribution_metrics["zero_calibration"]["p_0"]["absolute_error"],
        "ge_3_abs_error": distribution_metrics["tail_calibration"]["p_ge_3"]["absolute_error"],
        "ge_5_abs_error": distribution_metrics["tail_calibration"]["p_ge_5"]["absolute_error"],
        "ge_10_abs_error": distribution_metrics["tail_calibration"]["p_ge_10"]["absolute_error"],
        "beats_control_on_crps": control_comparison["improvement_flags"]["beats_baseline_on_crps"],
        "beats_control_on_negative_log_score": control_comparison["improvement_flags"]["beats_baseline_on_negative_log_score"],
        "rmse_within_2pct_of_control": control_comparison["guardrails"]["rmse_within_2pct"],
        "tail_calibration_stable_vs_control": control_comparison["guardrails"]["tail_calibration_stable"],
    }
    if stage3_comparison is not None:
        row["beats_stage3_on_crps"] = stage3_comparison["improvement_flags"]["beats_baseline_on_crps"]
        row["beats_stage3_on_negative_log_score"] = stage3_comparison["improvement_flags"]["beats_baseline_on_negative_log_score"]
    return row


__all__ = [
    "AwayRunDistributionSummary",
    "build_distribution_comparison",
    "central_interval",
    "flatten_mcmc_report_row",
    "normalize_probability_vector",
    "summarize_away_run_distribution",
    "tail_probability",
]
