from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.stats import nbinom, poisson


MIN_MEAN_PREDICTION = 1e-9
MIN_PROBABILITY = 1e-12
DEFAULT_SUPPORT_TAIL_PROBABILITY = 1e-8
DEFAULT_CALIBRATION_BIN_COUNT = 10
TAIL_EVENT_THRESHOLDS = (3, 5, 10)
ZERO_EVENT_THRESHOLDS = (0, 1)

EventKind = Literal["eq", "ge"]


@dataclass(frozen=True, slots=True)
class NegativeBinomialFit:
    dispersion_size: float
    overdispersion_alpha: float
    objective_value: float
    converged: bool


@dataclass(frozen=True, slots=True)
class ZeroAdjustmentFit:
    delta: float
    objective_value: float
    converged: bool


def clip_mean_predictions(mean_predictions: np.ndarray | list[float]) -> np.ndarray:
    means = np.asarray(mean_predictions, dtype=float)
    if means.ndim != 1:
        raise ValueError("mean_predictions must be one-dimensional")
    return np.clip(means, MIN_MEAN_PREDICTION, None)


def coerce_count_targets(actual_counts: np.ndarray | list[int] | pd.Series) -> np.ndarray:
    counts = np.asarray(actual_counts, dtype=int)
    if counts.ndim != 1:
        raise ValueError("actual_counts must be one-dimensional")
    if np.any(counts < 0):
        raise ValueError("actual_counts must be non-negative")
    return counts


def negative_binomial_shape_params(
    mean_predictions: np.ndarray | list[float],
    *,
    dispersion_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    means = clip_mean_predictions(mean_predictions)
    resolved_dispersion = max(float(dispersion_size), MIN_MEAN_PREDICTION)
    probability = resolved_dispersion / (resolved_dispersion + means)
    return (
        np.full_like(means, resolved_dispersion, dtype=float),
        np.clip(probability, MIN_PROBABILITY, 1.0 - MIN_PROBABILITY),
    )


def fit_negative_binomial_dispersion(
    actual_counts: np.ndarray | list[int] | pd.Series,
    mean_predictions: np.ndarray | list[float],
    *,
    min_dispersion_size: float = 1e-3,
    max_dispersion_size: float = 1e6,
) -> NegativeBinomialFit:
    counts = coerce_count_targets(actual_counts)
    means = clip_mean_predictions(mean_predictions)

    def objective(log_dispersion_size: float) -> float:
        dispersion_size = math.exp(float(log_dispersion_size))
        shape, probability = negative_binomial_shape_params(
            means,
            dispersion_size=dispersion_size,
        )
        log_likelihood = nbinom.logpmf(counts, shape, probability)
        return float(-np.sum(np.nan_to_num(log_likelihood, nan=-1e12, neginf=-1e12)))

    result = minimize_scalar(
        objective,
        bounds=(
            math.log(max(float(min_dispersion_size), MIN_MEAN_PREDICTION)),
            math.log(max(float(max_dispersion_size), float(min_dispersion_size) * 10.0)),
        ),
        method="bounded",
    )
    dispersion_size = math.exp(float(result.x))
    return NegativeBinomialFit(
        dispersion_size=float(dispersion_size),
        overdispersion_alpha=float(1.0 / max(dispersion_size, MIN_MEAN_PREDICTION)),
        objective_value=float(result.fun),
        converged=bool(result.success),
    )


def fit_zero_adjustment(
    actual_counts: np.ndarray | list[int] | pd.Series,
    baseline_zero_probabilities: np.ndarray | list[float],
    *,
    min_delta: float = -12.0,
    max_delta: float = 12.0,
) -> ZeroAdjustmentFit:
    counts = coerce_count_targets(actual_counts)
    baseline = np.clip(
        np.asarray(baseline_zero_probabilities, dtype=float),
        MIN_PROBABILITY,
        1.0 - MIN_PROBABILITY,
    )
    observed_zero = (counts == 0).astype(float)

    def objective(delta: float) -> float:
        adjusted_zero_probability = expit(logit(baseline) + float(delta))
        log_likelihood = (
            observed_zero * np.log(np.clip(adjusted_zero_probability, MIN_PROBABILITY, 1.0))
            + (1.0 - observed_zero)
            * np.log(np.clip(1.0 - adjusted_zero_probability, MIN_PROBABILITY, 1.0))
        )
        return float(-np.sum(log_likelihood))

    result = minimize_scalar(
        objective,
        bounds=(float(min_delta), float(max_delta)),
        method="bounded",
    )
    return ZeroAdjustmentFit(
        delta=float(result.x),
        objective_value=float(result.fun),
        converged=bool(result.success),
    )


def resolve_support_max(
    actual_counts: np.ndarray | list[int] | pd.Series,
    mean_predictions: np.ndarray | list[float],
    *,
    family: Literal["poisson", "negative_binomial", "zero_adjusted_negative_binomial"],
    dispersion_size: float | None = None,
    tail_probability: float = DEFAULT_SUPPORT_TAIL_PROBABILITY,
) -> int:
    counts = coerce_count_targets(actual_counts)
    means = clip_mean_predictions(mean_predictions)
    resolved_tail_probability = min(max(float(tail_probability), 1e-12), 1e-3)
    quantile_probability = 1.0 - resolved_tail_probability

    if family == "poisson":
        quantiles = poisson.ppf(quantile_probability, means)
        fallback = means + (10.0 * np.sqrt(means))
    else:
        if dispersion_size is None:
            raise ValueError("dispersion_size is required for negative binomial support resolution")
        shape, probability = negative_binomial_shape_params(means, dispersion_size=dispersion_size)
        quantiles = nbinom.ppf(quantile_probability, shape, probability)
        variance = means + ((means**2) / max(float(dispersion_size), MIN_MEAN_PREDICTION))
        fallback = means + (10.0 * np.sqrt(variance))

    if not np.all(np.isfinite(quantiles)):
        quantiles = fallback
    return int(max(np.max(counts, initial=0), math.ceil(float(np.max(quantiles))), 15))


def build_support(support_max: int) -> np.ndarray:
    resolved_support_max = max(int(support_max), 0)
    return np.arange(resolved_support_max + 1, dtype=int)


def normalize_pmf_matrix(pmf_matrix: np.ndarray) -> np.ndarray:
    pmf = np.asarray(pmf_matrix, dtype=float)
    if pmf.ndim != 2:
        raise ValueError("pmf_matrix must be two-dimensional")
    pmf = np.clip(pmf, 0.0, None)
    row_sums = pmf.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0.0, 1.0, row_sums)
    return pmf / row_sums


def poisson_pmf_matrix(
    mean_predictions: np.ndarray | list[float],
    support: np.ndarray | list[int] | range,
) -> np.ndarray:
    means = clip_mean_predictions(mean_predictions)
    resolved_support = np.asarray(list(support), dtype=int)
    pmf = poisson.pmf(resolved_support[None, :], means[:, None])
    return normalize_pmf_matrix(pmf)


def negative_binomial_pmf_matrix(
    mean_predictions: np.ndarray | list[float],
    support: np.ndarray | list[int] | range,
    *,
    dispersion_size: float,
) -> np.ndarray:
    means = clip_mean_predictions(mean_predictions)
    resolved_support = np.asarray(list(support), dtype=int)
    shape, probability = negative_binomial_shape_params(means, dispersion_size=dispersion_size)
    pmf = nbinom.pmf(resolved_support[None, :], shape[:, None], probability[:, None])
    return normalize_pmf_matrix(pmf)


def zero_adjusted_negative_binomial_pmf_matrix(
    mean_predictions: np.ndarray | list[float],
    support: np.ndarray | list[int] | range,
    *,
    dispersion_size: float,
    zero_adjustment_delta: float,
) -> np.ndarray:
    baseline_pmf = negative_binomial_pmf_matrix(
        mean_predictions,
        support,
        dispersion_size=dispersion_size,
    )
    baseline_zero_probability = np.clip(
        baseline_pmf[:, 0],
        MIN_PROBABILITY,
        1.0 - MIN_PROBABILITY,
    )
    adjusted_zero_probability = expit(logit(baseline_zero_probability) + float(zero_adjustment_delta))
    adjusted_pmf = baseline_pmf.copy()
    adjusted_pmf[:, 0] = adjusted_zero_probability

    positive_mass = np.clip(1.0 - baseline_zero_probability, MIN_PROBABILITY, 1.0)
    positive_reweight = ((1.0 - adjusted_zero_probability) / positive_mass)[:, None]
    adjusted_pmf[:, 1:] = baseline_pmf[:, 1:] * positive_reweight
    return normalize_pmf_matrix(adjusted_pmf)


def realized_probabilities(
    actual_counts: np.ndarray | list[int] | pd.Series,
    pmf_matrix: np.ndarray,
    support: np.ndarray | list[int] | range,
) -> np.ndarray:
    counts = coerce_count_targets(actual_counts)
    resolved_support = np.asarray(list(support), dtype=int)
    pmf = normalize_pmf_matrix(pmf_matrix)
    support_lookup = {int(value): index for index, value in enumerate(resolved_support.tolist())}
    missing_counts = sorted({int(value) for value in counts if int(value) not in support_lookup})
    if missing_counts:
        raise ValueError(f"support does not include realized counts {missing_counts}")
    indices = np.array([support_lookup[int(value)] for value in counts], dtype=int)
    return pmf[np.arange(len(counts)), indices]


def discrete_log_score(
    actual_counts: np.ndarray | list[int] | pd.Series,
    pmf_matrix: np.ndarray,
    support: np.ndarray | list[int] | range,
) -> dict[str, float]:
    probabilities = realized_probabilities(actual_counts, pmf_matrix, support)
    log_probabilities = np.log(np.clip(probabilities, MIN_PROBABILITY, 1.0))
    return {
        "mean_log_score": float(np.mean(log_probabilities)),
        "mean_negative_log_score": float(-np.mean(log_probabilities)),
    }


def discrete_crps(
    actual_counts: np.ndarray | list[int] | pd.Series,
    pmf_matrix: np.ndarray,
    support: np.ndarray | list[int] | range,
) -> dict[str, float]:
    counts = coerce_count_targets(actual_counts)
    resolved_support = np.asarray(list(support), dtype=int)
    cdf = np.cumsum(normalize_pmf_matrix(pmf_matrix), axis=1)
    observed_cdf = (resolved_support[None, :] >= counts[:, None]).astype(float)
    per_row = np.sum((cdf - observed_cdf) ** 2, axis=1)
    return {
        "mean_crps": float(np.mean(per_row)),
    }


def event_probability(
    pmf_matrix: np.ndarray,
    support: np.ndarray | list[int] | range,
    *,
    kind: EventKind,
    threshold: int,
) -> np.ndarray:
    resolved_support = np.asarray(list(support), dtype=int)
    pmf = normalize_pmf_matrix(pmf_matrix)
    if kind == "eq":
        mask = resolved_support == int(threshold)
    elif kind == "ge":
        mask = resolved_support >= int(threshold)
    else:
        raise ValueError(f"unsupported event kind: {kind}")
    return np.sum(pmf[:, mask], axis=1)


def build_probability_bins(
    observed: np.ndarray | list[float],
    predicted_probabilities: np.ndarray | list[float],
    *,
    bin_count: int = DEFAULT_CALIBRATION_BIN_COUNT,
) -> list[dict[str, float | int]]:
    observed_array = np.asarray(observed, dtype=float)
    probabilities = np.clip(np.asarray(predicted_probabilities, dtype=float), 0.0, 1.0)
    if len(probabilities) == 0:
        return []
    unique_probability_count = int(pd.Series(probabilities).nunique(dropna=False))
    if unique_probability_count <= 1:
        return [
            {
                "bin_index": 0,
                "count": int(len(probabilities)),
                "mean_predicted_probability": float(np.mean(probabilities)),
                "empirical_rate": float(np.mean(observed_array)),
                "absolute_error": float(abs(np.mean(probabilities) - np.mean(observed_array))),
            }
        ]

    resolved_bin_count = max(1, min(int(bin_count), unique_probability_count, len(probabilities)))
    assignments = pd.qcut(
        pd.Series(probabilities),
        q=resolved_bin_count,
        labels=False,
        duplicates="drop",
    )

    bins: list[dict[str, float | int]] = []
    assignment_series = pd.Series(assignments, dtype="Int64")
    for bin_index in sorted(value for value in assignment_series.dropna().unique().tolist()):
        mask = assignment_series == int(bin_index)
        bin_probabilities = probabilities[mask.to_numpy()]
        bin_observed = observed_array[mask.to_numpy()]
        bins.append(
            {
                "bin_index": int(bin_index),
                "count": int(mask.sum()),
                "mean_predicted_probability": float(np.mean(bin_probabilities)),
                "empirical_rate": float(np.mean(bin_observed)),
                "absolute_error": float(abs(np.mean(bin_probabilities) - np.mean(bin_observed))),
            }
        )
    return bins


def summarize_event_calibration(
    actual_counts: np.ndarray | list[int] | pd.Series,
    predicted_probabilities: np.ndarray | list[float],
    *,
    kind: EventKind,
    threshold: int,
    bin_count: int = DEFAULT_CALIBRATION_BIN_COUNT,
) -> dict[str, object]:
    counts = coerce_count_targets(actual_counts)
    probabilities = np.clip(np.asarray(predicted_probabilities, dtype=float), 0.0, 1.0)
    if kind == "eq":
        observed = (counts == int(threshold)).astype(float)
        event_name = f"p_{int(threshold)}"
    elif kind == "ge":
        observed = (counts >= int(threshold)).astype(float)
        event_name = f"p_ge_{int(threshold)}"
    else:
        raise ValueError(f"unsupported event kind: {kind}")

    bins = build_probability_bins(observed, probabilities, bin_count=bin_count)
    return {
        "event": event_name,
        "mean_predicted_probability": float(np.mean(probabilities)),
        "predicted_probability_std": float(np.std(probabilities)),
        "empirical_rate": float(np.mean(observed)),
        "absolute_error": float(abs(np.mean(probabilities) - np.mean(observed))),
        "brier_score": float(np.mean((probabilities - observed) ** 2)),
        "bin_count": int(len(bins)),
        "bins": bins,
    }


def _first_true_indices(mask: np.ndarray) -> np.ndarray:
    any_true = np.any(mask, axis=1)
    indices = np.argmax(mask, axis=1)
    fallback_indices = np.full(mask.shape[0], mask.shape[1] - 1, dtype=int)
    return np.where(any_true, indices, fallback_indices)


def summarize_interval_coverage(
    actual_counts: np.ndarray | list[int] | pd.Series,
    pmf_matrix: np.ndarray,
    support: np.ndarray | list[int] | range,
    *,
    nominal_coverage: float,
) -> dict[str, float]:
    counts = coerce_count_targets(actual_counts)
    resolved_support = np.asarray(list(support), dtype=int)
    pmf = normalize_pmf_matrix(pmf_matrix)
    cdf = np.cumsum(pmf, axis=1)
    lower_quantile = (1.0 - float(nominal_coverage)) / 2.0
    upper_quantile = 1.0 - lower_quantile

    lower_indices = _first_true_indices(cdf >= lower_quantile)
    upper_indices = _first_true_indices(cdf >= upper_quantile)

    lower_bounds = resolved_support[lower_indices]
    upper_bounds = resolved_support[upper_indices]
    covered = (counts >= lower_bounds) & (counts <= upper_bounds)
    return {
        "nominal_coverage": float(nominal_coverage),
        "empirical_coverage": float(np.mean(covered)),
        "coverage_error": float(np.mean(covered) - nominal_coverage),
        "mean_width": float(np.mean(upper_bounds - lower_bounds)),
    }


def summarize_distribution_metrics(
    actual_counts: np.ndarray | list[int] | pd.Series,
    pmf_matrix: np.ndarray,
    support: np.ndarray | list[int] | range,
    *,
    calibration_bin_count: int = DEFAULT_CALIBRATION_BIN_COUNT,
) -> dict[str, object]:
    counts = coerce_count_targets(actual_counts)
    resolved_support = np.asarray(list(support), dtype=int)
    pmf = normalize_pmf_matrix(pmf_matrix)

    metrics: dict[str, object] = {}
    metrics.update(discrete_crps(counts, pmf, resolved_support))
    metrics.update(discrete_log_score(counts, pmf, resolved_support))

    metrics["zero_calibration"] = {
        "p_0": summarize_event_calibration(
            counts,
            event_probability(pmf, resolved_support, kind="eq", threshold=0),
            kind="eq",
            threshold=0,
            bin_count=calibration_bin_count,
        ),
        "p_ge_1": summarize_event_calibration(
            counts,
            event_probability(pmf, resolved_support, kind="ge", threshold=1),
            kind="ge",
            threshold=1,
            bin_count=calibration_bin_count,
        ),
    }
    metrics["tail_calibration"] = {
        f"p_ge_{threshold}": summarize_event_calibration(
            counts,
            event_probability(pmf, resolved_support, kind="ge", threshold=threshold),
            kind="ge",
            threshold=threshold,
            bin_count=calibration_bin_count,
        )
        for threshold in TAIL_EVENT_THRESHOLDS
    }
    metrics["interval_coverage"] = {
        "central_50": summarize_interval_coverage(
            counts,
            pmf,
            resolved_support,
            nominal_coverage=0.50,
        ),
        "central_80": summarize_interval_coverage(
            counts,
            pmf,
            resolved_support,
            nominal_coverage=0.80,
        ),
        "central_95": summarize_interval_coverage(
            counts,
            pmf,
            resolved_support,
            nominal_coverage=0.95,
        ),
    }

    predicted_mean = np.sum(pmf * resolved_support[None, :], axis=1)
    metrics["prediction_summary"] = {
        "mean_predicted_runs": float(np.mean(predicted_mean)),
        "mean_predicted_p_0": float(np.mean(event_probability(pmf, resolved_support, kind="eq", threshold=0))),
        "mean_predicted_p_ge_3": float(np.mean(event_probability(pmf, resolved_support, kind="ge", threshold=3))),
        "mean_predicted_p_ge_5": float(np.mean(event_probability(pmf, resolved_support, kind="ge", threshold=5))),
        "mean_predicted_p_ge_10": float(np.mean(event_probability(pmf, resolved_support, kind="ge", threshold=10))),
    }
    return metrics


def dataclass_to_dict(payload: NegativeBinomialFit | ZeroAdjustmentFit) -> dict[str, float | bool]:
    return asdict(payload)


__all__ = [
    "DEFAULT_CALIBRATION_BIN_COUNT",
    "DEFAULT_SUPPORT_TAIL_PROBABILITY",
    "MIN_MEAN_PREDICTION",
    "MIN_PROBABILITY",
    "NegativeBinomialFit",
    "TAIL_EVENT_THRESHOLDS",
    "ZeroAdjustmentFit",
    "ZERO_EVENT_THRESHOLDS",
    "build_probability_bins",
    "build_support",
    "clip_mean_predictions",
    "coerce_count_targets",
    "dataclass_to_dict",
    "discrete_crps",
    "discrete_log_score",
    "event_probability",
    "fit_negative_binomial_dispersion",
    "fit_zero_adjustment",
    "negative_binomial_pmf_matrix",
    "negative_binomial_shape_params",
    "normalize_pmf_matrix",
    "poisson_pmf_matrix",
    "realized_probabilities",
    "resolve_support_max",
    "summarize_distribution_metrics",
    "summarize_event_calibration",
    "summarize_interval_coverage",
    "zero_adjusted_negative_binomial_pmf_matrix",
]
