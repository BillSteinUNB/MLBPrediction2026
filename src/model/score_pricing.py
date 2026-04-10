from __future__ import annotations

import math

from scipy.stats import nbinom, skellam

DEFAULT_TOTAL_OVERDISPERSION_FACTOR = 2.3


def _is_whole_number_line(value: float, *, tolerance: float = 1e-9) -> bool:
    return math.isclose(float(value), round(float(value)), abs_tol=tolerance)


def normal_cdf(value: float) -> float:
    """Return the standard normal cumulative distribution function."""

    return 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))


def projected_margin(*, home_runs_mean: float, away_runs_mean: float) -> float:
    """Return the expected run margin from the home team's perspective."""

    return float(home_runs_mean) - float(away_runs_mean)


def projected_total(*, home_runs_mean: float, away_runs_mean: float) -> float:
    """Return the expected total runs."""

    return float(home_runs_mean) + float(away_runs_mean)


def _resolve_difference_std(
    *,
    home_runs_std: float,
    away_runs_std: float,
    correlation: float = 0.0,
) -> float | None:
    resolved_home_std = float(home_runs_std)
    resolved_away_std = float(away_runs_std)

    if resolved_home_std < 0 or resolved_away_std < 0:
        raise ValueError("standard deviations must be non-negative")

    resolved_correlation = float(correlation)
    if resolved_correlation < -1.0 or resolved_correlation > 1.0:
        raise ValueError("correlation must be between -1 and 1")

    variance = (
        resolved_home_std**2
        + resolved_away_std**2
        - 2.0 * resolved_correlation * resolved_home_std * resolved_away_std
    )
    if variance <= 0:
        return None

    return math.sqrt(variance)


def _resolve_total_variance(
    *,
    home_runs_mean: float,
    away_runs_mean: float,
    home_runs_std: float,
    away_runs_std: float,
    correlation: float = 0.0,
    overdispersion_factor: float = DEFAULT_TOTAL_OVERDISPERSION_FACTOR,
) -> float | None:
    resolved_correlation = float(correlation)
    if resolved_correlation < -1.0 or resolved_correlation > 1.0:
        raise ValueError("correlation must be between -1 and 1")

    variance = (
        float(home_runs_std) ** 2
        + float(away_runs_std) ** 2
        + 2.0 * resolved_correlation * float(home_runs_std) * float(away_runs_std)
    )
    total_mean = projected_total(
        home_runs_mean=home_runs_mean,
        away_runs_mean=away_runs_mean,
    )
    variance_floor = max(float(total_mean) * float(overdispersion_factor), float(total_mean))
    resolved_variance = max(float(variance), variance_floor)
    if resolved_variance <= 0:
        return None
    return resolved_variance


def _probability_above_threshold(*, mean: float, std_dev: float, threshold: float) -> float | None:
    if std_dev <= 0:
        return None

    z_score = (float(threshold) - float(mean)) / float(std_dev)
    probability = 1.0 - normal_cdf(z_score)
    return max(0.0, min(1.0, float(probability)))


def _nbinom_shape_params(*, mean: float, variance: float) -> tuple[float, float]:
    resolved_mean = max(float(mean), 1e-6)
    resolved_variance = max(float(variance), resolved_mean)
    if resolved_variance <= resolved_mean:
        resolved_variance = resolved_mean + 1e-6
    shape = (resolved_mean**2) / (resolved_variance - resolved_mean)
    probability = shape / (shape + resolved_mean)
    return float(shape), float(probability)


def moneyline_probabilities(
    *,
    home_runs_mean: float,
    away_runs_mean: float,
    home_runs_std: float,
    away_runs_std: float,
    correlation: float = 0.0,
) -> tuple[float | None, float | None]:
    """Return home/away moneyline win probabilities."""

    difference_mean = projected_margin(
        home_runs_mean=home_runs_mean,
        away_runs_mean=away_runs_mean,
    )
    difference_std = _resolve_difference_std(
        home_runs_std=home_runs_std,
        away_runs_std=away_runs_std,
        correlation=correlation,
    )
    if difference_std is None:
        return None, None

    home_probability = _probability_above_threshold(
        mean=difference_mean,
        std_dev=difference_std,
        threshold=0.0,
    )
    if home_probability is None:
        return None, None

    away_probability = 1.0 - home_probability
    return home_probability, away_probability


def spread_cover_probabilities(
    *,
    home_runs_mean: float,
    away_runs_mean: float,
    home_runs_std: float,
    away_runs_std: float,
    home_point: float,
    correlation: float = 0.0,
) -> tuple[float | None, float | None]:
    """Return home/away run-line win probabilities for a given home point."""

    home_probability, away_probability, _ = spread_outcome_probabilities(
        home_runs_mean=home_runs_mean,
        away_runs_mean=away_runs_mean,
        home_runs_std=home_runs_std,
        away_runs_std=away_runs_std,
        home_point=home_point,
        correlation=correlation,
    )
    return home_probability, away_probability


def spread_outcome_probabilities(
    *,
    home_runs_mean: float,
    away_runs_mean: float,
    home_runs_std: float,
    away_runs_std: float,
    home_point: float,
    correlation: float = 0.0,
) -> tuple[float | None, float | None, float | None]:
    """Return home/away run-line win probabilities plus push probability."""

    _ = home_runs_std, away_runs_std, correlation
    home_mean = max(float(home_runs_mean), 1e-6)
    away_mean = max(float(away_runs_mean), 1e-6)
    resolved_home_point = float(home_point)

    if _is_whole_number_line(resolved_home_point):
        push_margin = int(round(-resolved_home_point))
        push_probability = float(skellam.pmf(push_margin, home_mean, away_mean))
        home_probability = 1.0 - float(skellam.cdf(push_margin, home_mean, away_mean))
        away_probability = float(skellam.cdf(push_margin - 1, home_mean, away_mean))
    else:
        threshold = math.floor(-resolved_home_point)
        home_probability = 1.0 - float(skellam.cdf(threshold, home_mean, away_mean))
        away_probability = float(skellam.cdf(threshold, home_mean, away_mean))
        push_probability = 0.0

    home_probability = max(0.0, min(1.0, home_probability))
    away_probability = max(0.0, min(1.0, away_probability))
    push_probability = max(0.0, min(1.0, push_probability))
    return home_probability, away_probability, push_probability


def totals_probabilities(
    *,
    home_runs_mean: float,
    away_runs_mean: float,
    home_runs_std: float,
    away_runs_std: float,
    total_point: float,
    correlation: float = 0.0,
) -> tuple[float | None, float | None]:
    """Return over/under win probabilities for a posted total."""

    over_probability, under_probability, _ = totals_outcome_probabilities(
        home_runs_mean=home_runs_mean,
        away_runs_mean=away_runs_mean,
        home_runs_std=home_runs_std,
        away_runs_std=away_runs_std,
        total_point=total_point,
        correlation=correlation,
    )
    return over_probability, under_probability


def totals_outcome_probabilities(
    *,
    home_runs_mean: float,
    away_runs_mean: float,
    home_runs_std: float,
    away_runs_std: float,
    total_point: float,
    correlation: float = 0.0,
) -> tuple[float | None, float | None, float | None]:
    """Return over/under win probabilities plus push probability for a posted total."""

    total_mean = projected_total(
        home_runs_mean=home_runs_mean,
        away_runs_mean=away_runs_mean,
    )
    total_variance = _resolve_total_variance(
        home_runs_mean=home_runs_mean,
        away_runs_mean=away_runs_mean,
        home_runs_std=home_runs_std,
        away_runs_std=away_runs_std,
        correlation=correlation,
    )
    if total_variance is None:
        return None, None, None

    shape, probability = _nbinom_shape_params(mean=total_mean, variance=total_variance)
    resolved_total_point = float(total_point)

    if _is_whole_number_line(resolved_total_point):
        push_total = int(round(resolved_total_point))
        push_probability = float(nbinom.pmf(push_total, shape, probability))
        over_probability = 1.0 - float(nbinom.cdf(push_total, shape, probability))
        under_probability = float(nbinom.cdf(push_total - 1, shape, probability))
    else:
        threshold = math.floor(resolved_total_point)
        over_probability = 1.0 - float(nbinom.cdf(threshold, shape, probability))
        under_probability = float(nbinom.cdf(threshold, shape, probability))
        push_probability = 0.0

    over_probability = max(0.0, min(1.0, over_probability))
    under_probability = max(0.0, min(1.0, under_probability))
    push_probability = max(0.0, min(1.0, push_probability))
    return over_probability, under_probability, push_probability


def moneyline_probability(
    *,
    home_runs_mean: float,
    away_runs_mean: float,
    home_runs_std: float,
    away_runs_std: float,
    correlation: float = 0.0,
) -> float | None:
    """Return the home team moneyline win probability."""

    home_probability, _ = moneyline_probabilities(
        home_runs_mean=home_runs_mean,
        away_runs_mean=away_runs_mean,
        home_runs_std=home_runs_std,
        away_runs_std=away_runs_std,
        correlation=correlation,
    )
    return home_probability


def spread_cover_probability(
    *,
    home_runs_mean: float,
    away_runs_mean: float,
    home_runs_std: float,
    away_runs_std: float,
    home_point: float,
    correlation: float = 0.0,
) -> float | None:
    """Return the home team run-line cover probability."""

    home_probability, _ = spread_cover_probabilities(
        home_runs_mean=home_runs_mean,
        away_runs_mean=away_runs_mean,
        home_runs_std=home_runs_std,
        away_runs_std=away_runs_std,
        home_point=home_point,
        correlation=correlation,
    )
    return home_probability


def totals_over_probability(
    *,
    home_runs_mean: float,
    away_runs_mean: float,
    home_runs_std: float,
    away_runs_std: float,
    total_point: float,
    correlation: float = 0.0,
) -> float | None:
    """Return the over probability for a posted total."""

    over_probability, _ = totals_probabilities(
        home_runs_mean=home_runs_mean,
        away_runs_mean=away_runs_mean,
        home_runs_std=home_runs_std,
        away_runs_std=away_runs_std,
        total_point=total_point,
        correlation=correlation,
    )
    return over_probability


def totals_under_probability(
    *,
    home_runs_mean: float,
    away_runs_mean: float,
    home_runs_std: float,
    away_runs_std: float,
    total_point: float,
    correlation: float = 0.0,
) -> float | None:
    """Return the under probability for a posted total."""

    _, under_probability = totals_probabilities(
        home_runs_mean=home_runs_mean,
        away_runs_mean=away_runs_mean,
        home_runs_std=home_runs_std,
        away_runs_std=away_runs_std,
        total_point=total_point,
        correlation=correlation,
    )
    return under_probability


__all__ = [
    "DEFAULT_TOTAL_OVERDISPERSION_FACTOR",
    "moneyline_probability",
    "moneyline_probabilities",
    "normal_cdf",
    "projected_margin",
    "projected_total",
    "spread_cover_probability",
    "spread_cover_probabilities",
    "spread_outcome_probabilities",
    "totals_over_probability",
    "totals_probabilities",
    "totals_outcome_probabilities",
    "totals_under_probability",
]
