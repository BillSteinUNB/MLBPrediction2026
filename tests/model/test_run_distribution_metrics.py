from __future__ import annotations

import math

import numpy as np

from src.model.run_distribution_metrics import (
    build_support,
    discrete_crps,
    discrete_log_score,
    event_probability,
    fit_negative_binomial_dispersion,
    fit_zero_adjustment,
    negative_binomial_pmf_matrix,
    poisson_pmf_matrix,
    summarize_distribution_metrics,
    summarize_interval_coverage,
    zero_adjusted_negative_binomial_pmf_matrix,
)


def test_poisson_pmf_matrix_normalizes_and_matches_zero_probability() -> None:
    means = np.array([0.5, 2.0], dtype=float)
    support = build_support(12)

    pmf = poisson_pmf_matrix(means, support)

    assert pmf.shape == (2, 13)
    np.testing.assert_allclose(pmf.sum(axis=1), np.ones(2), atol=1e-10)
    assert math.isclose(pmf[0, 0], math.exp(-0.5), rel_tol=1e-6)
    assert math.isclose(pmf[1, 0], math.exp(-2.0), rel_tol=1e-6)


def test_negative_binomial_dispersion_fit_returns_positive_value_for_overdispersed_counts() -> None:
    means = np.array([1.4, 1.7, 2.1, 2.8, 3.3, 3.9, 4.4, 4.8], dtype=float)
    actual = np.array([0, 5, 0, 7, 1, 9, 0, 8], dtype=int)

    fit = fit_negative_binomial_dispersion(actual, means)
    pmf = negative_binomial_pmf_matrix(means, build_support(20), dispersion_size=fit.dispersion_size)

    assert fit.dispersion_size > 0.0
    assert fit.overdispersion_alpha > 0.0
    np.testing.assert_allclose(pmf.sum(axis=1), np.ones(len(means)), atol=1e-10)


def test_zero_adjustment_fit_increases_zero_mass_when_training_sample_is_zero_heavy() -> None:
    means = np.full(8, 2.0, dtype=float)
    actual = np.array([0, 0, 0, 0, 0, 1, 2, 0], dtype=int)
    support = build_support(15)

    nb_fit = fit_negative_binomial_dispersion(actual, means)
    baseline_pmf = negative_binomial_pmf_matrix(means, support, dispersion_size=nb_fit.dispersion_size)
    zero_fit = fit_zero_adjustment(actual, event_probability(baseline_pmf, support, kind="eq", threshold=0))
    adjusted_pmf = zero_adjusted_negative_binomial_pmf_matrix(
        means,
        support,
        dispersion_size=nb_fit.dispersion_size,
        zero_adjustment_delta=zero_fit.delta,
    )

    assert zero_fit.delta > 0.0
    assert adjusted_pmf[:, 0].mean() > baseline_pmf[:, 0].mean()
    np.testing.assert_allclose(adjusted_pmf.sum(axis=1), np.ones(len(means)), atol=1e-10)


def test_discrete_scores_match_manual_two_point_case() -> None:
    actual = np.array([0, 1], dtype=int)
    support = np.array([0, 1], dtype=int)
    pmf = np.array(
        [
            [0.75, 0.25],
            [0.20, 0.80],
        ],
        dtype=float,
    )

    log_scores = discrete_log_score(actual, pmf, support)
    crps = discrete_crps(actual, pmf, support)

    expected_mean_negative_log = float(((-math.log(0.75)) + (-math.log(0.80))) / 2.0)
    expected_mean_crps = float((((0.75 - 1.0) ** 2) + ((0.20 - 0.0) ** 2)) / 2.0)
    assert math.isclose(log_scores["mean_negative_log_score"], expected_mean_negative_log, rel_tol=1e-9)
    assert math.isclose(crps["mean_crps"], expected_mean_crps, rel_tol=1e-9)


def test_interval_coverage_reports_empirical_coverage_and_width() -> None:
    actual = np.array([0, 1, 2], dtype=int)
    support = np.array([0, 1, 2], dtype=int)
    pmf = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.15, 0.70, 0.15],
            [0.10, 0.20, 0.70],
        ],
        dtype=float,
    )

    coverage = summarize_interval_coverage(actual, pmf, support, nominal_coverage=0.50)

    assert math.isclose(coverage["empirical_coverage"], 1.0, rel_tol=1e-9)
    assert math.isclose(coverage["mean_width"], 2.0 / 3.0, rel_tol=1e-9)


def test_summarize_distribution_metrics_exposes_zero_tail_and_interval_sections() -> None:
    actual = np.array([0, 1, 3, 5], dtype=int)
    means = np.array([0.4, 1.2, 2.8, 4.9], dtype=float)
    support = build_support(20)

    pmf = poisson_pmf_matrix(means, support)
    summary = summarize_distribution_metrics(actual, pmf, support, calibration_bin_count=4)

    assert "mean_crps" in summary
    assert "mean_negative_log_score" in summary
    assert summary["zero_calibration"]["p_0"]["event"] == "p_0"
    assert summary["zero_calibration"]["p_ge_1"]["event"] == "p_ge_1"
    assert summary["tail_calibration"]["p_ge_3"]["event"] == "p_ge_3"
    assert summary["tail_calibration"]["p_ge_5"]["event"] == "p_ge_5"
    assert summary["tail_calibration"]["p_ge_10"]["event"] == "p_ge_10"
    assert summary["interval_coverage"]["central_50"]["nominal_coverage"] == 0.5
    assert summary["interval_coverage"]["central_80"]["nominal_coverage"] == 0.8
    assert summary["interval_coverage"]["central_95"]["nominal_coverage"] == 0.95
