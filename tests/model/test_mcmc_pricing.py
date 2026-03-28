from __future__ import annotations

import math

import numpy as np

from src.model.mcmc_pricing import (
    build_distribution_comparison,
    normalize_probability_vector,
    summarize_away_run_distribution,
    tail_probability,
)


def test_normalize_probability_vector_reweights_non_negative_mass() -> None:
    normalized = normalize_probability_vector(np.array([0.0, 2.0, 3.0], dtype=float))

    np.testing.assert_allclose(normalized, np.array([0.0, 0.4, 0.6], dtype=float), atol=1e-12)
    assert math.isclose(float(normalized.sum()), 1.0, rel_tol=1e-12)


def test_summarize_away_run_distribution_reports_expected_runs_tails_and_intervals() -> None:
    summary = summarize_away_run_distribution(
        support=np.array([0, 1, 2, 3], dtype=int),
        probabilities=np.array([0.10, 0.20, 0.30, 0.40], dtype=float),
    )

    assert math.isclose(summary.expected_away_runs, 2.0, rel_tol=1e-12)
    assert math.isclose(summary.shutout_probability, 0.10, rel_tol=1e-12)
    assert math.isclose(summary.tail_probabilities["p_ge_3"], 0.40, rel_tol=1e-12)
    assert math.isclose(
        tail_probability([0, 1, 2, 3], [0.10, 0.20, 0.30, 0.40], threshold=2),
        0.70,
        rel_tol=1e-12,
    )
    assert summary.central_intervals["central_80"]["lower_bound"] == 0
    assert summary.central_intervals["central_80"]["upper_bound"] == 3
    assert summary.away_run_pmf[-1] == {"runs": 3, "probability": 0.4}


def test_build_distribution_comparison_flags_metric_improvements() -> None:
    comparison = build_distribution_comparison(
        challenger_label="stage4_mcmc",
        challenger_mean_metrics={
            "rmse": 3.28,
            "predicted_mean": 4.39,
            "actual_mean": 4.30,
        },
        challenger_distribution_metrics={
            "mean_crps": 1.77,
            "mean_negative_log_score": 2.46,
            "zero_calibration": {"p_0": {"absolute_error": 0.02}},
            "tail_calibration": {
                "p_ge_3": {"absolute_error": 0.02},
                "p_ge_5": {"absolute_error": 0.03},
                "p_ge_10": {"absolute_error": 0.04},
            },
        },
        baseline_label="control",
        baseline_mean_metrics={
            "rmse": 3.29,
            "predicted_mean": 4.34,
            "actual_mean": 4.30,
        },
        baseline_distribution_metrics={
            "mean_crps": 1.79,
            "mean_negative_log_score": 2.48,
            "zero_calibration": {"p_0": {"absolute_error": 0.03}},
            "tail_calibration": {
                "p_ge_3": {"absolute_error": 0.04},
                "p_ge_5": {"absolute_error": 0.05},
                "p_ge_10": {"absolute_error": 0.04},
            },
        },
    )

    assert comparison["challenger_label"] == "stage4_mcmc"
    assert comparison["baseline_label"] == "control"
    assert comparison["improvement_flags"]["beats_baseline_on_crps"] is True
    assert comparison["improvement_flags"]["beats_baseline_on_negative_log_score"] is True
    assert comparison["guardrails"]["rmse_within_2pct"] is True
