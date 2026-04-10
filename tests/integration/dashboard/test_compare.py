"""Tests for compare endpoint."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def sample_runs_dir(models_dir: Path) -> Path:
    """Create a models directory with sample run summary files."""
    import json

    # Create sample directory structure
    run_dir = models_dir / "test_exp"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create sample training_run_a.json and training_run_b.json (matching required naming pattern)
    summary_a = {
        "experiment_name": "test_exp",
        "model_version": "1.0",
        "holdout_season": 2023,
        "feature_columns": ["feat1"] * 42,
        "models": {
            "xgboost_v1": {
                "target_column": "f5_ml",
                "variant": "baseline",
                "holdout_metrics": {
                    "accuracy": 0.75,
                    "log_loss": 0.45,
                    "roc_auc": 0.82,
                    "brier": 0.22,
                    "ece": 0.03,
                    "reliability_gap": 0.02,
                },
            }
        },
    }

    summary_b = {
        "experiment_name": "test_exp",
        "model_version": "1.0",
        "holdout_season": 2023,
        "feature_columns": ["feat1"] * 42,
        "models": {
            "xgboost_v1": {
                "target_column": "f5_ml",
                "variant": "baseline",
                "holdout_metrics": {
                    "accuracy": 0.78,
                    "log_loss": 0.42,
                    "roc_auc": 0.85,
                    "brier": 0.20,
                    "ece": 0.025,
                    "reliability_gap": 0.015,
                },
            }
        },
    }

    # Use naming pattern: training_run_*, stacking_run_*, or calibration_run_*
    run_a_path = run_dir / "training_run_a.json"
    run_b_path = run_dir / "training_run_b.json"
    run_a_path.write_text(json.dumps(summary_a))
    run_b_path.write_text(json.dumps(summary_b))

    return models_dir


def test_compare_same_lane(
    client: TestClient, sample_runs_dir: Path, experiments_dir: Path
) -> None:
    """Test comparing two runs in the same lane."""
    response = client.get(
        "/api/compare?run_a=test_exp/training_run_a.json&run_b=test_exp/training_run_b.json"
    )

    assert response.status_code == 200
    data = response.json()

    # Verify structure
    assert "run_a" in data
    assert "run_b" in data
    assert "metric_deltas" in data
    assert "same_lane" in data

    # Verify same_lane flag is true (same season, target, variant)
    assert data["same_lane"] is True

    # Verify run details
    assert data["run_a"]["accuracy"] == 0.75
    assert data["run_b"]["accuracy"] == 0.78

    # Verify metric deltas (run_b - run_a)
    assert data["metric_deltas"]["accuracy"] == pytest.approx(0.03, rel=1e-2)
    assert data["metric_deltas"]["roc_auc"] == pytest.approx(0.03, rel=1e-2)
    assert data["metric_deltas"]["log_loss"] == pytest.approx(-0.03, rel=1e-2)  # negative is better
    assert data["metric_deltas"]["brier"] == pytest.approx(-0.02, rel=1e-2)  # negative is better


def test_compare_different_lanes(
    client: TestClient, sample_runs_dir: Path, experiments_dir: Path
) -> None:
    """Test comparing runs in different lanes (different holdout_season)."""
    # Create a run with different holdout_season
    import json
    from pathlib import Path

    models_dir = sample_runs_dir
    run_dir = models_dir / "test_exp"

    summary_c = {
        "experiment_name": "test_exp",
        "model_version": "1.0",
        "holdout_season": 2022,  # Different season
        "feature_columns": ["feat1"] * 42,
        "models": {
            "xgboost_v1": {
                "target_column": "f5_ml",
                "variant": "baseline",
                "holdout_metrics": {
                    "accuracy": 0.78,
                    "log_loss": 0.42,
                    "roc_auc": 0.85,
                    "brier": 0.20,
                    "ece": 0.025,
                    "reliability_gap": 0.015,
                },
            }
        },
    }

    run_c_path = run_dir / "training_run_c.json"
    run_c_path.write_text(json.dumps(summary_c))

    response = client.get(
        "/api/compare?run_a=test_exp/training_run_a.json&run_b=test_exp/training_run_c.json"
    )

    assert response.status_code == 200
    data = response.json()

    # Verify same_lane is false (different holdout_season)
    assert data["same_lane"] is False


def test_compare_run_not_found(
    client: TestClient, sample_runs_dir: Path, experiments_dir: Path
) -> None:
    """Test 404 when run is not found."""
    # Try to compare with non-existent run
    response = client.get("/api/compare?run_a=test_exp/training_run_a.json&run_b=nonexistent.json")

    # Should return 404 when a run is not found
    assert response.status_code == 404
    data = response.json()
    assert "Run not found" in data["detail"]


def test_compare_winner_determination(
    client: TestClient, sample_runs_dir: Path, experiments_dir: Path
) -> None:
    """Test that winner is correctly determined."""
    import json
    from pathlib import Path

    models_dir = sample_runs_dir
    run_dir = models_dir / "test_exp"

    # run_e is clearly better on all metrics
    summary_e = {
        "experiment_name": "test_exp",
        "model_version": "1.0",
        "holdout_season": 2023,
        "feature_columns": ["feat1"] * 42,
        "models": {
            "xgboost_v1": {
                "target_column": "f5_ml",
                "variant": "baseline",
                "holdout_metrics": {
                    "accuracy": 0.85,
                    "log_loss": 0.30,
                    "roc_auc": 0.90,
                    "brier": 0.18,
                    "ece": 0.02,
                    "reliability_gap": 0.01,
                },
            }
        },
    }

    # run_f is clearly worse on all metrics
    summary_f = {
        "experiment_name": "test_exp",
        "model_version": "1.0",
        "holdout_season": 2023,
        "feature_columns": ["feat1"] * 42,
        "models": {
            "xgboost_v1": {
                "target_column": "f5_ml",
                "variant": "baseline",
                "holdout_metrics": {
                    "accuracy": 0.70,
                    "log_loss": 0.50,
                    "roc_auc": 0.75,
                    "brier": 0.30,
                    "ece": 0.05,
                    "reliability_gap": 0.04,
                },
            }
        },
    }

    run_e_path = run_dir / "training_run_e.json"
    run_f_path = run_dir / "training_run_f.json"
    run_e_path.write_text(json.dumps(summary_e))
    run_f_path.write_text(json.dumps(summary_f))

    response = client.get(
        "/api/compare?run_a=test_exp/training_run_e.json&run_b=test_exp/training_run_f.json"
    )

    assert response.status_code == 200
    data = response.json()

    # run_e should be the winner (better on all metrics)
    assert data["winner"] == "a"
