"""Tests for dashboard overview endpoint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _write_json(path: Path, payload: dict) -> None:
    """Helper to write JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _training_payload(
    *, model_version: str, target: str, roc_auc: float, summary_rel_path: str
) -> dict:
    """Create a training payload."""
    return {
        "model_version": model_version,
        "data_version_hash": "abc123",
        "holdout_season": 2024,
        "feature_columns": ["feature_a", "feature_b"],
        "models": {
            f"{target}_model": {
                "model_name": f"{target}_model",
                "target_column": target,
                "model_version": model_version,
                "model_path": summary_rel_path,
                "best_params": {"n_estimators": 100},
                "holdout_metrics": {
                    "accuracy": 0.61,
                    "log_loss": 0.65,
                    "roc_auc": roc_auc,
                },
                "feature_importance_rankings": [
                    {"feature": "feature_a", "importance": 0.8},
                    {"feature": "feature_b", "importance": 0.2},
                ],
                "train_row_count": 100,
                "holdout_row_count": 20,
            }
        },
    }


def test_overview_no_data(client: TestClient, models_dir: Path, experiments_dir: Path) -> None:
    """Test overview endpoint with no run data."""
    response = client.get("/api/overview")

    assert response.status_code == 200
    data = response.json()
    assert data["total_runs"] == 0
    assert data["active_lanes"] == 0
    assert data["best_run"] is None
    assert data["latest_run"] is None
    assert data["recent_runs"] == []


def test_overview_single_run(client: TestClient, models_dir: Path, experiments_dir: Path) -> None:
    """Test overview endpoint with a single run."""
    # Create corresponding run summary JSON with proper naming (*_run_*.json pattern)
    run_path = models_dir / "exp1" / "training_run_001.json"
    payload = _training_payload(
        model_version="1.0",
        target="f5_ml",
        roc_auc=0.72,
        summary_rel_path="exp1/training_run_001.json",
    )
    _write_json(run_path, payload)

    response = client.get("/api/overview")

    assert response.status_code == 200
    data = response.json()
    assert data["total_runs"] == 1
    assert data["active_lanes"] == 1
    assert data["best_run"] is not None
    assert data["best_run"]["roc_auc"] == 0.72
    assert data["latest_run"] is not None
    assert len(data["recent_runs"]) == 0  # No improvements or regressions


def test_overview_multiple_runs_with_improvements(
    client: TestClient, models_dir: Path, experiments_dir: Path
) -> None:
    """Test overview endpoint with multiple runs showing improvements."""
    # Create run summary JSONs with proper naming (training_run_*.json pattern)
    # v1: baseline
    run_v1 = models_dir / "exp1" / "training_run_001.json"
    payload_v1 = {
        "model_version": "1.0",
        "data_version_hash": "abc123",
        "holdout_season": 2024,
        "feature_columns": ["feature_a", "feature_b"],
        "models": {
            "f5_ml_model": {
                "model_name": "f5_ml_model",
                "target_column": "f5_ml",
                "model_version": "1.0",
                "best_params": {"n_estimators": 100},
                "holdout_metrics": {"accuracy": 0.60, "log_loss": 0.66, "roc_auc": 0.70},
                "feature_importance_rankings": [
                    {"feature": "feature_a", "importance": 0.8},
                    {"feature": "feature_b", "importance": 0.2},
                ],
                "train_row_count": 100,
                "holdout_row_count": 20,
            }
        },
    }
    _write_json(run_v1, payload_v1)

    # v2: improvement
    run_v2 = models_dir / "exp1" / "training_run_002.json"
    payload_v2 = {
        "model_version": "1.1",
        "data_version_hash": "abc123",
        "holdout_season": 2024,
        "feature_columns": ["feature_a", "feature_b"],
        "models": {
            "f5_ml_model": {
                "model_name": "f5_ml_model",
                "target_column": "f5_ml",
                "model_version": "1.1",
                "best_params": {"n_estimators": 100},
                "holdout_metrics": {"accuracy": 0.61, "log_loss": 0.65, "roc_auc": 0.72},
                "feature_importance_rankings": [
                    {"feature": "feature_a", "importance": 0.8},
                    {"feature": "feature_b", "importance": 0.2},
                ],
                "train_row_count": 100,
                "holdout_row_count": 20,
            }
        },
    }
    _write_json(run_v2, payload_v2)

    # v3: bigger improvement
    run_v3 = models_dir / "exp1" / "training_run_003.json"
    payload_v3 = {
        "model_version": "1.2",
        "data_version_hash": "abc123",
        "holdout_season": 2024,
        "feature_columns": ["feature_a", "feature_b"],
        "models": {
            "f5_ml_model": {
                "model_name": "f5_ml_model",
                "target_column": "f5_ml",
                "model_version": "1.2",
                "best_params": {"n_estimators": 100},
                "holdout_metrics": {"accuracy": 0.62, "log_loss": 0.64, "roc_auc": 0.74},
                "feature_importance_rankings": [
                    {"feature": "feature_a", "importance": 0.8},
                    {"feature": "feature_b", "importance": 0.2},
                ],
                "train_row_count": 100,
                "holdout_row_count": 20,
            }
        },
    }
    _write_json(run_v3, payload_v3)

    response = client.get("/api/overview")

    assert response.status_code == 200
    data = response.json()
    assert data["total_runs"] == 3
    assert data["active_lanes"] == 1
    assert data["best_run"]["roc_auc"] == 0.74
    # Should include runs with improvements
    assert len(data["recent_runs"]) > 0


def test_overview_multiple_lanes(
    client: TestClient, models_dir: Path, experiments_dir: Path
) -> None:
    """Test overview endpoint with runs in different lanes."""
    # Create runs across different target columns and variants
    # Lane 1: f5_ml baseline
    run_f5ml_base = models_dir / "exp1" / "training_run_001.json"
    payload_f5ml_base = {
        "model_version": "1.0",
        "data_version_hash": "abc123",
        "holdout_season": 2024,
        "feature_columns": ["feature_a"],
        "models": {
            "f5_ml_model": {
                "model_name": "f5_ml_model",
                "target_column": "f5_ml",
                "model_version": "1.0",
                "best_params": {},
                "holdout_metrics": {"accuracy": 0.61, "log_loss": 0.65, "roc_auc": 0.72},
                "feature_importance_rankings": [{"feature": "feature_a", "importance": 1.0}],
                "train_row_count": 100,
                "holdout_row_count": 20,
            }
        },
    }
    _write_json(run_f5ml_base, payload_f5ml_base)

    # Lane 2: run_line baseline
    run_rl_base = models_dir / "exp1" / "training_run_002.json"
    payload_rl_base = {
        "model_version": "1.0",
        "data_version_hash": "abc123",
        "holdout_season": 2024,
        "feature_columns": ["feature_a"],
        "models": {
            "run_line_model": {
                "model_name": "run_line_model",
                "target_column": "run_line",
                "model_version": "1.0",
                "best_params": {},
                "holdout_metrics": {"accuracy": 0.58, "log_loss": 0.68, "roc_auc": 0.68},
                "feature_importance_rankings": [{"feature": "feature_a", "importance": 1.0}],
                "train_row_count": 100,
                "holdout_row_count": 20,
            }
        },
    }
    _write_json(run_rl_base, payload_rl_base)

    # Lane 3: f5_ml v2 variant
    run_f5ml_v2 = models_dir / "exp1" / "training_run_003.json"
    payload_f5ml_v2 = {
        "model_version": "1.1",
        "data_version_hash": "abc123",
        "holdout_season": 2024,
        "feature_columns": ["feature_a"],
        "models": {
            "f5_ml_model": {
                "model_name": "f5_ml_model",
                "target_column": "f5_ml",
                "model_version": "1.1",
                "best_params": {},
                "holdout_metrics": {"accuracy": 0.62, "log_loss": 0.64, "roc_auc": 0.74},
                "feature_importance_rankings": [{"feature": "feature_a", "importance": 1.0}],
                "train_row_count": 100,
                "holdout_row_count": 20,
            }
        },
    }
    _write_json(run_f5ml_v2, payload_f5ml_v2)

    response = client.get("/api/overview")

    assert response.status_code == 200
    data = response.json()
    assert data["total_runs"] == 3
    # Should have 2 different lanes (holdout_season:target_column:variant combinations)
    # Lane 1: 2024:f5_ml:base (runs 1 and 3)
    # Lane 2: 2024:run_line:base (run 2)
    assert data["active_lanes"] == 2
    assert data["best_run"]["roc_auc"] == 0.74


def test_overview_response_structure(
    client: TestClient, models_dir: Path, experiments_dir: Path
) -> None:
    """Test that overview response has all required fields."""
    # Create a single run
    run_path = models_dir / "exp1" / "training_run_001.json"
    payload = _training_payload(
        model_version="1.0",
        target="f5_ml",
        roc_auc=0.72,
        summary_rel_path="exp1/training_run_001.json",
    )
    _write_json(run_path, payload)

    response = client.get("/api/overview")

    assert response.status_code == 200
    data = response.json()

    # Check required fields exist
    assert "total_runs" in data
    assert "active_lanes" in data
    assert "best_run" in data
    assert "latest_run" in data
    assert "recent_runs" in data

    # Check types
    assert isinstance(data["total_runs"], int)
    assert isinstance(data["active_lanes"], int)
    assert isinstance(data["recent_runs"], list)
