"""Tests for dashboard promotions endpoint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _create_mock_run_json(
    models_dir: Path,
    experiment_name: str = "xgb_baseline",
    model_name: str = "xgboost",
    target_column: str = "f5_ml_result",
    holdout_season: int = 2024,
    model_version: str = "1.0.20260321T100000",
    accuracy: float = 0.75,
    log_loss: float = 0.55,
    roc_auc: float = 0.82,
    **kwargs,
) -> Path:
    """Helper to create a mock run JSON file in the models directory.

    Creates a file matching pattern training_run_<model_version>.json which
    results in variant="base" when processed by build_experiment_metrics_dataframe.

    Returns the path to the created JSON file.
    """
    # Create experiment directory
    exp_dir = models_dir / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create run JSON file with training_run_ prefix for variant="base"
    run_json_path = exp_dir / f"training_run_{model_version}.json"

    payload = {
        "model_version": model_version,
        "holdout_season": holdout_season,
        "feature_columns": ["feat1", "feat2", "feat3"] * 17,  # 51 features
        "models": {
            model_name: {
                "target_column": target_column,
                "holdout_metrics": {
                    "accuracy": accuracy,
                    "log_loss": log_loss,
                    "roc_auc": roc_auc,
                    "brier": 0.12,
                    "ece": 0.03,
                    "reliability_gap": 0.05,
                },
                "feature_importance_rankings": [
                    {"feature": "batting_wrc", "importance": 0.15},
                    {"feature": "pitching_xfip", "importance": 0.12},
                ],
                "best_params": {"max_depth": 6, "learning_rate": 0.1},
            }
        },
    }
    payload.update(kwargs)
    run_json_path.write_text(json.dumps(payload))

    return run_json_path


def test_promotions_empty_list(client: TestClient, models_dir: Path, experiments_dir: Path) -> None:
    """Test GET /api/promotions returns empty list when file missing."""
    response = client.get("/api/promotions")

    assert response.status_code == 200
    data = response.json()
    assert data == []


def test_create_promotion_success(
    client: TestClient, models_dir: Path, experiments_dir: Path
) -> None:
    """Test POST /api/promotions creates promotion successfully."""
    # Setup: Create a run
    _create_mock_run_json(
        models_dir,
        experiment_name="xgb_v1",
        holdout_season=2024,
        model_version="1.0.20260320T100000",
    )

    # Get the run to find the actual summary_path (which is the absolute path)
    from src.dashboard.adapters import ExperimentDataAdapter

    adapter = ExperimentDataAdapter(models_dir=models_dir, experiments_dir=experiments_dir)
    runs = adapter.get_all_runs(models_dir)
    assert len(runs) == 1
    summary_path = runs[0].summary_path

    # Act: Create promotion
    request_body = {
        "run_id": summary_path,
        "target_stage": "staging",
        "reason": "Performance improved by 2%",
    }
    response = client.post("/api/promotions", json=request_body)

    # Assert
    assert response.status_code == 201
    data = response.json()
    assert data["promotion_id"] is not None
    assert data["run_id"] == summary_path
    assert data["from_stage"] == "development"
    assert data["to_stage"] == "staging"
    assert data["promoted_timestamp"] is not None
    assert data["metadata"]["reason"] == "Performance improved by 2%"


def test_create_promotion_run_not_found(
    client: TestClient, models_dir: Path, experiments_dir: Path
) -> None:
    """Test POST /api/promotions returns 404 when run not found."""
    request_body = {
        "run_id": "/nonexistent/training_run_v1.json",
        "target_stage": "staging",
    }
    response = client.post("/api/promotions", json=request_body)

    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


def test_promotions_persists_to_file(
    client: TestClient, models_dir: Path, experiments_dir: Path
) -> None:
    """Test that promotions are persisted to promotions.json."""
    # Setup: Create a run
    _create_mock_run_json(
        models_dir,
        experiment_name="xgb_v2",
        holdout_season=2024,
        model_version="1.0.20260321T100000",
    )

    # Get the run to find the actual summary_path
    from src.dashboard.adapters import ExperimentDataAdapter

    adapter = ExperimentDataAdapter(models_dir=models_dir, experiments_dir=experiments_dir)
    runs = adapter.get_all_runs(models_dir)
    assert len(runs) == 1
    summary_path = runs[0].summary_path

    # Act: Create promotion
    request_body = {
        "run_id": summary_path,
        "target_stage": "production",
    }
    response = client.post("/api/promotions", json=request_body)
    assert response.status_code == 201

    # Assert: File exists and contains data
    promotions_file = experiments_dir / "promotions.json"
    assert promotions_file.exists()

    with open(promotions_file, encoding="utf-8") as f:
        promotions = json.load(f)

    assert len(promotions) == 1
    assert promotions[0]["run_id"] == summary_path
    assert promotions[0]["to_stage"] == "production"


def test_promotions_list_after_create(
    client: TestClient, models_dir: Path, experiments_dir: Path
) -> None:
    """Test GET /api/promotions returns created promotion."""
    # Setup: Create a run
    _create_mock_run_json(
        models_dir,
        experiment_name="xgb_v3",
        holdout_season=2024,
        model_version="1.0.20260322T100000",
    )

    # Get the run to find the actual summary_path
    from src.dashboard.adapters import ExperimentDataAdapter

    adapter = ExperimentDataAdapter(models_dir=models_dir, experiments_dir=experiments_dir)
    runs = adapter.get_all_runs(models_dir)
    assert len(runs) == 1
    summary_path = runs[0].summary_path

    # Act: Create promotion
    request_body = {
        "run_id": summary_path,
        "target_stage": "staging",
    }
    response = client.post("/api/promotions", json=request_body)
    assert response.status_code == 201
    created_promotion = response.json()

    # Act: List promotions
    response = client.get("/api/promotions")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["promotion_id"] == created_promotion["promotion_id"]
    assert data[0]["run_id"] == summary_path


def test_create_promotion_without_reason(
    client: TestClient, models_dir: Path, experiments_dir: Path
) -> None:
    """Test POST /api/promotions works without optional reason."""
    # Setup: Create a run
    _create_mock_run_json(
        models_dir,
        experiment_name="xgb_v4",
        holdout_season=2024,
        model_version="1.0.20260323T100000",
    )

    # Get the run to find the actual summary_path
    from src.dashboard.adapters import ExperimentDataAdapter

    adapter = ExperimentDataAdapter(models_dir=models_dir, experiments_dir=experiments_dir)
    runs = adapter.get_all_runs(models_dir)
    assert len(runs) == 1
    summary_path = runs[0].summary_path

    # Act: Create promotion without reason
    request_body = {
        "run_id": summary_path,
        "target_stage": "staging",
    }
    response = client.post("/api/promotions", json=request_body)

    # Assert
    assert response.status_code == 201
    data = response.json()
    assert data["metadata"] is None
