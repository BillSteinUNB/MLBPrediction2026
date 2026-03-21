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


def test_get_runs_empty(client: TestClient, models_dir: Path) -> None:
    """GET /api/runs returns empty list when no runs exist."""
    response = client.get("/api/runs")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 0


def test_get_runs_list(client: TestClient, models_dir: Path, experiments_dir: Path) -> None:
    """GET /api/runs returns list of RunSummary objects."""
    # Create mock run JSON files
    _create_mock_run_json(
        models_dir,
        experiment_name="xgb_v1",
        holdout_season=2024,
        model_version="1.0.20260320T100000",
    )
    _create_mock_run_json(
        models_dir,
        experiment_name="xgb_v2",
        holdout_season=2024,
        model_version="1.0.20260321T100000",
        roc_auc=0.85,
    )

    response = client.get("/api/runs")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2

    # Verify first run
    first = data[0]
    assert first["experiment_name"] == "xgb_v1"
    assert first["model_name"] == "xgboost"
    assert first["target_column"] == "f5_ml_result"
    assert first["holdout_season"] == 2024
    assert first["variant"] == "base"
    assert first["roc_auc"] == 0.82

    # Verify second run
    second = data[1]
    assert second["experiment_name"] == "xgb_v2"
    assert second["roc_auc"] == 0.85
    assert second["variant"] == "base"


def test_get_runs_filter_holdout_season(
    client: TestClient, models_dir: Path, experiments_dir: Path
) -> None:
    """GET /api/runs?holdout_season=2024 filters by holdout_season."""
    _create_mock_run_json(models_dir, experiment_name="xgb_v1", holdout_season=2023)
    _create_mock_run_json(models_dir, experiment_name="xgb_v2", holdout_season=2024)
    _create_mock_run_json(models_dir, experiment_name="xgb_v3", holdout_season=2024)

    response = client.get("/api/runs?holdout_season=2024")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert all(run["holdout_season"] == 2024 for run in data)


def test_get_runs_filter_target_column(
    client: TestClient, models_dir: Path, experiments_dir: Path
) -> None:
    """GET /api/runs?target_column=f5_ml_result filters by target_column."""
    _create_mock_run_json(models_dir, experiment_name="xgb_v1", target_column="f5_ml_result")
    _create_mock_run_json(models_dir, experiment_name="xgb_v2", target_column="f5_rl_result")
    _create_mock_run_json(models_dir, experiment_name="xgb_v3", target_column="f5_ml_result")

    response = client.get("/api/runs?target_column=f5_ml_result")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert all(run["target_column"] == "f5_ml_result" for run in data)


def test_get_runs_filter_variant(
    client: TestClient, models_dir: Path, experiments_dir: Path
) -> None:
    """GET /api/runs?variant=base filters by variant."""
    # All training runs have variant="base"
    _create_mock_run_json(models_dir, experiment_name="xgb_v1")
    _create_mock_run_json(models_dir, experiment_name="xgb_v2")

    response = client.get("/api/runs?variant=base")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert all(run["variant"] == "base" for run in data)


def test_get_runs_filter_combined(
    client: TestClient, models_dir: Path, experiments_dir: Path
) -> None:
    """GET /api/runs with multiple filters applies all constraints."""
    _create_mock_run_json(
        models_dir,
        experiment_name="xgb_v1",
        holdout_season=2024,
        target_column="f5_ml_result",
        model_version="1.0.20260320T100000",
    )
    _create_mock_run_json(
        models_dir,
        experiment_name="xgb_v2",
        holdout_season=2024,
        target_column="f5_rl_result",
        model_version="1.0.20260321T100000",
    )
    _create_mock_run_json(
        models_dir,
        experiment_name="xgb_v3",
        holdout_season=2023,
        target_column="f5_ml_result",
        model_version="1.0.20260322T100000",
    )
    _create_mock_run_json(
        models_dir,
        experiment_name="xgb_v4",
        holdout_season=2024,
        target_column="f5_ml_result",
        model_version="1.0.20260323T100000",
    )

    response = client.get("/api/runs?holdout_season=2024&target_column=f5_ml_result&variant=base")

    assert response.status_code == 200
    data = response.json()
    # Should only match xgb_v1 and xgb_v4 (both match all three filters)
    assert len(data) == 2
    for run in data:
        assert run["holdout_season"] == 2024
        assert run["target_column"] == "f5_ml_result"
        assert run["variant"] == "base"


def test_get_run_detail_exists(client: TestClient, models_dir: Path, experiments_dir: Path) -> None:
    """GET /api/runs/detail?summary_path=... returns RunDetail for existing run."""
    # Create the run in the models directory
    run_path = _create_mock_run_json(
        models_dir,
        experiment_name="xgb_v1",
        model_version="1.0.20260321T100000",
    )

    # Get the run to find the actual summary_path
    from src.dashboard.adapters import ExperimentDataAdapter

    adapter = ExperimentDataAdapter(models_dir=models_dir)
    runs = adapter.get_all_runs(models_dir)

    assert len(runs) == 1
    run = runs[0]
    summary_path = run.summary_path

    response = client.get(f"/api/runs/detail?summary_path={summary_path}")

    assert response.status_code == 200
    data = response.json()
    assert data["experiment_name"] == "xgb_v1"
    assert "xgb_v1" in data["summary_path"]
    assert data["feature_importance"] is not None
    assert len(data["feature_importance"]) == 2
    assert data["feature_importance"][0]["feature"] == "batting_wrc"
    assert data["best_params"] is not None
    assert data["best_params"]["max_depth"] == 6


def test_get_run_detail_not_found(client: TestClient, models_dir: Path) -> None:
    """GET /api/runs/detail?summary_path=... returns 404 when run doesn't exist."""
    response = client.get("/api/runs/detail?summary_path=nonexistent/path/to/run.json")

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data or "error" in data
