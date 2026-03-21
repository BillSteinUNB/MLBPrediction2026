"""Tests for run detail endpoint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dashboard.adapters import ExperimentDataAdapter
from src.dashboard.schemas import RunDetail


def _write_json(path: Path, payload: dict) -> None:
    """Helper to write JSON payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _training_artifact(*, model_name: str = "xgboost", target: str = "target_col") -> dict:
    """Create a training run artifact with feature importance."""
    return {
        "model_version": "20260101T000000Z_test",
        "holdout_season": 2025,
        "models": {
            model_name: {
                "target_column": target,
                "feature_importance_rankings": [
                    {"feature": "feature_1", "importance": 0.25},
                    {"feature": "feature_2", "importance": 0.18},
                    {"feature": "feature_3", "importance": 0.15},
                ],
                "best_params": {"max_depth": 5, "n_estimators": 100},
                "holdout_metrics": {
                    "reliability_diagram": [
                        {
                            "bin_index": 0,
                            "predicted_mean": 0.1,
                            "true_fraction": 0.12,
                            "count": 50,
                        },
                        {
                            "bin_index": 1,
                            "predicted_mean": 0.3,
                            "true_fraction": 0.28,
                            "count": 60,
                        },
                    ],
                    "quality_gates": {"brier": 0.12, "ece": 0.03},
                },
                "train_row_count": 1000,
                "holdout_row_count": 300,
                "calibration_method": "platt",
                "meta_feature_columns": ["meta_1", "meta_2"],
            }
        },
    }


def test_get_run_detail_training_run(models_dir: Path) -> None:
    """Test retrieving run detail for a training run with feature importance."""
    # Use build_experiment_metrics_dataframe to automatically detect runs
    artifact_path = models_dir / "exp-one" / "training_run_20260101T000000Z_a.json"
    _write_json(artifact_path, _training_artifact())

    adapter = ExperimentDataAdapter(models_dir=models_dir)
    runs = adapter.get_all_runs(models_dir)

    assert len(runs) > 0
    run = runs[0]
    detail = adapter.get_run_detail(models_dir, run.summary_path)

    assert detail is not None
    assert isinstance(detail, RunDetail)
    assert detail.run_kind == "training"
    assert detail.model_name == "xgboost"
    assert detail.feature_importance is not None
    assert len(detail.feature_importance) == 3
    assert detail.feature_importance[0].feature == "feature_1"
    assert detail.feature_importance[0].importance == 0.25


def test_get_run_detail_calibration_run(models_dir: Path) -> None:
    """Test retrieving run detail for a calibration run with reliability diagram."""
    artifact_path = models_dir / "exp-one" / "calibration_run_20260101T000000Z_b.json"
    _write_json(artifact_path, _training_artifact(model_name="calib_model"))

    adapter = ExperimentDataAdapter(models_dir=models_dir)
    runs = adapter.get_all_runs(models_dir)

    assert len(runs) > 0
    run = runs[0]
    detail = adapter.get_run_detail(models_dir, run.summary_path)

    assert detail is not None
    assert detail.reliability_diagram is not None
    assert len(detail.reliability_diagram) == 2
    assert detail.reliability_diagram[0].bin_index == 0
    assert detail.reliability_diagram[0].predicted_mean == 0.1


def test_get_run_detail_quality_gates(models_dir: Path) -> None:
    """Test that quality gates are extracted correctly."""
    artifact_path = models_dir / "exp-one" / "training_run_20260101T000000Z_c.json"
    _write_json(artifact_path, _training_artifact())

    adapter = ExperimentDataAdapter(models_dir=models_dir)
    runs = adapter.get_all_runs(models_dir)

    assert len(runs) > 0
    run = runs[0]
    detail = adapter.get_run_detail(models_dir, run.summary_path)

    assert detail is not None
    assert detail.quality_gates is not None
    assert detail.quality_gates["brier"] == 0.12
    assert detail.quality_gates["ece"] == 0.03


def test_get_run_detail_best_params(models_dir: Path) -> None:
    """Test that best_params are extracted correctly."""
    artifact_path = models_dir / "exp-one" / "training_run_20260101T000000Z_d.json"
    _write_json(artifact_path, _training_artifact())

    adapter = ExperimentDataAdapter(models_dir=models_dir)
    runs = adapter.get_all_runs(models_dir)

    assert len(runs) > 0
    run = runs[0]
    detail = adapter.get_run_detail(models_dir, run.summary_path)

    assert detail is not None
    assert detail.best_params is not None
    assert detail.best_params["max_depth"] == 5
    assert detail.best_params["n_estimators"] == 100


def test_get_run_detail_stacking_metrics(models_dir: Path) -> None:
    """Test that stacking_metrics are extracted (holdout_metrics)."""
    artifact_path = models_dir / "exp-one" / "training_run_20260101T000000Z_e.json"
    _write_json(artifact_path, _training_artifact())

    adapter = ExperimentDataAdapter(models_dir=models_dir)
    runs = adapter.get_all_runs(models_dir)

    assert len(runs) > 0
    run = runs[0]
    detail = adapter.get_run_detail(models_dir, run.summary_path)

    assert detail is not None
    assert detail.stacking_metrics is not None
    assert "reliability_diagram" in detail.stacking_metrics
    assert "quality_gates" in detail.stacking_metrics


def test_get_run_detail_not_found(models_dir: Path) -> None:
    """Test that get_run_detail returns None when summary_path not found."""
    adapter = ExperimentDataAdapter(models_dir=models_dir)
    detail = adapter.get_run_detail(models_dir, "nonexistent.json")
    assert detail is None


def test_get_run_detail_endpoint_404(client) -> None:
    """Test that endpoint returns 404 for non-existent run."""
    response = client.get("/api/runs/detail?summary_path=nonexistent.json")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_run_detail_endpoint_missing_query_param(client) -> None:
    """Test that endpoint requires summary_path query parameter."""
    response = client.get("/api/runs/detail")
    # FastAPI returns 422 for missing required query parameter
    assert response.status_code in (422, 404)


def test_get_run_detail_meta_feature_columns(models_dir: Path) -> None:
    """Test that meta_feature_columns are extracted correctly."""
    artifact_path = models_dir / "exp-one" / "training_run_20260101T000000Z_f.json"
    _write_json(artifact_path, _training_artifact())

    adapter = ExperimentDataAdapter(models_dir=models_dir)
    runs = adapter.get_all_runs(models_dir)

    assert len(runs) > 0
    run = runs[0]
    detail = adapter.get_run_detail(models_dir, run.summary_path)

    assert detail is not None
    assert detail.meta_feature_columns is not None
    assert "meta_1" in detail.meta_feature_columns
    assert "meta_2" in detail.meta_feature_columns
