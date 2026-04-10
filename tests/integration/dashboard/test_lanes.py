"""Tests for lanes endpoint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def create_run_file(
    models_dir: Path,
    holdout_season: int,
    target_column: str,
    variant: str,
    model_name: str = "xgboost_v1",
    timestamp: str = "1234567890",
    roc_auc: float = 0.85,
) -> str:
    """Helper to create a run summary file and return its path."""
    run_folder = models_dir / f"exp_{holdout_season}_{target_column}"
    run_folder.mkdir(parents=True, exist_ok=True)

    summary_file = run_folder / f"calibration_run_{timestamp}.json"
    summary_path = str(summary_file).replace("\\", "/")

    summary_data = {
        "holdout_season": holdout_season,
        "model_version": timestamp,
        "calibration_method": variant,
        "models": {
            model_name: {
                "target_column": target_column,
                "calibration_method": variant,
                "holdout_metrics": {
                    "calibrated_roc_auc": roc_auc,
                    "calibrated_accuracy": 0.8,
                    "calibrated_log_loss": 0.4,
                    "calibrated_brier": 0.15,
                    "calibrated_ece": 0.05,
                    "max_reliability_gap": 0.1,
                },
            }
        },
    }

    summary_file.write_text(json.dumps(summary_data))

    return summary_path


def create_metrics_csv(experiments_dir: Path, runs_data: list[dict]) -> None:
    """Helper to create experiment_metrics.csv."""
    metrics_file = experiments_dir / "experiment_metrics.csv"

    # Create header
    headers = [
        "experiment_name",
        "summary_path",
        "run_kind",
        "model_name",
        "target_column",
        "model_version",
        "variant",
        "run_timestamp",
        "holdout_season",
        "feature_column_count",
        "accuracy",
        "log_loss",
        "roc_auc",
        "brier",
        "ece",
        "reliability_gap",
    ]

    lines = [",".join(headers)]

    for run in runs_data:
        values = [
            run.get("experiment_name", "exp1"),
            run.get("summary_path", ""),
            run.get("run_kind", "calibration"),
            run.get("model_name", "xgboost_v1"),
            run.get("target_column", "ML_F5"),
            run.get("model_version", "1234567890"),
            run.get("variant", "baseline"),
            run.get("run_timestamp", "1234567890"),
            str(run.get("holdout_season", 2023)),
            str(run.get("feature_column_count", 100)),
            str(run.get("accuracy", 0.8)),
            str(run.get("log_loss", 0.4)),
            str(run.get("roc_auc", "")),
            str(run.get("brier", 0.15)),
            str(run.get("ece", 0.05)),
            str(run.get("reliability_gap", 0.1)),
        ]
        lines.append(",".join(values))

    metrics_file.write_text("\n".join(lines))


def test_lanes_empty(client: TestClient, models_dir: Path, experiments_dir: Path) -> None:
    """Test lanes endpoint with no runs."""
    response = client.get("/api/lanes")

    assert response.status_code == 200
    lanes = response.json()
    assert isinstance(lanes, list)
    assert len(lanes) == 0


def test_lanes_single_lane_single_run(
    client: TestClient, models_dir: Path, experiments_dir: Path
) -> None:
    """Test lanes endpoint with a single lane containing one run."""
    # Create a single run
    summary_path = create_run_file(
        models_dir,
        holdout_season=2023,
        target_column="ML_F5",
        variant="baseline",
        model_name="xgboost_v1",
        timestamp="1000000000",
        roc_auc=0.85,
    )

    create_metrics_csv(
        experiments_dir,
        [
            {
                "experiment_name": "exp1",
                "summary_path": summary_path,
                "model_name": "xgboost_v1",
                "target_column": "ML_F5",
                "variant": "baseline",
                "holdout_season": 2023,
                "run_timestamp": "1000000000",
                "roc_auc": 0.85,
            }
        ],
    )

    response = client.get("/api/lanes")

    assert response.status_code == 200
    lanes = response.json()
    assert len(lanes) == 1

    lane = lanes[0]
    assert lane["lane_id"] == "2023:ML_F5:baseline"
    assert lane["model_name"] == "xgboost_v1"
    assert lane["variant"] == "baseline"
    assert lane["best_run"] is not None
    assert lane["latest_run"] is not None
    assert lane["best_run"]["roc_auc"] == 0.85
    assert lane["latest_run"]["roc_auc"] == 0.85


def test_lanes_single_lane_multiple_runs(
    client: TestClient, models_dir: Path, experiments_dir: Path
) -> None:
    """Test lanes endpoint with a single lane containing multiple runs."""
    import time

    # Create multiple runs for the same lane
    path1 = create_run_file(
        models_dir,
        holdout_season=2023,
        target_column="ML_F5",
        variant="baseline",
        model_name="xgboost_v1",
        timestamp="1000000000",
        roc_auc=0.80,
    )

    # Add slight delay to ensure mtime differs
    time.sleep(0.01)

    path2 = create_run_file(
        models_dir,
        holdout_season=2023,
        target_column="ML_F5",
        variant="baseline",
        model_name="xgboost_v2",
        timestamp="2000000000",
        roc_auc=0.88,
    )

    create_metrics_csv(
        experiments_dir,
        [
            {
                "experiment_name": "exp1",
                "summary_path": path1,
                "model_name": "xgboost_v1",
                "target_column": "ML_F5",
                "variant": "baseline",
                "holdout_season": 2023,
                "run_timestamp": "1000000000",
                "roc_auc": 0.80,
            },
            {
                "experiment_name": "exp2",
                "summary_path": path2,
                "model_name": "xgboost_v2",
                "target_column": "ML_F5",
                "variant": "baseline",
                "holdout_season": 2023,
                "run_timestamp": "2000000000",
                "roc_auc": 0.88,
            },
        ],
    )

    response = client.get("/api/lanes")

    assert response.status_code == 200
    lanes = response.json()
    assert len(lanes) == 1

    lane = lanes[0]
    assert lane["lane_id"] == "2023:ML_F5:baseline"
    # Latest should be the one created more recently
    assert lane["latest_run"]["model_name"] == "xgboost_v2"
    assert lane["latest_run"]["roc_auc"] == 0.88
    # Best should be the one with higher ROC-AUC (0.88)
    assert lane["best_run"]["roc_auc"] == 0.88


def test_lanes_multiple_lanes(client: TestClient, models_dir: Path, experiments_dir: Path) -> None:
    """Test lanes endpoint with multiple lanes."""
    paths = []

    # Lane 1: holdout_season=2023, target_column=ML_F5, variant=baseline
    path1 = create_run_file(
        models_dir,
        holdout_season=2023,
        target_column="ML_F5",
        variant="baseline",
        model_name="xgboost_v1",
        timestamp="1000000000",
        roc_auc=0.85,
    )
    paths.append(path1)

    # Lane 2: holdout_season=2023, target_column=RL_F5, variant=baseline
    path2 = create_run_file(
        models_dir,
        holdout_season=2023,
        target_column="RL_F5",
        variant="baseline",
        model_name="xgboost_v1",
        timestamp="1000000001",
        roc_auc=0.82,
    )
    paths.append(path2)

    # Lane 3: holdout_season=2024, target_column=ML_F5, variant=baseline
    path3 = create_run_file(
        models_dir,
        holdout_season=2024,
        target_column="ML_F5",
        variant="baseline",
        model_name="xgboost_v1",
        timestamp="1000000002",
        roc_auc=0.87,
    )
    paths.append(path3)

    # Lane 4: holdout_season=2023, target_column=ML_F5, variant=tuned
    path4 = create_run_file(
        models_dir,
        holdout_season=2023,
        target_column="ML_F5",
        variant="tuned",
        model_name="xgboost_v1",
        timestamp="1000000003",
        roc_auc=0.89,
    )
    paths.append(path4)

    runs_data = [
        {
            "summary_path": paths[0],
            "model_name": "xgboost_v1",
            "target_column": "ML_F5",
            "variant": "baseline",
            "holdout_season": 2023,
            "run_timestamp": "1000000000",
            "roc_auc": 0.85,
        },
        {
            "summary_path": paths[1],
            "model_name": "xgboost_v1",
            "target_column": "RL_F5",
            "variant": "baseline",
            "holdout_season": 2023,
            "run_timestamp": "1000000001",
            "roc_auc": 0.82,
        },
        {
            "summary_path": paths[2],
            "model_name": "xgboost_v1",
            "target_column": "ML_F5",
            "variant": "baseline",
            "holdout_season": 2024,
            "run_timestamp": "1000000002",
            "roc_auc": 0.87,
        },
        {
            "summary_path": paths[3],
            "model_name": "xgboost_v1",
            "target_column": "ML_F5",
            "variant": "tuned",
            "holdout_season": 2023,
            "run_timestamp": "1000000003",
            "roc_auc": 0.89,
        },
    ]

    create_metrics_csv(experiments_dir, runs_data)

    response = client.get("/api/lanes")

    assert response.status_code == 200
    lanes = response.json()
    assert len(lanes) == 4

    # Verify lanes are sorted by lane_id
    lane_ids = [lane["lane_id"] for lane in lanes]
    assert lane_ids == sorted(lane_ids)

    # Verify specific lanes
    lane_2023_ml_baseline = next(
        (lane for lane in lanes if lane["lane_id"] == "2023:ML_F5:baseline"), None
    )
    assert lane_2023_ml_baseline is not None
    assert lane_2023_ml_baseline["best_run"]["roc_auc"] == 0.85
    assert lane_2023_ml_baseline["latest_run"]["roc_auc"] == 0.85

    lane_2023_ml_tuned = next(
        (lane for lane in lanes if lane["lane_id"] == "2023:ML_F5:tuned"), None
    )
    assert lane_2023_ml_tuned is not None
    assert lane_2023_ml_tuned["best_run"]["roc_auc"] == 0.89


def test_lanes_response_schema(client: TestClient, models_dir: Path, experiments_dir: Path) -> None:
    """Test that lanes response conforms to expected schema."""
    summary_path = create_run_file(
        models_dir,
        holdout_season=2023,
        target_column="ML_F5",
        variant="baseline",
        model_name="xgboost_v1",
        timestamp="1000000000",
        roc_auc=0.85,
    )

    create_metrics_csv(
        experiments_dir,
        [
            {
                "experiment_name": "exp1",
                "summary_path": summary_path,
                "model_name": "xgboost_v1",
                "target_column": "ML_F5",
                "variant": "baseline",
                "holdout_season": 2023,
                "run_timestamp": "1000000000",
                "roc_auc": 0.85,
            }
        ],
    )

    response = client.get("/api/lanes")

    assert response.status_code == 200
    lanes = response.json()

    for lane in lanes:
        # Check required fields
        assert "lane_id" in lane
        assert "model_name" in lane
        assert "variant" in lane
        assert "best_run" in lane
        assert "latest_run" in lane

        # Check types
        assert isinstance(lane["lane_id"], str)
        assert isinstance(lane["model_name"], str)
        assert isinstance(lane["variant"], str)

        # Check best_run and latest_run structure
        for run in [lane["best_run"], lane["latest_run"]]:
            if run is not None:
                assert "model_name" in run
                assert "target_column" in run
                assert "variant" in run
                assert "run_timestamp" in run
