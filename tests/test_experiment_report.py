from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.ops.experiment_report import build_experiment_metrics_dataframe, write_experiment_report


def test_build_experiment_metrics_dataframe_computes_deltas(tmp_path: Path) -> None:
    older_dir = tmp_path / "exp-a"
    newer_dir = tmp_path / "exp-b"
    older_dir.mkdir()
    newer_dir.mkdir()

    older_training = {
        "model_version": "20260320T120000Z_oldhash",
        "data_version_hash": "oldhash",
        "holdout_season": 2024,
        "feature_columns": ["a", "b"],
        "models": {
            "f5_ml_model": {
                "target_column": "f5_ml_result",
                "holdout_metrics": {
                    "accuracy": 0.55,
                    "log_loss": 0.68,
                    "roc_auc": 0.58,
                },
            }
        },
    }
    newer_training = {
        "model_version": "20260321T120000Z_newhash",
        "data_version_hash": "newhash",
        "holdout_season": 2024,
        "feature_columns": ["a", "b", "c"],
        "models": {
            "f5_ml_model": {
                "target_column": "f5_ml_result",
                "holdout_metrics": {
                    "accuracy": 0.57,
                    "log_loss": 0.66,
                    "roc_auc": 0.60,
                },
            }
        },
    }

    (older_dir / "training_run_20260320T120000Z_oldhash.json").write_text(
        json.dumps(older_training),
        encoding="utf-8",
    )
    (newer_dir / "training_run_20260321T120000Z_newhash.json").write_text(
        json.dumps(newer_training),
        encoding="utf-8",
    )

    frame = build_experiment_metrics_dataframe(tmp_path)

    assert len(frame) == 2
    latest = frame.iloc[-1]
    assert latest["experiment_name"] == "exp-b"
    assert latest["delta_vs_prev_accuracy"] == pytest.approx(0.02)
    assert latest["delta_vs_prev_log_loss"] == pytest.approx(-0.02)
    assert latest["delta_vs_prev_roc_auc"] == pytest.approx(0.02)
    assert bool(latest["is_best_accuracy"]) is True
    assert bool(latest["is_best_log_loss"]) is True
    assert bool(latest["is_best_roc_auc"]) is True


def test_write_experiment_report_creates_csv_and_html(tmp_path: Path) -> None:
    run_dir = tmp_path / "exp-a"
    run_dir.mkdir()
    payload = {
        "model_version": "20260321T120000Z_hash",
        "data_version_hash": "hash",
        "holdout_season": 2024,
        "feature_columns": ["a"],
        "models": {
            "f5_ml_model": {
                "target_column": "f5_ml_result",
                "holdout_metrics": {
                    "accuracy": 0.57,
                    "log_loss": 0.66,
                    "roc_auc": 0.60,
                },
            }
        },
    }
    (run_dir / "training_run_20260321T120000Z_hash.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    outputs = write_experiment_report(models_dir=tmp_path, output_dir=tmp_path / "reports")

    for path in outputs.values():
        assert path.exists()
    assert "dashboard_html" in outputs
