from __future__ import annotations

import json

import pandas as pd

from src.ops.experiment_tracker import append_experiment_record


def test_append_experiment_record_writes_jsonl_and_csv(tmp_path) -> None:
    log_path = append_experiment_record(
        {
            "run_type": "training",
            "experiment_name": "tracker-smoke",
            "models": {
                "f5_ml_model": {
                    "holdout_metrics": {"roc_auc": 0.59, "log_loss": 0.68},
                }
            },
        },
        tracking_dir=tmp_path,
    )

    csv_path = tmp_path / "experiment_log.csv"
    assert log_path.exists()
    assert csv_path.exists()

    with log_path.open("r", encoding="utf-8") as handle:
        payload = json.loads(handle.readline())
    frame = pd.read_csv(csv_path)

    assert payload["run_type"] == "training"
    assert payload["experiment_name"] == "tracker-smoke"
    assert frame.loc[0, "models_f5_ml_model_holdout_metrics_roc_auc"] == 0.59
