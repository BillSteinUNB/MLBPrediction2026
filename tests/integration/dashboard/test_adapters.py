from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dashboard.adapters import ExperimentDataAdapter
from src.dashboard.schemas import Promotion, RunSummary


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _training_payload(
    *, model_version: str, target: str, roc_auc: float, summary_rel_path: str
) -> dict:
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


def _stacking_payload(*, model_version: str, target: str) -> dict:
    return {
        "model_version": model_version,
        "data_version_hash": "abc123",
        "holdout_season": 2024,
        "raw_meta_feature_columns": ["park_runs_factor"],
        "models": {
            f"{target}_stacking_model": {
                "model_name": f"{target}_stacking_model",
                "target_column": target,
                "model_version": model_version,
                "meta_feature_columns": ["xgb_probability", "park_runs_factor"],
                "holdout_metrics": {
                    "base_brier": 0.22,
                    "stacked_brier": 0.21,
                    "stacked_brier_improvement": 0.01,
                    "base_log_loss": 0.66,
                    "stacked_log_loss": 0.64,
                    "base_accuracy": 0.6,
                    "stacked_accuracy": 0.63,
                    "base_roc_auc": 0.62,
                    "stacked_roc_auc": 0.65,
                },
            }
        },
    }


def _calibration_payload(*, model_version: str, target: str) -> dict:
    return {
        "model_version": model_version,
        "data_version_hash": "abc123",
        "holdout_season": 2024,
        "calibration_method": "identity",
        "models": {
            f"{target}_calibrated_model": {
                "model_name": f"{target}_calibrated_model",
                "target_column": target,
                "calibration_method": "identity",
                "model_version": model_version,
                "train_row_count": 120,
                "holdout_row_count": 20,
                "holdout_metrics": {
                    "stacked_brier": 0.21,
                    "calibrated_brier": 0.2,
                    "brier_improvement": 0.01,
                    "stacked_ece": 0.03,
                    "calibrated_ece": 0.02,
                    "stacked_accuracy": 0.63,
                    "calibrated_accuracy": 0.64,
                    "stacked_log_loss": 0.64,
                    "calibrated_log_loss": 0.62,
                    "stacked_roc_auc": 0.65,
                    "calibrated_roc_auc": 0.67,
                    "reliability_diagram": [
                        {
                            "bin_index": 0,
                            "mean_predicted_probability": 0.4,
                            "empirical_positive_rate": 0.5,
                            "count": 10,
                        }
                    ],
                    "max_reliability_gap": 0.1,
                    "quality_gates": {"brier_lt_0_25": True},
                },
            }
        },
    }


@pytest.fixture
def seeded_models_dir(models_dir: Path) -> Path:
    lane_dir = models_dir / "exp-one"
    _write_json(
        lane_dir / "training_run_20260101T000000Z_a.json",
        _training_payload(
            model_version="20260101T000000Z_a",
            target="f5_ml_result",
            roc_auc=0.61,
            summary_rel_path="data\\models\\exp-one\\f5_ml_model_20260101T000000Z_a.joblib",
        ),
    )
    _write_json(
        lane_dir / "training_run_20260201T000000Z_b.json",
        _training_payload(
            model_version="20260201T000000Z_b",
            target="f5_ml_result",
            roc_auc=0.66,
            summary_rel_path="data\\models\\exp-one\\f5_ml_model_20260201T000000Z_b.joblib",
        ),
    )
    _write_json(
        lane_dir / "stacking_run_20260201T000000Z_b.json",
        _stacking_payload(model_version="20260201T000000Z_b", target="f5_ml_result"),
    )
    _write_json(
        lane_dir / "calibration_run_20260201T000000Z_b.json",
        _calibration_payload(model_version="20260201T000000Z_b", target="f5_ml_result"),
    )
    _write_json(
        lane_dir / "training_run_20260201T000000Z_b_rl.json",
        _training_payload(
            model_version="20260201T000000Z_b",
            target="f5_rl_result",
            roc_auc=0.58,
            summary_rel_path="data\\models\\exp-one\\f5_rl_model_20260201T000000Z_b.joblib",
        ),
    )
    return models_dir


def test_get_all_runs_uses_mtime_cache(
    monkeypatch: pytest.MonkeyPatch, seeded_models_dir: Path
) -> None:
    adapter = ExperimentDataAdapter(models_dir=seeded_models_dir)
    calls = {"count": 0}

    from src.ops import experiment_report

    original = experiment_report.build_experiment_metrics_dataframe

    def wrapped(models_dir: str | Path):
        calls["count"] += 1
        return original(models_dir)

    monkeypatch.setattr("src.dashboard.adapters.build_experiment_metrics_dataframe", wrapped)

    runs_first = adapter.get_all_runs(seeded_models_dir)
    runs_second = adapter.get_all_runs(seeded_models_dir)

    assert len(runs_first) >= 4
    assert runs_second
    assert calls["count"] == 1
    assert all("\\" not in run.summary_path for run in runs_first)


def test_get_run_detail_merges_json_data(seeded_models_dir: Path) -> None:
    adapter = ExperimentDataAdapter(models_dir=seeded_models_dir)
    runs = adapter.get_all_runs(seeded_models_dir)
    calibration_run = next(run for run in runs if run.run_kind == "calibration")

    detail = adapter.get_run_detail(seeded_models_dir, calibration_run.summary_path)

    assert detail is not None
    assert detail.summary_path == calibration_run.summary_path
    assert detail.target_column == "f5_ml_result"
    assert detail.reliability_diagram
    assert detail.reliability_diagram[0].predicted_mean == 0.4
    assert detail.reliability_diagram[0].true_fraction == 0.5
    assert detail.quality_gates == {"brier_lt_0_25": True}
    assert detail.calibration_method == "identity"


def test_get_lanes_and_overview(seeded_models_dir: Path) -> None:
    adapter = ExperimentDataAdapter(models_dir=seeded_models_dir)
    runs = adapter.get_all_runs(seeded_models_dir)

    lanes = adapter.get_lanes(runs)
    overview = adapter.get_overview(runs)

    assert len(lanes) >= 2
    assert all(lane.latest_run is not None for lane in lanes)
    assert overview.total_runs == len(runs)
    assert overview.active_lanes == len(lanes)
    assert overview.latest_run is not None
    assert overview.best_run is not None
    assert 1 <= len(overview.recent_runs) <= 10


def test_compare_runs_returns_deltas_and_winner() -> None:
    adapter = ExperimentDataAdapter()
    run_a = RunSummary(
        experiment_name="exp-one",
        summary_path="data/models/exp-one/training_a.json",
        run_kind="training",
        model_name="f5_ml_model",
        target_column="f5_ml_result",
        model_version="v1",
        variant="base",
        run_timestamp="20260101T000000Z",
        holdout_season=2024,
        accuracy=0.6,
        log_loss=0.66,
        roc_auc=0.61,
        brier=0.23,
    )
    run_b = RunSummary(
        experiment_name="exp-two",
        summary_path="data/models/exp-two/training_b.json",
        run_kind="training",
        model_name="f5_ml_model",
        target_column="f5_ml_result",
        model_version="v2",
        variant="base",
        run_timestamp="20260201T000000Z",
        holdout_season=2024,
        accuracy=0.64,
        log_loss=0.62,
        roc_auc=0.67,
        brier=0.2,
    )

    result = adapter.compare_runs(run_a, run_b)

    assert result.metric_deltas["roc_auc"] == pytest.approx(0.06)
    assert result.metric_deltas["log_loss"] == pytest.approx(-0.04)
    assert result.winner == "b"


def test_read_and_write_promotions(experiments_dir: Path) -> None:
    adapter = ExperimentDataAdapter(experiments_dir=experiments_dir)

    assert adapter.read_promotions(experiments_dir) == []

    created = adapter.write_promotion(
        experiments_dir,
        Promotion(
            promotion_id="p1",
            run_id="data\\models\\exp-one\\training_run.json",
            from_stage="staging",
            to_stage="production",
            promoted_timestamp="2026-03-21T16:38:29Z",
            metadata={"actor": "tester"},
        ),
    )

    promotions = adapter.read_promotions(experiments_dir)

    assert created.run_id == "data/models/exp-one/training_run.json"
    assert len(promotions) == 1
    assert promotions[0].promotion_id == "p1"
    assert promotions[0].run_id == "data/models/exp-one/training_run.json"


def test_malformed_json_is_skipped_with_empty_results(
    models_dir: Path, experiments_dir: Path
) -> None:
    adapter = ExperimentDataAdapter(models_dir=models_dir, experiments_dir=experiments_dir)

    bad_summary = models_dir / "exp-bad" / "training_run_20260101T000000Z_bad.json"
    bad_summary.parent.mkdir(parents=True, exist_ok=True)
    bad_summary.write_text("{this-is-not-json", encoding="utf-8")

    promotions_file = experiments_dir / "promotions.json"
    promotions_file.write_text("{also-not-json", encoding="utf-8")

    detail = adapter.get_run_detail(models_dir, bad_summary)
    promotions = adapter.read_promotions(experiments_dir)

    assert detail is None
    assert promotions == []
