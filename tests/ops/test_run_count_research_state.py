from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from uuid import uuid4

import pytest

from scripts.report_run_count_research_state import (
    extract_registry_row,
    generate_registry,
    select_current_control,
)


def _write_metadata(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_workspace(name: str) -> Path:
    workspace = Path("tests") / "ops" / ".tmp" / f"{name}_{uuid4().hex}"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def test_generate_registry_writes_expected_outputs() -> None:
    workspace = _make_workspace("generate_registry")
    models_root = workspace / "models"
    output_dir = workspace / "reports"

    try:
        weaker_path = models_root / "2026-away-flat-xgbonly-forceddelta8-b" / (
            "full_game_away_runs_model_20260328T020000Z_demo.metadata.json"
        )
        control_path = models_root / "2026-away-flat-xgbonly-forceddelta8-a" / (
            "full_game_away_runs_model_20260328T010000Z_demo.metadata.json"
        )

        base_payload = {
            "model_name": "full_game_away_runs_model",
            "target_column": "final_away_score",
            "data_version_hash": "demo-hash",
            "holdout_season": 2025,
            "feature_columns": [
                "weather_air_density_factor",
                "away_team_log5_30g",
                "plate_umpire_runs_factor",
                "catcher_adjusted_framing_runs_30g",
                "away_lineup_woba_delta_7v30g",
            ],
            "cv_metric_name": "poisson_deviance",
            "cv_aggregation_mode": "mean",
            "blend_mode": "xgb_only",
            "feature_selection_mode": "flat",
            "forced_delta_count": 8,
            "forced_delta_feature_count": 1,
            "feature_importance_rankings": [
                {"feature": f"feature_{index}", "importance": 0.1 - (index * 0.01)}
                for index in range(6)
            ],
        }

        weaker_payload = {
            **base_payload,
            "model_version": "20260328T020000Z_demo",
            "holdout_metrics": {
                "r2": 0.031,
                "rmse": 3.31,
                "mae": 2.58,
                "poisson_deviance": 2.51,
            },
            "cv_best_score": 2.34,
            "cv_best_rmse": None,
        }
        control_payload = {
            **base_payload,
            "model_version": "20260328T010000Z_demo",
            "holdout_metrics": {
                "r2": 0.0382,
                "rmse": 3.29,
                "mae": 2.57,
                "poisson_deviance": 2.50,
            },
            "cv_best_score": 2.31,
            "cv_best_rmse": None,
        }

        _write_metadata(weaker_path, weaker_payload)
        _write_metadata(control_path, control_payload)

        os.utime(weaker_path, (1_700_000_020, 1_700_000_020))
        os.utime(control_path, (1_700_000_010, 1_700_000_010))

        rows = generate_registry(
            models_root=models_root,
            output_dir=output_dir,
            target_model="full_game_away_runs_model",
            repo_root=workspace,
        )

        assert len(rows) == 2
        assert rows[0]["artifact_path"].endswith("20260328T020000Z_demo.metadata.json")

        registry_json = json.loads(
            (output_dir / "full_game_away_runs_registry.json").read_text(encoding="utf-8")
        )
        assert registry_json["row_count"] == 2
        top_5_features = registry_json["rows"][0]["top_5_features"]
        assert [item["feature"] for item in top_5_features] == [
            "feature_0",
            "feature_1",
            "feature_2",
            "feature_3",
            "feature_4",
        ]
        assert [item["importance"] for item in top_5_features] == pytest.approx(
            [0.1, 0.09, 0.08, 0.07, 0.06]
        )

        current_control = json.loads(
            (output_dir / "current_control.json").read_text(encoding="utf-8")
        )
        assert current_control["selected_artifact_path"].endswith(
            "20260328T010000Z_demo.metadata.json"
        )
        assert current_control["warning"] is None
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_extract_registry_row_counts_feature_families() -> None:
    workspace = _make_workspace("feature_counts")
    metadata_path = workspace / "models" / "exp" / "full_game_away_runs_model_demo.metadata.json"
    try:
        _write_metadata(
            metadata_path,
            {
                "model_name": "full_game_away_runs_model",
                "target_column": "final_away_score",
                "model_version": "demo",
                "data_version_hash": "hash",
                "holdout_season": 2025,
                "feature_columns": [
                    "weather_air_density_factor",
                    "weather_temp_factor",
                    "away_team_log5_30g",
                    "home_team_log5_30g",
                    "plate_umpire_zone_tightness",
                    "catcher_adjusted_framing_runs_30g",
                    "away_lineup_woba_delta_7v30g",
                    "home_starter_xera_delta_7v30s",
                ],
                "holdout_metrics": {
                    "r2": 0.01,
                    "rmse": 3.4,
                    "mae": 2.6,
                    "poisson_deviance": 2.6,
                },
                "forced_delta_feature_count": 2,
                "forced_delta_features": [
                    "away_lineup_woba_delta_7v30g",
                    "home_starter_xera_delta_7v30s",
                ],
            },
        )

        row = extract_registry_row(metadata_path, repo_root=workspace)

        assert row["weather_feature_count"] == 2
        assert row["log5_feature_count"] == 2
        assert row["plate_umpire_feature_count"] == 1
        assert row["framing_feature_count"] == 1
        assert row["delta_feature_count"] == 2
        assert row["selected_feature_count"] == 8
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_select_current_control_prefers_exact_stage_one_logic() -> None:
    rows = [
        {
            "artifact_path": "data/models/2026-away-flat-xgbonly-forceddelta8-a/run.metadata.json",
            "experiment_dir": "data/models/2026-away-flat-xgbonly-forceddelta8-a",
            "blend_mode": "xgb_only",
            "feature_selection_mode": "flat",
            "holdout_season": 2025,
            "holdout_r2": 0.0382,
            "holdout_rmse": 3.29,
            "holdout_poisson_deviance": 2.50,
            "data_version_hash": "hash-a",
        },
        {
            "artifact_path": "data/models/2026-away-flat-xgbonly-forceddelta8-b/run.metadata.json",
            "experiment_dir": "data/models/2026-away-flat-xgbonly-forceddelta8-b",
            "blend_mode": "xgb_only",
            "feature_selection_mode": "flat",
            "holdout_season": 2025,
            "holdout_r2": 0.0379,
            "holdout_rmse": 3.30,
            "holdout_poisson_deviance": 2.51,
            "data_version_hash": "hash-b",
        },
        {
            "artifact_path": "data/models/2026-away-flat-learned-forceddelta8-c/run.metadata.json",
            "experiment_dir": "data/models/2026-away-flat-learned-forceddelta8-c",
            "blend_mode": "learned",
            "feature_selection_mode": "flat",
            "holdout_season": 2025,
            "holdout_r2": 0.05,
            "holdout_rmse": 3.26,
            "holdout_poisson_deviance": 2.48,
            "data_version_hash": "hash-c",
        },
    ]

    control = select_current_control(rows)

    assert control["selected_artifact_path"].endswith("forceddelta8-a/run.metadata.json")
    assert control["selection_reason"] == "matched forceddelta8 + xgb_only + flat + holdout 2025 + highest holdout_r2"
    assert control["warning"] is None


def test_missing_optional_fields_do_not_crash_registry_generation() -> None:
    workspace = _make_workspace("missing_optional_fields")
    metadata_path = workspace / "models" / "exp" / "full_game_away_runs_model_demo.metadata.json"
    try:
        _write_metadata(
            metadata_path,
            {
                "model_name": "full_game_away_runs_model",
                "target_column": "final_away_score",
                "model_version": "demo",
                "data_version_hash": "hash",
                "holdout_season": 2025,
                "feature_columns": ["away_team_log5_30g"],
                "holdout_metrics": {"r2": 0.02, "rmse": 3.3, "mae": 2.5},
            },
        )

        row = extract_registry_row(metadata_path, repo_root=workspace)

        assert row["cv_metric_name"] is None
        assert row["cv_best_score"] is None
        assert row["cv_best_rmse"] is None
        assert row["feature_selection_mode"] is None
        assert row["blend_mode"] is None
        assert row["forced_delta_count"] is None
        assert row["forced_delta_feature_count"] is None
        assert row["holdout_poisson_deviance"] is None
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
