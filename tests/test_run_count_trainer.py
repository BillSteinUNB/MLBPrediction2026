from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.clients.weather_client import fetch_game_weather
from src.model.artifact_runtime import collect_runtime_versions
import src.model.run_count_trainer as run_count_trainer_module
from src.model.run_count_trainer import (
    BlendedRunCountRegressor,
    DEFAULT_RUN_COUNT_SEARCH_ITERATIONS,
    DEFAULT_RUN_COUNT_MODEL_SPECS,
    _build_estimator,
    _resolve_run_count_candidate_feature_columns,
    _select_run_count_feature_columns,
    create_time_series_split,
    train_run_count_models,
)


def _training_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = datetime(2024, 4, 1, 19, 5, tzinfo=UTC)

    for index in range(18):
        scheduled_start = start + timedelta(days=index)
        season = 2024 if index < 12 else 2025
        home_strength = 0.35 + (0.04 * (index % 6))
        away_strength = 0.30 + (0.03 * ((index + 2) % 5))
        park_runs_factor = 0.96 + (0.01 * (index % 4))
        bullpen_pitch_count_3d = 18.0 + index
        weather_composite = 0.95 + (0.01 * (index % 3))

        f5_home_score = 2 + int(home_strength * 4) + (index % 2)
        f5_away_score = 1 + int(away_strength * 4) + ((index + 1) % 2)
        final_home_score = f5_home_score + (index % 3)
        final_away_score = f5_away_score + ((index + 2) % 3)

        rows.append(
            {
                "game_pk": 10_000 + index,
                "season": season,
                "game_date": scheduled_start.date().isoformat(),
                "scheduled_start": scheduled_start.isoformat(),
                "as_of_timestamp": (scheduled_start - timedelta(days=1)).isoformat(),
                "home_team": "NYY" if index % 2 == 0 else "BOS",
                "away_team": "BOS" if index % 2 == 0 else "NYY",
                "venue": "Yankee Stadium" if index % 2 == 0 else "Fenway Park",
                "game_type": "R",
                "build_timestamp": scheduled_start.isoformat(),
                "data_version_hash": "synthetic-data-hash",
                "home_team_log5": home_strength,
                "away_team_log5": away_strength,
                "home_team_pythagorean_wp_30g": 0.40 + (0.02 * (index % 5)),
                "away_team_pythagorean_wp_30g": 0.38 + (0.018 * ((index + 1) % 5)),
                "park_runs_factor": park_runs_factor,
                "weather_composite": weather_composite,
                "bullpen_pitch_count_3d": bullpen_pitch_count_3d,
                "bullpen_pitch_count_7d": 34.0 + (index * 0.5),
                "defense_metric": 0.50 + (0.01 * (index % 4)),
                "offense_metric": 0.48 + (0.015 * ((index + 2) % 4)),
                "f5_home_score": f5_home_score,
                "f5_away_score": f5_away_score,
                "final_home_score": final_home_score,
                "final_away_score": final_away_score,
            }
        )

    return pd.DataFrame(rows)


def test_create_time_series_split_preserves_temporal_order() -> None:
    splitter = create_time_series_split(row_count=18, requested_splits=5)

    assert splitter.n_splits == 5
    for train_indices, test_indices in splitter.split(range(18)):
        assert max(train_indices) < min(test_indices)


def test_train_run_count_models_trains_four_models_and_saves_metadata(tmp_path: Path) -> None:
    input_path = tmp_path / "training_data.parquet"
    _training_frame().to_parquet(input_path, index=False)

    result = train_run_count_models(
        training_data=input_path,
        output_dir=tmp_path / "models",
        holdout_season=2025,
        search_space={
            "max_depth": [2],
            "n_estimators": [10, 20],
            "learning_rate": [0.1],
        },
        time_series_splits=3,
        search_iterations=2,
        random_state=7,
        early_stopping_rounds=5,
        validation_fraction=0.25,
    )

    assert set(result.models) == {spec["model_name"] for spec in DEFAULT_RUN_COUNT_MODEL_SPECS}
    assert result.summary_path.exists()
    assert result.model_version

    for model_name, artifact in result.models.items():
        assert artifact.model_path.exists()
        assert artifact.metadata_path.exists()
        assert result.model_version in artifact.model_path.name
        loaded_model = joblib.load(artifact.model_path)
        assert isinstance(loaded_model, BlendedRunCountRegressor)
        assert artifact.train_row_count == 12
        assert artifact.holdout_row_count == 6
        assert artifact.holdout_metrics["mae"] is not None
        assert artifact.holdout_metrics["rmse"] is not None
        assert artifact.holdout_metrics["r2"] is not None
        assert artifact.holdout_metrics["naive_mean_prediction"] is not None
        assert artifact.holdout_metrics["naive_mae"] is not None
        assert artifact.holdout_metrics["naive_rmse"] is not None
        assert artifact.holdout_metrics["rmse_improvement_vs_naive_pct"] is not None
        assert artifact.feature_importance_rankings

        metadata = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
        assert metadata["model_name"] == model_name
        assert metadata["target_column"] == artifact.target_column
        assert metadata["runtime_versions"] == collect_runtime_versions()
        assert metadata["search_backend"] == "optuna"
        assert metadata["model_family"] == "xgboost_lightgbm_blend"
        assert metadata["blend_weights"]["xgboost"] == pytest.approx(0.6)
        assert metadata["blend_weights"]["lightgbm"] == pytest.approx(0.4)
        assert metadata["optuna_trial_count"] == 2
        assert metadata["requested_n_estimators"] == 20
        assert metadata["final_n_estimators"] <= metadata["requested_n_estimators"]
        assert metadata["feature_importance_rankings"] == artifact.feature_importance_rankings


def test_run_count_feature_pruning_drops_redundant_windows_and_team_offense() -> None:
    frame = _training_frame().assign(
        home_team_wrc_plus_7g=lambda df: df["home_team_log5"] * 100,
        home_lineup_wrc_plus_7g=lambda df: (df["home_team_log5"] * 100) + 1,
        home_team_xwoba_7g=lambda df: df["home_team_log5"] * 0.01,
        home_lineup_xwoba_7g=lambda df: (df["home_team_log5"] * 0.01) + 0.001,
        away_team_woba_14g=lambda df: df["away_team_log5"],
        away_lineup_woba_14g=lambda df: df["away_team_log5"] + 0.01,
        home_starter_xfip_60s=lambda df: 3.5 + (df.index * 0.01),
    )

    candidate_columns = _resolve_run_count_candidate_feature_columns(frame)

    assert "home_team_wrc_plus_7g" not in candidate_columns
    assert "home_lineup_wrc_plus_7g" in candidate_columns
    assert "home_team_xwoba_7g" not in candidate_columns
    assert "home_lineup_xwoba_7g" in candidate_columns
    assert "away_team_woba_14g" not in candidate_columns
    assert "away_lineup_woba_14g" not in candidate_columns
    assert "home_starter_xfip_60s" not in candidate_columns


def test_run_count_target_specific_selection_caps_feature_count() -> None:
    frame = _training_frame()
    candidate_columns = _resolve_run_count_candidate_feature_columns(frame)

    selected = _select_run_count_feature_columns(
        frame,
        target_column="f5_home_score",
        candidate_feature_columns=candidate_columns,
        max_feature_count=3,
    )

    assert len(selected) == 3
    assert selected == sorted(selected)


def test_run_count_estimator_uses_poisson_objective() -> None:
    estimator = _build_estimator(random_state=7)

    assert estimator.get_params()["objective"] == "count:poisson"
    assert estimator.get_params()["eval_metric"] == "rmse"


def test_main_rebuilds_training_data_with_live_weather_fetcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _StopBuild(Exception):
        pass

    def _fake_build_training_dataset(**kwargs):
        captured.update(kwargs)
        raise _StopBuild

    monkeypatch.setattr(run_count_trainer_module, "build_training_dataset", _fake_build_training_dataset)

    with pytest.raises(_StopBuild):
        run_count_trainer_module.main(
            [
                "--training-data",
                str(tmp_path / "missing_training_data.parquet"),
                "--output-dir",
                str(tmp_path / "models"),
            ]
        )

    assert captured["weather_fetcher"] is fetch_game_weather
    assert DEFAULT_RUN_COUNT_SEARCH_ITERATIONS == 150
