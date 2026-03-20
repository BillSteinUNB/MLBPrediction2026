from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import joblib
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit

from src.clients.weather_client import fetch_game_weather
import src.model.xgboost_trainer as xgboost_trainer_module
from src.model.xgboost_trainer import create_time_series_split, train_f5_models


def _training_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = datetime(2024, 4, 1, 19, 5, tzinfo=UTC)

    for index in range(18):
        scheduled_start = start + timedelta(days=index)
        season = 2024 if index < 12 else 2025
        signal = 1.0 if index % 2 == 0 else 0.0
        runline_signal = 1.0 if index % 3 == 0 else 0.0
        tied_after_5 = 1 if index == 14 else 0

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
                "home_team_log5": 0.35 + (0.45 * signal),
                "away_team_log5": 0.65 - (0.45 * signal),
                "home_team_pythagorean_wp_30g": 0.40 + (0.35 * signal),
                "away_team_pythagorean_wp_30g": 0.60 - (0.35 * signal),
                "park_runs_factor": 1.0 + (0.02 * (index % 3)),
                "weather_composite": 1.0,
                "bullpen_pitch_count_3d": 20.0 + index,
                "runline_signal": runline_signal,
                "f5_tied_after_5": tied_after_5,
                "f5_ml_result": 0 if tied_after_5 else int(signal),
                "f5_rl_result": int(runline_signal),
            }
        )

    return pd.DataFrame(rows)


def test_create_time_series_split_preserves_temporal_order() -> None:
    splitter = create_time_series_split(row_count=18, requested_splits=5)

    assert isinstance(splitter, TimeSeriesSplit)
    assert splitter.n_splits == 5

    for train_indices, test_indices in splitter.split(range(18)):
        assert max(train_indices) < min(test_indices)


def test_train_f5_models_trains_two_models_and_saves_versioned_joblib(tmp_path) -> None:
    input_path = tmp_path / "training_data.parquet"
    _training_frame().to_parquet(input_path, index=False)

    result = train_f5_models(
        training_data=input_path,
        output_dir=tmp_path / "models",
        holdout_season=2025,
        search_space={
            "max_depth": [3],
            "n_estimators": [20, 30],
            "learning_rate": [0.1],
        },
        time_series_splits=3,
        search_iterations=2,
        random_state=7,
    )

    assert set(result.models) == {"f5_ml_model", "f5_rl_model"}
    assert result.model_version

    for model_name, artifact in result.models.items():
        assert artifact.model_path.exists()
        assert artifact.model_path.name.startswith(f"{model_name}_")
        assert result.model_version in artifact.model_path.name
        loaded_model = joblib.load(artifact.model_path)
        assert loaded_model.__class__.__name__ == "XGBClassifier"
        assert artifact.holdout_metrics["accuracy"] >= 0.5


def test_train_f5_models_records_best_params_feature_importance_and_filters_ml_ties(tmp_path) -> None:
    input_path = tmp_path / "training_data.parquet"
    _training_frame().to_parquet(input_path, index=False)

    result = train_f5_models(
        training_data=input_path,
        output_dir=tmp_path / "models",
        holdout_season=2025,
        search_space={
            "max_depth": [3],
            "n_estimators": [20],
            "learning_rate": [0.1],
        },
        time_series_splits=3,
        search_iterations=1,
        random_state=11,
    )

    ml_artifact = result.models["f5_ml_model"]
    metadata = json.loads(ml_artifact.metadata_path.read_text(encoding="utf-8"))

    assert metadata["best_params"] == {
        "learning_rate": 0.1,
        "max_depth": 3,
        "n_estimators": 20,
    }
    assert metadata["train_row_count"] == 12
    assert metadata["holdout_row_count"] == 5
    assert metadata["feature_importance_rankings"]
    assert metadata["feature_importance_rankings"][0]["importance"] >= 0.0
    assert ml_artifact.feature_importance_rankings == metadata["feature_importance_rankings"]


def test_main_rebuilds_training_data_with_live_weather_fetcher(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _StopBuild(Exception):
        pass

    def _fake_build_training_dataset(**kwargs):
        captured.update(kwargs)
        raise _StopBuild

    monkeypatch.setattr(xgboost_trainer_module, "build_training_dataset", _fake_build_training_dataset)

    with pytest.raises(_StopBuild):
        xgboost_trainer_module.main(
            [
                "--training-data",
                str(tmp_path / "missing_training_data.parquet"),
                "--output-dir",
                str(tmp_path / "models"),
            ]
        )

    assert captured["weather_fetcher"] is fetch_game_weather
