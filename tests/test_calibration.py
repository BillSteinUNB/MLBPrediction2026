from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import pytest

from src.clients.weather_client import fetch_game_weather
import src.model.calibration as calibration_module
from src.model.calibration import (
    build_reliability_diagram,
    compute_expected_calibration_error,
    predict_calibrated,
    train_calibrated_models,
)


def _training_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = datetime(2024, 3, 28, 19, 5, tzinfo=UTC)

    for index in range(180):
        scheduled_start = start + timedelta(days=index)
        season = 2024 if index < 144 else 2025
        moneyline_signal = index % 2
        runline_signal = 1 if index % 4 in {0, 1} else 0
        home_log5 = 0.14 + (0.72 * moneyline_signal)
        home_pythagorean = 0.18 + (0.64 * moneyline_signal)
        park_runs_factor = 0.95 + (0.08 * ((index * 3) % 6) / 5)
        weather_composite = 0.98 + (0.04 * ((index * 2) % 5) / 4)

        ml_result = moneyline_signal
        rl_result = runline_signal

        rows.append(
            {
                "game_pk": 30_000 + index,
                "season": season,
                "game_date": scheduled_start.date().isoformat(),
                "scheduled_start": scheduled_start.isoformat(),
                "as_of_timestamp": (scheduled_start - timedelta(days=1)).isoformat(),
                "home_team": "NYY" if index % 2 == 0 else "BOS",
                "away_team": "BOS" if index % 2 == 0 else "NYY",
                "venue": "Yankee Stadium" if index % 2 == 0 else "Fenway Park",
                "game_type": "R",
                "build_timestamp": scheduled_start.isoformat(),
                "data_version_hash": "synthetic-calibration-data-hash",
                "home_team_log5_30g": home_log5,
                "away_team_log5_30g": 1.0 - home_log5,
                "home_team_f5_pythagorean_wp_30g": home_pythagorean,
                "away_team_f5_pythagorean_wp_30g": 1.0 - home_pythagorean,
                "park_runs_factor": park_runs_factor,
                "park_hr_factor": 0.97 + (0.05 * ((index + 1) % 5) / 4),
                "weather_composite": weather_composite,
                "bullpen_pitch_count_3d": 15.0 + (index % 11),
                "feature_noise_a": float((index * 13) % 7),
                "feature_noise_b": float((index * 17) % 9),
                "f5_tied_after_5": 0,
                "f5_ml_result": ml_result,
                "f5_rl_result": rl_result,
            }
        )

    return pd.DataFrame(rows)


def test_compute_expected_calibration_error_is_zero_for_perfect_constant_bin() -> None:
    y_true = np.array([0, 1, 0, 1], dtype=int)
    probabilities = np.array([0.5, 0.5, 0.5, 0.5], dtype=float)

    reliability = build_reliability_diagram(y_true, probabilities, bin_count=10)
    ece = compute_expected_calibration_error(y_true, probabilities, bin_count=10)

    occupied_bins = [entry for entry in reliability if entry["count"]]

    assert len(reliability) == 10
    assert len(occupied_bins) == 1
    assert occupied_bins[0]["count"] == 4
    assert occupied_bins[0]["mean_predicted_probability"] == 0.5
    assert occupied_bins[0]["empirical_positive_rate"] == 0.5
    assert ece == 0.0


def test_train_calibrated_models_trains_and_saves_versioned_bundle(tmp_path) -> None:
    input_path = tmp_path / "training_data.parquet"
    frame = _training_frame()
    frame.to_parquet(input_path, index=False)

    result = train_calibrated_models(
        training_data=input_path,
        output_dir=tmp_path / "models",
        holdout_season=2025,
        calibration_fraction=0.10,
        base_search_space={
            "max_depth": [1],
            "n_estimators": [12],
            "learning_rate": [0.15],
        },
        time_series_splits=4,
        search_iterations=1,
        random_state=23,
    )

    assert set(result.models) == {"f5_ml_calibrated_model", "f5_rl_calibrated_model"}
    assert result.model_version

    holdout_frame = frame.loc[frame["season"] == 2025].reset_index(drop=True)

    for artifact in result.models.values():
        assert artifact.model_path.exists()
        assert artifact.stacking_model_path.exists()
        assert artifact.model_path.name.startswith(f"{artifact.model_name}_")
        assert result.model_version in artifact.model_path.name
        assert artifact.calibration_row_count == 15

        loaded_model = joblib.load(artifact.model_path)
        probabilities = loaded_model.predict_calibrated(holdout_frame)

        assert loaded_model.__class__.__name__ == "CalibratedStackingModel"
        assert loaded_model.calibration_method == "platt"
        assert probabilities.shape == (len(holdout_frame),)
        assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()


def test_train_calibrated_models_reports_quality_gates_and_reliability_data(tmp_path) -> None:
    input_path = tmp_path / "training_data.parquet"
    _training_frame().to_parquet(input_path, index=False)

    result = train_calibrated_models(
        training_data=input_path,
        output_dir=tmp_path / "models",
        holdout_season=2025,
        calibration_fraction=0.10,
        calibration_method="isotonic",
        base_search_space={
            "max_depth": [1],
            "n_estimators": [12],
            "learning_rate": [0.15],
        },
        time_series_splits=4,
        search_iterations=1,
        random_state=29,
    )

    ml_artifact = result.models["f5_ml_calibrated_model"]
    metadata = json.loads(ml_artifact.metadata_path.read_text(encoding="utf-8"))

    assert metadata["calibration_method"] == "isotonic"
    assert metadata["calibration_fraction"] == 0.10
    assert metadata["calibration_row_count"] == 15
    assert metadata["holdout_metrics"]["calibrated_brier"] < 0.25
    assert metadata["holdout_metrics"]["calibrated_ece"] < 0.05
    assert metadata["holdout_metrics"]["max_reliability_gap"] <= 0.05
    assert metadata["holdout_metrics"]["quality_gates"] == {
        "brier_lt_0_25": True,
        "ece_lt_0_05": True,
        "reliability_gap_le_0_05": True,
    }
    assert len(metadata["holdout_metrics"]["reliability_diagram"]) == 10


def test_predict_calibrated_accepts_model_path_and_loaded_bundle(tmp_path) -> None:
    input_path = tmp_path / "training_data.parquet"
    frame = _training_frame()
    frame.to_parquet(input_path, index=False)

    result = train_calibrated_models(
        training_data=input_path,
        output_dir=tmp_path / "models",
        holdout_season=2025,
        calibration_fraction=0.10,
        base_search_space={
            "max_depth": [1],
            "n_estimators": [12],
            "learning_rate": [0.15],
        },
        time_series_splits=4,
        search_iterations=1,
        random_state=31,
    )

    artifact = result.models["f5_rl_calibrated_model"]
    loaded_model = joblib.load(artifact.model_path)
    holdout_frame = frame.loc[frame["season"] == 2025].reset_index(drop=True)

    from_path = predict_calibrated(artifact.model_path, holdout_frame)
    from_model = predict_calibrated(loaded_model, holdout_frame)
    from_method = loaded_model.predict_calibrated(holdout_frame)

    np.testing.assert_allclose(from_path, from_model)
    np.testing.assert_allclose(from_model, from_method)


def test_train_calibrated_models_supports_identity_method(tmp_path) -> None:
    input_path = tmp_path / "training_data.parquet"
    frame = _training_frame()
    frame.to_parquet(input_path, index=False)

    result = train_calibrated_models(
        training_data=input_path,
        output_dir=tmp_path / "models",
        holdout_season=2025,
        calibration_fraction=0.10,
        calibration_method="identity",
        base_search_space={
            "max_depth": [1],
            "n_estimators": [12],
            "learning_rate": [0.15],
        },
        time_series_splits=4,
        search_iterations=1,
        random_state=41,
    )

    artifact = result.models["f5_ml_calibrated_model"]
    holdout_frame = frame.loc[frame["season"] == 2025].reset_index(drop=True)
    loaded_model = joblib.load(artifact.model_path)
    stacking_model = joblib.load(artifact.stacking_model_path)

    calibrated = loaded_model.predict_calibrated(holdout_frame)
    stacked = stacking_model.predict_proba(holdout_frame)[:, 1]

    assert loaded_model.calibration_method == "identity"
    np.testing.assert_allclose(calibrated, stacked)


def test_cli_training_produces_loadable_calibrated_bundle(tmp_path) -> None:
    input_path = tmp_path / "training_data.parquet"
    frame = _training_frame()
    frame.to_parquet(input_path, index=False)

    output_dir = tmp_path / "models"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.model.calibration",
            "--training-data",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--holdout-season",
            "2025",
            "--calibration-fraction",
            "0.1",
            "--search-iterations",
            "1",
            "--time-series-splits",
            "3",
            "--random-state",
            "37",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    artifact_path = next(output_dir.glob("f5_ml_calibrated_model_*.joblib"))
    loaded_model = joblib.load(artifact_path)
    holdout_frame = frame.loc[frame["season"] == 2025].reset_index(drop=True)
    probabilities = loaded_model.predict_calibrated(holdout_frame)

    assert probabilities.shape == (len(holdout_frame),)
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()


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

    monkeypatch.setattr(calibration_module, "build_training_dataset", _fake_build_training_dataset)

    with pytest.raises(_StopBuild):
        calibration_module.main(
            [
                "--training-data",
                str(tmp_path / "missing_training_data.parquet"),
                "--output-dir",
                str(tmp_path / "models"),
            ]
        )

    assert captured["weather_fetcher"] is fetch_game_weather
