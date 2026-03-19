from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import joblib
import pandas as pd

from src.model.stacking import train_stacking_models


def _training_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = datetime(2024, 3, 28, 19, 5, tzinfo=UTC)

    for index in range(72):
        scheduled_start = start + timedelta(days=index)
        season = 2024 if index < 48 else 2025
        home_log5 = 0.28 + (0.38 * ((index % 12) / 11))
        home_pythagorean = 0.30 + (0.34 * (((index * 5) % 12) / 11))
        park_runs_factor = 0.94 + (0.03 * (index % 5))
        latent_score = (
            4.0 * (home_log5 - 0.5)
            + 3.0 * (home_pythagorean - 0.5)
            + 1.5 * (park_runs_factor - 1.0)
        )
        ml_result = int(latent_score > -0.08)
        rl_result = int(latent_score > 0.10)

        rows.append(
            {
                "game_pk": 20_000 + index,
                "season": season,
                "game_date": scheduled_start.date().isoformat(),
                "scheduled_start": scheduled_start.isoformat(),
                "as_of_timestamp": (scheduled_start - timedelta(days=1)).isoformat(),
                "home_team": "NYY" if index % 2 == 0 else "BOS",
                "away_team": "BOS" if index % 2 == 0 else "NYY",
                "venue": "Yankee Stadium" if index % 2 == 0 else "Fenway Park",
                "game_type": "R",
                "build_timestamp": scheduled_start.isoformat(),
                "data_version_hash": "synthetic-stacking-data-hash",
                "home_team_log5_30g": home_log5,
                "away_team_log5_30g": 1.0 - home_log5,
                "home_team_f5_pythagorean_wp_30g": home_pythagorean,
                "away_team_f5_pythagorean_wp_30g": 1.0 - home_pythagorean,
                "park_runs_factor": park_runs_factor,
                "park_hr_factor": 0.98 + (0.04 * ((index + 2) % 4)),
                "weather_composite": 1.0 + (0.01 * (index % 3)),
                "bullpen_pitch_count_3d": 12.0 + index,
                "feature_noise_a": float((index * 13) % 7),
                "feature_noise_b": float((index * 17) % 9),
                "f5_tied_after_5": 0,
                "f5_ml_result": ml_result,
                "f5_rl_result": rl_result,
            }
        )

    return pd.DataFrame(rows)


def test_train_stacking_models_trains_two_meta_learners_and_saves_versioned_joblib(
    tmp_path,
) -> None:
    input_path = tmp_path / "training_data.parquet"
    frame = _training_frame()
    frame.to_parquet(input_path, index=False)

    result = train_stacking_models(
        training_data=input_path,
        output_dir=tmp_path / "models",
        holdout_season=2025,
        base_search_space={
            "max_depth": [1],
            "n_estimators": [1],
            "learning_rate": [0.01],
        },
        time_series_splits=4,
        search_iterations=1,
        random_state=11,
    )

    assert set(result.models) == {"f5_ml_stacking_model", "f5_rl_stacking_model"}
    assert result.model_version

    holdout_frame = frame.loc[frame["season"] == 2025].reset_index(drop=True)

    for artifact in result.models.values():
        assert artifact.model_path.exists()
        assert artifact.base_model_path.exists()
        assert artifact.model_path.name.startswith(f"{artifact.model_name}_")
        assert result.model_version in artifact.model_path.name
        assert artifact.meta_feature_columns[0] == "xgb_probability"
        assert set(artifact.raw_meta_feature_columns) == {
            "home_team_f5_pythagorean_wp_30g",
            "home_team_log5_30g",
            "park_runs_factor",
        }

        loaded_model = joblib.load(artifact.model_path)
        probabilities = loaded_model.predict_proba(holdout_frame)[:, 1]

        assert loaded_model.__class__.__name__ == "StackingEnsembleModel"
        assert probabilities.shape == (len(holdout_frame),)
        assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()


def test_train_stacking_models_uses_cross_val_predict_oof_features_and_improves_brier(
    tmp_path,
) -> None:
    input_path = tmp_path / "training_data.parquet"
    _training_frame().to_parquet(input_path, index=False)

    result = train_stacking_models(
        training_data=input_path,
        output_dir=tmp_path / "models",
        holdout_season=2025,
        base_search_space={
            "max_depth": [1],
            "n_estimators": [1],
            "learning_rate": [0.01],
        },
        time_series_splits=4,
        search_iterations=1,
        random_state=17,
    )

    ml_artifact = result.models["f5_ml_stacking_model"]
    metadata = json.loads(ml_artifact.metadata_path.read_text(encoding="utf-8"))

    assert metadata["oof_prediction_strategy"] == "cross_val_predict"
    assert metadata["oof_row_count"] > 0
    assert metadata["holdout_metrics"]["stacked_brier"] <= metadata["holdout_metrics"]["base_brier"]
    assert metadata["holdout_metrics"]["stacked_brier"] < 0.25
