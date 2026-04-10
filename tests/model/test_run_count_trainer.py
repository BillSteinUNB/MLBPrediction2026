from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.clients.weather_client import fetch_game_weather
from src.model.artifact_runtime import collect_runtime_versions
from src.model.data_builder import (
    RUN_COUNT_REQUIRED_TEMPORAL_DELTA_COLUMNS,
    _json_bytes,
    _run_count_training_schema_metadata,
    _write_parquet_with_metadata,
)
import src.model.run_count_trainer as run_count_trainer_module
from src.model.run_count_trainer import (
    BlendedRunCountRegressor,
    DEFAULT_RUN_COUNT_BLEND_MODE,
    DEFAULT_RUN_COUNT_CV_AGGREGATION_MODE,
    DEFAULT_RUN_COUNT_LIGHTGBM_PARAM_MODE,
    DEFAULT_RUN_COUNT_SEARCH_ITERATIONS,
    DEFAULT_RUN_COUNT_MODEL_SPECS,
    _aggregate_run_count_cv_fold_scores,
    _build_estimator,
    _learn_run_count_blend_weights,
    _resolve_blended_model_params,
    _resolve_run_count_candidate_feature_columns,
    _resolve_run_count_blend_selection,
    _resolve_run_count_feature_bucket,
    _run_optuna_search,
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
                "home_lineup_woba_delta_7v30g": (home_strength - 0.30) * 0.18,
                "away_lineup_woba_delta_7v30g": (away_strength - 0.28) * 0.18,
                "home_starter_xera_delta_7v30s": (0.34 - home_strength) * 2.5,
                "away_starter_xera_delta_7v30s": (0.33 - away_strength) * 2.5,
                "f5_home_score": f5_home_score,
                "f5_away_score": f5_away_score,
                "final_home_score": final_home_score,
                "final_away_score": final_away_score,
            }
        )

    frame = pd.DataFrame(rows)
    for column_index, column_name in enumerate(RUN_COUNT_REQUIRED_TEMPORAL_DELTA_COLUMNS, start=1):
        frame[column_name] = (frame.index + column_index) * 0.01
    schema_metadata = _run_count_training_schema_metadata()
    frame.attrs.update(schema_metadata)
    frame.attrs["run_count_training_schema"] = schema_metadata
    return frame


def _write_training_parquet(
    dataframe: pd.DataFrame,
    output_path: Path,
    *,
    schema_metadata: dict[str, object] | None = None,
) -> None:
    metadata_payload = schema_metadata or _run_count_training_schema_metadata()
    _write_parquet_with_metadata(
        dataframe,
        output_path,
        parquet_metadata={
            b"mlbprediction2026.run_count_training_schema": _json_bytes(metadata_payload)
        },
    )


def _family_decision_by_name(
    decisions: list[dict[str, object]],
    family_name: str,
) -> dict[str, object]:
    return next(decision for decision in decisions if decision["family"] == family_name)


def test_create_time_series_split_preserves_temporal_order() -> None:
    splitter = create_time_series_split(row_count=18, requested_splits=5)

    assert splitter.n_splits == 5
    for train_indices, test_indices in splitter.split(range(18)):
        assert max(train_indices) < min(test_indices)


def test_train_run_count_models_trains_four_models_and_saves_metadata(tmp_path: Path) -> None:
    input_path = tmp_path / "training_data.parquet"
    _write_training_parquet(
        _training_frame().assign(
            home_lineup_woba_7g=lambda df: df["final_home_score"] * 0.09,
            home_lineup_woba_30g=lambda df: df["final_home_score"] * 0.07,
            away_starter_xfip_7s=lambda df: 5.5 - (df["final_home_score"] * 0.3),
            away_starter_xfip_30s=lambda df: 5.2 - (df["final_home_score"] * 0.2),
            away_starter_xfip_delta_7v30s=lambda df: df["away_starter_xfip_7s"]
            - df["away_starter_xfip_30s"],
        ),
        input_path,
    )

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
        assert artifact.holdout_metrics["poisson_deviance"] is not None
        assert artifact.holdout_metrics["r2"] is not None
        assert artifact.holdout_metrics["naive_mean_prediction"] is not None
        assert artifact.holdout_metrics["naive_mae"] is not None
        assert artifact.holdout_metrics["naive_rmse"] is not None
        assert artifact.holdout_metrics["naive_poisson_deviance"] is not None
        assert artifact.holdout_metrics["rmse_improvement_vs_naive_pct"] is not None
        assert artifact.holdout_metrics["poisson_deviance_improvement_vs_naive_pct"] is not None
        assert artifact.cv_metric_name == "poisson_deviance"
        assert artifact.cv_aggregation_mode == DEFAULT_RUN_COUNT_CV_AGGREGATION_MODE
        assert artifact.cv_best_score is not None
        assert artifact.cv_best_rmse is None
        assert len(artifact.cv_fold_scores) == 3
        assert artifact.cv_fold_weights == [1.0, 1.0, 1.0]
        assert artifact.lightgbm_param_mode == DEFAULT_RUN_COUNT_LIGHTGBM_PARAM_MODE
        assert artifact.optuna_parallel_workers == 2
        assert artifact.blend_mode == DEFAULT_RUN_COUNT_BLEND_MODE
        assert set(artifact.blend_candidate_scores) == {
            "xgb_only",
            "lgbm_only",
            "fixed",
            "learned",
        }
        assert artifact.blend_weights["xgboost"] >= 0.0
        assert artifact.blend_weights["lightgbm"] >= 0.0
        assert artifact.blend_weights["xgboost"] + artifact.blend_weights["lightgbm"] == pytest.approx(
            1.0
        )
        assert artifact.feature_importance_rankings

        metadata = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
        assert metadata["model_name"] == model_name
        assert metadata["target_column"] == artifact.target_column
        assert metadata["runtime_versions"] == collect_runtime_versions()
        assert metadata["search_backend"] == "optuna"
        assert metadata["model_family"] == "xgboost_lightgbm_blend"
        assert metadata["lightgbm_param_mode"] == DEFAULT_RUN_COUNT_LIGHTGBM_PARAM_MODE
        assert set(metadata["best_params_by_model"]) == {"xgboost", "lightgbm"}
        assert "lightgbm__num_leaves" in metadata["best_params"]
        assert "lightgbm__min_child_samples" in metadata["best_params"]
        assert "lightgbm__feature_fraction" in metadata["best_params"]
        assert metadata["best_params_by_model"]["lightgbm"]["num_leaves"] in (15, 31, 63)
        assert metadata["best_params_by_model"]["lightgbm"]["min_child_samples"] in (10, 20, 30)
        assert metadata["best_params_by_model"]["lightgbm"]["feature_fraction"] in (
            0.7,
            0.85,
            1.0,
        )
        assert metadata["blend_mode"] == artifact.blend_mode
        assert metadata["blend_weights"] == pytest.approx(artifact.blend_weights)
        assert metadata["learned_blend_weights"] == pytest.approx(artifact.learned_blend_weights)
        assert metadata["blend_candidate_scores"] == pytest.approx(artifact.blend_candidate_scores)
        assert metadata["blend_optimization_metric_name"] == "poisson_deviance"
        assert metadata["blend_oof_row_count"] > 0
        assert metadata["cv_metric_name"] == "poisson_deviance"
        assert metadata["cv_aggregation_mode"] == DEFAULT_RUN_COUNT_CV_AGGREGATION_MODE
        assert metadata["cv_best_score"] == pytest.approx(artifact.cv_best_score)
        assert metadata["cv_best_rmse"] is None
        assert metadata["cv_fold_scores"] == pytest.approx(artifact.cv_fold_scores)
        assert metadata["cv_fold_weights"] == pytest.approx(artifact.cv_fold_weights)
        assert metadata["cv_diagnostics"]["metric_name"] == "poisson_deviance"
        assert metadata["cv_diagnostics"]["aggregation_mode"] == artifact.cv_aggregation_mode
        assert metadata["cv_diagnostics"]["fold_scores"] == pytest.approx(artifact.cv_fold_scores)
        assert metadata["optuna_parallel_workers"] == artifact.optuna_parallel_workers
        assert metadata["optuna_trial_count"] == 2
        assert metadata["requested_n_estimators"] in (10, 20)
        assert metadata["final_n_estimators"] <= metadata["requested_n_estimators"]
        assert metadata["feature_importance_rankings"] == artifact.feature_importance_rankings
        assert metadata["feature_selection_mode"] == "grouped"
        assert metadata["feature_selection_bucket_counts"]
        assert metadata["feature_selection_bucket_targets"]
        assert metadata["selected_features_by_bucket"]
        assert metadata["omitted_top_features_by_bucket"]
        assert "delta" in metadata["feature_selection_bucket_counts"]
        assert "delta" in metadata["feature_selection_bucket_targets"]
        assert "delta" in metadata["selected_features_by_bucket"]
        assert "delta" in metadata["omitted_top_features_by_bucket"]
        assert metadata["feature_selection_bucket_counts"]["delta"] >= 1
        assert metadata["forced_delta_feature_count"] == 0
        assert metadata["forced_delta_features"] == []
        assert metadata["feature_selection_diagnostics"]["mode"] == "grouped"
        assert metadata["feature_selection_diagnostics"]["forced_delta_features"] == []
        assert metadata["feature_selection_diagnostics"]["family_decisions"]
        assert metadata["feature_selection_family_decisions"]
        assert artifact.feature_selection_family_decisions == metadata["feature_selection_family_decisions"]
        assert artifact.forced_delta_features == []


def test_train_run_count_models_rejects_missing_required_delta_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "stale_training_data.parquet"
    stale_frame = _training_frame().drop(columns=["home_lineup_woba_delta_7v30g"])
    _write_training_parquet(stale_frame, input_path)

    with pytest.raises(ValueError, match="missing required columns"):
        train_run_count_models(
            training_data=input_path,
            output_dir=tmp_path / "models",
            holdout_season=2025,
            search_space={"max_depth": [2], "n_estimators": [10], "learning_rate": [0.1]},
            time_series_splits=3,
            search_iterations=1,
            random_state=7,
        )


def test_resolve_blended_model_params_supports_independent_lightgbm_overrides() -> None:
    xgboost_params, lightgbm_params = _resolve_blended_model_params(
        {
            "max_depth": 4,
            "n_estimators": 200,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
            "min_child_weight": 2,
            "lightgbm__num_leaves": 63,
            "lightgbm__min_child_samples": 30,
            "lightgbm__feature_fraction": 0.7,
        },
        lightgbm_param_mode="independent",
    )

    assert "lightgbm__num_leaves" not in xgboost_params
    assert lightgbm_params["num_leaves"] == 63
    assert lightgbm_params["min_child_samples"] == 30
    assert lightgbm_params["feature_fraction"] == pytest.approx(0.7)
    assert lightgbm_params["colsample_bytree"] == pytest.approx(0.7)


def test_train_run_count_models_can_preserve_derived_lightgbm_mode(tmp_path: Path) -> None:
    input_path = tmp_path / "training_data_derived.parquet"
    _write_training_parquet(
        _training_frame().assign(
            home_lineup_woba_7g=lambda df: df["final_home_score"] * 0.09,
            home_lineup_woba_30g=lambda df: df["final_home_score"] * 0.07,
            away_starter_xfip_7s=lambda df: 5.5 - (df["final_home_score"] * 0.3),
            away_starter_xfip_30s=lambda df: 5.2 - (df["final_home_score"] * 0.2),
            away_starter_xfip_delta_7v30s=lambda df: df["away_starter_xfip_7s"]
            - df["away_starter_xfip_30s"],
        ),
        input_path,
    )

    result = train_run_count_models(
        training_data=input_path,
        output_dir=tmp_path / "models_derived",
        holdout_season=2025,
        search_space={"max_depth": [2], "n_estimators": [20], "learning_rate": [0.1]},
        time_series_splits=3,
        search_iterations=1,
        random_state=7,
        early_stopping_rounds=5,
        validation_fraction=0.25,
        lightgbm_param_mode="derived",
    )

    for artifact in result.models.values():
        metadata = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
        assert artifact.lightgbm_param_mode == "derived"
        assert metadata["lightgbm_param_mode"] == "derived"
        assert "lightgbm__num_leaves" not in metadata["best_params"]
        assert "num_leaves" not in metadata["best_params_by_model"]["xgboost"]
        assert "lightgbm__num_leaves" not in metadata["search_space"]
        assert metadata["best_params_by_model"]["lightgbm"]["min_child_samples"] >= 5


def test_train_run_count_models_supports_xgb_only_ablation(tmp_path: Path) -> None:
    input_path = tmp_path / "training_data_xgb_only.parquet"
    _write_training_parquet(
        _training_frame().assign(
            home_lineup_woba_7g=lambda df: df["final_home_score"] * 0.09,
            home_lineup_woba_30g=lambda df: df["final_home_score"] * 0.07,
            away_starter_xfip_7s=lambda df: 5.5 - (df["final_home_score"] * 0.3),
            away_starter_xfip_30s=lambda df: 5.2 - (df["final_home_score"] * 0.2),
            away_starter_xfip_delta_7v30s=lambda df: df["away_starter_xfip_7s"]
            - df["away_starter_xfip_30s"],
        ),
        input_path,
    )

    result = train_run_count_models(
        training_data=input_path,
        output_dir=tmp_path / "models_xgb_only",
        holdout_season=2025,
        search_space={"max_depth": [2], "n_estimators": [10], "learning_rate": [0.1]},
        time_series_splits=3,
        search_iterations=1,
        random_state=7,
        early_stopping_rounds=5,
        validation_fraction=0.25,
        blend_mode="xgb_only",
    )

    for artifact in result.models.values():
        assert artifact.blend_mode == "xgb_only"
        assert artifact.blend_weights == pytest.approx({"xgboost": 1.0, "lightgbm": 0.0})
        loaded_model = joblib.load(artifact.model_path)
        assert loaded_model.xgboost_weight == pytest.approx(1.0)
        assert loaded_model.lightgbm_weight == pytest.approx(0.0)


def test_train_run_count_models_rejects_schema_version_mismatch(tmp_path: Path) -> None:
    input_path = tmp_path / "wrong_schema_training_data.parquet"
    _write_training_parquet(
        _training_frame(),
        input_path,
        schema_metadata={
            **_run_count_training_schema_metadata(),
            "schema_version": 1,
        },
    )

    with pytest.raises(ValueError, match="schema mismatch"):
        train_run_count_models(
            training_data=input_path,
            output_dir=tmp_path / "models",
            holdout_season=2025,
            search_space={"max_depth": [2], "n_estimators": [10], "learning_rate": [0.1]},
            time_series_splits=3,
            search_iterations=1,
            random_state=7,
        )


def test_run_count_feature_bucket_routes_delta_features_to_dedicated_bucket() -> None:
    assert _resolve_run_count_feature_bucket("home_lineup_woba_delta_7v30g") == "delta"
    assert _resolve_run_count_feature_bucket("away_starter_xera_delta_7v30s") == "delta"
    assert _resolve_run_count_feature_bucket("home_lineup_woba_7g") == "short_form"
    assert _resolve_run_count_feature_bucket("away_starter_xera_30s") == "medium_form"
    assert _resolve_run_count_feature_bucket("park_runs_factor") == "context"


def test_run_count_feature_selection_tracks_delta_bucket_diagnostics() -> None:
    frame = _training_frame().assign(
        home_lineup_iso_delta_7v30g=lambda df: df["home_lineup_woba_delta_7v30g"] * 1.2,
        away_lineup_iso_delta_7v30g=lambda df: df["away_lineup_woba_delta_7v30g"] * 1.1,
        home_starter_xfip_delta_7v30s=lambda df: df["home_starter_xera_delta_7v30s"] * 0.8,
        away_starter_xfip_delta_7v30s=lambda df: df["away_starter_xera_delta_7v30s"] * 0.8,
    )
    candidate_resolution = _resolve_run_count_candidate_feature_columns(frame)

    selected = _select_run_count_feature_columns(
        frame,
        target_column="f5_home_score",
        candidate_feature_columns=candidate_resolution.candidate_columns,
        max_feature_count=80,
    )

    assert "delta" in selected.bucket_counts
    assert "delta" in selected.bucket_targets
    assert "delta" in selected.selected_features_by_bucket
    assert "delta" in selected.omitted_top_features_by_bucket
    assert selected.bucket_targets["delta"] > 0
    assert selected.bucket_counts["delta"] >= 1
    assert all(
        _resolve_run_count_feature_bucket(feature_name) == "delta"
        for feature_name in selected.selected_features_by_bucket["delta"]
    )


def test_run_count_feature_pruning_drops_redundant_windows_and_team_offense() -> None:
    frame = _training_frame().assign(
        home_team_wrc_plus_7g=lambda df: df["home_team_log5"] * 100,
        home_lineup_wrc_plus_7g=lambda df: (df["home_team_log5"] * 100) + 1,
        home_team_xwoba_7g=lambda df: df["home_team_log5"],
        home_lineup_xwoba_7g=lambda df: df["home_team_log5"] + 0.01,
        away_team_woba_14g=lambda df: df["away_team_log5"],
        away_lineup_woba_14g=lambda df: df["away_team_log5"] + 0.01,
        home_starter_xfip_60s=lambda df: 3.5 + (df.index * 0.01),
    )

    candidate_resolution = _resolve_run_count_candidate_feature_columns(frame)
    candidate_columns = candidate_resolution.candidate_columns

    assert "home_team_wrc_plus_7g" not in candidate_columns
    assert "home_lineup_wrc_plus_7g" in candidate_columns
    assert "home_team_xwoba_7g" not in candidate_columns
    assert "home_lineup_xwoba_7g" in candidate_columns
    assert "away_team_woba_14g" not in candidate_columns
    assert "away_lineup_woba_14g" not in candidate_columns
    assert "home_starter_xfip_60s" not in candidate_columns


def test_run_count_feature_pruning_keeps_team_offense_when_lineup_replacement_has_no_variance() -> None:
    frame = _training_frame().assign(
        home_team_wrc_plus_7g=lambda df: df["home_team_log5"] * 100,
        home_lineup_wrc_plus_7g=0.0,
    )

    candidate_resolution = _resolve_run_count_candidate_feature_columns(frame)
    candidate_columns = candidate_resolution.candidate_columns

    assert "home_team_wrc_plus_7g" in candidate_columns
    assert "home_lineup_wrc_plus_7g" not in candidate_columns


def test_run_count_target_specific_selection_caps_feature_count() -> None:
    frame = _training_frame()
    candidate_resolution = _resolve_run_count_candidate_feature_columns(frame)
    candidate_columns = candidate_resolution.candidate_columns

    selected = _select_run_count_feature_columns(
        frame,
        target_column="f5_home_score",
        candidate_feature_columns=candidate_columns,
        max_feature_count=80,
    )

    assert len(selected.feature_columns) <= 80
    assert selected.feature_columns == sorted(selected.feature_columns)


def test_run_count_feature_pruning_excludes_confirmed_near_constant_columns() -> None:
    frame = _training_frame().assign(
        abs_active=lambda df: 1.0 + ((df.index % 2) * 0.001),
        abs_walk_rate_delta=lambda df: (df.index % 2) * 0.001,
        abs_strikeout_rate_delta=lambda df: (df.index % 3) * 0.001,
        weather_precip_probability=lambda df: (df.index % 2) * 0.001,
        weather_data_missing=lambda df: (df.index % 2).astype(float),
    )

    candidate_resolution = _resolve_run_count_candidate_feature_columns(frame)

    assert "abs_active" not in candidate_resolution.candidate_columns
    assert "abs_walk_rate_delta" not in candidate_resolution.candidate_columns
    assert "abs_strikeout_rate_delta" not in candidate_resolution.candidate_columns
    assert "weather_precip_probability" not in candidate_resolution.candidate_columns
    assert "weather_data_missing" not in candidate_resolution.candidate_columns
    assert candidate_resolution.excluded_candidate_counts["near_constant"] == 5


def test_resolve_run_count_optuna_workers_defaults_to_two(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLB_OPTUNA_N_JOBS", raising=False)

    assert run_count_trainer_module._resolve_run_count_optuna_workers() == 2


def test_run_count_grouped_selection_picks_one_family_member_per_lineup_and_starter_group() -> None:
    frame = _training_frame().assign(
        home_lineup_woba_7g=lambda df: df["f5_home_score"] * 0.12,
        home_lineup_woba_30g=lambda df: (df["f5_home_score"] * 0.08) + (df.index * 0.02),
        home_lineup_woba_delta_7v30g=lambda df: (df["f5_home_score"] * 0.03) - (df.index * 0.04),
        home_starter_xfip_7s=lambda df: 6.0 - (df["f5_home_score"] * 0.45),
        home_starter_xfip_30s=lambda df: 5.7 - (df["f5_home_score"] * 0.22) + (df.index * 0.04),
        home_starter_xfip_delta_7v30s=lambda df: df["home_starter_xfip_7s"]
        - df["home_starter_xfip_30s"]
        + (df.index * 0.03),
        away_starter_xera_7s=lambda df: 3.0 + (df["f5_home_score"] * 0.08) - (df.index * 0.05),
        away_starter_xera_30s=lambda df: 2.8 + (df["f5_home_score"] * 0.33),
        away_starter_xera_delta_7v30s=lambda df: df["away_starter_xera_7s"]
        - df["away_starter_xera_30s"]
        + (df.index * 0.02),
    )

    selected = _select_run_count_feature_columns(
        frame,
        target_column="f5_home_score",
        candidate_feature_columns=[
            "home_lineup_woba_7g",
            "home_lineup_woba_30g",
            "home_lineup_woba_delta_7v30g",
            "home_starter_xfip_7s",
            "home_starter_xfip_30s",
            "home_starter_xfip_delta_7v30s",
            "away_starter_xera_7s",
            "away_starter_xera_30s",
            "away_starter_xera_delta_7v30s",
        ],
        max_feature_count=80,
    )

    assert "home_lineup_woba_7g" in selected.feature_columns
    assert "home_lineup_woba_30g" not in selected.feature_columns
    assert "home_lineup_woba_delta_7v30g" not in selected.feature_columns
    assert "home_starter_xfip_7s" in selected.feature_columns
    assert "home_starter_xfip_30s" not in selected.feature_columns
    assert "home_starter_xfip_delta_7v30s" not in selected.feature_columns
    assert "away_starter_xera_30s" in selected.feature_columns
    assert "away_starter_xera_7s" not in selected.feature_columns
    assert "away_starter_xera_delta_7v30s" not in selected.feature_columns

    lineup_decision = _family_decision_by_name(selected.family_decisions, "home_lineup_woba")
    assert lineup_decision["winner"] == "home_lineup_woba_7g"
    assert lineup_decision["winner_bucket"] == "short_form"
    assert lineup_decision["selected"] is True

    xfip_decision = _family_decision_by_name(selected.family_decisions, "home_starter_xfip")
    assert xfip_decision["winner"] == "home_starter_xfip_7s"
    assert xfip_decision["winner_bucket"] == "short_form"
    assert xfip_decision["selected"] is True

    xera_decision = _family_decision_by_name(selected.family_decisions, "away_starter_xera")
    assert xera_decision["winner"] == "away_starter_xera_30s"
    assert xera_decision["winner_bucket"] == "medium_form"
    assert xera_decision["selected"] is True


def test_run_count_flat_selection_can_force_delta_features() -> None:
    frame = _training_frame().assign(
        home_lineup_woba_7g=lambda df: df["f5_home_score"] * 0.12,
        home_lineup_woba_30g=lambda df: df["f5_home_score"] * 0.11,
        home_lineup_woba_delta_7v30g=lambda df: (df.index * 0.01) + 0.001,
        away_starter_xera_30s=lambda df: 2.8 + (df["f5_home_score"] * 0.33),
        away_starter_xera_delta_7v30s=lambda df: (df.index * 0.015) + 0.001,
    )

    selected = run_count_trainer_module._select_run_count_feature_columns_flat(
        frame,
        target_column="f5_home_score",
        candidate_feature_columns=[
            "home_lineup_woba_7g",
            "home_lineup_woba_30g",
            "home_lineup_woba_delta_7v30g",
            "away_starter_xera_30s",
            "away_starter_xera_delta_7v30s",
        ],
        max_feature_count=3,
        forced_delta_count=2,
    )

    assert len(selected.feature_columns) == 3
    assert set(selected.forced_delta_features) == {
        "home_lineup_woba_delta_7v30g",
        "away_starter_xera_delta_7v30s",
    }
    assert set(selected.forced_delta_features).issubset(set(selected.feature_columns))


def test_train_run_count_models_rejects_forced_delta_with_grouped_mode(tmp_path: Path) -> None:
    input_path = tmp_path / "training_data_forced_grouped.parquet"
    _write_training_parquet(_training_frame(), input_path)

    with pytest.raises(ValueError, match="forced_delta_feature_count is currently supported only with flat"):
        train_run_count_models(
            training_data=input_path,
            output_dir=tmp_path / "models_grouped_forced",
            holdout_season=2025,
            search_space={"max_depth": [2], "n_estimators": [10], "learning_rate": [0.1]},
            time_series_splits=3,
            search_iterations=1,
            random_state=7,
            feature_selection_mode="grouped",
            forced_delta_feature_count=4,
        )


def test_learn_run_count_blend_weights_finds_nonnegative_oof_optimum() -> None:
    learned_xgboost_weight, learned_lightgbm_weight = _learn_run_count_blend_weights(
        actual=[1.0, 2.0, 3.0, 4.0],
        xgboost_predictions=[1.0, 2.0, 3.0, 4.0],
        lightgbm_predictions=[4.0, 4.0, 4.0, 4.0],
    )

    assert learned_xgboost_weight >= 0.0
    assert learned_lightgbm_weight >= 0.0
    assert learned_xgboost_weight + learned_lightgbm_weight == pytest.approx(1.0)
    assert learned_xgboost_weight == pytest.approx(1.0)
    assert learned_lightgbm_weight == pytest.approx(0.0)


def test_resolve_run_count_blend_selection_reports_candidate_scores() -> None:
    selection = _resolve_run_count_blend_selection(
        actual=[1.0, 2.0, 3.0, 4.0],
        xgboost_predictions=[1.0, 2.0, 3.0, 4.0],
        lightgbm_predictions=[4.0, 4.0, 4.0, 4.0],
        blend_mode="xgb_only",
    )

    assert selection.blend_mode == "xgb_only"
    assert selection.xgboost_weight == pytest.approx(1.0)
    assert selection.lightgbm_weight == pytest.approx(0.0)
    assert selection.learned_xgboost_weight == pytest.approx(1.0)
    assert selection.learned_lightgbm_weight == pytest.approx(0.0)
    assert set(selection.candidate_scores) == {"xgb_only", "lgbm_only", "fixed", "learned"}
    assert selection.candidate_scores["xgb_only"] <= selection.candidate_scores["fixed"]
    assert selection.candidate_scores["xgb_only"] <= selection.candidate_scores["lgbm_only"]


def test_blended_run_count_regressor_supports_xgb_only_prediction_mode() -> None:
    class _StubModel:
        def __init__(self, predictions: list[float]) -> None:
            self._predictions = predictions

        def predict(self, _dataframe: pd.DataFrame) -> list[float]:
            return list(self._predictions)

    estimator = BlendedRunCountRegressor(
        xgboost_model=_StubModel([1.5, 2.5]),  # type: ignore[arg-type]
        lightgbm_model=_StubModel([99.0, 99.0]),  # type: ignore[arg-type]
        xgboost_weight=1.0,
        lightgbm_weight=0.0,
    )

    predictions = estimator.predict(pd.DataFrame({"feature": [1, 2]}))

    assert predictions.tolist() == pytest.approx([1.5, 2.5])


def test_run_count_estimator_uses_poisson_objective() -> None:
    estimator = _build_estimator(random_state=7)

    assert estimator.get_params()["objective"] == "count:poisson"
    assert estimator.get_params()["eval_metric"] == "poisson-nloglik"


def test_run_count_cv_aggregation_defaults_to_mean() -> None:
    aggregated = _aggregate_run_count_cv_fold_scores(
        [0.9, 0.6, 0.3],
        aggregation_mode="mean",
    )

    assert aggregated == pytest.approx(0.6)


def test_run_count_cv_aggregation_can_weight_recent_folds_more_heavily() -> None:
    aggregated = _aggregate_run_count_cv_fold_scores(
        [0.9, 0.6, 0.3],
        aggregation_mode="recent_weighted",
        fold_weights=[1.0, 2.0, 3.0],
    )

    assert aggregated == pytest.approx((0.9 + 1.2 + 0.9) / 6.0)


def test_train_run_count_models_persists_recent_weighted_cv_metadata(tmp_path: Path) -> None:
    input_path = tmp_path / "training_data_recent.parquet"
    _write_training_parquet(
        _training_frame().assign(
            home_lineup_woba_7g=lambda df: df["final_home_score"] * 0.09,
            home_lineup_woba_30g=lambda df: df["final_home_score"] * 0.07,
            away_starter_xfip_7s=lambda df: 5.5 - (df["final_home_score"] * 0.3),
            away_starter_xfip_30s=lambda df: 5.2 - (df["final_home_score"] * 0.2),
            away_starter_xfip_delta_7v30s=lambda df: df["away_starter_xfip_7s"]
            - df["away_starter_xfip_30s"],
        ),
        input_path,
    )

    result = train_run_count_models(
        training_data=input_path,
        output_dir=tmp_path / "models_recent",
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
        cv_aggregation_mode="recent_weighted",
    )

    for artifact in result.models.values():
        metadata = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
        assert artifact.cv_aggregation_mode == "recent_weighted"
        assert artifact.cv_fold_weights == [1.0, 2.0, 3.0]
        assert metadata["cv_aggregation_mode"] == "recent_weighted"
        assert metadata["cv_fold_weights"] == pytest.approx([1.0, 2.0, 3.0])
        assert len(metadata["cv_fold_scores"]) == 3
        assert metadata["cv_diagnostics"]["aggregation_mode"] == "recent_weighted"


def test_run_count_optuna_search_passes_parallel_worker_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_frame = _training_frame().loc[lambda df: df["season"] == 2024].reset_index(drop=True)
    captured: dict[str, object] = {}

    class _FakeBestTrial:
        value = 1.23
        number = 4
        params = {"max_depth": 2, "n_estimators": 20, "learning_rate": 0.1}
        user_attrs = {
            "cv_aggregation_mode": "mean",
            "cv_fold_scores": [1.3, 1.2, 1.19],
            "cv_fold_weights": [1.0, 1.0, 1.0],
        }

    class _FakeStudy:
        def __init__(self) -> None:
            self.trials: list[object] = []
            self.best_trial = _FakeBestTrial()

        def optimize(self, _objective, **kwargs) -> None:
            captured["n_jobs"] = kwargs.get("n_jobs")
            self.trials = [object(), object()]

    monkeypatch.setattr(run_count_trainer_module.optuna, "create_study", lambda **_kwargs: _FakeStudy())

    (
        best_params,
        best_score,
        best_trial_number,
        trial_count,
        _,
        _,
        resolved_splits,
        cv_diagnostics,
    ) = _run_optuna_search(
        train_frame=train_frame,
        candidate_feature_columns=["home_team_log5", "away_team_log5", "park_runs_factor"],
        target_column="final_away_score",
        model_name="full_game_away_runs_model",
        output_dir=tmp_path,
        holdout_season=2025,
        data_version_hash="synthetic-data-hash",
        search_space={"max_depth": [2], "n_estimators": [20], "learning_rate": [0.1]},
        time_series_splits=3,
        search_iterations=2,
        random_state=7,
        optuna_workers=3,
        cv_aggregation_mode="mean",
        lightgbm_param_mode="derived",
        feature_selection_mode="flat",
        forced_delta_feature_count=0,
        max_feature_count=80,
    )

    assert captured["n_jobs"] == 3
    assert best_params == {"max_depth": 2, "n_estimators": 20, "learning_rate": 0.1}
    assert best_score == pytest.approx(1.23)
    assert best_trial_number == 4
    assert trial_count == 2
    assert resolved_splits == 3
    assert cv_diagnostics.aggregation_mode == "mean"
    assert cv_diagnostics.fold_scores == pytest.approx([1.3, 1.2, 1.19])
    assert cv_diagnostics.fold_weights == pytest.approx([1.0, 1.0, 1.0])


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
    assert DEFAULT_RUN_COUNT_SEARCH_ITERATIONS == 500
