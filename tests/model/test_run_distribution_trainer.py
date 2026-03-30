from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

from src.model.data_builder import (
    RUN_COUNT_REQUIRED_TEMPORAL_DELTA_COLUMNS,
    _json_bytes,
    _run_count_training_schema_metadata,
    _write_parquet_with_metadata,
)
from src.model.run_distribution_trainer import (
    DEFAULT_DISTRIBUTION_FAMILY_NAME,
    RunDistributionModel,
    build_control_comparison,
    build_variable_nb_support,
    negative_binomial_pmf_matrix_by_row,
    summarize_array,
    train_run_distribution_model,
    zero_adjusted_negative_binomial_pmf_matrix_by_row,
)
from src.model.run_research_features import augment_run_research_features


def _training_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = datetime(2024, 4, 1, 19, 5, tzinfo=UTC)
    for index in range(24):
        scheduled_start = start + timedelta(days=index)
        season = 2024 if index < 16 else 2025
        home_strength = 0.34 + (0.03 * (index % 5))
        away_strength = 0.29 + (0.04 * ((index + 1) % 5))
        park_runs_factor = 0.95 + (0.01 * (index % 4))
        weather_air_density_factor = 0.97 + (0.015 * (index % 3))
        shutout_risk = 1 if index in {2, 7, 10, 18, 20} else 0
        final_away_score = 0 if shutout_risk else 2 + (index % 5)
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
                "data_version_hash": "synthetic-stage3-hash",
                "home_team_log5_30g": home_strength,
                "away_team_log5_30g": away_strength,
                "home_starter_k_pct_30s": 0.20 + (0.01 * (index % 4)),
                "home_starter_k_pct_7s": 0.19 + (0.01 * (index % 3)),
                "away_team_woba_7g": away_strength + (0.01 * (index % 3)),
                "away_lineup_woba_7g": away_strength + (0.015 * (index % 4)),
                "away_lineup_woba_30g": away_strength + (0.005 * (index % 4)),
                "weather_air_density_factor": weather_air_density_factor,
                "park_runs_factor": park_runs_factor,
                "home_team_bullpen_xfip": 4.0 + (0.1 * (index % 3)),
                "away_team_bullpen_xfip": 3.8 + (0.1 * ((index + 1) % 3)),
                "home_lineup_woba_delta_7v30g": (0.02 * (index % 4)) - 0.02,
                "away_lineup_woba_delta_7v30g": (0.025 * (index % 4)) - 0.03,
                "home_starter_xera_delta_7v30s": (0.03 * (index % 4)) - 0.02,
                "away_starter_xera_delta_7v30s": (0.02 * (index % 5)) - 0.03,
                "f5_home_score": 2 + (index % 3),
                "f5_away_score": 1 + (index % 2),
                "final_home_score": 3 + (index % 4),
                "final_away_score": final_away_score,
            }
        )
    frame = pd.DataFrame(rows)
    for column_index, column_name in enumerate(RUN_COUNT_REQUIRED_TEMPORAL_DELTA_COLUMNS, start=1):
        if column_name not in frame.columns:
            frame[column_name] = (frame.index + column_index) * 0.01
    schema_metadata = _run_count_training_schema_metadata()
    frame.attrs.update(schema_metadata)
    frame.attrs["run_count_training_schema"] = schema_metadata
    return frame


def _write_training_parquet(dataframe: pd.DataFrame, output_path: Path) -> None:
    metadata_payload = _run_count_training_schema_metadata()
    _write_parquet_with_metadata(
        dataframe,
        output_path,
        parquet_metadata={
            b"mlbprediction2026.run_count_training_schema": _json_bytes(metadata_payload)
        },
    )


def _write_mean_artifact(
    *,
    frame: pd.DataFrame,
    output_dir: Path,
    holdout_season: int,
) -> Path:
    model = PoissonRegressor(alpha=0.0, max_iter=500)
    feature_columns = [
        "home_team_log5_30g",
        "away_team_log5_30g",
        "home_starter_k_pct_30s",
        "park_runs_factor",
        "weather_air_density_factor",
        "away_lineup_woba_30g",
    ]
    train_frame = frame.loc[frame["season"] < holdout_season].copy()
    model.fit(train_frame.loc[:, feature_columns], train_frame["final_away_score"])

    model_version = "20260328T000000Z_synthetic"
    model_path = output_dir / f"full_game_away_runs_model_{model_version}.joblib"
    metadata_path = output_dir / f"full_game_away_runs_model_{model_version}.metadata.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metadata_payload = {
        "model_name": "full_game_away_runs_model",
        "target_column": "final_away_score",
        "model_version": model_version,
        "data_version_hash": "synthetic-stage3-hash",
        "holdout_season": holdout_season,
        "feature_columns": feature_columns,
        "best_params": {
            "max_depth": 2,
            "n_estimators": 24,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
        "holdout_metrics": {
            "rmse": 2.9,
            "predicted_mean": 3.0,
            "actual_mean": 3.0,
        },
        "model_family": "xgboost_lightgbm_blend",
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    return metadata_path


def test_variable_negative_binomial_helpers_normalize_per_row() -> None:
    support = build_variable_nb_support(
        actual_counts=[0, 4, 7],
        mean_predictions=[1.2, 3.5, 5.0],
        dispersion_size=[2.0, 1.4, 3.0],
        tail_probability=1e-8,
    )
    nb = negative_binomial_pmf_matrix_by_row(
        [1.2, 3.5, 5.0],
        support,
        dispersion_size=[2.0, 1.4, 3.0],
    )
    zanb = zero_adjusted_negative_binomial_pmf_matrix_by_row(
        [1.2, 3.5, 5.0],
        support,
        dispersion_size=[2.0, 1.4, 3.0],
        adjusted_zero_probability=[0.35, 0.10, 0.05],
    )

    assert nb.shape == zanb.shape
    assert np.allclose(nb.sum(axis=1), 1.0)
    assert np.allclose(zanb.sum(axis=1), 1.0)
    assert abs(zanb[0, 0] - 0.35) < 1e-9


def test_train_run_distribution_model_writes_stage3_artifact_and_reports(tmp_path: Path) -> None:
    training_path = tmp_path / "training_data.parquet"
    frame = _training_frame()
    _write_training_parquet(frame, training_path)
    mean_metadata_path = _write_mean_artifact(
        frame=frame,
        output_dir=tmp_path / "mean_head",
        holdout_season=2025,
    )

    artifact = train_run_distribution_model(
        training_data=training_path,
        output_dir=tmp_path / "dist_models",
        mean_artifact_metadata_path=mean_metadata_path,
        holdout_season=2025,
        feature_selection_mode="flat",
        forced_delta_feature_count=0,
        time_series_splits=3,
        xgb_n_jobs=1,
        head_param_overrides={
            "max_depth": 2,
            "n_estimators": 20,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "min_child_weight": 1,
            "reg_lambda": 1.0,
        },
        distribution_report_dir=tmp_path / "reports",
    )

    assert artifact.model_path.exists()
    assert artifact.metadata_path.exists()
    assert artifact.summary_path.exists()
    assert artifact.distribution_report_json_path.exists()
    assert artifact.distribution_report_csv_path.exists()
    assert artifact.control_comparison_json_path.exists()
    assert (tmp_path / "reports" / f"{artifact.model_version}.holdout_predictions.csv").exists()
    assert artifact.model_version
    assert artifact.distribution_metrics["mean_crps"] >= 0.0

    loaded_model = joblib.load(artifact.model_path)
    assert isinstance(loaded_model, RunDistributionModel)
    holdout_frame = augment_run_research_features(frame.loc[frame["season"] == 2025].copy()).dataframe
    components = loaded_model.predict_components(holdout_frame)
    assert set(components) == {
        "mu",
        "control_mu",
        "mu_delta",
        "overdispersion_alpha",
        "dispersion",
        "baseline_zero_probability",
        "adjusted_zero_probability",
        "p_zero_extra",
    }
    assert len(components["mu"]) == len(holdout_frame)
    assert np.all(components["mu"] > 0.0)
    assert np.all(components["dispersion"] > 0.0)

    metadata = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
    assert metadata["distribution_lane_stage"] == 3
    assert metadata["distribution_family"] == DEFAULT_DISTRIBUTION_FAMILY_NAME
    assert metadata["fitted_distribution_family_name"] == DEFAULT_DISTRIBUTION_FAMILY_NAME
    assert metadata["count_training_positive_only"] is True
    assert metadata["count_training_row_count"] == int(
        (
            frame.loc[frame["season"] < 2025, "final_away_score"]
            .astype(int)
            .gt(0)
            .sum()
        )
    )
    assert metadata["zero_training_row_count"] == int((frame["season"] < 2025).sum())
    assert metadata["research_feature_metadata"]["feature_families"]["ttop"]
    assert metadata["research_feature_metadata"]["feature_families"]["pitch_archetype"]
    assert metadata["research_feature_metadata"]["market_priors"]["coverage_pct"] == 0.0
    assert metadata["mean_metrics"]["rmse"] == artifact.holdout_metrics["rmse"]
    assert "distribution_metrics" in metadata
    assert metadata["mu_delta_mode"] == "off"
    assert metadata["mu_delta_enabled"] is False
    assert "mu_delta_blend_weight" in metadata
    assert "zero_calibration" in metadata
    assert "tail_calibration" in metadata
    assert "selected_features_by_head" in metadata
    assert "mu_delta" in metadata["selected_features_by_head"]
    assert "dispersion_summary_statistics_holdout" in metadata
    assert "zero_mass_summary_statistics_holdout" in metadata


def test_train_run_distribution_model_supports_mu_delta_off_mode(tmp_path: Path) -> None:
    training_path = tmp_path / "training_data.parquet"
    frame = _training_frame()
    _write_training_parquet(frame, training_path)
    mean_metadata_path = _write_mean_artifact(
        frame=frame,
        output_dir=tmp_path / "mean_head",
        holdout_season=2025,
    )

    artifact = train_run_distribution_model(
        training_data=training_path,
        output_dir=tmp_path / "dist_models",
        mean_artifact_metadata_path=mean_metadata_path,
        holdout_season=2025,
        feature_selection_mode="flat",
        forced_delta_feature_count=0,
        time_series_splits=3,
        xgb_n_jobs=1,
        distribution_report_dir=tmp_path / "reports",
        mu_delta_mode="off",
    )

    metadata = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
    assert metadata["mu_delta_mode"] == "off"
    assert metadata["mu_delta_enabled"] is False
    assert metadata["mu_delta_blend_weight"] == 0.0
    assert metadata["selected_features_by_head"]["mu_delta"] == []
    assert metadata["feature_selection_diagnostics_by_head"]["mu_delta"]["feature_columns"] == []
    assert "market_feature_variation_summary" in metadata
    assert "slice_summaries" in metadata
    assert metadata["output_paths"]["holdout_predictions_csv"]
    assert "calibration_bins" in metadata
    assert metadata["calibration_bins"]["shutout_probability"]
    assert metadata["calibration_bins"]["tail_probabilities"]["p_ge_3"]
    assert metadata["comparison_to_control"]["guardrails"]["rmse_within_2pct"] is True


def test_train_run_distribution_model_trains_dispersion_head_on_positive_rows_only(
    monkeypatch,
    tmp_path: Path,
) -> None:
    training_path = tmp_path / "training_data.parquet"
    frame = _training_frame()
    _write_training_parquet(frame, training_path)
    mean_metadata_path = _write_mean_artifact(
        frame=frame,
        output_dir=tmp_path / "mean_head",
        holdout_season=2025,
    )

    captured: dict[str, object] = {}

    from src.model import run_distribution_trainer as trainer_module

    original_fit = trainer_module._fit_dispersion_estimator

    def _capturing_fit_dispersion_estimator(**kwargs):
        train_frame = kwargs["train_frame"]
        captured["row_count"] = int(len(train_frame))
        captured["targets"] = train_frame["final_away_score"].astype(int).tolist()
        return original_fit(**kwargs)

    monkeypatch.setattr(
        trainer_module,
        "_fit_dispersion_estimator",
        _capturing_fit_dispersion_estimator,
    )

    train_run_distribution_model(
        training_data=training_path,
        output_dir=tmp_path / "dist_models",
        mean_artifact_metadata_path=mean_metadata_path,
        holdout_season=2025,
        feature_selection_mode="flat",
        forced_delta_feature_count=0,
        time_series_splits=3,
        xgb_n_jobs=1,
        head_param_overrides={
            "max_depth": 2,
            "n_estimators": 20,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "min_child_weight": 1,
            "reg_lambda": 1.0,
        },
        distribution_report_dir=tmp_path / "reports",
    )

    expected_positive_rows = int(
        (
            frame.loc[frame["season"] < 2025, "final_away_score"]
            .astype(int)
            .gt(0)
            .sum()
        )
    )
    assert captured["row_count"] == expected_positive_rows
    assert captured["targets"]
    assert min(captured["targets"]) > 0


def test_build_control_comparison_flags_guardrails() -> None:
    comparison = build_control_comparison(
        stage3_mean_metrics={
            "rmse": 3.30,
            "predicted_mean": 4.40,
            "actual_mean": 4.30,
        },
        control_mean_metrics={
            "rmse": 3.25,
            "predicted_mean": 4.32,
            "actual_mean": 4.30,
        },
        stage3_distribution_metrics={
            "mean_crps": 1.78,
            "mean_negative_log_score": 2.47,
            "zero_calibration": {"p_0": {"absolute_error": 0.02}},
            "tail_calibration": {
                "p_ge_3": {"absolute_error": 0.03},
                "p_ge_5": {"absolute_error": 0.04},
                "p_ge_10": {"absolute_error": 0.02},
            },
        },
        control_distribution_metrics={
            "mean_crps": 1.80,
            "mean_negative_log_score": 2.49,
            "zero_calibration": {"p_0": {"absolute_error": 0.03}},
            "tail_calibration": {
                "p_ge_3": {"absolute_error": 0.04},
                "p_ge_5": {"absolute_error": 0.05},
                "p_ge_10": {"absolute_error": 0.03},
            },
        },
    )

    assert comparison["improvement_flags"]["beats_control_on_crps"] is True
    assert comparison["improvement_flags"]["beats_control_on_negative_log_score"] is True
    assert comparison["guardrails"]["tail_calibration_stable"] is True


def test_summarize_array_reports_quantiles() -> None:
    summary = summarize_array([1.0, 2.0, 3.0, 4.0], name="dispersion_size")

    assert summary["name"] == "dispersion_size"
    assert summary["mean"] == 2.5
    assert summary["median"] == 2.5
    assert summary["min"] == 1.0
    assert summary["max"] == 4.0
