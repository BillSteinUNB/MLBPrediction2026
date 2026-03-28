from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import optuna
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping as lgb_early_stopping
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from src.clients.weather_client import fetch_game_weather
from src.model.artifact_runtime import collect_runtime_versions
from src.model.data_builder import (
    DEFAULT_OUTPUT_PATH,
    RUN_COUNT_REQUIRED_TEMPORAL_DELTA_COLUMNS,
    RUN_COUNT_TRAINING_SCHEMA_NAME,
    RUN_COUNT_TRAINING_SCHEMA_VERSION,
    build_training_dataset,
    inspect_run_count_training_data,
    resolve_feature_fill_value,
    validate_run_count_training_data,
)
from src.model.xgboost_trainer import (
    DEFAULT_MODEL_OUTPUT_DIR,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TIME_SERIES_SPLITS,
    DEFAULT_TOP_FEATURE_COUNT,
    DEFAULT_VALIDATION_FRACTION,
    DEFAULT_XGBOOST_N_JOBS,
    _build_model_version,
    _extract_feature_importance_rankings,
    _load_training_dataframe,
    _normalize_best_params,
    _refit_estimator_with_temporal_early_stopping,
    _resolve_data_version_hash,
    _resolve_experiment_output_dir,
    _resolve_holdout_season,
    _resolve_numeric_feature_columns,
    _resolve_search_iterations,
    _build_optuna_progress_callback,
    _build_optuna_study_name,
    _complete_optuna_params,
    _split_temporal_validation_frame,
    create_time_series_split,
)


logger = logging.getLogger(__name__)
_console = Console()

DEFAULT_XGBOOST_BLEND_WEIGHT = 0.6
DEFAULT_LIGHTGBM_BLEND_WEIGHT = 0.4
DEFAULT_RUN_COUNT_BLEND_MODE = "learned"
RUN_COUNT_BLEND_MODES = ("learned", "fixed", "xgb_only", "lgbm_only")

DEFAULT_RUN_COUNT_SEARCH_ITERATIONS = 500
DEFAULT_RUN_COUNT_MAX_FEATURE_COUNT = 80
DEFAULT_RUN_COUNT_SHORT_FORM_FEATURE_COUNT = 24
DEFAULT_RUN_COUNT_MEDIUM_FORM_FEATURE_COUNT = 28
DEFAULT_RUN_COUNT_DELTA_FEATURE_COUNT = 12
DEFAULT_RUN_COUNT_CONTEXT_FEATURE_COUNT = 16
DEFAULT_RUN_COUNT_FORCED_DELTA_FEATURE_COUNT = 0
DEFAULT_RUN_COUNT_EARLY_STOPPING_ROUNDS = 40
DEFAULT_RUN_COUNT_FEATURE_SELECTION_MODE = "grouped"
DEFAULT_RUN_COUNT_CV_METRIC_NAME = "poisson_deviance"
DEFAULT_RUN_COUNT_CV_AGGREGATION_MODE = "mean"
DEFAULT_RUN_COUNT_LIGHTGBM_PARAM_MODE = "independent"
DEFAULT_RUN_COUNT_XGBOOST_EVAL_METRIC = "poisson-nloglik"
DEFAULT_RUN_COUNT_LIGHTGBM_EVAL_METRIC = "poisson"
DEFAULT_MIN_POISSON_PREDICTION = 1e-9
_RUN_COUNT_OFFENSE_METRICS = (
    "wrc_plus",
    "woba",
    "xwoba",
    "woba_minus_xwoba",
    "iso",
    "babip",
    "k_pct",
    "bb_pct",
)
DEFAULT_RUN_COUNT_SEARCH_SPACE: dict[str, list[float | int]] = {
    "max_depth": [3, 4, 5, 6, 7, 8],
    "n_estimators": [200, 300, 400, 500, 600, 700, 800, 900, 1000],
    "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 2, 3, 4, 5, 6, 7],
    "gamma": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5],
    "reg_alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
    "reg_lambda": [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0],
}
DEFAULT_RUN_COUNT_LIGHTGBM_SEARCH_SPACE: dict[str, list[float | int]] = {
    "lightgbm__num_leaves": [15, 31, 63],
    "lightgbm__min_child_samples": [10, 20, 30],
    "lightgbm__feature_fraction": [0.7, 0.85, 1.0],
}

DEFAULT_RUN_COUNT_MODEL_SPECS: tuple[dict[str, str], ...] = (
    {"model_name": "f5_home_runs_model", "target_column": "f5_home_score"},
    {"model_name": "f5_away_runs_model", "target_column": "f5_away_score"},
    {"model_name": "full_game_home_runs_model", "target_column": "final_home_score"},
    {"model_name": "full_game_away_runs_model", "target_column": "final_away_score"},
)


def _resolve_run_count_optuna_workers() -> int:
    configured = os.getenv("MLB_OPTUNA_N_JOBS")
    if configured is None:
        return 1
    try:
        return max(1, int(configured))
    except ValueError:
        logger.warning("Ignoring invalid MLB_OPTUNA_N_JOBS value: %s", configured)
        return 1


DEFAULT_RUN_COUNT_OPTUNA_WORKERS = _resolve_run_count_optuna_workers()


@dataclass(frozen=True, slots=True)
class BlendedRunCountRegressor:
    xgboost_model: XGBRegressor
    lightgbm_model: LGBMRegressor
    xgboost_weight: float = DEFAULT_XGBOOST_BLEND_WEIGHT
    lightgbm_weight: float = DEFAULT_LIGHTGBM_BLEND_WEIGHT

    def predict(self, dataframe: pd.DataFrame) -> pd.Series:
        return _blend_run_count_predictions(
            self.xgboost_model.predict(dataframe),
            self.lightgbm_model.predict(dataframe),
            xgboost_weight=self.xgboost_weight,
            lightgbm_weight=self.lightgbm_weight,
        )


@dataclass(frozen=True, slots=True)
class RunCountTrainingArtifact:
    model_name: str
    target_column: str
    model_version: str
    model_path: Path
    metadata_path: Path
    best_params: dict[str, float | int]
    lightgbm_param_mode: str
    optuna_parallel_workers: int
    cv_metric_name: str
    cv_aggregation_mode: str
    cv_best_score: float
    cv_best_rmse: float | None
    cv_fold_scores: list[float]
    cv_fold_weights: list[float]
    blend_mode: str
    blend_weights: dict[str, float]
    learned_blend_weights: dict[str, float]
    blend_candidate_scores: dict[str, float]
    holdout_metrics: dict[str, float | None]
    feature_columns: list[str]
    feature_importance_rankings: list[dict[str, float | str]]
    train_row_count: int
    holdout_row_count: int
    holdout_season: int
    requested_n_estimators: int
    final_n_estimators: int
    best_iteration: int | None
    early_stopping_rounds: int
    validation_fraction: float
    early_stopping_train_row_count: int
    early_stopping_validation_row_count: int
    feature_selection_bucket_counts: dict[str, int]
    feature_selection_bucket_targets: dict[str, int]
    selected_features_by_bucket: dict[str, list[str]]
    forced_delta_features: list[str]
    omitted_top_features_by_bucket: dict[str, list[dict[str, float | str]]]
    feature_selection_family_decisions: list[dict[str, Any]]
    excluded_candidate_counts: dict[str, int]
    feature_health_diagnostics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RunCountTrainingResult:
    model_version: str
    data_version_hash: str
    holdout_season: int
    feature_columns: list[str]
    summary_path: Path
    models: dict[str, RunCountTrainingArtifact]


@dataclass(frozen=True, slots=True)
class RunCountCandidateResolution:
    candidate_columns: list[str]
    excluded_candidate_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class RunCountFeatureSelectionResult:
    feature_columns: list[str]
    bucket_counts: dict[str, int]
    bucket_targets: dict[str, int]
    selected_features_by_bucket: dict[str, list[str]]
    forced_delta_features: list[str]
    omitted_top_features_by_bucket: dict[str, list[dict[str, float | str]]]
    family_decisions: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class RunCountCvDiagnostics:
    aggregation_mode: str
    fold_scores: list[float]
    fold_weights: list[float]
    aggregated_score: float


@dataclass(frozen=True, slots=True)
class RunCountBlendSelection:
    blend_mode: str
    xgboost_weight: float
    lightgbm_weight: float
    learned_xgboost_weight: float
    learned_lightgbm_weight: float
    candidate_scores: dict[str, float]
    optimization_metric_name: str
    oof_row_count: int


def train_run_count_models(
    *,
    training_data: pd.DataFrame | str | Path,
    output_dir: str | Path = DEFAULT_MODEL_OUTPUT_DIR,
    holdout_season: int | None = None,
    search_space: Mapping[str, Sequence[float | int]] = DEFAULT_RUN_COUNT_SEARCH_SPACE,
    time_series_splits: int = DEFAULT_TIME_SERIES_SPLITS,
    search_iterations: int = DEFAULT_RUN_COUNT_SEARCH_ITERATIONS,
    random_state: int = DEFAULT_RANDOM_STATE,
    optuna_workers: int = DEFAULT_RUN_COUNT_OPTUNA_WORKERS,
    top_feature_count: int = DEFAULT_TOP_FEATURE_COUNT,
    early_stopping_rounds: int = DEFAULT_RUN_COUNT_EARLY_STOPPING_ROUNDS,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    feature_selection_mode: str = DEFAULT_RUN_COUNT_FEATURE_SELECTION_MODE,
    forced_delta_feature_count: int = DEFAULT_RUN_COUNT_FORCED_DELTA_FEATURE_COUNT,
    cv_aggregation_mode: str = DEFAULT_RUN_COUNT_CV_AGGREGATION_MODE,
    lightgbm_param_mode: str = DEFAULT_RUN_COUNT_LIGHTGBM_PARAM_MODE,
    blend_mode: str = DEFAULT_RUN_COUNT_BLEND_MODE,
    model_specs: Sequence[Mapping[str, str]] = DEFAULT_RUN_COUNT_MODEL_SPECS,
) -> RunCountTrainingResult:
    """Train and persist run-count regressors for full-game and F5 score targets."""

    validated_training_data = validate_run_count_training_data(training_data)
    training_data_inspection = inspect_run_count_training_data(validated_training_data)
    dataset = _load_training_dataframe(validated_training_data)
    candidate_resolution = _resolve_run_count_candidate_feature_columns(dataset)
    candidate_feature_columns = candidate_resolution.candidate_columns
    if not candidate_feature_columns:
        raise ValueError("Training data does not contain any numeric feature columns")

    effective_holdout_season = _resolve_holdout_season(dataset, holdout_season)
    data_version_hash = _resolve_data_version_hash(dataset)
    model_version = _build_model_version(data_version_hash)

    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, RunCountTrainingArtifact] = {}
    for spec in model_specs:
        model_name = str(spec["model_name"])
        target_column = str(spec["target_column"])
        logger.info(
            "Training %s for holdout season %s with %s search iterations and %s time-series splits",
            model_name,
            effective_holdout_season,
            _resolve_search_iterations(search_space, search_iterations),
            min(time_series_splits, max(len(dataset) - 1, 1)),
        )
        artifact = _train_single_model(
            dataset=dataset,
            model_name=model_name,
            target_column=target_column,
            candidate_feature_columns=candidate_feature_columns,
            excluded_candidate_counts=candidate_resolution.excluded_candidate_counts,
            output_dir=resolved_output_dir,
            model_version=model_version,
            holdout_season=effective_holdout_season,
            search_space=search_space,
            time_series_splits=time_series_splits,
            search_iterations=search_iterations,
            random_state=random_state,
            optuna_workers=optuna_workers,
            top_feature_count=top_feature_count,
            data_version_hash=data_version_hash,
            early_stopping_rounds=early_stopping_rounds,
            validation_fraction=validation_fraction,
            feature_selection_mode=feature_selection_mode,
            forced_delta_feature_count=forced_delta_feature_count,
            cv_aggregation_mode=cv_aggregation_mode,
            lightgbm_param_mode=lightgbm_param_mode,
            blend_mode=blend_mode,
            training_data_inspection=training_data_inspection,
        )
        artifacts[artifact.model_name] = artifact

    summary_path = resolved_output_dir / f"run_count_training_run_{model_version}.json"
    summary_payload = {
        "model_version": model_version,
        "data_version_hash": data_version_hash,
        "holdout_season": effective_holdout_season,
        "feature_columns": sorted(
            {column for artifact in artifacts.values() for column in artifact.feature_columns}
        ),
        "models": {name: _artifact_to_json_ready(artifact) for name, artifact in artifacts.items()},
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return RunCountTrainingResult(
        model_version=model_version,
        data_version_hash=data_version_hash,
        holdout_season=effective_holdout_season,
        feature_columns=sorted(
            {column for artifact in artifacts.values() for column in artifact.feature_columns}
        ),
        summary_path=summary_path,
        models=artifacts,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Train the run-count regressors from the persisted training parquet."""

    parser = argparse.ArgumentParser(description="Train MLB run-count models with temporal CV")
    parser.add_argument("--training-data", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_MODEL_OUTPUT_DIR))
    parser.add_argument("--experiment-name")
    parser.add_argument("--holdout-season", type=int, default=2025)
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--refresh-training-data", action="store_true")
    parser.add_argument("--allow-backfill-years", action="store_true")
    parser.add_argument("--time-series-splits", type=int, default=DEFAULT_TIME_SERIES_SPLITS)
    parser.add_argument(
        "--search-iterations", type=int, default=DEFAULT_RUN_COUNT_SEARCH_ITERATIONS
    )
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--optuna-workers", type=int, default=DEFAULT_RUN_COUNT_OPTUNA_WORKERS)
    parser.add_argument(
        "--early-stopping-rounds", type=int, default=DEFAULT_RUN_COUNT_EARLY_STOPPING_ROUNDS
    )
    parser.add_argument("--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION)
    parser.add_argument(
        "--feature-selection-mode",
        choices=("grouped", "bucketed", "flat"),
        default=DEFAULT_RUN_COUNT_FEATURE_SELECTION_MODE,
    )
    parser.add_argument(
        "--forced-delta-count",
        type=int,
        default=DEFAULT_RUN_COUNT_FORCED_DELTA_FEATURE_COUNT,
    )
    parser.add_argument(
        "--cv-aggregation-mode",
        choices=("mean", "recent_weighted"),
        default=DEFAULT_RUN_COUNT_CV_AGGREGATION_MODE,
    )
    parser.add_argument(
        "--lightgbm-param-mode",
        choices=("independent", "derived"),
        default=DEFAULT_RUN_COUNT_LIGHTGBM_PARAM_MODE,
    )
    parser.add_argument(
        "--blend-mode",
        choices=RUN_COUNT_BLEND_MODES,
        default=DEFAULT_RUN_COUNT_BLEND_MODE,
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    training_path = Path(args.training_data)
    if args.refresh_training_data or not training_path.exists():
        logger.info("Building training data at %s", training_path)
        build_training_dataset(
            start_year=args.start_year,
            end_year=args.end_year,
            output_path=training_path,
            allow_backfill_years=args.allow_backfill_years,
            refresh=args.refresh_training_data,
            weather_fetcher=fetch_game_weather,
        )
    else:
        logger.info(
            "Using existing training parquet at %s; expected schema is %s v%s. "
            "For feature-engineering changes, prefer scripts/build_parquet.py.",
            training_path,
            RUN_COUNT_TRAINING_SCHEMA_NAME,
            RUN_COUNT_TRAINING_SCHEMA_VERSION,
        )

    resolved_output_dir = _resolve_experiment_output_dir(args.output_dir, args.experiment_name)
    result = train_run_count_models(
        training_data=training_path,
        output_dir=resolved_output_dir,
        holdout_season=args.holdout_season,
        time_series_splits=args.time_series_splits,
        search_iterations=args.search_iterations,
        random_state=args.random_state,
        optuna_workers=args.optuna_workers,
        early_stopping_rounds=args.early_stopping_rounds,
        validation_fraction=args.validation_fraction,
        feature_selection_mode=args.feature_selection_mode,
        forced_delta_feature_count=args.forced_delta_count,
        cv_aggregation_mode=args.cv_aggregation_mode,
        lightgbm_param_mode=args.lightgbm_param_mode,
        blend_mode=args.blend_mode,
    )
    print(json.dumps(_run_result_to_json_ready(result), indent=2))
    return 0


def _train_single_model(
    *,
    dataset: pd.DataFrame,
    model_name: str,
    target_column: str,
    candidate_feature_columns: Sequence[str],
    excluded_candidate_counts: Mapping[str, int],
    output_dir: Path,
    model_version: str,
    holdout_season: int,
    search_space: Mapping[str, Sequence[float | int]],
    time_series_splits: int,
    search_iterations: int,
    random_state: int,
    optuna_workers: int,
    top_feature_count: int,
    data_version_hash: str,
    early_stopping_rounds: int,
    validation_fraction: float,
    feature_selection_mode: str = DEFAULT_RUN_COUNT_FEATURE_SELECTION_MODE,
    forced_delta_feature_count: int = DEFAULT_RUN_COUNT_FORCED_DELTA_FEATURE_COUNT,
    cv_aggregation_mode: str = DEFAULT_RUN_COUNT_CV_AGGREGATION_MODE,
    lightgbm_param_mode: str = DEFAULT_RUN_COUNT_LIGHTGBM_PARAM_MODE,
    blend_mode: str = DEFAULT_RUN_COUNT_BLEND_MODE,
    training_data_inspection: Any | None = None,
) -> RunCountTrainingArtifact:
    frame = _prepare_run_count_frame(dataset, target_column=target_column)
    train_frame = frame.loc[frame["season"] < holdout_season].copy()
    holdout_frame = frame.loc[frame["season"] == holdout_season].copy()
    if train_frame.empty:
        raise ValueError(f"No training rows found before holdout season {holdout_season}")
    if holdout_frame.empty:
        raise ValueError(f"No holdout rows found for season {holdout_season}")
    if feature_selection_mode == "flat":
        selection_result = _select_run_count_feature_columns_flat(
            train_frame,
            target_column=target_column,
            candidate_feature_columns=candidate_feature_columns,
            max_feature_count=DEFAULT_RUN_COUNT_MAX_FEATURE_COUNT,
            forced_delta_count=forced_delta_feature_count,
        )
    elif feature_selection_mode == "bucketed":
        if forced_delta_feature_count > 0:
            raise ValueError("forced_delta_feature_count is currently supported only with flat selection mode")
        selection_result = _select_run_count_feature_columns_bucketed(
            train_frame,
            target_column=target_column,
            candidate_feature_columns=candidate_feature_columns,
            max_feature_count=DEFAULT_RUN_COUNT_MAX_FEATURE_COUNT,
        )
    else:
        if forced_delta_feature_count > 0:
            raise ValueError("forced_delta_feature_count is currently supported only with flat selection mode")
        selection_result = _select_run_count_feature_columns(
            train_frame,
            target_column=target_column,
            candidate_feature_columns=candidate_feature_columns,
            max_feature_count=DEFAULT_RUN_COUNT_MAX_FEATURE_COUNT,
        )
    feature_columns = selection_result.feature_columns
    if not feature_columns:
        raise ValueError(f"No run-count features selected for target {target_column}")

    (
        best_params,
        cv_best_score,
        optuna_best_trial_number,
        optuna_trial_count,
        optuna_study_name,
        optuna_storage_path,
        resolved_time_series_splits,
        cv_diagnostics,
    ) = _run_optuna_search(
        train_frame=train_frame,
        feature_columns=feature_columns,
        target_column=target_column,
        model_name=model_name,
        output_dir=output_dir,
        holdout_season=holdout_season,
        data_version_hash=data_version_hash,
        search_space=search_space,
        time_series_splits=time_series_splits,
        search_iterations=search_iterations,
        random_state=random_state,
        optuna_workers=optuna_workers,
        cv_aggregation_mode=cv_aggregation_mode,
        lightgbm_param_mode=lightgbm_param_mode,
    )
    (
        best_estimator,
        blend_selection,
        requested_n_estimators,
        final_n_estimators,
        best_iteration,
        early_stopping_train_row_count,
        early_stopping_validation_row_count,
    ) = _refit_blended_estimator_with_temporal_early_stopping(
        best_params=best_params,
        training_frame=train_frame,
        feature_columns=feature_columns,
        target_column=target_column,
        random_state=random_state,
        early_stopping_rounds=early_stopping_rounds,
        validation_fraction=validation_fraction,
        time_series_splits=resolved_time_series_splits,
        lightgbm_param_mode=lightgbm_param_mode,
        blend_mode=blend_mode,
    )
    xgboost_best_params, lightgbm_best_params = _resolve_blended_model_params(
        best_params,
        lightgbm_param_mode=lightgbm_param_mode,
    )

    holdout_predictions = best_estimator.predict(holdout_frame[list(feature_columns)])
    holdout_metrics = _compute_holdout_metrics(
        train_frame=train_frame,
        holdout_frame=holdout_frame,
        target_column=target_column,
        holdout_predictions=holdout_predictions,
    )
    feature_importance_rankings = _extract_blended_feature_importance_rankings(
        best_estimator,
        feature_columns,
        top_feature_count=top_feature_count,
    )
    feature_health_diagnostics = _build_run_count_feature_health_diagnostics(
        train_frame=train_frame,
        holdout_frame=holdout_frame,
        feature_columns=feature_columns,
        selection_result=selection_result,
        training_data_inspection=training_data_inspection,
    )

    logger.info("%s best hyperparameters: %s", model_name, best_params)
    logger.info("%s holdout metrics: %s", model_name, holdout_metrics)

    model_path = output_dir / f"{model_name}_{model_version}.joblib"
    joblib.dump(best_estimator, model_path)
    metadata_path = model_path.with_suffix(".metadata.json")
    metadata_payload = {
        "model_name": model_name,
        "target_column": target_column,
        "model_version": model_version,
        "data_version_hash": data_version_hash,
        "holdout_season": holdout_season,
        "train_row_count": int(len(train_frame)),
        "holdout_row_count": int(len(holdout_frame)),
        "feature_columns": list(feature_columns),
        "best_params": best_params,
        "best_params_by_model": {
            "xgboost": xgboost_best_params,
            "lightgbm": lightgbm_best_params,
        },
        "runtime_versions": collect_runtime_versions(),
        "model_family": "xgboost_lightgbm_blend",
        "lightgbm_param_mode": lightgbm_param_mode,
        "feature_selection_mode": feature_selection_mode,
        "forced_delta_feature_count": int(forced_delta_feature_count),
        "blend_mode": blend_selection.blend_mode,
        "blend_weights": {
            "xgboost": blend_selection.xgboost_weight,
            "lightgbm": blend_selection.lightgbm_weight,
        },
        "learned_blend_weights": {
            "xgboost": blend_selection.learned_xgboost_weight,
            "lightgbm": blend_selection.learned_lightgbm_weight,
        },
        "blend_candidate_scores": blend_selection.candidate_scores,
        "blend_optimization_metric_name": blend_selection.optimization_metric_name,
        "blend_oof_row_count": blend_selection.oof_row_count,
        "search_backend": "optuna",
        "optuna_parallel_workers": int(optuna_workers),
        "cv_metric_name": DEFAULT_RUN_COUNT_CV_METRIC_NAME,
        "cv_aggregation_mode": cv_diagnostics.aggregation_mode,
        "cv_best_score": cv_best_score,
        "cv_best_rmse": None,
        "cv_fold_scores": cv_diagnostics.fold_scores,
        "cv_fold_weights": cv_diagnostics.fold_weights,
        "cv_diagnostics": {
            "metric_name": DEFAULT_RUN_COUNT_CV_METRIC_NAME,
            "aggregation_mode": cv_diagnostics.aggregation_mode,
            "best_score": cv_best_score,
            "fold_scores": cv_diagnostics.fold_scores,
            "fold_weights": cv_diagnostics.fold_weights,
        },
        "optuna_best_trial_number": optuna_best_trial_number,
        "optuna_trial_count": optuna_trial_count,
        "optuna_study_name": optuna_study_name,
        "optuna_storage_path": str(optuna_storage_path),
        "holdout_metrics": holdout_metrics,
        "requested_n_estimators": requested_n_estimators,
        "final_n_estimators": final_n_estimators,
        "best_iteration": best_iteration,
        "early_stopping_rounds": int(early_stopping_rounds),
        "validation_fraction": float(validation_fraction),
        "early_stopping_train_row_count": int(early_stopping_train_row_count),
        "early_stopping_validation_row_count": int(early_stopping_validation_row_count),
        "feature_importance_rankings": feature_importance_rankings,
        "feature_selection_bucket_counts": selection_result.bucket_counts,
        "feature_selection_bucket_targets": selection_result.bucket_targets,
        "selected_features_by_bucket": selection_result.selected_features_by_bucket,
        "forced_delta_features": list(selection_result.forced_delta_features),
        "omitted_top_features_by_bucket": selection_result.omitted_top_features_by_bucket,
        "feature_selection_family_decisions": selection_result.family_decisions,
        "excluded_candidate_counts": dict(excluded_candidate_counts),
        "feature_health_diagnostics": feature_health_diagnostics,
        "feature_selection_diagnostics": {
            "mode": feature_selection_mode,
            "bucket_counts": selection_result.bucket_counts,
            "bucket_targets": selection_result.bucket_targets,
            "selected_features_by_bucket": selection_result.selected_features_by_bucket,
            "forced_delta_features": list(selection_result.forced_delta_features),
            "omitted_top_features_by_bucket": selection_result.omitted_top_features_by_bucket,
            "family_decisions": selection_result.family_decisions,
            "excluded_candidate_counts": dict(excluded_candidate_counts),
            "feature_health_diagnostics": feature_health_diagnostics,
        },
        "search_space": {
            key: list(values)
            for key, values in {
                **search_space,
                **(
                    DEFAULT_RUN_COUNT_LIGHTGBM_SEARCH_SPACE
                    if lightgbm_param_mode == "independent"
                    else {}
                ),
            }.items()
        },
        "time_series_splits": resolved_time_series_splits,
        "trained_at": datetime.now(UTC).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

    artifact = RunCountTrainingArtifact(
        model_name=model_name,
        target_column=target_column,
        model_version=model_version,
        model_path=model_path,
        metadata_path=metadata_path,
        best_params=best_params,
        lightgbm_param_mode=lightgbm_param_mode,
        optuna_parallel_workers=int(optuna_workers),
        cv_metric_name=DEFAULT_RUN_COUNT_CV_METRIC_NAME,
        cv_aggregation_mode=cv_diagnostics.aggregation_mode,
        cv_best_score=cv_best_score,
        cv_best_rmse=None,
        cv_fold_scores=list(cv_diagnostics.fold_scores),
        cv_fold_weights=list(cv_diagnostics.fold_weights),
        blend_mode=blend_selection.blend_mode,
        blend_weights={
            "xgboost": blend_selection.xgboost_weight,
            "lightgbm": blend_selection.lightgbm_weight,
        },
        learned_blend_weights={
            "xgboost": blend_selection.learned_xgboost_weight,
            "lightgbm": blend_selection.learned_lightgbm_weight,
        },
        blend_candidate_scores=dict(blend_selection.candidate_scores),
        holdout_metrics=holdout_metrics,
        feature_columns=list(feature_columns),
        feature_importance_rankings=feature_importance_rankings,
        train_row_count=len(train_frame),
        holdout_row_count=len(holdout_frame),
        holdout_season=holdout_season,
        requested_n_estimators=requested_n_estimators,
        final_n_estimators=final_n_estimators,
        best_iteration=best_iteration,
        early_stopping_rounds=int(early_stopping_rounds),
        validation_fraction=float(validation_fraction),
        early_stopping_train_row_count=early_stopping_train_row_count,
        early_stopping_validation_row_count=early_stopping_validation_row_count,
        feature_selection_bucket_counts=dict(selection_result.bucket_counts),
        feature_selection_bucket_targets=dict(selection_result.bucket_targets),
        selected_features_by_bucket={
            bucket: list(columns)
            for bucket, columns in selection_result.selected_features_by_bucket.items()
        },
        forced_delta_features=list(selection_result.forced_delta_features),
        omitted_top_features_by_bucket={
            bucket: [dict(item) for item in items]
            for bucket, items in selection_result.omitted_top_features_by_bucket.items()
        },
        feature_selection_family_decisions=[dict(item) for item in selection_result.family_decisions],
        excluded_candidate_counts=dict(excluded_candidate_counts),
        feature_health_diagnostics=feature_health_diagnostics,
    )
    summary_path = output_dir / f"{model_name}_training_run_{model_version}.json"
    summary_path.write_text(
        json.dumps(
            {
                "model_version": model_version,
                "data_version_hash": data_version_hash,
                "holdout_season": holdout_season,
                "artifact": _artifact_to_json_ready(artifact),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return artifact


def _prepare_run_count_frame(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
) -> pd.DataFrame:
    frame = dataframe.copy()
    frame[target_column] = pd.to_numeric(frame[target_column], errors="coerce")
    frame = frame.loc[frame[target_column].notna()].copy()
    return frame.sort_values(["scheduled_start", "game_pk"]).reset_index(drop=True)


def _build_run_count_feature_health_diagnostics(
    *,
    train_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    selection_result: RunCountFeatureSelectionResult,
    training_data_inspection: Any | None,
) -> dict[str, Any]:
    return {
        "expected_delta_columns": _build_expected_delta_column_diagnostics(training_data_inspection),
        "selected_feature_counts": {
            "bucket": {
                str(bucket): int(count) for bucket, count in selection_result.bucket_counts.items()
            },
            "family": _count_selected_features_by_family(feature_columns),
        },
        "forced_delta_features": list(selection_result.forced_delta_features),
        "omitted_top_ranked_delta_features": [
            dict(item) for item in selection_result.omitted_top_features_by_bucket.get("delta", [])[:5]
        ],
        "selected_feature_fill_health": _build_selected_feature_fill_health(
            train_frame=train_frame,
            holdout_frame=holdout_frame,
            feature_columns=feature_columns,
        ),
        "selected_feature_drift": _build_selected_feature_drift_summary(
            train_frame=train_frame,
            holdout_frame=holdout_frame,
            feature_columns=feature_columns,
        ),
    }


def _build_expected_delta_column_diagnostics(training_data_inspection: Any | None) -> dict[str, Any]:
    missing_columns = tuple(
        getattr(training_data_inspection, "missing_temporal_delta_columns", ()) or ()
    )
    expected_count = len(RUN_COUNT_REQUIRED_TEMPORAL_DELTA_COLUMNS)
    return {
        "all_present": not missing_columns,
        "expected_count": expected_count,
        "present_count": expected_count - len(missing_columns),
        "missing_columns": list(missing_columns),
    }


def _count_selected_features_by_family(feature_columns: Sequence[str]) -> dict[str, int]:
    family_counts = Counter(_resolve_run_count_feature_health_family(column) for column in feature_columns)
    return {
        family_name: int(count)
        for family_name, count in sorted(family_counts.items(), key=lambda item: (-item[1], item[0]))
    }


def _resolve_run_count_feature_health_family(column: str) -> str:
    if column.startswith("home_lineup_"):
        return "home_lineup"
    if column.startswith("away_lineup_"):
        return "away_lineup"
    if column.startswith("home_starter_"):
        return "home_starter"
    if column.startswith("away_starter_"):
        return "away_starter"
    if column.startswith("home_team_bullpen_"):
        return "home_bullpen"
    if column.startswith("away_team_bullpen_"):
        return "away_bullpen"
    if column.startswith("home_team_"):
        return "home_team_context"
    if column.startswith("away_team_"):
        return "away_team_context"
    if column.startswith("plate_umpire_"):
        return "umpire"
    if column.startswith("weather_"):
        return "weather"
    if column in {
        "park_runs_factor",
        "park_hr_factor",
        "abs_active",
        "abs_walk_rate_delta",
        "abs_strikeout_rate_delta",
        "home_timezone_crossings_east",
        "away_timezone_crossings_east",
        "home_is_day_after_night_game",
        "away_is_day_after_night_game",
    }:
        return "schedule_context"
    if column in {
        "home_offense_vs_away_starter_woba_gap",
        "away_offense_vs_home_starter_woba_gap",
    }:
        return "matchup_interaction"
    return "other"


def _build_selected_feature_fill_health(
    *,
    train_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> dict[str, Any]:
    train_summary = _summarize_feature_fill_health_for_frame(train_frame, feature_columns)
    holdout_summary = _summarize_feature_fill_health_for_frame(holdout_frame, feature_columns)
    return {
        "default_fill_share_note": (
            "default_fill_share is a post-fill proxy based on each feature's configured fill value; "
            "missing_share is exact."
        ),
        "train": train_summary,
        "holdout": holdout_summary,
    }


def _summarize_feature_fill_health_for_frame(
    dataframe: pd.DataFrame,
    feature_columns: Sequence[str],
) -> dict[str, Any]:
    feature_summaries: list[dict[str, Any]] = []
    default_fill_shares: list[float] = []
    missing_shares: list[float] = []
    for column in feature_columns:
        share_summary = _build_feature_share_summary(dataframe, column)
        feature_summaries.append(share_summary)
        default_fill_shares.append(float(share_summary["default_fill_share"]))
        missing_shares.append(float(share_summary["missing_share"]))

    return {
        "row_count": int(len(dataframe)),
        "feature_count": int(len(feature_columns)),
        "features_with_default_fill": int(sum(share > 0.0 for share in default_fill_shares)),
        "features_with_missing": int(sum(share > 0.0 for share in missing_shares)),
        "mean_default_fill_share": _round_metric(sum(default_fill_shares) / len(default_fill_shares))
        if default_fill_shares
        else 0.0,
        "mean_missing_share": _round_metric(sum(missing_shares) / len(missing_shares))
        if missing_shares
        else 0.0,
        "top_default_fill_share": _top_feature_share_entries(
            feature_summaries,
            share_key="default_fill_share",
        ),
        "top_missing_share": _top_feature_share_entries(
            feature_summaries,
            share_key="missing_share",
        ),
    }


def _build_feature_share_summary(dataframe: pd.DataFrame, column: str) -> dict[str, Any]:
    numeric_series = pd.to_numeric(dataframe[column], errors="coerce")
    missing_share = float(numeric_series.isna().mean()) if len(numeric_series) else 0.0
    fill_value = float(resolve_feature_fill_value(column))
    if pd.isna(fill_value):
        default_fill_share = missing_share
        fill_value_payload: float | None = None
    else:
        default_fill_share = (
            float((numeric_series - fill_value).abs().le(1e-12).fillna(False).mean())
            if len(numeric_series)
            else 0.0
        )
        fill_value_payload = _round_metric(fill_value)
    return {
        "feature": column,
        "family": _resolve_run_count_feature_health_family(column),
        "default_fill_share": _round_metric(default_fill_share),
        "missing_share": _round_metric(missing_share),
        "default_fill_value": fill_value_payload,
    }


def _top_feature_share_entries(
    feature_summaries: Sequence[Mapping[str, Any]],
    *,
    share_key: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    ranked_entries = sorted(
        (
            summary
            for summary in feature_summaries
            if float(summary.get(share_key, 0.0) or 0.0) > 0.0
        ),
        key=lambda summary: (
            -float(summary.get(share_key, 0.0) or 0.0),
            str(summary.get("feature", "")),
        ),
    )
    return [
        {
            "feature": str(entry["feature"]),
            "family": str(entry["family"]),
            share_key: _round_metric(float(entry[share_key])),
        }
        for entry in ranked_entries[:limit]
    ]


def _build_selected_feature_drift_summary(
    *,
    train_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> dict[str, Any]:
    drift_entries: list[dict[str, Any]] = []
    large_shift_count = 0
    severe_shift_count = 0
    for column in feature_columns:
        entry = _build_feature_drift_entry(train_frame, holdout_frame, column)
        drift_entries.append(entry)
        standardized_mean_shift = float(entry.get("standardized_mean_shift") or 0.0)
        if standardized_mean_shift >= 0.5:
            large_shift_count += 1
        if standardized_mean_shift >= 1.0:
            severe_shift_count += 1

    ranked_entries = sorted(
        drift_entries,
        key=lambda entry: (
            -float(entry.get("standardized_mean_shift") or 0.0),
            -float(entry.get("missing_share_delta") or 0.0),
            str(entry.get("feature", "")),
        ),
    )
    return {
        "compared_feature_count": int(len(feature_columns)),
        "features_with_standardized_mean_shift_ge_0_5": int(large_shift_count),
        "features_with_standardized_mean_shift_ge_1_0": int(severe_shift_count),
        "top_standardized_mean_shift": ranked_entries[:5],
    }


def _build_feature_drift_entry(
    train_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    column: str,
) -> dict[str, Any]:
    train_series = pd.to_numeric(train_frame[column], errors="coerce")
    holdout_series = pd.to_numeric(holdout_frame[column], errors="coerce")
    train_non_null = train_series.dropna()
    holdout_non_null = holdout_series.dropna()

    train_mean = float(train_non_null.mean()) if not train_non_null.empty else 0.0
    holdout_mean = float(holdout_non_null.mean()) if not holdout_non_null.empty else 0.0
    train_std = float(train_non_null.std(ddof=0)) if len(train_non_null) > 1 else 0.0
    if train_std <= 1e-9:
        standardized_mean_shift = 0.0 if abs(holdout_mean - train_mean) <= 1e-9 else None
    else:
        standardized_mean_shift = abs(holdout_mean - train_mean) / train_std

    train_share_summary = _build_feature_share_summary(train_frame, column)
    holdout_share_summary = _build_feature_share_summary(holdout_frame, column)
    return {
        "feature": column,
        "family": _resolve_run_count_feature_health_family(column),
        "train_mean": _round_metric(train_mean),
        "holdout_mean": _round_metric(holdout_mean),
        "standardized_mean_shift": _round_metric(standardized_mean_shift),
        "missing_share_delta": _round_metric(
            abs(
                float(holdout_share_summary["missing_share"])
                - float(train_share_summary["missing_share"])
            )
        ),
        "default_fill_share_delta": _round_metric(
            abs(
                float(holdout_share_summary["default_fill_share"])
                - float(train_share_summary["default_fill_share"])
            )
        ),
    }


def _round_metric(value: float | None, digits: int = 4) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _run_optuna_search(
    *,
    train_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    model_name: str,
    output_dir: Path,
    holdout_season: int,
    data_version_hash: str,
    search_space: Mapping[str, Sequence[float | int]],
    time_series_splits: int,
    search_iterations: int,
    random_state: int,
    optuna_workers: int,
    cv_aggregation_mode: str,
    lightgbm_param_mode: str,
) -> tuple[dict[str, float | int], float, int, int, str, Path, int, RunCountCvDiagnostics]:
    resolved_iterations = _resolve_search_iterations(search_space, search_iterations)
    splitter = create_time_series_split(
        row_count=len(train_frame),
        requested_splits=time_series_splits,
    )
    resolved_time_series_splits = int(splitter.n_splits)
    study_name = _build_optuna_study_name(
        model_name=model_name,
        target_column=target_column,
        data_version_hash=data_version_hash,
        holdout_season=holdout_season,
    )
    storage_path = output_dir / "optuna_studies.db"
    storage_url = f"sqlite:///{storage_path.resolve().as_posix()}"

    logger.info(
        "Running Optuna for %s with %s requested trials, %s time-series splits, and %s workers",
        model_name,
        resolved_iterations,
        resolved_time_series_splits,
        max(1, int(optuna_workers)),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=min(10, resolved_iterations),
            n_warmup_steps=2,
        ),
        storage=storage_url,
        load_if_exists=True,
    )

    existing_trial_count = len(study.trials)
    remaining_trials = max(0, resolved_iterations - existing_trial_count)
    if remaining_trials > 0:
        logger.info(
            "Optuna study %s has %s existing trials; running %s additional trials",
            study_name,
            existing_trial_count,
            remaining_trials,
        )
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=30),
            TextColumn("{task.percentage:>3.0f}%"),
            TextColumn("best {task.fields[best]}"),
            TextColumn("latest {task.fields[latest]}"),
            TextColumn("state {task.fields[state]}"),
            TimeElapsedColumn(),
            console=_console,
        ) as progress:
            task_id = progress.add_task(
                f"Training {model_name}",
                total=10,
                completed=min(10, existing_trial_count),
                best="n/a",
                latest="n/a",
                state="queued",
            )
            progress_callback = _build_optuna_progress_callback(
                progress=progress,
                task_id=task_id,
                model_name=model_name,
                target_trial_count=resolved_iterations,
            )
            study.optimize(
                lambda trial: _objective_poisson_deviance(
                    trial,
                    train_frame=train_frame,
                    feature_columns=feature_columns,
                    target_column=target_column,
                    search_space=search_space,
                    splitter=splitter,
                    random_state=random_state,
                    cv_aggregation_mode=cv_aggregation_mode,
                    lightgbm_param_mode=lightgbm_param_mode,
                ),
                n_trials=remaining_trials,
                n_jobs=max(1, int(optuna_workers)),
                callbacks=[progress_callback],
                gc_after_trial=True,
                show_progress_bar=False,
            )
            try:
                best_value = study.best_value
            except (ValueError, AttributeError):
                best_value = getattr(getattr(study, "best_trial", None), "value", None)
            progress.update(
                task_id,
                completed=10,
                best="n/a" if best_value is None else f"{best_value:.6f}",
                latest="done",
                state="complete",
            )
    else:
        logger.info(
            "Optuna study %s already has %s trials; resuming without new trials",
            study_name,
            existing_trial_count,
        )

    best_trial = study.best_trial
    best_params = _complete_run_count_best_params(
        best_trial.params,
        search_space=search_space,
        lightgbm_param_mode=lightgbm_param_mode,
    )
    cv_diagnostics = RunCountCvDiagnostics(
        aggregation_mode=str(
            best_trial.user_attrs.get("cv_aggregation_mode", DEFAULT_RUN_COUNT_CV_AGGREGATION_MODE)
        ),
        fold_scores=[
            float(score) for score in best_trial.user_attrs.get("cv_fold_scores", [])
        ],
        fold_weights=[
            float(weight) for weight in best_trial.user_attrs.get("cv_fold_weights", [])
        ],
        aggregated_score=float(best_trial.value),
    )
    return (
        best_params,
        float(best_trial.value),
        int(best_trial.number),
        len(study.trials),
        study_name,
        storage_path,
        resolved_time_series_splits,
        cv_diagnostics,
    )


def _objective_poisson_deviance(
    trial: optuna.trial.Trial,
    *,
    train_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    search_space: Mapping[str, Sequence[float | int]],
    splitter: TimeSeriesSplit,
    random_state: int,
    cv_aggregation_mode: str,
    lightgbm_param_mode: str,
) -> float:
    params = _suggest_optuna_regressor_params(
        trial,
        search_space=search_space,
        lightgbm_param_mode=lightgbm_param_mode,
    )
    xgboost_params, lightgbm_params = _resolve_blended_model_params(
        params,
        lightgbm_param_mode=lightgbm_param_mode,
    )
    feature_frame = train_frame[list(feature_columns)]
    target_series = train_frame[target_column]
    fold_losses: list[float] = []
    fold_weights = _resolve_run_count_cv_fold_weights(
        fold_count=int(splitter.n_splits),
        aggregation_mode=cv_aggregation_mode,
    )

    for fold_index, (train_indices, test_indices) in enumerate(
        splitter.split(train_frame), start=1
    ):
        xgboost_estimator = _build_estimator(random_state=random_state)
        xgboost_estimator.set_params(**xgboost_params)
        xgboost_estimator.fit(
            feature_frame.iloc[train_indices],
            target_series.iloc[train_indices],
        )
        lightgbm_estimator = _build_lightgbm_estimator(random_state=random_state)
        lightgbm_estimator.set_params(**lightgbm_params)
        lightgbm_estimator.fit(
            feature_frame.iloc[train_indices],
            target_series.iloc[train_indices],
        )
        predictions = _blend_run_count_predictions(
            xgboost_estimator.predict(feature_frame.iloc[test_indices]),
            lightgbm_estimator.predict(feature_frame.iloc[test_indices]),
        )
        fold_loss = _compute_poisson_deviance(target_series.iloc[test_indices], predictions)
        fold_losses.append(fold_loss)
        aggregated_loss = _aggregate_run_count_cv_fold_scores(
            fold_losses,
            aggregation_mode=cv_aggregation_mode,
            fold_weights=fold_weights[: len(fold_losses)],
        )
        trial.report(aggregated_loss, step=fold_index)
        if trial.should_prune():
            raise optuna.TrialPruned(
                f"Pruned at fold {fold_index} with aggregated Poisson deviance "
                f"{aggregated_loss:.6f}"
            )

    final_score = _aggregate_run_count_cv_fold_scores(
        fold_losses,
        aggregation_mode=cv_aggregation_mode,
        fold_weights=fold_weights,
    )
    trial.set_user_attr("cv_aggregation_mode", cv_aggregation_mode)
    trial.set_user_attr("cv_fold_scores", [float(loss) for loss in fold_losses])
    trial.set_user_attr("cv_fold_weights", [float(weight) for weight in fold_weights])
    return final_score


def _resolve_run_count_cv_fold_weights(
    *,
    fold_count: int,
    aggregation_mode: str,
) -> list[float]:
    if fold_count <= 0:
        raise ValueError("fold_count must be positive")
    if aggregation_mode == "mean":
        return [1.0] * fold_count
    if aggregation_mode == "recent_weighted":
        return [float(index) for index in range(1, fold_count + 1)]
    raise ValueError(f"Unsupported cv aggregation mode: {aggregation_mode}")


def _aggregate_run_count_cv_fold_scores(
    fold_scores: Sequence[float],
    *,
    aggregation_mode: str,
    fold_weights: Sequence[float] | None = None,
) -> float:
    if not fold_scores:
        raise ValueError("fold_scores must contain at least one value")

    scores = [float(score) for score in fold_scores]
    if aggregation_mode == "mean":
        return float(sum(scores) / len(scores))
    if aggregation_mode != "recent_weighted":
        raise ValueError(f"Unsupported cv aggregation mode: {aggregation_mode}")

    if fold_weights is None:
        weights = _resolve_run_count_cv_fold_weights(
            fold_count=len(scores),
            aggregation_mode=aggregation_mode,
        )
    else:
        weights = [float(weight) for weight in fold_weights]
    if len(weights) != len(scores):
        raise ValueError("fold_weights must match fold_scores length")
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("fold_weights must sum to a positive value")
    return float(sum(score * weight for score, weight in zip(scores, weights, strict=False)) / total_weight)


def _suggest_optuna_regressor_params(
    trial: optuna.trial.Trial,
    *,
    search_space: Mapping[str, Sequence[float | int]],
    lightgbm_param_mode: str = DEFAULT_RUN_COUNT_LIGHTGBM_PARAM_MODE,
) -> dict[str, float | int]:
    if _matches_default_run_count_search_space(search_space):
        params: dict[str, float | int] = {
            "max_depth": int(trial.suggest_int("max_depth", 3, 8)),
            "n_estimators": int(trial.suggest_int("n_estimators", 200, 1000, step=50)),
            "learning_rate": float(trial.suggest_float("learning_rate", 0.005, 0.1, log=True)),
            "subsample": float(trial.suggest_float("subsample", 0.6, 1.0)),
            "colsample_bytree": float(trial.suggest_float("colsample_bytree", 0.5, 1.0)),
            "min_child_weight": int(trial.suggest_int("min_child_weight", 1, 7)),
            "gamma": float(trial.suggest_float("gamma", 0.0, 0.5)),
            "reg_alpha": float(trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True)),
            "reg_lambda": float(trial.suggest_float("reg_lambda", 0.1, 10.0, log=True)),
        }
        if lightgbm_param_mode == "independent":
            params.update(_suggest_optuna_lightgbm_params(trial))
        elif lightgbm_param_mode != "derived":
            raise ValueError(f"Unsupported LightGBM parameter mode: {lightgbm_param_mode}")
        return params

    params: dict[str, float | int] = {}
    for key, values in search_space.items():
        options = list(values)
        if not options:
            raise ValueError(f"Search space for {key} must contain at least one value")
        if len(options) == 1:
            params[key] = options[0]
            continue
        params[key] = trial.suggest_categorical(key, options)
    if lightgbm_param_mode == "independent":
        params.update(_suggest_optuna_lightgbm_params(trial))
    elif lightgbm_param_mode != "derived":
        raise ValueError(f"Unsupported LightGBM parameter mode: {lightgbm_param_mode}")
    return _normalize_best_params(params)


def _suggest_optuna_lightgbm_params(trial: optuna.trial.Trial) -> dict[str, float | int]:
    return {
        "lightgbm__num_leaves": int(
            trial.suggest_categorical("lightgbm__num_leaves", [15, 31, 63])
        ),
        "lightgbm__min_child_samples": int(
            trial.suggest_categorical("lightgbm__min_child_samples", [10, 20, 30])
        ),
        "lightgbm__feature_fraction": float(
            trial.suggest_categorical("lightgbm__feature_fraction", [0.7, 0.85, 1.0])
        ),
    }


def _compute_holdout_metrics(
    *,
    train_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    target_column: str,
    holdout_predictions: Sequence[float],
) -> dict[str, float | None]:
    actual = pd.Series(pd.to_numeric(holdout_frame[target_column], errors="coerce"))
    predicted = pd.Series(list(holdout_predictions), index=holdout_frame.index, dtype=float)
    naive_prediction = float(pd.to_numeric(train_frame[target_column], errors="coerce").mean())
    naive_series = pd.Series(
        [naive_prediction] * len(actual),
        index=holdout_frame.index,
        dtype=float,
    )
    model_rmse = float(mean_squared_error(actual, predicted) ** 0.5)
    naive_rmse = float(mean_squared_error(actual, naive_series) ** 0.5)
    model_mae = float(mean_absolute_error(actual, predicted))
    naive_mae = float(mean_absolute_error(actual, naive_series))
    model_poisson_deviance = _compute_poisson_deviance(actual, predicted)
    naive_poisson_deviance = _compute_poisson_deviance(actual, naive_series)
    rmse_improvement_pct = (
        ((naive_rmse - model_rmse) / naive_rmse) * 100.0 if naive_rmse > 0 else None
    )
    mae_improvement_pct = ((naive_mae - model_mae) / naive_mae) * 100.0 if naive_mae > 0 else None
    poisson_deviance_improvement_pct = (
        ((naive_poisson_deviance - model_poisson_deviance) / naive_poisson_deviance) * 100.0
        if naive_poisson_deviance > 0
        else None
    )
    return {
        "mae": model_mae,
        "rmse": model_rmse,
        "poisson_deviance": model_poisson_deviance,
        "r2": float(r2_score(actual, predicted)),
        "actual_mean": float(actual.mean()),
        "predicted_mean": float(predicted.mean()),
        "naive_mean_prediction": naive_prediction,
        "naive_mae": naive_mae,
        "naive_rmse": naive_rmse,
        "naive_poisson_deviance": naive_poisson_deviance,
        "mae_improvement_vs_naive_pct": mae_improvement_pct,
        "rmse_improvement_vs_naive_pct": rmse_improvement_pct,
        "poisson_deviance_improvement_vs_naive_pct": poisson_deviance_improvement_pct,
    }


def _refit_blended_estimator_with_temporal_early_stopping(
    *,
    best_params: Mapping[str, float | int],
    training_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    random_state: int,
    early_stopping_rounds: int,
    validation_fraction: float,
    time_series_splits: int,
    lightgbm_param_mode: str,
    blend_mode: str,
) -> tuple[BlendedRunCountRegressor, RunCountBlendSelection, int, int, int | None, int, int]:
    if blend_mode not in RUN_COUNT_BLEND_MODES:
        raise ValueError(f"Unsupported blend mode: {blend_mode}")

    xgboost_params, lightgbm_params = _resolve_blended_model_params(
        best_params,
        lightgbm_param_mode=lightgbm_param_mode,
    )
    xgboost_estimator = _build_estimator(random_state=random_state)
    xgboost_estimator.set_params(**xgboost_params)
    (
        fitted_xgboost_estimator,
        requested_n_estimators,
        final_n_estimators,
        best_iteration,
        early_stopping_train_row_count,
        early_stopping_validation_row_count,
    ) = _refit_estimator_with_temporal_early_stopping(
        estimator=xgboost_estimator,
        training_frame=training_frame,
        feature_columns=feature_columns,
        target_column=target_column,
        early_stopping_rounds=early_stopping_rounds,
        validation_fraction=validation_fraction,
    )

    lightgbm_estimator = _build_lightgbm_estimator(random_state=random_state)
    lightgbm_estimator.set_params(**lightgbm_params)
    fitted_lightgbm_estimator = _refit_lightgbm_with_temporal_early_stopping(
        estimator=lightgbm_estimator,
        training_frame=training_frame,
        feature_columns=feature_columns,
        target_column=target_column,
        early_stopping_rounds=early_stopping_rounds,
        validation_fraction=validation_fraction,
    )

    blend_selection = _select_run_count_blend(
        best_params=best_params,
        training_frame=training_frame,
        feature_columns=feature_columns,
        target_column=target_column,
        random_state=random_state,
        time_series_splits=time_series_splits,
        lightgbm_param_mode=lightgbm_param_mode,
        blend_mode=blend_mode,
    )

    blended = BlendedRunCountRegressor(
        xgboost_model=fitted_xgboost_estimator,
        lightgbm_model=fitted_lightgbm_estimator,
        xgboost_weight=blend_selection.xgboost_weight,
        lightgbm_weight=blend_selection.lightgbm_weight,
    )

    return (
        blended,
        blend_selection,
        requested_n_estimators,
        final_n_estimators,
        best_iteration,
        early_stopping_train_row_count,
        early_stopping_validation_row_count,
    )


def _select_run_count_blend(
    *,
    best_params: Mapping[str, float | int],
    training_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    random_state: int,
    time_series_splits: int,
    lightgbm_param_mode: str,
    blend_mode: str,
) -> RunCountBlendSelection:
    oof_predictions = _generate_run_count_oof_predictions(
        best_params=best_params,
        training_frame=training_frame,
        feature_columns=feature_columns,
        target_column=target_column,
        random_state=random_state,
        time_series_splits=time_series_splits,
        lightgbm_param_mode=lightgbm_param_mode,
    )
    return _resolve_run_count_blend_selection(
        actual=oof_predictions["actual"],
        xgboost_predictions=oof_predictions["xgboost"],
        lightgbm_predictions=oof_predictions["lightgbm"],
        blend_mode=blend_mode,
    )


def _generate_run_count_oof_predictions(
    *,
    best_params: Mapping[str, float | int],
    training_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    random_state: int,
    time_series_splits: int,
    lightgbm_param_mode: str,
) -> dict[str, pd.Series]:
    feature_frame = training_frame[list(feature_columns)]
    target_series = training_frame[target_column]
    xgboost_params, lightgbm_params = _resolve_blended_model_params(
        best_params,
        lightgbm_param_mode=lightgbm_param_mode,
    )
    splitter = create_time_series_split(
        row_count=len(training_frame), requested_splits=time_series_splits
    )

    oof_xgb_preds: list[float] = []
    oof_lgbm_preds: list[float] = []
    oof_actuals: list[float] = []

    for train_indices, test_indices in splitter.split(training_frame):
        xgb_fold = _build_estimator(random_state=random_state)
        xgb_fold.set_params(**xgboost_params)
        xgb_fold.fit(feature_frame.iloc[train_indices], target_series.iloc[train_indices])

        lgbm_fold = _build_lightgbm_estimator(random_state=random_state)
        lgbm_fold.set_params(**lightgbm_params)
        lgbm_fold.fit(feature_frame.iloc[train_indices], target_series.iloc[train_indices])

        oof_xgb_preds.extend(xgb_fold.predict(feature_frame.iloc[test_indices]))
        oof_lgbm_preds.extend(lgbm_fold.predict(feature_frame.iloc[test_indices]))
        oof_actuals.extend(target_series.iloc[test_indices].tolist())

    return {
        "actual": pd.Series(oof_actuals, dtype=float),
        "xgboost": pd.Series(oof_xgb_preds, dtype=float),
        "lightgbm": pd.Series(oof_lgbm_preds, dtype=float),
    }


def _resolve_run_count_blend_selection(
    *,
    actual: Sequence[float],
    xgboost_predictions: Sequence[float],
    lightgbm_predictions: Sequence[float],
    blend_mode: str,
) -> RunCountBlendSelection:
    learned_xgboost_weight, learned_lightgbm_weight = _learn_run_count_blend_weights(
        actual=actual,
        xgboost_predictions=xgboost_predictions,
        lightgbm_predictions=lightgbm_predictions,
    )
    candidate_weights = {
        "xgb_only": (1.0, 0.0),
        "lgbm_only": (0.0, 1.0),
        "fixed": (DEFAULT_XGBOOST_BLEND_WEIGHT, DEFAULT_LIGHTGBM_BLEND_WEIGHT),
        "learned": (learned_xgboost_weight, learned_lightgbm_weight),
    }
    candidate_scores = {
        candidate_name: _compute_poisson_deviance(
            actual,
            _blend_run_count_predictions(
                xgboost_predictions,
                lightgbm_predictions,
                xgboost_weight=weights[0],
                lightgbm_weight=weights[1],
            ),
        )
        for candidate_name, weights in candidate_weights.items()
    }
    selected_weights = candidate_weights.get(blend_mode)
    if selected_weights is None:
        raise ValueError(f"Unsupported blend mode: {blend_mode}")
    return RunCountBlendSelection(
        blend_mode=blend_mode,
        xgboost_weight=selected_weights[0],
        lightgbm_weight=selected_weights[1],
        learned_xgboost_weight=learned_xgboost_weight,
        learned_lightgbm_weight=learned_lightgbm_weight,
        candidate_scores=candidate_scores,
        optimization_metric_name=DEFAULT_RUN_COUNT_CV_METRIC_NAME,
        oof_row_count=len(pd.Series(actual)),
    )


def _learn_run_count_blend_weights(
    *,
    actual: Sequence[float],
    xgboost_predictions: Sequence[float],
    lightgbm_predictions: Sequence[float],
    resolution: int = 1000,
) -> tuple[float, float]:
    if resolution <= 0:
        raise ValueError("resolution must be positive")

    actual_series = pd.Series(actual, dtype=float)
    xgboost_series = pd.Series(xgboost_predictions, dtype=float)
    lightgbm_series = pd.Series(lightgbm_predictions, dtype=float)

    best_xgboost_weight = DEFAULT_XGBOOST_BLEND_WEIGHT
    best_loss = float("inf")
    for step in range(resolution + 1):
        xgboost_weight = step / resolution
        lightgbm_weight = 1.0 - xgboost_weight
        loss = _compute_poisson_deviance(
            actual_series,
            _blend_run_count_predictions(
                xgboost_series,
                lightgbm_series,
                xgboost_weight=xgboost_weight,
                lightgbm_weight=lightgbm_weight,
            ),
        )
        if loss < best_loss:
            best_loss = loss
            best_xgboost_weight = xgboost_weight

    learned_lightgbm_weight = 1.0 - best_xgboost_weight
    logger.info(
        "OOF blend learning selected xgb_weight=%.3f, lightgbm_weight=%.3f, loss=%.6f",
        best_xgboost_weight,
        learned_lightgbm_weight,
        best_loss,
    )
    return float(best_xgboost_weight), float(learned_lightgbm_weight)


def _refit_lightgbm_with_temporal_early_stopping(
    *,
    estimator: LGBMRegressor,
    training_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    early_stopping_rounds: int,
    validation_fraction: float,
) -> LGBMRegressor:
    requested_n_estimators = int(estimator.get_params().get("n_estimators", 100))
    train_slice, validation_slice = _split_temporal_validation_frame(
        training_frame,
        validation_fraction=validation_fraction,
    )
    if early_stopping_rounds <= 0 or validation_slice.empty:
        fitted_estimator = clone(estimator)
        fitted_estimator.fit(training_frame[list(feature_columns)], training_frame[target_column])
        return fitted_estimator

    early_stop_estimator = clone(estimator)
    early_stop_estimator.fit(
        train_slice[list(feature_columns)],
        train_slice[target_column],
        eval_set=[(validation_slice[list(feature_columns)], validation_slice[target_column])],
        eval_metric=DEFAULT_RUN_COUNT_LIGHTGBM_EVAL_METRIC,
        callbacks=[lgb_early_stopping(stopping_rounds=int(early_stopping_rounds), verbose=False)],
    )
    best_iteration = getattr(early_stop_estimator, "best_iteration_", None)
    final_n_estimators = requested_n_estimators
    if best_iteration is not None:
        final_n_estimators = max(1, int(best_iteration))

    final_estimator = clone(estimator)
    final_estimator.set_params(n_estimators=final_n_estimators)
    final_estimator.fit(training_frame[list(feature_columns)], training_frame[target_column])
    return final_estimator


def _build_estimator(*, random_state: int) -> XGBRegressor:
    return XGBRegressor(
        objective="count:poisson",
        eval_metric=DEFAULT_RUN_COUNT_XGBOOST_EVAL_METRIC,
        random_state=random_state,
        tree_method="hist",
        n_jobs=DEFAULT_XGBOOST_N_JOBS,
        verbosity=0,
    )


def _build_lightgbm_estimator(*, random_state: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="poisson",
        random_state=random_state,
        n_jobs=DEFAULT_XGBOOST_N_JOBS,
        verbosity=-1,
    )


def _blend_run_count_predictions(
    xgboost_predictions: Sequence[float],
    lightgbm_predictions: Sequence[float],
    *,
    xgboost_weight: float = DEFAULT_XGBOOST_BLEND_WEIGHT,
    lightgbm_weight: float = DEFAULT_LIGHTGBM_BLEND_WEIGHT,
) -> pd.Series:
    xgboost_series = pd.Series(xgboost_predictions, dtype=float)
    lightgbm_series = pd.Series(lightgbm_predictions, dtype=float).where(
        lambda series: series.notna(), xgboost_series
    )
    blended = (xgboost_weight * xgboost_series + lightgbm_weight * lightgbm_series).where(
        lambda series: series.notna(), xgboost_series
    )
    return blended.clip(lower=0.0)


def _compute_poisson_deviance(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> float:
    actual_series = pd.Series(pd.to_numeric(actual, errors="coerce"), dtype=float)
    predicted_series = pd.Series(pd.to_numeric(predicted, errors="coerce"), dtype=float).clip(
        lower=DEFAULT_MIN_POISSON_PREDICTION
    )
    return float(mean_poisson_deviance(actual_series, predicted_series))


def _lightgbm_params_from_xgboost_params(
    params: Mapping[str, float | int],
) -> dict[str, float | int]:
    return {
        "max_depth": int(params.get("max_depth", -1)),
        "n_estimators": int(params.get("n_estimators", 100)),
        "learning_rate": float(params.get("learning_rate", 0.05)),
        "subsample": float(params.get("subsample", 1.0)),
        "colsample_bytree": float(params.get("colsample_bytree", 1.0)),
        "min_child_samples": max(5, int(round(float(params.get("min_child_weight", 1)) * 4))),
        "min_split_gain": float(params.get("gamma", 0.0)),
        "reg_alpha": float(params.get("reg_alpha", 0.0)),
        "reg_lambda": float(params.get("reg_lambda", 0.0)),
    }


def _complete_run_count_best_params(
    params: Mapping[str, float | int],
    *,
    search_space: Mapping[str, Sequence[float | int]],
    lightgbm_param_mode: str,
) -> dict[str, float | int]:
    completed = _complete_optuna_params(params, search_space=search_space)
    if lightgbm_param_mode == "derived":
        return completed
    if lightgbm_param_mode != "independent":
        raise ValueError(f"Unsupported LightGBM parameter mode: {lightgbm_param_mode}")

    for key, values in DEFAULT_RUN_COUNT_LIGHTGBM_SEARCH_SPACE.items():
        if key in params:
            completed[key] = params[key]
        elif len(values) == 1:
            completed[key] = values[0]
    return _normalize_best_params(completed)


def _resolve_blended_model_params(
    params: Mapping[str, float | int],
    *,
    lightgbm_param_mode: str,
) -> tuple[dict[str, float | int], dict[str, float | int]]:
    xgboost_params = {
        str(key): value for key, value in params.items() if not str(key).startswith("lightgbm__")
    }
    lightgbm_params = _lightgbm_params_from_xgboost_params(xgboost_params)
    if lightgbm_param_mode == "derived":
        return xgboost_params, lightgbm_params
    if lightgbm_param_mode != "independent":
        raise ValueError(f"Unsupported LightGBM parameter mode: {lightgbm_param_mode}")

    lightgbm_overrides = {
        str(key).removeprefix("lightgbm__"): value
        for key, value in params.items()
        if str(key).startswith("lightgbm__")
    }
    if "feature_fraction" in lightgbm_overrides:
        lightgbm_overrides["colsample_bytree"] = lightgbm_overrides["feature_fraction"]
    lightgbm_params.update(lightgbm_overrides)
    return _normalize_best_params(xgboost_params), _normalize_best_params(lightgbm_params)


def _matches_default_run_count_search_space(
    search_space: Mapping[str, Sequence[float | int]],
) -> bool:
    if set(search_space) != set(DEFAULT_RUN_COUNT_SEARCH_SPACE):
        return False
    for key, values in DEFAULT_RUN_COUNT_SEARCH_SPACE.items():
        if list(search_space[key]) != list(values):
            return False
    return True


def _resolve_run_count_candidate_feature_columns(
    dataframe: pd.DataFrame,
) -> RunCountCandidateResolution:
    candidate_columns = _resolve_numeric_feature_columns(dataframe)
    filtered_columns: list[str] = []
    candidate_set = set(candidate_columns)
    excluded_14_window_count = 0
    excluded_60_window_count = 0
    excluded_redundant_team_offense_count = 0
    for column in candidate_columns:
        if any(token in column for token in ("_14g", "_14s")):
            excluded_14_window_count += 1
            continue
        if any(token in column for token in ("_60g", "_60s")):
            excluded_60_window_count += 1
            continue
        if _is_redundant_team_offense_feature(column, candidate_set, dataframe):
            excluded_redundant_team_offense_count += 1
            continue
        filtered_columns.append(column)
    return RunCountCandidateResolution(
        candidate_columns=filtered_columns,
        excluded_candidate_counts={
            "14_window": excluded_14_window_count,
            "60_window": excluded_60_window_count,
            "redundant_team_offense": excluded_redundant_team_offense_count,
        },
    )


def _is_redundant_team_offense_feature(
    column: str,
    candidate_columns: set[str],
    dataframe: pd.DataFrame,
) -> bool:
    if not column.startswith(("home_team_", "away_team_")):
        return False
    if not any(f"_{metric}_" in column for metric in _RUN_COUNT_OFFENSE_METRICS):
        return False
    lineup_column = column.replace("_team_", "_lineup_", 1)
    if lineup_column not in candidate_columns or lineup_column not in dataframe.columns:
        return False

    lineup_std = pd.to_numeric(dataframe[lineup_column], errors="coerce").std()
    if pd.isna(lineup_std) or float(lineup_std) < 0.001:
        return False
    return True


def _select_run_count_feature_columns(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    candidate_feature_columns: Sequence[str],
    max_feature_count: int = DEFAULT_RUN_COUNT_MAX_FEATURE_COUNT,
) -> RunCountFeatureSelectionResult:
    """Select features with family-level competition before bucket allocation."""

    if max_feature_count <= 0:
        raise ValueError("max_feature_count must be positive")

    bucket_targets = _resolve_run_count_bucket_targets(max_feature_count=max_feature_count)
    feature_scores = _score_run_count_candidate_features(
        dataframe,
        target_column=target_column,
        candidate_feature_columns=candidate_feature_columns,
    )

    family_candidates: dict[str, list[tuple[float, str, str]]] = {}
    for score, column in feature_scores:
        family_key = _resolve_run_count_feature_family(column)
        bucket_name = _resolve_run_count_feature_bucket(column)
        family_candidates.setdefault(family_key, []).append((score, column, bucket_name))

    ranked_features_by_bucket: dict[str, list[tuple[float, str]]] = {
        "short_form": [],
        "medium_form": [],
        "delta": [],
        "context": [],
    }
    family_decisions: list[dict[str, Any]] = []
    for family_key, candidates in sorted(family_candidates.items()):
        ranked_candidates = sorted(candidates, key=lambda item: (-item[0], item[1]))
        winner_score, winner_name, winner_bucket = ranked_candidates[0]
        ranked_features_by_bucket[winner_bucket].append((winner_score, winner_name))
        family_decisions.append(
            {
                "family": family_key,
                "winner": winner_name,
                "winner_bucket": winner_bucket,
                "winner_score": winner_score,
                "selected": False,
                "candidates": [
                    {"feature": name, "score": score, "bucket": bucket_name, "selected": False}
                    for score, name, bucket_name in ranked_candidates
                ],
            }
        )
        if len(ranked_candidates) > 1:
            logger.info(
                "Run-count selector family %s chose %s (score=%.4f) over %s",
                family_key,
                winner_name,
                winner_score,
                ", ".join(name for _, name, _ in ranked_candidates[1:]),
            )

    for bucket_name in ranked_features_by_bucket:
        ranked_features_by_bucket[bucket_name].sort(key=lambda item: (-item[0], item[1]))

    return _finalize_run_count_feature_selection(
        ranked_features_by_bucket=ranked_features_by_bucket,
        bucket_targets=bucket_targets,
        family_decisions=family_decisions,
        forced_delta_features=[],
    )


def _select_run_count_feature_columns_bucketed(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    candidate_feature_columns: Sequence[str],
    max_feature_count: int = DEFAULT_RUN_COUNT_MAX_FEATURE_COUNT,
) -> RunCountFeatureSelectionResult:
    if max_feature_count <= 0:
        raise ValueError("max_feature_count must be positive")

    bucket_targets = _resolve_run_count_bucket_targets(max_feature_count=max_feature_count)
    ranked_features_by_bucket = _rank_run_count_candidate_features(
        dataframe,
        target_column=target_column,
        candidate_feature_columns=candidate_feature_columns,
    )

    return _finalize_run_count_feature_selection(
        ranked_features_by_bucket=ranked_features_by_bucket,
        bucket_targets=bucket_targets,
        family_decisions=[],
        forced_delta_features=[],
    )


def _resolve_run_count_bucket_targets(*, max_feature_count: int) -> dict[str, int]:
    base_targets = {
        "short_form": DEFAULT_RUN_COUNT_SHORT_FORM_FEATURE_COUNT,
        "medium_form": DEFAULT_RUN_COUNT_MEDIUM_FORM_FEATURE_COUNT,
        "delta": DEFAULT_RUN_COUNT_DELTA_FEATURE_COUNT,
        "context": DEFAULT_RUN_COUNT_CONTEXT_FEATURE_COUNT,
    }
    base_total = sum(base_targets.values())
    if max_feature_count == base_total:
        return base_targets
    if max_feature_count < base_total:
        raise ValueError(
            f"max_feature_count {max_feature_count} is smaller than configured bucket total {base_total}"
        )
    return {
        "short_form": base_targets["short_form"],
        "medium_form": base_targets["medium_form"],
        "delta": base_targets["delta"],
        "context": base_targets["context"] + (max_feature_count - base_total),
    }


def _rank_run_count_candidate_features(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    candidate_feature_columns: Sequence[str],
) -> dict[str, list[tuple[float, str]]]:
    rankings_by_bucket: dict[str, list[tuple[float, str]]] = {
        "short_form": [],
        "medium_form": [],
        "delta": [],
        "context": [],
    }
    for score, column in _score_run_count_candidate_features(
        dataframe,
        target_column=target_column,
        candidate_feature_columns=candidate_feature_columns,
    ):
        bucket_name = _resolve_run_count_feature_bucket(column)
        rankings_by_bucket[bucket_name].append((score, column))

    for bucket_name in rankings_by_bucket:
        rankings_by_bucket[bucket_name].sort(key=lambda item: (-item[0], item[1]))
    return rankings_by_bucket


def _score_run_count_candidate_features(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    candidate_feature_columns: Sequence[str],
) -> list[tuple[float, str]]:
    target_series = pd.to_numeric(dataframe[target_column], errors="coerce")
    scored: list[tuple[float, str]] = []
    for column in candidate_feature_columns:
        feature_series = pd.to_numeric(dataframe[column], errors="coerce")
        correlation = feature_series.corr(target_series)
        score = abs(float(correlation)) if pd.notna(correlation) else 0.0
        scored.append((score, column))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored


def _resolve_run_count_feature_bucket(column: str) -> str:
    if column.endswith("_delta_7v30g") or column.endswith("_delta_7v30s"):
        return "delta"
    if "_7g" in column or "_7s" in column:
        return "short_form"
    if "_30g" in column or "_30s" in column:
        return "medium_form"
    return "context"


def _resolve_run_count_feature_family(column: str) -> str:
    for prefix, short_suffix, medium_suffix, delta_suffix in (
        ("home_lineup_", "_7g", "_30g", "_delta_7v30g"),
        ("away_lineup_", "_7g", "_30g", "_delta_7v30g"),
        ("home_starter_", "_7s", "_30s", "_delta_7v30s"),
        ("away_starter_", "_7s", "_30s", "_delta_7v30s"),
    ):
        if not column.startswith(prefix):
            continue
        metric = _strip_run_count_family_suffix(
            column[len(prefix) :],
            short_suffix=short_suffix,
            medium_suffix=medium_suffix,
            delta_suffix=delta_suffix,
        )
        if metric is not None:
            return f"{prefix}{metric}"
    return column


def _strip_run_count_family_suffix(
    feature_name: str,
    *,
    short_suffix: str,
    medium_suffix: str,
    delta_suffix: str,
) -> str | None:
    for suffix in (short_suffix, medium_suffix, delta_suffix):
        if feature_name.endswith(suffix):
            return feature_name[: -len(suffix)]
    return None


def _select_run_count_feature_columns_flat(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    candidate_feature_columns: Sequence[str],
    max_feature_count: int = DEFAULT_RUN_COUNT_MAX_FEATURE_COUNT,
    forced_delta_count: int = DEFAULT_RUN_COUNT_FORCED_DELTA_FEATURE_COUNT,
) -> RunCountFeatureSelectionResult:
    """Select top-N features by absolute Pearson correlation (no bucketing).

    This is the selection strategy used by Run 3 which achieved 5.16% holdout R².
    All features compete in a single ranked pool rather than being siloed into
    short_form / medium_form / context buckets.
    """
    if max_feature_count <= 0:
        raise ValueError("max_feature_count must be positive")
    if forced_delta_count < 0:
        raise ValueError("forced_delta_count must be non-negative")
    if forced_delta_count > max_feature_count:
        raise ValueError("forced_delta_count cannot exceed max_feature_count")

    scored = _score_run_count_candidate_features(
        dataframe,
        target_column=target_column,
        candidate_feature_columns=candidate_feature_columns,
    )

    selected_set: set[str] = set()
    forced_delta_features: list[str] = []
    if forced_delta_count > 0:
        ranked_delta_features = [
            feature_name
            for score, feature_name in scored
            if _resolve_run_count_feature_bucket(feature_name) == "delta" and score > 0.0
        ]
        forced_delta_features = ranked_delta_features[:forced_delta_count]
        selected_set.update(forced_delta_features)

    for _, feature_name in scored:
        if len(selected_set) >= max_feature_count:
            break
        selected_set.add(feature_name)

    selected = sorted(selected_set)
    selected_set = set(selected)
    omitted = [{"feature": name, "score": score} for score, name in scored if name not in selected_set][
        :10
    ]

    return RunCountFeatureSelectionResult(
        feature_columns=selected,
        bucket_counts={"flat": len(selected)},
        bucket_targets={"flat": max_feature_count},
        selected_features_by_bucket={"flat": selected},
        forced_delta_features=sorted(forced_delta_features),
        omitted_top_features_by_bucket={"flat": omitted},
        family_decisions=[],
    )


def _finalize_run_count_feature_selection(
    *,
    ranked_features_by_bucket: Mapping[str, Sequence[tuple[float, str]]],
    bucket_targets: Mapping[str, int],
    family_decisions: list[dict[str, Any]],
    forced_delta_features: Sequence[str],
) -> RunCountFeatureSelectionResult:
    selected_by_bucket: dict[str, list[str]] = {bucket: [] for bucket in bucket_targets}
    selected_feature_set: set[str] = set()
    max_feature_count = sum(bucket_targets.values())
    for bucket_name, bucket_target in bucket_targets.items():
        for _, feature_name in ranked_features_by_bucket[bucket_name]:
            if len(selected_by_bucket[bucket_name]) >= bucket_target:
                break
            selected_by_bucket[bucket_name].append(feature_name)
            selected_feature_set.add(feature_name)

    if len(selected_feature_set) < max_feature_count:
        for bucket_name in ("short_form", "medium_form", "delta", "context"):
            for _, feature_name in ranked_features_by_bucket[bucket_name]:
                if len(selected_feature_set) >= max_feature_count:
                    break
                if feature_name in selected_feature_set:
                    continue
                selected_by_bucket[bucket_name].append(feature_name)
                selected_feature_set.add(feature_name)

    selected_columns = sorted(selected_feature_set)
    for family_decision in family_decisions:
        winner_name = str(family_decision["winner"])
        family_decision["selected"] = winner_name in selected_feature_set
        updated_candidates: list[dict[str, float | str | bool]] = []
        for candidate in family_decision["candidates"]:
            updated_candidate = dict(candidate)
            updated_candidate["selected"] = str(updated_candidate["feature"]) in selected_feature_set
            updated_candidates.append(updated_candidate)
        family_decision["candidates"] = updated_candidates

    return RunCountFeatureSelectionResult(
        feature_columns=selected_columns,
        bucket_counts={bucket: len(columns) for bucket, columns in selected_by_bucket.items()},
        bucket_targets=dict(bucket_targets),
        selected_features_by_bucket={
            bucket: sorted(columns) for bucket, columns in selected_by_bucket.items()
        },
        forced_delta_features=sorted(str(feature) for feature in forced_delta_features),
        omitted_top_features_by_bucket=_build_omitted_top_features_by_bucket(
            ranked_features_by_bucket=ranked_features_by_bucket,
            selected_feature_set=selected_feature_set,
        ),
        family_decisions=family_decisions,
    )


def _build_omitted_top_features_by_bucket(
    *,
    ranked_features_by_bucket: Mapping[str, Sequence[tuple[float, str]]],
    selected_feature_set: set[str],
) -> dict[str, list[dict[str, float | str]]]:
    omitted_features_by_bucket: dict[str, list[dict[str, float | str]]] = {}
    for bucket_name, rankings in ranked_features_by_bucket.items():
        omitted_rankings = [
            {"feature": feature_name, "score": score}
            for score, feature_name in rankings
            if feature_name not in selected_feature_set
        ]
        omitted_features_by_bucket[bucket_name] = omitted_rankings[:10]
    return omitted_features_by_bucket


def _extract_blended_feature_importance_rankings(
    estimator: BlendedRunCountRegressor,
    feature_columns: Sequence[str],
    *,
    top_feature_count: int,
) -> list[dict[str, float | str]]:
    xgboost_rankings = _extract_feature_importance_rankings(
        estimator.xgboost_model,
        feature_columns,
        top_feature_count=len(feature_columns),
    )
    xgboost_scores = {str(item["feature"]): float(item["importance"]) for item in xgboost_rankings}

    lightgbm_importances = getattr(estimator.lightgbm_model, "feature_importances_", None)
    lightgbm_scores: dict[str, float] = {}
    if lightgbm_importances is not None:
        total_importance = float(sum(float(value) for value in lightgbm_importances))
        for feature_name, importance in zip(feature_columns, lightgbm_importances, strict=False):
            if total_importance > 0:
                lightgbm_scores[str(feature_name)] = float(importance) / total_importance
            else:
                lightgbm_scores[str(feature_name)] = 0.0

    blended_scores: list[tuple[str, float]] = []
    for feature_name in feature_columns:
        normalized_name = str(feature_name)
        blended_importance = estimator.xgboost_weight * xgboost_scores.get(
            normalized_name, 0.0
        ) + estimator.lightgbm_weight * lightgbm_scores.get(normalized_name, 0.0)
        blended_scores.append((normalized_name, float(blended_importance)))

    blended_scores.sort(key=lambda item: (-item[1], item[0]))
    return [
        {"feature": feature_name, "importance": importance}
        for feature_name, importance in blended_scores[:top_feature_count]
    ]


def _artifact_to_json_ready(artifact: RunCountTrainingArtifact) -> dict[str, Any]:
    payload = asdict(artifact)
    payload["model_path"] = str(artifact.model_path)
    payload["metadata_path"] = str(artifact.metadata_path)
    return payload


def _run_result_to_json_ready(result: RunCountTrainingResult) -> dict[str, Any]:
    return {
        "model_version": result.model_version,
        "data_version_hash": result.data_version_hash,
        "holdout_season": result.holdout_season,
        "feature_columns": result.feature_columns,
        "summary_path": str(result.summary_path),
        "models": {
            name: _artifact_to_json_ready(artifact) for name, artifact in result.models.items()
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
