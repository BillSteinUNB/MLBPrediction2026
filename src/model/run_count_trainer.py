from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import optuna
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping as lgb_early_stopping
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from src.clients.weather_client import fetch_game_weather
from src.model.artifact_runtime import collect_runtime_versions
from src.model.data_builder import DEFAULT_OUTPUT_PATH, build_training_dataset
from src.model.xgboost_trainer import (
    DEFAULT_EARLY_STOPPING_ROUNDS,
    DEFAULT_MODEL_OUTPUT_DIR,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SEARCH_SPACE,
    DEFAULT_TIME_SERIES_SPLITS,
    DEFAULT_TOP_FEATURE_COUNT,
    DEFAULT_VALIDATION_FRACTION,
    DEFAULT_XGBOOST_N_JOBS,
    DEFAULT_XGBOOST_DEVICE,
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

DEFAULT_XGBOOST_BLEND_WEIGHT = 0.6
DEFAULT_LIGHTGBM_BLEND_WEIGHT = 0.4

DEFAULT_RUN_COUNT_SEARCH_ITERATIONS = 150
DEFAULT_RUN_COUNT_MAX_FEATURE_COUNT = 80
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

DEFAULT_RUN_COUNT_MODEL_SPECS: tuple[dict[str, str], ...] = (
    {"model_name": "f5_home_runs_model", "target_column": "f5_home_score"},
    {"model_name": "f5_away_runs_model", "target_column": "f5_away_score"},
    {"model_name": "full_game_home_runs_model", "target_column": "final_home_score"},
    {"model_name": "full_game_away_runs_model", "target_column": "final_away_score"},
)


@dataclass(frozen=True, slots=True)
class BlendedRunCountRegressor:
    xgboost_model: XGBRegressor
    lightgbm_model: LGBMRegressor
    xgboost_weight: float = DEFAULT_XGBOOST_BLEND_WEIGHT
    lightgbm_weight: float = DEFAULT_LIGHTGBM_BLEND_WEIGHT

    def predict(self, dataframe: pd.DataFrame) -> pd.Series:
        xgboost_predictions = pd.Series(self.xgboost_model.predict(dataframe), dtype=float)
        lightgbm_predictions = pd.Series(self.lightgbm_model.predict(dataframe), dtype=float)
        lightgbm_predictions = lightgbm_predictions.where(
            lightgbm_predictions.notna(), xgboost_predictions
        )
        blended = (
            self.xgboost_weight * xgboost_predictions + self.lightgbm_weight * lightgbm_predictions
        )
        blended = blended.where(blended.notna(), xgboost_predictions)
        return blended.clip(lower=0.0)


@dataclass(frozen=True, slots=True)
class RunCountTrainingArtifact:
    model_name: str
    target_column: str
    model_version: str
    model_path: Path
    metadata_path: Path
    best_params: dict[str, float | int]
    cv_best_rmse: float
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


@dataclass(frozen=True, slots=True)
class RunCountTrainingResult:
    model_version: str
    data_version_hash: str
    holdout_season: int
    feature_columns: list[str]
    summary_path: Path
    models: dict[str, RunCountTrainingArtifact]


def train_run_count_models(
    *,
    training_data: pd.DataFrame | str | Path,
    output_dir: str | Path = DEFAULT_MODEL_OUTPUT_DIR,
    holdout_season: int | None = None,
    search_space: Mapping[str, Sequence[float | int]] = DEFAULT_RUN_COUNT_SEARCH_SPACE,
    time_series_splits: int = DEFAULT_TIME_SERIES_SPLITS,
    search_iterations: int = DEFAULT_RUN_COUNT_SEARCH_ITERATIONS,
    random_state: int = DEFAULT_RANDOM_STATE,
    top_feature_count: int = DEFAULT_TOP_FEATURE_COUNT,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
) -> RunCountTrainingResult:
    """Train and persist run-count regressors for full-game and F5 score targets."""

    dataset = _load_training_dataframe(training_data)
    candidate_feature_columns = _resolve_run_count_candidate_feature_columns(dataset)
    if not candidate_feature_columns:
        raise ValueError("Training data does not contain any numeric feature columns")

    effective_holdout_season = _resolve_holdout_season(dataset, holdout_season)
    data_version_hash = _resolve_data_version_hash(dataset)
    model_version = _build_model_version(data_version_hash)

    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, RunCountTrainingArtifact] = {}
    for spec in DEFAULT_RUN_COUNT_MODEL_SPECS:
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
            output_dir=resolved_output_dir,
            model_version=model_version,
            holdout_season=effective_holdout_season,
            search_space=search_space,
            time_series_splits=time_series_splits,
            search_iterations=search_iterations,
            random_state=random_state,
            top_feature_count=top_feature_count,
            data_version_hash=data_version_hash,
            early_stopping_rounds=early_stopping_rounds,
            validation_fraction=validation_fraction,
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
    parser.add_argument("--early-stopping-rounds", type=int, default=DEFAULT_EARLY_STOPPING_ROUNDS)
    parser.add_argument("--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION)
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

    resolved_output_dir = _resolve_experiment_output_dir(args.output_dir, args.experiment_name)
    result = train_run_count_models(
        training_data=training_path,
        output_dir=resolved_output_dir,
        holdout_season=args.holdout_season,
        time_series_splits=args.time_series_splits,
        search_iterations=args.search_iterations,
        random_state=args.random_state,
        early_stopping_rounds=args.early_stopping_rounds,
        validation_fraction=args.validation_fraction,
    )
    print(json.dumps(_run_result_to_json_ready(result), indent=2))
    return 0


def _train_single_model(
    *,
    dataset: pd.DataFrame,
    model_name: str,
    target_column: str,
    candidate_feature_columns: Sequence[str],
    output_dir: Path,
    model_version: str,
    holdout_season: int,
    search_space: Mapping[str, Sequence[float | int]],
    time_series_splits: int,
    search_iterations: int,
    random_state: int,
    top_feature_count: int,
    data_version_hash: str,
    early_stopping_rounds: int,
    validation_fraction: float,
) -> RunCountTrainingArtifact:
    frame = _prepare_run_count_frame(dataset, target_column=target_column)
    train_frame = frame.loc[frame["season"] < holdout_season].copy()
    holdout_frame = frame.loc[frame["season"] == holdout_season].copy()
    if train_frame.empty:
        raise ValueError(f"No training rows found before holdout season {holdout_season}")
    if holdout_frame.empty:
        raise ValueError(f"No holdout rows found for season {holdout_season}")
    feature_columns = _select_run_count_feature_columns(
        train_frame,
        target_column=target_column,
        candidate_feature_columns=candidate_feature_columns,
        max_feature_count=DEFAULT_RUN_COUNT_MAX_FEATURE_COUNT,
    )
    if not feature_columns:
        raise ValueError(f"No run-count features selected for target {target_column}")

    (
        best_params,
        cv_best_rmse,
        optuna_best_trial_number,
        optuna_trial_count,
        optuna_study_name,
        optuna_storage_path,
        resolved_time_series_splits,
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
    )
    (
        best_estimator,
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
        "runtime_versions": collect_runtime_versions(),
        "model_family": "xgboost_lightgbm_blend",
        "blend_weights": {
            "xgboost": DEFAULT_XGBOOST_BLEND_WEIGHT,
            "lightgbm": DEFAULT_LIGHTGBM_BLEND_WEIGHT,
        },
        "search_backend": "optuna",
        "cv_best_rmse": cv_best_rmse,
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
        "search_space": {key: list(values) for key, values in search_space.items()},
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
        cv_best_rmse=cv_best_rmse,
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
) -> tuple[dict[str, float | int], float, int, int, str, Path, int]:
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
        "Running Optuna for %s with %s requested trials and %s time-series splits",
        model_name,
        resolved_iterations,
        resolved_time_series_splits,
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
        progress_callback = _build_optuna_progress_callback(
            model_name=model_name,
            target_trial_count=resolved_iterations,
        )
        study.optimize(
            lambda trial: _objective_rmse(
                trial,
                train_frame=train_frame,
                feature_columns=feature_columns,
                target_column=target_column,
                search_space=search_space,
                splitter=splitter,
                random_state=random_state,
            ),
            n_trials=remaining_trials,
            callbacks=[progress_callback],
            gc_after_trial=True,
            show_progress_bar=False,
        )
    else:
        logger.info(
            "Optuna study %s already has %s trials; resuming without new trials",
            study_name,
            existing_trial_count,
        )

    best_trial = study.best_trial
    best_params = _complete_optuna_params(best_trial.params, search_space=search_space)
    return (
        best_params,
        float(best_trial.value),
        int(best_trial.number),
        len(study.trials),
        study_name,
        storage_path,
        resolved_time_series_splits,
    )


def _objective_rmse(
    trial: optuna.trial.Trial,
    *,
    train_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    search_space: Mapping[str, Sequence[float | int]],
    splitter: TimeSeriesSplit,
    random_state: int,
) -> float:
    params = _suggest_optuna_regressor_params(trial, search_space=search_space)
    feature_frame = train_frame[list(feature_columns)]
    target_series = train_frame[target_column]
    fold_losses: list[float] = []

    for fold_index, (train_indices, test_indices) in enumerate(
        splitter.split(train_frame), start=1
    ):
        xgboost_estimator = _build_estimator(random_state=random_state)
        xgboost_estimator.set_params(**params)
        xgboost_estimator.fit(
            feature_frame.iloc[train_indices],
            target_series.iloc[train_indices],
        )
        lightgbm_estimator = _build_lightgbm_estimator(random_state=random_state)
        lightgbm_estimator.set_params(**_lightgbm_params_from_xgboost_params(params))
        lightgbm_estimator.fit(
            feature_frame.iloc[train_indices],
            target_series.iloc[train_indices],
        )
        xgboost_predictions = pd.Series(
            xgboost_estimator.predict(feature_frame.iloc[test_indices]),
            dtype=float,
        )
        lightgbm_predictions = pd.Series(
            lightgbm_estimator.predict(feature_frame.iloc[test_indices]),
            dtype=float,
        ).where(lambda series: series.notna(), xgboost_predictions)
        predictions = (
            DEFAULT_XGBOOST_BLEND_WEIGHT * xgboost_predictions
            + DEFAULT_LIGHTGBM_BLEND_WEIGHT * lightgbm_predictions
        ).where(lambda series: series.notna(), xgboost_predictions)
        fold_loss = float(
            mean_squared_error(
                target_series.iloc[test_indices],
                predictions,
            )
            ** 0.5
        )
        fold_losses.append(fold_loss)
        trial.report(sum(fold_losses) / len(fold_losses), step=fold_index)
        if trial.should_prune():
            raise optuna.TrialPruned(
                f"Pruned at fold {fold_index} with mean RMSE {sum(fold_losses) / len(fold_losses):.6f}"
            )

    return float(sum(fold_losses) / len(fold_losses))


def _suggest_optuna_regressor_params(
    trial: optuna.trial.Trial,
    *,
    search_space: Mapping[str, Sequence[float | int]],
) -> dict[str, float | int]:
    if _matches_default_run_count_search_space(search_space):
        return {
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

    params: dict[str, float | int] = {}
    for key, values in search_space.items():
        options = list(values)
        if not options:
            raise ValueError(f"Search space for {key} must contain at least one value")
        if len(options) == 1:
            params[key] = options[0]
            continue
        params[key] = trial.suggest_categorical(key, options)
    return _normalize_best_params(params)


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
    rmse_improvement_pct = (
        ((naive_rmse - model_rmse) / naive_rmse) * 100.0 if naive_rmse > 0 else None
    )
    mae_improvement_pct = ((naive_mae - model_mae) / naive_mae) * 100.0 if naive_mae > 0 else None
    return {
        "mae": model_mae,
        "rmse": model_rmse,
        "r2": float(r2_score(actual, predicted)),
        "actual_mean": float(actual.mean()),
        "predicted_mean": float(predicted.mean()),
        "naive_mean_prediction": naive_prediction,
        "naive_mae": naive_mae,
        "naive_rmse": naive_rmse,
        "mae_improvement_vs_naive_pct": mae_improvement_pct,
        "rmse_improvement_vs_naive_pct": rmse_improvement_pct,
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
) -> tuple[BlendedRunCountRegressor, int, int, int | None, int, int]:
    xgboost_estimator = _build_estimator(random_state=random_state)
    xgboost_estimator.set_params(**best_params)
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
    lightgbm_estimator.set_params(**_lightgbm_params_from_xgboost_params(best_params))
    fitted_lightgbm_estimator = _refit_lightgbm_with_temporal_early_stopping(
        estimator=lightgbm_estimator,
        training_frame=training_frame,
        feature_columns=feature_columns,
        target_column=target_column,
        early_stopping_rounds=early_stopping_rounds,
        validation_fraction=validation_fraction,
    )

    return (
        BlendedRunCountRegressor(
            xgboost_model=fitted_xgboost_estimator,
            lightgbm_model=fitted_lightgbm_estimator,
        ),
        requested_n_estimators,
        final_n_estimators,
        best_iteration,
        early_stopping_train_row_count,
        early_stopping_validation_row_count,
    )


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
        eval_metric="rmse",
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
        eval_metric="rmse",
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


def _matches_default_run_count_search_space(
    search_space: Mapping[str, Sequence[float | int]],
) -> bool:
    if set(search_space) != set(DEFAULT_RUN_COUNT_SEARCH_SPACE):
        return False
    for key, values in DEFAULT_RUN_COUNT_SEARCH_SPACE.items():
        if list(search_space[key]) != list(values):
            return False
    return True


def _resolve_run_count_candidate_feature_columns(dataframe: pd.DataFrame) -> list[str]:
    candidate_columns = _resolve_numeric_feature_columns(dataframe)
    filtered_columns: list[str] = []
    candidate_set = set(candidate_columns)
    for column in candidate_columns:
        if any(token in column for token in ("_14g", "_60g", "_14s", "_60s")):
            continue
        if _is_redundant_team_offense_feature(column, candidate_set):
            continue
        filtered_columns.append(column)
    return filtered_columns


def _is_redundant_team_offense_feature(
    column: str,
    candidate_columns: set[str],
) -> bool:
    if not column.startswith(("home_team_", "away_team_")):
        return False
    if not any(f"_{metric}_" in column for metric in _RUN_COUNT_OFFENSE_METRICS):
        return False
    lineup_column = column.replace("_team_", "_lineup_", 1)
    return lineup_column in candidate_columns


def _select_run_count_feature_columns(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    candidate_feature_columns: Sequence[str],
    max_feature_count: int = DEFAULT_RUN_COUNT_MAX_FEATURE_COUNT,
) -> list[str]:
    if max_feature_count <= 0:
        raise ValueError("max_feature_count must be positive")
    if len(candidate_feature_columns) <= max_feature_count:
        return list(candidate_feature_columns)

    target_series = pd.to_numeric(dataframe[target_column], errors="coerce")
    rankings: list[tuple[float, str]] = []
    for column in candidate_feature_columns:
        feature_series = pd.to_numeric(dataframe[column], errors="coerce")
        correlation = feature_series.corr(target_series)
        score = abs(float(correlation)) if pd.notna(correlation) else 0.0
        rankings.append((score, column))

    rankings.sort(key=lambda item: (-item[0], item[1]))
    selected_columns = [column for _, column in rankings[:max_feature_count]]
    return sorted(selected_columns)


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
        blended_importance = DEFAULT_XGBOOST_BLEND_WEIGHT * xgboost_scores.get(
            normalized_name, 0.0
        ) + DEFAULT_LIGHTGBM_BLEND_WEIGHT * lightgbm_scores.get(normalized_name, 0.0)
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
