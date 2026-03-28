from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import prod
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import optuna
import pandas as pd
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from sklearn.base import clone
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from src.clients.weather_client import fetch_game_weather
from src.model.artifact_runtime import collect_runtime_versions
from src.model.data_builder import (
    DEFAULT_OUTPUT_PATH,
    _compute_data_version_hash,
    _feature_columns,
    build_training_dataset,
)
from src.model.promotion import build_promotion_reason
from src.ops.experiment_tracker import log_training_run


logger = logging.getLogger(__name__)
_console = Console()

DEFAULT_MODEL_OUTPUT_DIR = Path("data") / "models"
DEFAULT_TIME_SERIES_SPLITS = 5
DEFAULT_SEARCH_ITERATIONS = 100
DEFAULT_RANDOM_STATE = 2026
DEFAULT_TOP_FEATURE_COUNT = 25
DEFAULT_EARLY_STOPPING_ROUNDS = 20
DEFAULT_VALIDATION_FRACTION = 0.15
DEFAULT_SEARCH_SPACE: dict[str, list[float | int]] = {
    "max_depth": [3, 4, 5, 6, 7, 8],
    "n_estimators": [100, 200, 300, 400, 500],
    "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0.0, 0.1, 0.3, 0.5],
    "reg_alpha": [0.0, 0.01, 0.1, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 4.0],
}
_MODEL_SPECS = (
    {"model_name": "f5_ml_model", "target_column": "f5_ml_result", "drop_ties": True},
    {"model_name": "f5_rl_model", "target_column": "f5_rl_result", "drop_ties": False},
)


def _resolve_xgboost_n_jobs() -> int:
    configured = os.getenv("MLB_XGBOOST_N_JOBS")
    if configured is not None:
        try:
            return max(1, int(configured))
        except ValueError:
            logger.warning("Ignoring invalid MLB_XGBOOST_N_JOBS value: %s", configured)

    detected_cpu_count = os.cpu_count() or 1
    return max(1, detected_cpu_count - 1)


def _resolve_xgboost_device() -> str:
    device = os.getenv("MLB_XGBOOST_DEVICE", "cpu").strip().lower()
    if device not in {"cpu", "cuda", "gpu"}:
        logger.warning("Ignoring invalid MLB_XGBOOST_DEVICE value: %s — using cpu", device)
        return "cpu"
    return device


DEFAULT_XGBOOST_N_JOBS = _resolve_xgboost_n_jobs()
DEFAULT_XGBOOST_DEVICE = _resolve_xgboost_device()


@dataclass(frozen=True, slots=True)
class ModelTrainingArtifact:
    model_name: str
    target_column: str
    model_version: str
    model_path: Path
    metadata_path: Path
    best_params: dict[str, float | int]
    cv_best_log_loss: float
    holdout_metrics: dict[str, float | None]
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
    promoted_variant: str
    promotion_reason: str


@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    model_version: str
    data_version_hash: str
    holdout_season: int
    feature_columns: list[str]
    summary_path: Path
    models: dict[str, ModelTrainingArtifact]


def create_time_series_split(
    *,
    row_count: int,
    requested_splits: int = DEFAULT_TIME_SERIES_SPLITS,
) -> TimeSeriesSplit:
    """Create a temporal cross-validator that never shuffles future rows into training."""

    if row_count < 3:
        raise ValueError("TimeSeriesSplit requires at least 3 rows")

    actual_splits = min(requested_splits, row_count - 1)
    if actual_splits < 2:
        raise ValueError("TimeSeriesSplit requires at least 2 splits")
    return TimeSeriesSplit(n_splits=actual_splits)


def train_f5_models(
    *,
    training_data: pd.DataFrame | str | Path,
    output_dir: str | Path = DEFAULT_MODEL_OUTPUT_DIR,
    holdout_season: int | None = None,
    search_space: Mapping[str, Sequence[float | int]] = DEFAULT_SEARCH_SPACE,
    time_series_splits: int = DEFAULT_TIME_SERIES_SPLITS,
    search_iterations: int = DEFAULT_SEARCH_ITERATIONS,
    random_state: int = DEFAULT_RANDOM_STATE,
    top_feature_count: int = DEFAULT_TOP_FEATURE_COUNT,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
) -> TrainingRunResult:
    """Train and persist versioned XGBoost models for F5 moneyline and run line outcomes."""

    dataset = _load_training_dataframe(training_data)
    feature_columns = _resolve_numeric_feature_columns(dataset)
    if not feature_columns:
        raise ValueError("Training data does not contain any numeric feature columns")

    effective_holdout_season = _resolve_holdout_season(dataset, holdout_season)
    data_version_hash = _resolve_data_version_hash(dataset)
    model_version = _build_model_version(data_version_hash)

    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, ModelTrainingArtifact] = {}
    for spec in _MODEL_SPECS:
        logger.info(
            "Training %s for holdout season %s with %s search iterations and %s time-series splits",
            str(spec["model_name"]),
            effective_holdout_season,
            _resolve_search_iterations(search_space, search_iterations),
            min(time_series_splits, max(len(dataset) - 1, 1)),
        )
        artifact = _train_single_model(
            dataset=dataset,
            model_name=str(spec["model_name"]),
            target_column=str(spec["target_column"]),
            drop_ties=bool(spec["drop_ties"]),
            feature_columns=feature_columns,
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

    summary_path = resolved_output_dir / f"training_run_{model_version}.json"
    summary_payload = {
        "model_version": model_version,
        "data_version_hash": data_version_hash,
        "holdout_season": effective_holdout_season,
        "feature_columns": feature_columns,
        "promoted_variants": {
            name: artifact.promoted_variant for name, artifact in artifacts.items()
        },
        "models": {name: _artifact_to_json_ready(artifact) for name, artifact in artifacts.items()},
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return TrainingRunResult(
        model_version=model_version,
        data_version_hash=data_version_hash,
        holdout_season=effective_holdout_season,
        feature_columns=feature_columns,
        summary_path=summary_path,
        models=artifacts,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Train both F5 XGBoost models from the persisted training parquet."""

    parser = argparse.ArgumentParser(description="Train F5 XGBoost models with temporal CV")
    parser.add_argument("--training-data", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_MODEL_OUTPUT_DIR))
    parser.add_argument("--experiment-name")
    parser.add_argument("--holdout-season", type=int, default=2025)
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--refresh-training-data", action="store_true")
    parser.add_argument("--allow-backfill-years", action="store_true")
    parser.add_argument("--time-series-splits", type=int, default=DEFAULT_TIME_SERIES_SPLITS)
    parser.add_argument("--search-iterations", type=int, default=DEFAULT_SEARCH_ITERATIONS)
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

    result = train_f5_models(
        training_data=training_path,
        output_dir=resolved_output_dir,
        holdout_season=args.holdout_season,
        time_series_splits=args.time_series_splits,
        search_iterations=args.search_iterations,
        random_state=args.random_state,
        early_stopping_rounds=args.early_stopping_rounds,
        validation_fraction=args.validation_fraction,
    )
    log_training_run(
        result,
        experiment_name=args.experiment_name,
        output_dir=resolved_output_dir,
        training_data=training_path,
        start_year=args.start_year,
        end_year=args.end_year,
        refresh_training_data=args.refresh_training_data,
        allow_backfill_years=args.allow_backfill_years,
        search_iterations=args.search_iterations,
        time_series_splits=args.time_series_splits,
    )
    print(json.dumps(_run_result_to_json_ready(result), indent=2))
    return 0


def _train_single_model(
    *,
    dataset: pd.DataFrame,
    model_name: str,
    target_column: str,
    drop_ties: bool,
    feature_columns: list[str],
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
) -> ModelTrainingArtifact:
    frame = _prepare_training_frame(dataset, target_column=target_column, drop_ties=drop_ties)
    train_frame = frame.loc[frame["season"] < holdout_season].copy()
    holdout_frame = frame.loc[frame["season"] == holdout_season].copy()
    if train_frame.empty:
        raise ValueError(f"No training rows found before holdout season {holdout_season}")
    if holdout_frame.empty:
        raise ValueError(f"No holdout rows found for season {holdout_season}")

    (
        best_params,
        cv_best_log_loss,
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
    searched_estimator = _build_estimator(random_state=random_state)
    searched_estimator.set_params(**best_params)
    (
        best_estimator,
        requested_n_estimators,
        final_n_estimators,
        best_iteration,
        early_stopping_train_row_count,
        early_stopping_validation_row_count,
    ) = _refit_estimator_with_temporal_early_stopping(
        estimator=searched_estimator,
        training_frame=train_frame,
        feature_columns=feature_columns,
        target_column=target_column,
        early_stopping_rounds=early_stopping_rounds,
        validation_fraction=validation_fraction,
    )
    holdout_probabilities = best_estimator.predict_proba(holdout_frame[feature_columns])[:, 1]
    holdout_predictions = best_estimator.predict(holdout_frame[feature_columns])

    holdout_metrics = {
        "accuracy": float(accuracy_score(holdout_frame[target_column], holdout_predictions)),
        "log_loss": float(
            log_loss(holdout_frame[target_column], holdout_probabilities, labels=[0, 1])
        ),
        "roc_auc": _safe_roc_auc(holdout_frame[target_column], holdout_probabilities),
    }
    feature_importance_rankings = _extract_feature_importance_rankings(
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
        "feature_columns": feature_columns,
        "best_params": best_params,
        "runtime_versions": collect_runtime_versions(),
        "search_backend": "optuna",
        "cv_best_log_loss": cv_best_log_loss,
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
        "promoted_variant": "base",
        "promotion_reason": build_promotion_reason(
            promoted_variant="base",
            metrics_by_variant={
                "base": {
                    "log_loss": holdout_metrics["log_loss"],
                    "roc_auc": holdout_metrics["roc_auc"],
                    "accuracy": holdout_metrics["accuracy"],
                    "brier": None,
                }
            },
        ),
        "feature_importance_rankings": feature_importance_rankings,
        "search_space": {key: list(values) for key, values in search_space.items()},
        "time_series_splits": resolved_time_series_splits,
        "trained_at": datetime.now(UTC).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

    return ModelTrainingArtifact(
        model_name=model_name,
        target_column=target_column,
        model_version=model_version,
        model_path=model_path,
        metadata_path=metadata_path,
        best_params=best_params,
        cv_best_log_loss=cv_best_log_loss,
        holdout_metrics=holdout_metrics,
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
        promoted_variant="base",
        promotion_reason=build_promotion_reason(
            promoted_variant="base",
            metrics_by_variant={
                "base": {
                    "log_loss": holdout_metrics["log_loss"],
                    "roc_auc": holdout_metrics["roc_auc"],
                    "accuracy": holdout_metrics["accuracy"],
                    "brier": None,
                }
            },
        ),
    )


def _build_estimator(*, random_state: int) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        tree_method="hist",
        device=DEFAULT_XGBOOST_DEVICE,
        n_jobs=1 if DEFAULT_XGBOOST_DEVICE != "cpu" else DEFAULT_XGBOOST_N_JOBS,
        verbosity=0,
    )


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
                lambda trial: _objective_log_loss(
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


def _objective_log_loss(
    trial: optuna.trial.Trial,
    *,
    train_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    search_space: Mapping[str, Sequence[float | int]],
    splitter: TimeSeriesSplit,
    random_state: int,
) -> float:
    params = _suggest_optuna_params(trial, search_space=search_space)
    feature_frame = train_frame[list(feature_columns)]
    target_series = train_frame[target_column]
    fold_losses: list[float] = []

    for fold_index, (train_indices, test_indices) in enumerate(
        splitter.split(train_frame), start=1
    ):
        estimator = _build_estimator(random_state=random_state)
        estimator.set_params(**params)
        estimator.fit(
            feature_frame.iloc[train_indices],
            target_series.iloc[train_indices],
        )
        probabilities = estimator.predict_proba(feature_frame.iloc[test_indices])[:, 1]
        fold_loss = float(
            log_loss(
                target_series.iloc[test_indices],
                probabilities,
                labels=[0, 1],
            )
        )
        fold_losses.append(fold_loss)
        trial.report(sum(fold_losses) / len(fold_losses), step=fold_index)
        if trial.should_prune():
            raise optuna.TrialPruned(
                f"Pruned at fold {fold_index} with mean log loss {sum(fold_losses) / len(fold_losses):.6f}"
            )

    return float(sum(fold_losses) / len(fold_losses))


def _suggest_optuna_params(
    trial: optuna.trial.Trial,
    *,
    search_space: Mapping[str, Sequence[float | int]],
) -> dict[str, float | int]:
    if _matches_default_search_space(search_space):
        return {
            "max_depth": int(trial.suggest_int("max_depth", 3, 8)),
            "n_estimators": int(trial.suggest_int("n_estimators", 100, 500, step=50)),
            "learning_rate": float(trial.suggest_float("learning_rate", 0.005, 0.1, log=True)),
            "subsample": float(trial.suggest_float("subsample", 0.6, 1.0)),
            "colsample_bytree": float(trial.suggest_float("colsample_bytree", 0.5, 1.0)),
            "min_child_weight": int(trial.suggest_int("min_child_weight", 1, 10)),
            "gamma": float(trial.suggest_float("gamma", 0.0, 5.0)),
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


def _complete_optuna_params(
    best_trial_params: Mapping[str, Any],
    *,
    search_space: Mapping[str, Sequence[float | int]],
) -> dict[str, float | int]:
    completed: dict[str, Any] = dict(best_trial_params)
    for key, values in search_space.items():
        options = list(values)
        if len(options) == 1 and key not in completed:
            completed[key] = options[0]
    return _normalize_best_params(completed)


def _matches_default_search_space(search_space: Mapping[str, Sequence[float | int]]) -> bool:
    if set(search_space) != set(DEFAULT_SEARCH_SPACE):
        return False
    for key, values in DEFAULT_SEARCH_SPACE.items():
        if list(search_space[key]) != list(values):
            return False
    return True


def _build_optuna_study_name(
    *,
    model_name: str,
    target_column: str,
    data_version_hash: str,
    holdout_season: int,
) -> str:
    return f"{model_name}_{target_column}_{data_version_hash[:12]}_{holdout_season}"


def _build_optuna_progress_callback(
    *,
    progress: Progress,
    task_id: int,
    model_name: str,
    target_trial_count: int,
) -> Any:
    last_logged_bucket = -1

    def _callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        nonlocal last_logged_bucket
        completed_trial_count = len(study.trials)
        progress_fraction = min(1.0, completed_trial_count / max(target_trial_count, 1))
        current_bucket = min(10, int(progress_fraction * 10))
        if (
            completed_trial_count != target_trial_count
            and current_bucket <= 0
        ):
            return
        if current_bucket <= last_logged_bucket and completed_trial_count != target_trial_count:
            return

        if completed_trial_count == target_trial_count:
            current_bucket = 10
        last_logged_bucket = current_bucket
        try:
            best_value = study.best_value
        except ValueError:
            best_value = None
        latest_value = trial.value if trial.value is not None else None
        progress.update(
            task_id,
            completed=current_bucket,
            best="n/a" if best_value is None else f"{best_value:.6f}",
            latest="pruned" if latest_value is None else f"{latest_value:.6f}",
            state=trial.state.name.lower(),
        )

    return _callback


def _refit_estimator_with_temporal_early_stopping(
    *,
    estimator: XGBClassifier,
    training_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    early_stopping_rounds: int,
    validation_fraction: float,
) -> tuple[XGBClassifier, int, int, int | None, int, int]:
    requested_n_estimators = int(estimator.get_params().get("n_estimators", 100))
    train_slice, validation_slice = _split_temporal_validation_frame(
        training_frame,
        validation_fraction=validation_fraction,
    )
    if early_stopping_rounds <= 0 or validation_slice.empty:
        fitted_estimator = clone(estimator)
        fitted_estimator.fit(training_frame[list(feature_columns)], training_frame[target_column])
        return (
            fitted_estimator,
            requested_n_estimators,
            requested_n_estimators,
            None,
            int(len(training_frame)),
            0,
        )

    early_stop_estimator = clone(estimator)
    early_stop_estimator.set_params(early_stopping_rounds=int(early_stopping_rounds))
    early_stop_estimator.fit(
        train_slice[list(feature_columns)],
        train_slice[target_column],
        eval_set=[(validation_slice[list(feature_columns)], validation_slice[target_column])],
        verbose=False,
    )
    best_iteration = getattr(early_stop_estimator, "best_iteration", None)
    final_n_estimators = requested_n_estimators
    if best_iteration is not None:
        final_n_estimators = max(1, int(best_iteration) + 1)

    final_estimator = clone(estimator)
    final_estimator.set_params(n_estimators=final_n_estimators)
    final_estimator.fit(training_frame[list(feature_columns)], training_frame[target_column])
    return (
        final_estimator,
        requested_n_estimators,
        final_n_estimators,
        None if best_iteration is None else int(best_iteration),
        int(len(train_slice)),
        int(len(validation_slice)),
    )


def _split_temporal_validation_frame(
    dataframe: pd.DataFrame,
    *,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataframe.empty or validation_fraction <= 0:
        return dataframe.copy(), dataframe.iloc[0:0].copy()
    validation_row_count = min(
        max(1, int(round(len(dataframe) * validation_fraction))),
        max(len(dataframe) - 1, 0),
    )
    if validation_row_count <= 0:
        return dataframe.copy(), dataframe.iloc[0:0].copy()
    split_index = len(dataframe) - validation_row_count
    return (
        dataframe.iloc[:split_index].copy().reset_index(drop=True),
        dataframe.iloc[split_index:].copy().reset_index(drop=True),
    )


def _load_training_dataframe(training_data: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(training_data, pd.DataFrame):
        dataframe = training_data.copy()
    else:
        dataframe = pd.read_parquet(Path(training_data))

    if dataframe.empty:
        raise ValueError("Training data is empty")

    dataframe = dataframe.copy()
    dataframe["scheduled_start"] = pd.to_datetime(dataframe["scheduled_start"], utc=True)
    dataframe["season"] = pd.to_numeric(dataframe["season"], errors="raise").astype(int)
    return dataframe.sort_values(["scheduled_start", "game_pk"]).reset_index(drop=True)


def _resolve_numeric_feature_columns(dataframe: pd.DataFrame) -> list[str]:
    return [
        column
        for column in _feature_columns(dataframe)
        if pd.api.types.is_numeric_dtype(dataframe[column])
        and _has_numeric_variation(dataframe[column])
    ]


def _has_numeric_variation(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    first_value = non_null.iloc[0]
    return bool((non_null != first_value).any())


def _prepare_training_frame(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    drop_ties: bool,
) -> pd.DataFrame:
    frame = dataframe.copy()
    if drop_ties and "f5_tied_after_5" in frame.columns:
        frame = frame.loc[pd.to_numeric(frame["f5_tied_after_5"], errors="coerce").fillna(0) == 0]

    frame[target_column] = pd.to_numeric(frame[target_column], errors="raise").astype(int)
    return frame.sort_values(["scheduled_start", "game_pk"]).reset_index(drop=True)


def _resolve_holdout_season(dataframe: pd.DataFrame, requested_holdout_season: int | None) -> int:
    available_seasons = sorted(
        int(season) for season in dataframe["season"].dropna().unique().tolist()
    )
    if not available_seasons:
        raise ValueError("Training data does not contain any seasons")
    if requested_holdout_season is None:
        return available_seasons[-1]
    if requested_holdout_season not in available_seasons:
        raise ValueError(
            f"Requested holdout season {requested_holdout_season} not present in training data"
        )
    return requested_holdout_season


def _resolve_data_version_hash(dataframe: pd.DataFrame) -> str:
    if "data_version_hash" in dataframe.columns:
        non_null_hashes = dataframe["data_version_hash"].dropna().astype(str).unique().tolist()
        if len(non_null_hashes) == 1:
            return non_null_hashes[0]
    return _compute_data_version_hash(dataframe)


def _build_model_version(data_version_hash: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{data_version_hash[:8]}"


def _resolve_experiment_output_dir(
    output_dir: str | Path,
    experiment_name: str | None,
) -> Path:
    resolved_output_dir = Path(output_dir)
    if not experiment_name:
        return resolved_output_dir

    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", experiment_name.strip().lower()).strip("-.")
    if not slug:
        raise ValueError("experiment_name must contain at least one alphanumeric character")
    return resolved_output_dir / slug


def _resolve_search_iterations(
    search_space: Mapping[str, Sequence[float | int]],
    requested_iterations: int,
) -> int:
    if _matches_default_search_space(search_space):
        return max(1, int(requested_iterations))
    total_combinations = prod(max(len(values), 1) for values in search_space.values())
    return max(1, min(requested_iterations, total_combinations))


def _normalize_best_params(best_params: Mapping[str, Any]) -> dict[str, float | int]:
    normalized: dict[str, float | int] = {}
    for key, value in best_params.items():
        if isinstance(value, bool):
            normalized[key] = int(value)
        elif isinstance(value, int):
            normalized[key] = int(value)
        else:
            normalized[key] = float(value)
    return normalized


def _extract_feature_importance_rankings(
    estimator: XGBClassifier,
    feature_columns: Sequence[str],
    *,
    top_feature_count: int,
) -> list[dict[str, float | str]]:
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return []

    rankings = [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in zip(feature_columns, importances, strict=False)
    ]
    rankings.sort(key=lambda item: float(item["importance"]), reverse=True)
    return rankings[:top_feature_count]


def _safe_roc_auc(y_true: pd.Series, probabilities: Sequence[float]) -> float | None:
    if pd.Series(y_true).nunique() < 2:
        return None
    return float(roc_auc_score(y_true, probabilities))


def _artifact_to_json_ready(artifact: ModelTrainingArtifact) -> dict[str, Any]:
    payload = asdict(artifact)
    payload["model_path"] = str(artifact.model_path)
    payload["metadata_path"] = str(artifact.metadata_path)
    return payload


def _run_result_to_json_ready(result: TrainingRunResult) -> dict[str, Any]:
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
