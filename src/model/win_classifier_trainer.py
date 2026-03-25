from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

from src.clients.weather_client import fetch_game_weather
from src.model.artifact_runtime import collect_runtime_versions
from src.model.data_builder import DEFAULT_OUTPUT_PATH, build_training_dataset
from src.model.xgboost_trainer import (
    DEFAULT_EARLY_STOPPING_ROUNDS,
    DEFAULT_MODEL_OUTPUT_DIR,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TIME_SERIES_SPLITS,
    DEFAULT_TOP_FEATURE_COUNT,
    DEFAULT_VALIDATION_FRACTION,
    DEFAULT_XGBOOST_N_JOBS,
    _build_model_version,
    _build_optuna_progress_callback,
    _build_optuna_study_name,
    _complete_optuna_params,
    _extract_feature_importance_rankings,
    _load_training_dataframe,
    _normalize_best_params,
    _refit_estimator_with_temporal_early_stopping,
    _resolve_data_version_hash,
    _resolve_experiment_output_dir,
    _resolve_holdout_season,
    _resolve_numeric_feature_columns,
    _resolve_search_iterations,
    _safe_roc_auc,
    create_time_series_split,
)


logger = logging.getLogger(__name__)

DEFAULT_WIN_CLASSIFIER_SEARCH_ITERATIONS = 100
DEFAULT_WIN_CLASSIFIER_MAX_FEATURE_COUNT = 80
DEFAULT_CALIBRATION_CV = 5
_WIN_CLASSIFIER_INTERNAL_TARGET_COLUMN = "__home_win_target__"
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

DEFAULT_CLASSIFIER_SEARCH_SPACE: dict[str, list[float | int]] = {
    "max_depth": [3, 4, 5, 6],
    "n_estimators": [100, 200, 300, 400, 500],
    "learning_rate": [0.01, 0.02, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0.0, 0.1, 0.3, 0.5],
    "reg_alpha": [1e-3, 0.01, 0.1, 1.0],
    "reg_lambda": [0.1, 0.5, 1.0, 5.0],
    "scale_pos_weight": [1.0],
}

DEFAULT_WIN_CLASSIFIER_SPECS: tuple[dict[str, str], ...] = (
    {
        "model_name": "full_game_win_model",
        "home_target": "final_home_score",
        "away_target": "final_away_score",
    },
    {
        "model_name": "f5_win_model",
        "home_target": "f5_home_score",
        "away_target": "f5_away_score",
    },
)


@dataclass(frozen=True, slots=True)
class WinClassifierTrainingArtifact:
    model_name: str
    home_target_column: str
    away_target_column: str
    model_version: str
    model_path: Path
    metadata_path: Path
    best_params: dict[str, float | int]
    cv_best_log_loss: float
    holdout_metrics: dict[str, float | None]
    feature_columns: list[str]
    feature_importance_rankings: list[dict[str, float | str]]
    train_row_count: int
    holdout_row_count: int
    dropped_tie_row_count: int
    holdout_season: int
    requested_n_estimators: int
    final_n_estimators: int
    best_iteration: int | None
    early_stopping_rounds: int
    validation_fraction: float
    early_stopping_train_row_count: int
    early_stopping_validation_row_count: int
    calibration_method: str
    calibration_cv: int


@dataclass(frozen=True, slots=True)
class WinClassifierTrainingResult:
    model_version: str
    data_version_hash: str
    holdout_season: int
    feature_columns: list[str]
    summary_path: Path
    models: dict[str, WinClassifierTrainingArtifact]


def train_win_classifiers(
    *,
    training_data: pd.DataFrame | str | Path,
    output_dir: str | Path = DEFAULT_MODEL_OUTPUT_DIR,
    holdout_season: int | None = None,
    search_space: Mapping[str, Sequence[float | int]] = DEFAULT_CLASSIFIER_SEARCH_SPACE,
    time_series_splits: int = DEFAULT_TIME_SERIES_SPLITS,
    search_iterations: int = DEFAULT_WIN_CLASSIFIER_SEARCH_ITERATIONS,
    random_state: int = DEFAULT_RANDOM_STATE,
    max_feature_count: int = DEFAULT_WIN_CLASSIFIER_MAX_FEATURE_COUNT,
    top_feature_count: int = DEFAULT_TOP_FEATURE_COUNT,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
) -> WinClassifierTrainingResult:
    """Train and persist calibrated win-probability classifiers for MLB moneyline outcomes."""

    dataset = _load_training_dataframe(training_data)
    candidate_feature_columns = _resolve_win_classifier_candidate_feature_columns(dataset)
    if not candidate_feature_columns:
        raise ValueError("Training data does not contain any numeric feature columns")

    effective_holdout_season = _resolve_holdout_season(dataset, holdout_season)
    data_version_hash = _resolve_data_version_hash(dataset)
    model_version = _build_model_version(data_version_hash)

    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, WinClassifierTrainingArtifact] = {}
    for spec in DEFAULT_WIN_CLASSIFIER_SPECS:
        model_name = str(spec["model_name"])
        home_target_column = str(spec["home_target"])
        away_target_column = str(spec["away_target"])
        logger.info(
            "Training %s for holdout season %s with %s search iterations and %s time-series splits",
            model_name,
            effective_holdout_season,
            _resolve_search_iterations(search_space, search_iterations),
            min(time_series_splits, max(len(dataset) - 1, 1)),
        )
        artifact = _train_single_classifier(
            dataset=dataset,
            model_name=model_name,
            home_target_column=home_target_column,
            away_target_column=away_target_column,
            candidate_feature_columns=candidate_feature_columns,
            output_dir=resolved_output_dir,
            model_version=model_version,
            holdout_season=effective_holdout_season,
            search_space=search_space,
            time_series_splits=time_series_splits,
            search_iterations=search_iterations,
            random_state=random_state,
            max_feature_count=max_feature_count,
            top_feature_count=top_feature_count,
            data_version_hash=data_version_hash,
            early_stopping_rounds=early_stopping_rounds,
            validation_fraction=validation_fraction,
        )
        artifacts[artifact.model_name] = artifact

    summary_path = resolved_output_dir / f"win_classifier_training_run_{model_version}.json"
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

    return WinClassifierTrainingResult(
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
    """Train the win-probability classifiers from the persisted training parquet."""

    parser = argparse.ArgumentParser(description="Train MLB win probability classifiers")
    parser.add_argument("--training-data", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--experiment-name")
    parser.add_argument("--holdout-season", type=int, default=2025)
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--refresh-training-data", action="store_true")
    parser.add_argument("--allow-backfill-years", action="store_true")
    parser.add_argument("--time-series-splits", type=int, default=DEFAULT_TIME_SERIES_SPLITS)
    parser.add_argument("--search-iterations", type=int, default=DEFAULT_WIN_CLASSIFIER_SEARCH_ITERATIONS)
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

    base_output_dir = DEFAULT_MODEL_OUTPUT_DIR if args.output_dir is None else Path(args.output_dir)
    resolved_output_dir = _resolve_experiment_output_dir(base_output_dir, args.experiment_name)
    result = train_win_classifiers(
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


def _train_single_classifier(
    *,
    dataset: pd.DataFrame,
    model_name: str,
    home_target_column: str,
    away_target_column: str,
    candidate_feature_columns: Sequence[str],
    output_dir: Path,
    model_version: str,
    holdout_season: int,
    search_space: Mapping[str, Sequence[float | int]],
    time_series_splits: int,
    search_iterations: int,
    random_state: int,
    max_feature_count: int,
    top_feature_count: int,
    data_version_hash: str,
    early_stopping_rounds: int,
    validation_fraction: float,
) -> WinClassifierTrainingArtifact:
    frame, dropped_tie_row_count = _prepare_win_classifier_frame(
        dataset,
        home_target_column=home_target_column,
        away_target_column=away_target_column,
    )
    train_frame = frame.loc[frame["season"] < holdout_season].copy()
    holdout_frame = frame.loc[frame["season"] == holdout_season].copy()
    if train_frame.empty:
        raise ValueError(f"No training rows found before holdout season {holdout_season}")
    if holdout_frame.empty:
        raise ValueError(f"No holdout rows found for season {holdout_season}")

    feature_columns = _select_win_classifier_feature_columns(
        train_frame,
        target_column=_WIN_CLASSIFIER_INTERNAL_TARGET_COLUMN,
        candidate_feature_columns=candidate_feature_columns,
        max_feature_count=max_feature_count,
    )
    if not feature_columns:
        raise ValueError(f"No win-classifier features selected for {model_name}")

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
        target_column=_WIN_CLASSIFIER_INTERNAL_TARGET_COLUMN,
        model_name=model_name,
        output_dir=output_dir,
        holdout_season=holdout_season,
        data_version_hash=data_version_hash,
        search_space=search_space,
        time_series_splits=time_series_splits,
        search_iterations=search_iterations,
        random_state=random_state,
    )
    base_estimator = _build_classifier(random_state=random_state)
    base_estimator.set_params(**best_params)
    (
        best_estimator,
        requested_n_estimators,
        final_n_estimators,
        best_iteration,
        early_stopping_train_row_count,
        early_stopping_validation_row_count,
    ) = _refit_estimator_with_temporal_early_stopping(
        estimator=base_estimator,
        training_frame=train_frame,
        feature_columns=feature_columns,
        target_column=_WIN_CLASSIFIER_INTERNAL_TARGET_COLUMN,
        early_stopping_rounds=early_stopping_rounds,
        validation_fraction=validation_fraction,
    )

    calibration_cv = _resolve_calibration_cv(
        train_frame[_WIN_CLASSIFIER_INTERNAL_TARGET_COLUMN],
        requested_cv=DEFAULT_CALIBRATION_CV,
    )
    calibrated_model = CalibratedClassifierCV(
        best_estimator,
        method="isotonic",
        cv=calibration_cv,
    )
    calibrated_model.fit(
        train_frame[list(feature_columns)],
        train_frame[_WIN_CLASSIFIER_INTERNAL_TARGET_COLUMN],
    )

    holdout_probabilities = calibrated_model.predict_proba(holdout_frame[list(feature_columns)])[:, 1]
    holdout_metrics = _compute_classifier_holdout_metrics(
        actual=holdout_frame[_WIN_CLASSIFIER_INTERNAL_TARGET_COLUMN],
        predicted_probabilities=holdout_probabilities,
    )
    feature_importance_rankings = _extract_feature_importance_rankings(
        best_estimator,
        feature_columns,
        top_feature_count=top_feature_count,
    )

    logger.info("%s dropped %s tied rows before training", model_name, dropped_tie_row_count)
    logger.info("%s best hyperparameters: %s", model_name, best_params)
    logger.info("%s holdout metrics: %s", model_name, holdout_metrics)

    model_path = output_dir / f"{model_name}_{model_version}.joblib"
    joblib.dump(calibrated_model, model_path)
    metadata_path = model_path.with_suffix(".metadata.json")
    metadata_payload = {
        "model_name": model_name,
        "home_target_column": home_target_column,
        "away_target_column": away_target_column,
        "target_definition": f"{home_target_column} > {away_target_column}",
        "model_version": model_version,
        "data_version_hash": data_version_hash,
        "holdout_season": holdout_season,
        "train_row_count": int(len(train_frame)),
        "holdout_row_count": int(len(holdout_frame)),
        "dropped_tie_row_count": int(dropped_tie_row_count),
        "feature_columns": list(feature_columns),
        "best_params": best_params,
        "runtime_versions": collect_runtime_versions(),
        "model_family": "xgboost_classifier_calibrated",
        "search_backend": "optuna",
        "cv_best_log_loss": cv_best_log_loss,
        "optuna_best_trial_number": optuna_best_trial_number,
        "optuna_trial_count": optuna_trial_count,
        "optuna_study_name": optuna_study_name,
        "optuna_storage_path": str(optuna_storage_path),
        "calibration_method": "isotonic",
        "calibration_cv": int(calibration_cv),
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

    artifact = WinClassifierTrainingArtifact(
        model_name=model_name,
        home_target_column=home_target_column,
        away_target_column=away_target_column,
        model_version=model_version,
        model_path=model_path,
        metadata_path=metadata_path,
        best_params=best_params,
        cv_best_log_loss=cv_best_log_loss,
        holdout_metrics=holdout_metrics,
        feature_columns=list(feature_columns),
        feature_importance_rankings=feature_importance_rankings,
        train_row_count=len(train_frame),
        holdout_row_count=len(holdout_frame),
        dropped_tie_row_count=dropped_tie_row_count,
        holdout_season=holdout_season,
        requested_n_estimators=requested_n_estimators,
        final_n_estimators=final_n_estimators,
        best_iteration=best_iteration,
        early_stopping_rounds=int(early_stopping_rounds),
        validation_fraction=float(validation_fraction),
        early_stopping_train_row_count=early_stopping_train_row_count,
        early_stopping_validation_row_count=early_stopping_validation_row_count,
        calibration_method="isotonic",
        calibration_cv=calibration_cv,
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


def _prepare_win_classifier_frame(
    dataframe: pd.DataFrame,
    *,
    home_target_column: str,
    away_target_column: str,
) -> tuple[pd.DataFrame, int]:
    frame = dataframe.copy()
    frame[_WIN_CLASSIFIER_INTERNAL_TARGET_COLUMN] = _derive_win_target(
        frame,
        home_col=home_target_column,
        away_col=away_target_column,
    )
    dropped_tie_row_count = int(frame[_WIN_CLASSIFIER_INTERNAL_TARGET_COLUMN].isna().sum())
    frame = frame.loc[frame[_WIN_CLASSIFIER_INTERNAL_TARGET_COLUMN].notna()].copy()
    frame[_WIN_CLASSIFIER_INTERNAL_TARGET_COLUMN] = (
        pd.to_numeric(frame[_WIN_CLASSIFIER_INTERNAL_TARGET_COLUMN], errors="raise").astype(int)
    )
    return frame.sort_values(["scheduled_start", "game_pk"]).reset_index(drop=True), dropped_tie_row_count


def _derive_win_target(
    dataframe: pd.DataFrame,
    home_col: str,
    away_col: str,
) -> pd.Series:
    home = pd.to_numeric(dataframe[home_col], errors="coerce")
    away = pd.to_numeric(dataframe[away_col], errors="coerce")
    result = (home > away).astype(float)
    result[(home == away) | home.isna() | away.isna()] = float("nan")
    return result


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
            lambda trial: _optuna_objective(
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


def _optuna_objective(
    trial: optuna.trial.Trial,
    *,
    train_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    search_space: Mapping[str, Sequence[float | int]],
    splitter: Any,
    random_state: int,
) -> float:
    params = _suggest_optuna_classifier_params(trial, search_space=search_space)
    feature_frame = train_frame[list(feature_columns)]
    target_series = train_frame[target_column]
    fold_losses: list[float] = []

    for fold_index, (train_indices, test_indices) in enumerate(splitter.split(train_frame), start=1):
        estimator = _build_classifier(random_state=random_state)
        estimator.set_params(**params)
        estimator.fit(
            feature_frame.iloc[train_indices],
            target_series.iloc[train_indices],
        )
        fold_probabilities = estimator.predict_proba(feature_frame.iloc[test_indices])[:, 1]
        fold_loss = float(
            log_loss(
                target_series.iloc[test_indices],
                fold_probabilities,
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


def _suggest_optuna_classifier_params(
    trial: optuna.trial.Trial,
    *,
    search_space: Mapping[str, Sequence[float | int]],
) -> dict[str, float | int]:
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


def _compute_classifier_holdout_metrics(
    *,
    actual: Sequence[int] | pd.Series,
    predicted_probabilities: Sequence[float],
) -> dict[str, float | None]:
    actual_series = pd.Series(actual, dtype=int).reset_index(drop=True)
    probability_series = pd.Series(predicted_probabilities, dtype=float).clip(lower=0.0, upper=1.0)
    predicted_classes = (probability_series >= 0.5).astype(int)
    naive_accuracy = float(max(actual_series.mean(), 1.0 - actual_series.mean()))
    accuracy = float(accuracy_score(actual_series, predicted_classes))
    accuracy_improvement_vs_naive_pct = (
        ((accuracy - naive_accuracy) / naive_accuracy) * 100.0 if naive_accuracy > 0 else None
    )
    return {
        "accuracy": accuracy,
        "log_loss": float(log_loss(actual_series, probability_series, labels=[0, 1])),
        "brier_score": float(brier_score_loss(actual_series, probability_series)),
        "auc_roc": _safe_roc_auc(actual_series, probability_series),
        "naive_accuracy": naive_accuracy,
        "accuracy_improvement_vs_naive_pct": accuracy_improvement_vs_naive_pct,
        "actual_home_win_rate": float(actual_series.mean()),
        "predicted_home_win_rate": float(probability_series.mean()),
        "calibration_slope": _compute_calibration_slope(actual_series, probability_series),
    }


def _compute_calibration_slope(
    actual: pd.Series,
    probabilities: pd.Series,
    *,
    bin_count: int = 10,
) -> float | None:
    if actual.nunique() < 2:
        return None
    if probabilities.nunique() < 2:
        return None

    resolved_bin_count = max(2, min(bin_count, len(probabilities)))
    observed_rate, predicted_rate = calibration_curve(
        actual,
        probabilities,
        n_bins=resolved_bin_count,
        strategy="quantile",
    )
    if len(predicted_rate) < 2:
        return None
    if np.isclose(np.std(predicted_rate), 0.0):
        return None
    slope, _ = np.polyfit(predicted_rate, observed_rate, deg=1)
    return float(slope)


def _resolve_calibration_cv(target_series: pd.Series, *, requested_cv: int) -> int:
    class_counts = pd.Series(target_series).value_counts()
    if class_counts.empty:
        raise ValueError("Calibration target is empty")
    min_class_count = int(class_counts.min())
    resolved_cv = min(int(requested_cv), min_class_count)
    if resolved_cv < 2:
        raise ValueError("Need at least two examples from each class for isotonic calibration")
    if resolved_cv != requested_cv:
        logger.info(
            "Reducing calibration CV from %s to %s based on class counts",
            requested_cv,
            resolved_cv,
        )
    return resolved_cv


def _build_classifier(*, random_state: int) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        tree_method="hist",
        n_jobs=DEFAULT_XGBOOST_N_JOBS,
        verbosity=0,
        use_label_encoder=False,
    )


def _resolve_win_classifier_candidate_feature_columns(dataframe: pd.DataFrame) -> list[str]:
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


def _select_win_classifier_feature_columns(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    candidate_feature_columns: Sequence[str],
    max_feature_count: int = DEFAULT_WIN_CLASSIFIER_MAX_FEATURE_COUNT,
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


def _artifact_to_json_ready(artifact: WinClassifierTrainingArtifact) -> dict[str, Any]:
    payload = asdict(artifact)
    payload["model_path"] = str(artifact.model_path)
    payload["metadata_path"] = str(artifact.metadata_path)
    return payload


def _run_result_to_json_ready(result: WinClassifierTrainingResult) -> dict[str, Any]:
    return {
        "model_version": result.model_version,
        "data_version_hash": result.data_version_hash,
        "holdout_season": result.holdout_season,
        "feature_columns": result.feature_columns,
        "summary_path": str(result.summary_path),
        "models": {name: _artifact_to_json_ready(artifact) for name, artifact in result.models.items()},
    }


if __name__ == "__main__":
    raise SystemExit(main())
