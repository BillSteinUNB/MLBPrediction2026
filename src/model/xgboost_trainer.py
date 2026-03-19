from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import prod
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

from src.model.data_builder import (
    DEFAULT_OUTPUT_PATH,
    _compute_data_version_hash,
    _feature_columns,
    build_training_dataset,
)


logger = logging.getLogger(__name__)

DEFAULT_MODEL_OUTPUT_DIR = Path("data") / "models"
DEFAULT_TIME_SERIES_SPLITS = 5
DEFAULT_SEARCH_ITERATIONS = 15
DEFAULT_RANDOM_STATE = 2026
DEFAULT_TOP_FEATURE_COUNT = 25
DEFAULT_SEARCH_SPACE: dict[str, list[float | int]] = {
    "max_depth": [3, 4, 5, 6, 7, 8],
    "n_estimators": [100, 200, 300, 400, 500],
    "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1],
}
_MODEL_SPECS = (
    {"model_name": "f5_ml_model", "target_column": "f5_ml_result", "drop_ties": True},
    {"model_name": "f5_rl_model", "target_column": "f5_rl_result", "drop_ties": False},
)


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
        )
        artifacts[artifact.model_name] = artifact

    summary_path = resolved_output_dir / f"training_run_{model_version}.json"
    summary_payload = {
        "model_version": model_version,
        "data_version_hash": data_version_hash,
        "holdout_season": effective_holdout_season,
        "feature_columns": feature_columns,
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
    parser.add_argument("--holdout-season", type=int, default=2025)
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--refresh-training-data", action="store_true")
    parser.add_argument("--time-series-splits", type=int, default=DEFAULT_TIME_SERIES_SPLITS)
    parser.add_argument("--search-iterations", type=int, default=DEFAULT_SEARCH_ITERATIONS)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    training_path = Path(args.training_data)
    if args.refresh_training_data or not training_path.exists():
        logger.info("Building training data at %s", training_path)
        build_training_dataset(
            start_year=args.start_year,
            end_year=args.end_year,
            output_path=training_path,
            refresh=args.refresh_training_data,
        )

    result = train_f5_models(
        training_data=training_path,
        output_dir=args.output_dir,
        holdout_season=args.holdout_season,
        time_series_splits=args.time_series_splits,
        search_iterations=args.search_iterations,
        random_state=args.random_state,
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
) -> ModelTrainingArtifact:
    frame = _prepare_training_frame(dataset, target_column=target_column, drop_ties=drop_ties)
    train_frame = frame.loc[frame["season"] < holdout_season].copy()
    holdout_frame = frame.loc[frame["season"] == holdout_season].copy()
    if train_frame.empty:
        raise ValueError(f"No training rows found before holdout season {holdout_season}")
    if holdout_frame.empty:
        raise ValueError(f"No holdout rows found for season {holdout_season}")

    search = RandomizedSearchCV(
        estimator=_build_estimator(random_state=random_state),
        param_distributions={key: list(values) for key, values in search_space.items()},
        n_iter=_resolve_search_iterations(search_space, search_iterations),
        scoring="neg_log_loss",
        cv=create_time_series_split(
            row_count=len(train_frame),
            requested_splits=time_series_splits,
        ),
        random_state=random_state,
        refit=True,
        n_jobs=1,
    )

    search.fit(train_frame[feature_columns], train_frame[target_column])
    best_estimator: XGBClassifier = search.best_estimator_
    holdout_probabilities = best_estimator.predict_proba(holdout_frame[feature_columns])[:, 1]
    holdout_predictions = best_estimator.predict(holdout_frame[feature_columns])

    holdout_metrics = {
        "accuracy": float(accuracy_score(holdout_frame[target_column], holdout_predictions)),
        "log_loss": float(log_loss(holdout_frame[target_column], holdout_probabilities, labels=[0, 1])),
        "roc_auc": _safe_roc_auc(holdout_frame[target_column], holdout_probabilities),
    }
    best_params = _normalize_best_params(search.best_params_)
    cv_best_log_loss = float(-search.best_score_)
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
        "cv_best_log_loss": cv_best_log_loss,
        "holdout_metrics": holdout_metrics,
        "feature_importance_rankings": feature_importance_rankings,
        "search_space": {key: list(values) for key, values in search_space.items()},
        "time_series_splits": int(search.cv.n_splits),
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
    )


def _build_estimator(*, random_state: int) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        tree_method="hist",
        n_jobs=1,
        verbosity=0,
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
    ]


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
    available_seasons = sorted(int(season) for season in dataframe["season"].dropna().unique().tolist())
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


def _resolve_search_iterations(
    search_space: Mapping[str, Sequence[float | int]],
    requested_iterations: int,
) -> int:
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
        "models": {name: _artifact_to_json_ready(artifact) for name, artifact in result.models.items()},
    }


if __name__ == "__main__":
    raise SystemExit(main())
