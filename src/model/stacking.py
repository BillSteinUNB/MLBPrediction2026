from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import cross_val_predict
from xgboost import XGBClassifier

from src.clients.weather_client import fetch_game_weather
from src.model.data_builder import DEFAULT_OUTPUT_PATH, build_training_dataset
from src.model.xgboost_trainer import (
    DEFAULT_MODEL_OUTPUT_DIR,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SEARCH_ITERATIONS,
    DEFAULT_SEARCH_SPACE,
    DEFAULT_TIME_SERIES_SPLITS,
    _load_training_dataframe,
    _prepare_training_frame,
    _resolve_holdout_season,
    _resolve_numeric_feature_columns,
    create_time_series_split,
    train_f5_models,
)


logger = logging.getLogger(__name__)

DEFAULT_RAW_META_FEATURE_COLUMNS: tuple[str, ...] = (
    "home_team_f5_pythagorean_wp_30g",
    "home_team_log5_30g",
    "park_runs_factor",
)
DEFAULT_META_LEARNER_MAX_ITER = 1_000
_STACKING_MODEL_SPECS = (
    {
        "model_name": "f5_ml_stacking_model",
        "base_model_name": "f5_ml_model",
        "target_column": "f5_ml_result",
        "drop_ties": True,
    },
    {
        "model_name": "f5_rl_stacking_model",
        "base_model_name": "f5_rl_model",
        "target_column": "f5_rl_result",
        "drop_ties": False,
    },
)


@dataclass(frozen=True, slots=True)
class StackingModelArtifact:
    model_name: str
    base_model_name: str
    target_column: str
    model_version: str
    model_path: Path
    metadata_path: Path
    base_model_path: Path
    holdout_metrics: dict[str, float | None]
    train_row_count: int
    oof_row_count: int
    holdout_row_count: int
    warmup_row_count: int
    raw_meta_feature_columns: list[str]
    meta_feature_columns: list[str]
    oof_prediction_strategy: str
    persisted: bool
    skip_reason: str | None


@dataclass(frozen=True, slots=True)
class StackingRunResult:
    model_version: str
    data_version_hash: str
    holdout_season: int
    feature_columns: list[str]
    raw_meta_feature_columns: list[str]
    summary_path: Path
    models: dict[str, StackingModelArtifact]


@dataclass(frozen=True, slots=True)
class _OOFPredictionResult:
    probabilities: pd.Series
    warmup_row_count: int
    fold_count: int


class _WarmupAugmentedClassifier(BaseEstimator, ClassifierMixin):
    """Fit the cloned estimator on warmup rows plus the fold-specific training rows."""

    def __init__(
        self,
        *,
        base_estimator: XGBClassifier,
        warmup_X: pd.DataFrame,
        warmup_y: pd.Series,
    ) -> None:
        self.base_estimator = base_estimator
        self.warmup_X = warmup_X
        self.warmup_y = warmup_y

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> _WarmupAugmentedClassifier:
        feature_names = list(self.warmup_X.columns)
        train_X = _coerce_feature_frame(X, feature_names=feature_names)
        train_y = _coerce_target_series(y)

        fit_X = pd.concat([self.warmup_X.reset_index(drop=True), train_X], ignore_index=True)
        fit_y = pd.concat([self.warmup_y.reset_index(drop=True), train_y], ignore_index=True)

        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(fit_X, fit_y)
        self.classes_ = getattr(self.estimator_, "classes_", np.array([0, 1]))
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.estimator_.predict_proba(_coerce_feature_frame(X, feature_names=list(self.warmup_X.columns)))

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.estimator_.predict(_coerce_feature_frame(X, feature_names=list(self.warmup_X.columns)))


class _PartitionedTemporalCV:
    """Temporal partition with contiguous test blocks for cross_val_predict."""

    def __init__(self, test_fold_sizes: Sequence[int]) -> None:
        self.test_fold_sizes = tuple(int(size) for size in test_fold_sizes if int(size) > 0)
        if not self.test_fold_sizes:
            raise ValueError("At least one positive test fold size is required")

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return len(self.test_fold_sizes)

    def split(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        groups: Any = None,
    ):
        start = 0
        for fold_size in self.test_fold_sizes:
            stop = start + fold_size
            yield np.arange(start, dtype=int), np.arange(start, stop, dtype=int)
            start = stop


@dataclass(frozen=True, slots=True)
class StackingEnsembleModel:
    """Persisted stacking bundle combining the tuned XGBoost model and LR meta-learner."""

    model_name: str
    target_column: str
    base_estimator: XGBClassifier
    meta_learner: LogisticRegression
    base_feature_columns: list[str]
    raw_meta_feature_columns: list[str]
    meta_feature_columns: list[str]

    def predict_base_proba(self, dataframe: pd.DataFrame) -> np.ndarray:
        """Predict probabilities from the base XGBoost model."""

        feature_frame = _validate_required_columns(dataframe, self.base_feature_columns)
        return self.base_estimator.predict_proba(feature_frame)[:, 1]

    def build_meta_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Create LR meta-features from XGBoost probabilities plus selected raw baselines."""

        raw_feature_frame = _validate_required_columns(dataframe, self.raw_meta_feature_columns)
        meta_frame = raw_feature_frame.copy()
        meta_frame.insert(0, "xgb_probability", self.predict_base_proba(dataframe))
        return meta_frame[self.meta_feature_columns]

    def predict_proba(self, dataframe: pd.DataFrame) -> np.ndarray:
        """Predict stacked probabilities in the [0, 1] range."""

        return self.meta_learner.predict_proba(self.build_meta_features(dataframe))

    def predict(self, dataframe: pd.DataFrame) -> np.ndarray:
        """Predict binary labels from the stacked meta-learner."""

        return self.meta_learner.predict(self.build_meta_features(dataframe))


if __name__ == "__main__":
    sys.modules.setdefault("src.model.stacking", sys.modules[__name__])

StackingEnsembleModel.__module__ = "src.model.stacking"


def train_stacking_models(
    *,
    training_data: pd.DataFrame | str | Path,
    output_dir: str | Path = DEFAULT_MODEL_OUTPUT_DIR,
    holdout_season: int | None = None,
    raw_meta_feature_columns: Sequence[str] = DEFAULT_RAW_META_FEATURE_COLUMNS,
    base_search_space: Mapping[str, Sequence[float | int]] = DEFAULT_SEARCH_SPACE,
    time_series_splits: int = DEFAULT_TIME_SERIES_SPLITS,
    search_iterations: int = DEFAULT_SEARCH_ITERATIONS,
    random_state: int = DEFAULT_RANDOM_STATE,
    meta_learner_max_iter: int = DEFAULT_META_LEARNER_MAX_ITER,
    enforce_holdout_brier_gate: bool = True,
) -> StackingRunResult:
    """Train and persist LR stacking models on out-of-fold XGBoost probabilities."""

    dataset = _load_training_dataframe(training_data)
    feature_columns = _resolve_numeric_feature_columns(dataset)
    if not feature_columns:
        raise ValueError("Training data does not contain any numeric feature columns")

    effective_holdout_season = _resolve_holdout_season(dataset, holdout_season)
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    base_training_result = train_f5_models(
        training_data=dataset,
        output_dir=resolved_output_dir,
        holdout_season=effective_holdout_season,
        search_space=base_search_space,
        time_series_splits=time_series_splits,
        search_iterations=search_iterations,
        random_state=random_state,
    )

    resolved_raw_meta_feature_columns = _resolve_raw_meta_feature_columns(
        dataset,
        requested_columns=raw_meta_feature_columns,
    )
    model_version = base_training_result.model_version
    artifacts: dict[str, StackingModelArtifact] = {}

    for spec in _STACKING_MODEL_SPECS:
        base_artifact = base_training_result.models[str(spec["base_model_name"])]
        stacking_artifact = _train_single_stacking_model(
            dataset=dataset,
            model_name=str(spec["model_name"]),
            base_model_name=str(spec["base_model_name"]),
            target_column=str(spec["target_column"]),
            drop_ties=bool(spec["drop_ties"]),
            base_model_path=base_artifact.model_path,
            feature_columns=feature_columns,
            raw_meta_feature_columns=resolved_raw_meta_feature_columns,
            output_dir=resolved_output_dir,
            model_version=model_version,
            holdout_season=effective_holdout_season,
            time_series_splits=time_series_splits,
            random_state=random_state,
            meta_learner_max_iter=meta_learner_max_iter,
            enforce_holdout_brier_gate=enforce_holdout_brier_gate,
        )
        artifacts[stacking_artifact.model_name] = stacking_artifact

    summary_path = resolved_output_dir / f"stacking_run_{model_version}.json"
    summary_path.write_text(
        json.dumps(
            {
                "model_version": model_version,
                "data_version_hash": base_training_result.data_version_hash,
                "holdout_season": effective_holdout_season,
                "feature_columns": feature_columns,
                "raw_meta_feature_columns": resolved_raw_meta_feature_columns,
                "models": {
                    name: _stacking_artifact_to_json_ready(artifact)
                    for name, artifact in artifacts.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return StackingRunResult(
        model_version=model_version,
        data_version_hash=base_training_result.data_version_hash,
        holdout_season=effective_holdout_season,
        feature_columns=feature_columns,
        raw_meta_feature_columns=resolved_raw_meta_feature_columns,
        summary_path=summary_path,
        models=artifacts,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Train stacking ensembles from the persisted anti-leakage-safe training parquet."""

    parser = argparse.ArgumentParser(description="Train F5 stacking models with temporal OOF XGBoost")
    parser.add_argument("--training-data", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_MODEL_OUTPUT_DIR))
    parser.add_argument("--holdout-season", type=int, default=2025)
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--refresh-training-data", action="store_true")
    parser.add_argument("--allow-backfill-years", action="store_true")
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
            allow_backfill_years=args.allow_backfill_years,
            refresh=args.refresh_training_data,
            weather_fetcher=fetch_game_weather,
        )

    result = train_stacking_models(
        training_data=training_path,
        output_dir=args.output_dir,
        holdout_season=args.holdout_season,
        time_series_splits=args.time_series_splits,
        search_iterations=args.search_iterations,
        random_state=args.random_state,
    )
    print(json.dumps(_stacking_run_result_to_json_ready(result), indent=2))
    return 0


def _train_single_stacking_model(
    *,
    dataset: pd.DataFrame,
    model_name: str,
    base_model_name: str,
    target_column: str,
    drop_ties: bool,
    base_model_path: Path,
    feature_columns: list[str],
    raw_meta_feature_columns: list[str],
    output_dir: Path,
    model_version: str,
    holdout_season: int,
    time_series_splits: int,
    random_state: int,
    meta_learner_max_iter: int,
    enforce_holdout_brier_gate: bool,
) -> StackingModelArtifact:
    frame = _prepare_training_frame(dataset, target_column=target_column, drop_ties=drop_ties)
    train_frame = frame.loc[frame["season"] < holdout_season].copy().reset_index(drop=True)
    holdout_frame = frame.loc[frame["season"] == holdout_season].copy().reset_index(drop=True)
    if train_frame.empty:
        raise ValueError(f"No training rows found before holdout season {holdout_season}")
    if holdout_frame.empty:
        raise ValueError(f"No holdout rows found for season {holdout_season}")

    _validate_required_columns(train_frame, raw_meta_feature_columns)

    base_model = joblib.load(base_model_path)
    oof_result = _generate_temporal_oof_probabilities(
        estimator=base_model,
        feature_frame=train_frame[feature_columns],
        target=train_frame[target_column],
        requested_splits=time_series_splits,
    )
    oof_frame = train_frame.iloc[oof_result.warmup_row_count :].copy().reset_index(drop=True)
    oof_frame["xgb_probability"] = oof_result.probabilities.reset_index(drop=True)

    meta_feature_columns = ["xgb_probability", *raw_meta_feature_columns]
    meta_learner = LogisticRegression(max_iter=meta_learner_max_iter, random_state=random_state)
    meta_learner.fit(oof_frame[meta_feature_columns], oof_frame[target_column])

    holdout_base_probabilities = base_model.predict_proba(holdout_frame[feature_columns])[:, 1]
    holdout_meta_frame = holdout_frame[raw_meta_feature_columns].copy()
    holdout_meta_frame.insert(0, "xgb_probability", holdout_base_probabilities)
    holdout_stacked_probabilities = meta_learner.predict_proba(holdout_meta_frame[meta_feature_columns])[:, 1]
    holdout_stacked_predictions = meta_learner.predict(holdout_meta_frame[meta_feature_columns])
    holdout_target = holdout_frame[target_column]

    holdout_metrics = {
        "base_brier": float(brier_score_loss(holdout_target, holdout_base_probabilities)),
        "stacked_brier": float(brier_score_loss(holdout_target, holdout_stacked_probabilities)),
        "stacked_brier_improvement": float(
            brier_score_loss(holdout_target, holdout_base_probabilities)
            - brier_score_loss(holdout_target, holdout_stacked_probabilities)
        ),
        "base_log_loss": float(log_loss(holdout_target, holdout_base_probabilities, labels=[0, 1])),
        "stacked_log_loss": float(
            log_loss(holdout_target, holdout_stacked_probabilities, labels=[0, 1])
        ),
        "base_accuracy": float(
            accuracy_score(holdout_target, (holdout_base_probabilities >= 0.5).astype(int))
        ),
        "stacked_accuracy": float(accuracy_score(holdout_target, holdout_stacked_predictions)),
        "base_roc_auc": _safe_roc_auc(holdout_target, holdout_base_probabilities),
        "stacked_roc_auc": _safe_roc_auc(holdout_target, holdout_stacked_probabilities),
    }
    logger.info("%s holdout metrics: %s", model_name, holdout_metrics)

    model_path = output_dir / f"{model_name}_{model_version}.joblib"
    metadata_path = model_path.with_suffix(".metadata.json")
    persisted = (
        not enforce_holdout_brier_gate
        or holdout_metrics["stacked_brier"] <= holdout_metrics["base_brier"]
    )
    skip_reason: str | None = None

    if enforce_holdout_brier_gate and not persisted:
        skip_reason = (
            "Skipped persistence because holdout stacked_brier "
            f"{holdout_metrics['stacked_brier']:.6f} exceeded base_brier "
            f"{holdout_metrics['base_brier']:.6f}."
        )
        logger.warning("%s %s", model_name, skip_reason)

    stacking_model = StackingEnsembleModel(
        model_name=model_name,
        target_column=target_column,
        base_estimator=base_model,
        meta_learner=meta_learner,
        base_feature_columns=feature_columns,
        raw_meta_feature_columns=raw_meta_feature_columns,
        meta_feature_columns=meta_feature_columns,
    )

    metadata_payload = {
        "model_name": model_name,
        "base_model_name": base_model_name,
        "target_column": target_column,
        "model_version": model_version,
        "base_model_path": str(base_model_path),
        "train_row_count": int(len(train_frame)),
        "oof_row_count": int(len(oof_frame)),
        "warmup_row_count": int(oof_result.warmup_row_count),
        "holdout_row_count": int(len(holdout_frame)),
        "feature_columns": feature_columns,
        "raw_meta_feature_columns": raw_meta_feature_columns,
        "meta_feature_columns": meta_feature_columns,
        "oof_prediction_strategy": "cross_val_predict",
        "oof_fold_count": int(oof_result.fold_count),
        "holdout_metrics": holdout_metrics,
        "persisted": persisted,
        "skip_reason": skip_reason,
        "trained_at": datetime.now(UTC).isoformat(),
    }
    if persisted:
        joblib.dump(stacking_model, model_path)
        metadata_path.write_text(
            json.dumps(metadata_payload, indent=2),
            encoding="utf-8",
        )

    return StackingModelArtifact(
        model_name=model_name,
        base_model_name=base_model_name,
        target_column=target_column,
        model_version=model_version,
        model_path=model_path,
        metadata_path=metadata_path,
        base_model_path=base_model_path,
        holdout_metrics=holdout_metrics,
        train_row_count=len(train_frame),
        oof_row_count=len(oof_frame),
        holdout_row_count=len(holdout_frame),
        warmup_row_count=oof_result.warmup_row_count,
        raw_meta_feature_columns=raw_meta_feature_columns,
        meta_feature_columns=meta_feature_columns,
        oof_prediction_strategy="cross_val_predict",
        persisted=persisted,
        skip_reason=skip_reason,
    )


def _generate_temporal_oof_probabilities(
    *,
    estimator: XGBClassifier,
    feature_frame: pd.DataFrame,
    target: pd.Series,
    requested_splits: int,
) -> _OOFPredictionResult:
    splitter = create_time_series_split(
        row_count=len(feature_frame),
        requested_splits=requested_splits,
    )
    time_series_folds = list(splitter.split(feature_frame))
    if not time_series_folds:
        raise ValueError("Unable to create time-series folds for stacking")

    warmup_end = min(int(test_indices[0]) for _, test_indices in time_series_folds)
    warmup_end = _expand_warmup_until_both_classes(target, warmup_end)
    if warmup_end >= len(feature_frame):
        raise ValueError("Need at least one out-of-fold row after the warmup block")

    warmup_X = feature_frame.iloc[:warmup_end].copy().reset_index(drop=True)
    warmup_y = target.iloc[:warmup_end].copy().reset_index(drop=True)
    oof_X = feature_frame.iloc[warmup_end:].copy().reset_index(drop=True)
    oof_y = target.iloc[warmup_end:].copy().reset_index(drop=True)
    if oof_y.nunique() < 2:
        raise ValueError("Out-of-fold segment must contain both classes for temporal stacking")

    fold_sizes = _build_oof_fold_sizes(
        row_count=len(oof_X),
        requested_splits=splitter.n_splits,
    )
    augmented_estimator = _WarmupAugmentedClassifier(
        base_estimator=clone(estimator),
        warmup_X=warmup_X,
        warmup_y=warmup_y,
    )
    probabilities = cross_val_predict(
        augmented_estimator,
        oof_X,
        oof_y,
        cv=_PartitionedTemporalCV(fold_sizes),
        method="predict_proba",
    )[:, 1]

    return _OOFPredictionResult(
        probabilities=pd.Series(probabilities, index=feature_frame.index[warmup_end:]),
        warmup_row_count=warmup_end,
        fold_count=len(fold_sizes),
    )


def _resolve_raw_meta_feature_columns(
    dataframe: pd.DataFrame,
    *,
    requested_columns: Sequence[str],
) -> list[str]:
    missing_columns = [column for column in requested_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing raw meta-feature columns: {', '.join(missing_columns)}")
    return [str(column) for column in requested_columns]


def _expand_warmup_until_both_classes(target: pd.Series, warmup_end: int) -> int:
    resolved_warmup_end = max(1, warmup_end)
    while resolved_warmup_end < len(target) and target.iloc[:resolved_warmup_end].nunique() < 2:
        resolved_warmup_end += 1
    return resolved_warmup_end


def _build_oof_fold_sizes(*, row_count: int, requested_splits: int) -> list[int]:
    split_count = max(1, min(requested_splits, row_count))
    base_size, remainder = divmod(row_count, split_count)
    fold_sizes = [base_size for _ in range(split_count)]
    for index in range(remainder):
        fold_sizes[index] += 1
    return [size for size in fold_sizes if size > 0]


def _validate_required_columns(dataframe: pd.DataFrame, required_columns: Sequence[str]) -> pd.DataFrame:
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    return dataframe[list(required_columns)].copy()


def _coerce_feature_frame(
    features: pd.DataFrame | np.ndarray,
    *,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    if isinstance(features, pd.DataFrame):
        return features.loc[:, list(feature_names)].copy().reset_index(drop=True)
    return pd.DataFrame(features, columns=list(feature_names))


def _coerce_target_series(target: pd.Series | np.ndarray) -> pd.Series:
    if isinstance(target, pd.Series):
        return target.copy().reset_index(drop=True)
    return pd.Series(target)


def _safe_roc_auc(y_true: pd.Series, probabilities: Sequence[float]) -> float | None:
    if pd.Series(y_true).nunique() < 2:
        return None
    return float(roc_auc_score(y_true, probabilities))


def _stacking_artifact_to_json_ready(artifact: StackingModelArtifact) -> dict[str, Any]:
    payload = asdict(artifact)
    payload["model_path"] = str(artifact.model_path)
    payload["metadata_path"] = str(artifact.metadata_path)
    payload["base_model_path"] = str(artifact.base_model_path)
    return payload


def _stacking_run_result_to_json_ready(result: StackingRunResult) -> dict[str, Any]:
    return {
        "model_version": result.model_version,
        "data_version_hash": result.data_version_hash,
        "holdout_season": result.holdout_season,
        "feature_columns": result.feature_columns,
        "raw_meta_feature_columns": result.raw_meta_feature_columns,
        "summary_path": str(result.summary_path),
        "models": {
            name: _stacking_artifact_to_json_ready(artifact) for name, artifact in result.models.items()
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
