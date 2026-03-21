from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from src.clients.weather_client import fetch_game_weather
from src.model.data_builder import DEFAULT_OUTPUT_PATH, build_training_dataset
from src.model.stacking import (
    DEFAULT_META_LEARNER_MAX_ITER,
    DEFAULT_RAW_META_FEATURE_COLUMNS,
    StackingEnsembleModel,
    train_stacking_models,
)
from src.model.xgboost_trainer import (
    DEFAULT_MODEL_OUTPUT_DIR,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SEARCH_ITERATIONS,
    DEFAULT_SEARCH_SPACE,
    DEFAULT_TIME_SERIES_SPLITS,
    _load_training_dataframe,
    _prepare_training_frame,
    _resolve_holdout_season,
    _resolve_experiment_output_dir,
)


logger = logging.getLogger(__name__)

DEFAULT_CALIBRATION_FRACTION = 0.10
DEFAULT_CALIBRATION_BIN_COUNT = 10
DEFAULT_CALIBRATION_METHOD = "platt"
SUPPORTED_CALIBRATION_METHODS = ("identity", "isotonic", "platt", "blend")
TARGET_BRIER_SCORE = 0.25
TARGET_ECE = 0.05
TARGET_RELIABILITY_GAP = 0.05

_CALIBRATED_MODEL_SPECS = (
    {
        "model_name": "f5_ml_calibrated_model",
        "stacking_model_name": "f5_ml_stacking_model",
        "target_column": "f5_ml_result",
        "drop_ties": True,
    },
    {
        "model_name": "f5_rl_calibrated_model",
        "stacking_model_name": "f5_rl_stacking_model",
        "target_column": "f5_rl_result",
        "drop_ties": False,
    },
)


@dataclass(frozen=True, slots=True)
class CalibrationModelArtifact:
    model_name: str
    stacking_model_name: str
    target_column: str
    calibration_method: str
    model_version: str
    model_path: Path
    metadata_path: Path
    stacking_model_path: Path
    train_row_count: int
    calibration_row_count: int
    holdout_row_count: int
    calibration_fraction: float
    calibration_window_start: str | None
    calibration_window_end: str | None
    holdout_metrics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CalibrationRunResult:
    model_version: str
    data_version_hash: str
    holdout_season: int
    calibration_fraction: float
    calibration_method: str
    model_training_row_count: int
    calibration_row_count: int
    holdout_row_count: int
    summary_path: Path
    models: dict[str, CalibrationModelArtifact]


@dataclass(frozen=True, slots=True)
class CalibratedStackingModel:
    """Persisted inference bundle for stacked + isotonic calibrated probabilities."""

    model_name: str
    target_column: str
    stacking_model: StackingEnsembleModel
    calibrator: object

    @property
    def calibration_method(self) -> str:
        """Return the persisted probability calibrator name."""

        return str(getattr(self.calibrator, "method", "unknown"))

    def predict_stacked_probability(self, dataframe: pd.DataFrame) -> np.ndarray:
        """Predict the uncalibrated stacked probability for the positive class."""

        return np.asarray(self.stacking_model.predict_proba(dataframe)[:, 1], dtype=float)

    def predict_calibrated(self, dataframe: pd.DataFrame) -> np.ndarray:
        """Predict the calibrated positive-class probability for inference."""

        stacked_probability = self.predict_stacked_probability(dataframe)
        return np.asarray(self.calibrator.transform(stacked_probability), dtype=float)

    def predict_proba(self, dataframe: pd.DataFrame) -> np.ndarray:
        """Predict calibrated class probabilities in sklearn-compatible shape."""

        positive_probability = self.predict_calibrated(dataframe)
        return np.column_stack([1.0 - positive_probability, positive_probability])

    def predict(self, dataframe: pd.DataFrame) -> np.ndarray:
        """Predict calibrated class labels using the 0.5 threshold."""

        return (self.predict_calibrated(dataframe) >= 0.5).astype(int)


if __name__ == "__main__":
    sys.modules.setdefault("src.model.calibration", sys.modules[__name__])

CalibratedStackingModel.__module__ = "src.model.calibration"


@dataclass(frozen=True, slots=True)
class IdentityProbabilityCalibrator:
    """No-op probability calibrator used when the stacked probabilities already calibrate best."""

    method: str = "identity"

    def transform(self, probabilities: Sequence[float] | np.ndarray) -> np.ndarray:
        return _coerce_probability_array(probabilities)


@dataclass(frozen=True, slots=True)
class IsotonicProbabilityCalibrator:
    """Wrap an isotonic regressor with a common transform interface."""

    model: IsotonicRegression
    method: str = "isotonic"

    def transform(self, probabilities: Sequence[float] | np.ndarray) -> np.ndarray:
        return np.asarray(self.model.transform(_coerce_probability_array(probabilities)), dtype=float)


@dataclass(frozen=True, slots=True)
class PlattProbabilityCalibrator:
    """Logistic calibration wrapper for Platt scaling on probability logits."""

    model: LogisticRegression
    method: str = "platt"

    def transform(self, probabilities: Sequence[float] | np.ndarray) -> np.ndarray:
        probability_frame = _coerce_probability_array(probabilities).reshape(-1, 1)
        return np.asarray(self.model.predict_proba(probability_frame)[:, 1], dtype=float)


@dataclass(frozen=True, slots=True)
class BlendProbabilityCalibrator:
    """Simple equal-weight ensemble of isotonic and Platt probability calibrators."""

    isotonic: IsotonicProbabilityCalibrator
    platt: PlattProbabilityCalibrator
    method: str = "blend"

    def transform(self, probabilities: Sequence[float] | np.ndarray) -> np.ndarray:
        resolved_probability = _coerce_probability_array(probabilities)
        return np.asarray(
            (
                self.isotonic.transform(resolved_probability)
                + self.platt.transform(resolved_probability)
            )
            / 2.0,
            dtype=float,
        )


IdentityProbabilityCalibrator.__module__ = "src.model.calibration"
IsotonicProbabilityCalibrator.__module__ = "src.model.calibration"
PlattProbabilityCalibrator.__module__ = "src.model.calibration"
BlendProbabilityCalibrator.__module__ = "src.model.calibration"


@dataclass(frozen=True, slots=True)
class _DedicatedCalibrationSplit:
    model_training_frame: pd.DataFrame
    calibration_frame: pd.DataFrame
    holdout_frame: pd.DataFrame
    calibration_row_count: int


def train_calibrated_models(
    *,
    training_data: pd.DataFrame | str | Path,
    output_dir: str | Path = DEFAULT_MODEL_OUTPUT_DIR,
    holdout_season: int | None = None,
    calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION,
    calibration_method: str = DEFAULT_CALIBRATION_METHOD,
    raw_meta_feature_columns: Sequence[str] = DEFAULT_RAW_META_FEATURE_COLUMNS,
    base_search_space: Mapping[str, Sequence[float | int]] = DEFAULT_SEARCH_SPACE,
    time_series_splits: int = DEFAULT_TIME_SERIES_SPLITS,
    search_iterations: int = DEFAULT_SEARCH_ITERATIONS,
    random_state: int = DEFAULT_RANDOM_STATE,
    meta_learner_max_iter: int = DEFAULT_META_LEARNER_MAX_ITER,
    reliability_bin_count: int = DEFAULT_CALIBRATION_BIN_COUNT,
) -> CalibrationRunResult:
    """Train and persist isotonic-calibrated stacking bundles on a dedicated holdout slice."""

    dataset = _load_training_dataframe(training_data)
    effective_holdout_season = _resolve_holdout_season(dataset, holdout_season)
    dedicated_split = _split_dedicated_calibration_frame(
        dataset,
        holdout_season=effective_holdout_season,
        calibration_fraction=calibration_fraction,
    )

    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    stacking_training_frame = pd.concat(
        [dedicated_split.model_training_frame, dedicated_split.holdout_frame],
        ignore_index=True,
    )
    stacking_training_frame = stacking_training_frame.sort_values(
        ["scheduled_start", "game_pk"]
    ).reset_index(drop=True)

    stacking_result = train_stacking_models(
        training_data=stacking_training_frame,
        output_dir=resolved_output_dir,
        holdout_season=effective_holdout_season,
        raw_meta_feature_columns=raw_meta_feature_columns,
        base_search_space=base_search_space,
        time_series_splits=time_series_splits,
        search_iterations=search_iterations,
        random_state=random_state,
        meta_learner_max_iter=meta_learner_max_iter,
        enforce_holdout_brier_gate=False,
    )

    calibration_game_pks = set(dedicated_split.calibration_frame["game_pk"].tolist())
    model_training_game_pks = set(dedicated_split.model_training_frame["game_pk"].tolist())
    holdout_game_pks = set(dedicated_split.holdout_frame["game_pk"].tolist())

    artifacts: dict[str, CalibrationModelArtifact] = {}
    for spec in _CALIBRATED_MODEL_SPECS:
        artifact = _train_single_calibrated_model(
            dataset=dataset,
            model_name=str(spec["model_name"]),
            stacking_model_name=str(spec["stacking_model_name"]),
            target_column=str(spec["target_column"]),
            drop_ties=bool(spec["drop_ties"]),
            calibration_method=calibration_method,
            stacking_model_path=stacking_result.models[str(spec["stacking_model_name"])].model_path,
            output_dir=resolved_output_dir,
            model_version=stacking_result.model_version,
            calibration_fraction=calibration_fraction,
            calibration_game_pks=calibration_game_pks,
            model_training_game_pks=model_training_game_pks,
            holdout_game_pks=holdout_game_pks,
            reliability_bin_count=reliability_bin_count,
        )
        artifacts[artifact.model_name] = artifact

    summary_path = resolved_output_dir / f"calibration_run_{stacking_result.model_version}.json"
    summary_path.write_text(
        json.dumps(
            {
                "model_version": stacking_result.model_version,
                "data_version_hash": stacking_result.data_version_hash,
                "holdout_season": effective_holdout_season,
                "calibration_fraction": calibration_fraction,
                "calibration_method": calibration_method,
                "model_training_row_count": int(len(dedicated_split.model_training_frame)),
                "calibration_row_count": int(dedicated_split.calibration_row_count),
                "holdout_row_count": int(len(dedicated_split.holdout_frame)),
                "models": {
                    name: _calibration_artifact_to_json_ready(artifact)
                    for name, artifact in artifacts.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return CalibrationRunResult(
        model_version=stacking_result.model_version,
        data_version_hash=stacking_result.data_version_hash,
        holdout_season=effective_holdout_season,
        calibration_fraction=calibration_fraction,
        calibration_method=calibration_method,
        model_training_row_count=len(dedicated_split.model_training_frame),
        calibration_row_count=dedicated_split.calibration_row_count,
        holdout_row_count=len(dedicated_split.holdout_frame),
        summary_path=summary_path,
        models=artifacts,
    )


def build_reliability_diagram(
    y_true: Sequence[int] | pd.Series | np.ndarray,
    probabilities: Sequence[float] | pd.Series | np.ndarray,
    *,
    bin_count: int = DEFAULT_CALIBRATION_BIN_COUNT,
) -> list[dict[str, float | int | None]]:
    """Build uniform-bin reliability diagram data for calibrated probability review."""

    if bin_count < 1:
        raise ValueError("bin_count must be at least 1")

    target = np.asarray(y_true, dtype=float)
    predicted = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    if len(target) != len(predicted):
        raise ValueError("y_true and probabilities must have the same length")

    edges = np.linspace(0.0, 1.0, bin_count + 1)
    assignments = np.digitize(predicted, edges[1:-1], right=False)

    reliability_diagram: list[dict[str, float | int | None]] = []
    for index in range(bin_count):
        mask = assignments == index
        count = int(mask.sum())
        mean_probability: float | None
        positive_rate: float | None
        absolute_gap: float | None

        if count:
            mean_probability = float(predicted[mask].mean())
            positive_rate = float(target[mask].mean())
            absolute_gap = float(abs(mean_probability - positive_rate))
        else:
            mean_probability = None
            positive_rate = None
            absolute_gap = None

        reliability_diagram.append(
            {
                "bin_index": index,
                "bin_lower": float(edges[index]),
                "bin_upper": float(edges[index + 1]),
                "count": count,
                "mean_predicted_probability": mean_probability,
                "empirical_positive_rate": positive_rate,
                "absolute_gap": absolute_gap,
            }
        )

    return reliability_diagram


def compute_expected_calibration_error(
    y_true: Sequence[int] | pd.Series | np.ndarray,
    probabilities: Sequence[float] | pd.Series | np.ndarray,
    *,
    bin_count: int = DEFAULT_CALIBRATION_BIN_COUNT,
) -> float:
    """Compute expected calibration error using uniform-bin absolute probability gaps."""

    reliability_diagram = build_reliability_diagram(
        y_true,
        probabilities,
        bin_count=bin_count,
    )
    total_count = sum(int(entry["count"]) for entry in reliability_diagram)
    if total_count == 0:
        return 0.0

    return float(
        sum(
            (int(entry["count"]) / total_count) * float(entry["absolute_gap"] or 0.0)
            for entry in reliability_diagram
        )
    )


def predict_calibrated(
    model: CalibratedStackingModel | str | Path,
    dataframe: pd.DataFrame,
) -> np.ndarray:
    """Load a persisted calibrated bundle if needed and return calibrated probabilities."""

    calibrated_model = _load_calibrated_model(model)
    return calibrated_model.predict_calibrated(dataframe)


def main(argv: Sequence[str] | None = None) -> int:
    """Train isotonic-calibrated stacking models from the persisted training parquet."""

    parser = argparse.ArgumentParser(description="Train F5 isotonic calibration bundles")
    parser.add_argument("--training-data", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_MODEL_OUTPUT_DIR))
    parser.add_argument("--experiment-name")
    parser.add_argument("--holdout-season", type=int, default=2025)
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--refresh-training-data", action="store_true")
    parser.add_argument("--calibration-fraction", type=float, default=DEFAULT_CALIBRATION_FRACTION)
    parser.add_argument(
        "--calibration-method",
        default=DEFAULT_CALIBRATION_METHOD,
        choices=SUPPORTED_CALIBRATION_METHODS,
    )
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
            weather_fetcher=fetch_game_weather,
        )

    resolved_output_dir = _resolve_experiment_output_dir(args.output_dir, args.experiment_name)

    result = train_calibrated_models(
        training_data=training_path,
        output_dir=resolved_output_dir,
        holdout_season=args.holdout_season,
        calibration_fraction=args.calibration_fraction,
        calibration_method=args.calibration_method,
        time_series_splits=args.time_series_splits,
        search_iterations=args.search_iterations,
        random_state=args.random_state,
    )
    print(json.dumps(_calibration_run_result_to_json_ready(result), indent=2))
    return 0


def _train_single_calibrated_model(
    *,
    dataset: pd.DataFrame,
    model_name: str,
    stacking_model_name: str,
    target_column: str,
    drop_ties: bool,
    calibration_method: str,
    stacking_model_path: Path,
    output_dir: Path,
    model_version: str,
    calibration_fraction: float,
    calibration_game_pks: set[int],
    model_training_game_pks: set[int],
    holdout_game_pks: set[int],
    reliability_bin_count: int,
) -> CalibrationModelArtifact:
    frame = _prepare_training_frame(dataset, target_column=target_column, drop_ties=drop_ties)
    model_training_frame = frame.loc[frame["game_pk"].isin(model_training_game_pks)].copy()
    calibration_frame = frame.loc[frame["game_pk"].isin(calibration_game_pks)].copy()
    holdout_frame = frame.loc[frame["game_pk"].isin(holdout_game_pks)].copy()

    model_training_frame = model_training_frame.sort_values(["scheduled_start", "game_pk"])
    calibration_frame = calibration_frame.sort_values(["scheduled_start", "game_pk"])
    holdout_frame = holdout_frame.sort_values(["scheduled_start", "game_pk"])

    if calibration_frame.empty:
        raise ValueError(f"No calibration rows available for {model_name}")
    if holdout_frame.empty:
        raise ValueError(f"No holdout rows available for {model_name}")

    stacking_model = joblib.load(stacking_model_path)
    if not isinstance(stacking_model, StackingEnsembleModel):
        raise TypeError(f"Expected StackingEnsembleModel at {stacking_model_path}")

    calibration_target = calibration_frame[target_column].astype(int)
    holdout_target = holdout_frame[target_column].astype(int)

    calibration_stacked_probability = np.asarray(
        stacking_model.predict_proba(calibration_frame)[:, 1],
        dtype=float,
    )
    holdout_stacked_probability = np.asarray(
        stacking_model.predict_proba(holdout_frame)[:, 1],
        dtype=float,
    )

    calibrator = _fit_probability_calibrator(
        method=calibration_method,
        probabilities=calibration_stacked_probability,
        y_true=calibration_target,
    )
    resolved_calibration_method = str(getattr(calibrator, "method", calibration_method))
    holdout_calibrated_probability = np.asarray(
        calibrator.transform(holdout_stacked_probability),
        dtype=float,
    )

    holdout_metrics = _evaluate_holdout_metrics(
        y_true=holdout_target,
        stacked_probability=holdout_stacked_probability,
        calibrated_probability=holdout_calibrated_probability,
        reliability_bin_count=reliability_bin_count,
    )
    logger.info("%s holdout metrics: %s", model_name, holdout_metrics)

    calibrated_model = CalibratedStackingModel(
        model_name=model_name,
        target_column=target_column,
        stacking_model=stacking_model,
        calibrator=calibrator,
    )
    model_path = output_dir / f"{model_name}_{model_version}.joblib"
    joblib.dump(calibrated_model, model_path)

    calibration_window_start = (
        pd.Timestamp(calibration_frame["scheduled_start"].iloc[0]).isoformat()
        if not calibration_frame.empty
        else None
    )
    calibration_window_end = (
        pd.Timestamp(calibration_frame["scheduled_start"].iloc[-1]).isoformat()
        if not calibration_frame.empty
        else None
    )
    metadata_path = model_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "stacking_model_name": stacking_model_name,
                "target_column": target_column,
                "calibration_method": resolved_calibration_method,
                "model_version": model_version,
                "stacking_model_path": str(stacking_model_path),
                "train_row_count": int(len(model_training_frame)),
                "calibration_row_count": int(len(calibration_frame)),
                "holdout_row_count": int(len(holdout_frame)),
                "calibration_fraction": calibration_fraction,
                "calibration_window_start": calibration_window_start,
                "calibration_window_end": calibration_window_end,
                "calibrator_threshold_count": int(
                    len(
                        getattr(
                            getattr(calibrator, "model", calibrator),
                            "X_thresholds_",
                            [],
                        )
                    )
                ),
                "holdout_metrics": holdout_metrics,
                "trained_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return CalibrationModelArtifact(
        model_name=model_name,
        stacking_model_name=stacking_model_name,
        target_column=target_column,
        calibration_method=resolved_calibration_method,
        model_version=model_version,
        model_path=model_path,
        metadata_path=metadata_path,
        stacking_model_path=stacking_model_path,
        train_row_count=len(model_training_frame),
        calibration_row_count=len(calibration_frame),
        holdout_row_count=len(holdout_frame),
        calibration_fraction=calibration_fraction,
        calibration_window_start=calibration_window_start,
        calibration_window_end=calibration_window_end,
        holdout_metrics=holdout_metrics,
    )


def _split_dedicated_calibration_frame(
    dataframe: pd.DataFrame,
    *,
    holdout_season: int,
    calibration_fraction: float,
) -> _DedicatedCalibrationSplit:
    if calibration_fraction <= 0 or calibration_fraction >= 1:
        raise ValueError("calibration_fraction must be between 0 and 1")

    frame = dataframe.sort_values(["scheduled_start", "game_pk"]).reset_index(drop=True)
    pre_holdout_frame = frame.loc[frame["season"] < holdout_season].copy().reset_index(drop=True)
    holdout_frame = frame.loc[frame["season"] == holdout_season].copy().reset_index(drop=True)
    if pre_holdout_frame.empty:
        raise ValueError(f"No rows found before holdout season {holdout_season}")
    if holdout_frame.empty:
        raise ValueError(f"No rows found for holdout season {holdout_season}")
    if len(pre_holdout_frame) < 2:
        raise ValueError("Need at least two pre-holdout rows to reserve a calibration slice")

    calibration_row_count = min(
        max(1, ceil(len(pre_holdout_frame) * calibration_fraction)),
        len(pre_holdout_frame) - 1,
    )
    split_index = len(pre_holdout_frame) - calibration_row_count

    return _DedicatedCalibrationSplit(
        model_training_frame=pre_holdout_frame.iloc[:split_index].copy().reset_index(drop=True),
        calibration_frame=pre_holdout_frame.iloc[split_index:].copy().reset_index(drop=True),
        holdout_frame=holdout_frame,
        calibration_row_count=calibration_row_count,
    )


def _evaluate_holdout_metrics(
    *,
    y_true: pd.Series,
    stacked_probability: np.ndarray,
    calibrated_probability: np.ndarray,
    reliability_bin_count: int,
) -> dict[str, Any]:
    stacked_reliability = build_reliability_diagram(
        y_true,
        stacked_probability,
        bin_count=reliability_bin_count,
    )
    calibrated_reliability = build_reliability_diagram(
        y_true,
        calibrated_probability,
        bin_count=reliability_bin_count,
    )
    stacked_brier = float(brier_score_loss(y_true, stacked_probability))
    calibrated_brier = float(brier_score_loss(y_true, calibrated_probability))
    stacked_ece = compute_expected_calibration_error(
        y_true,
        stacked_probability,
        bin_count=reliability_bin_count,
    )
    calibrated_ece = compute_expected_calibration_error(
        y_true,
        calibrated_probability,
        bin_count=reliability_bin_count,
    )
    max_reliability_gap = _max_reliability_gap(calibrated_reliability)

    return {
        "stacked_brier": stacked_brier,
        "calibrated_brier": calibrated_brier,
        "brier_improvement": float(stacked_brier - calibrated_brier),
        "stacked_ece": stacked_ece,
        "calibrated_ece": calibrated_ece,
        "stacked_accuracy": float(accuracy_score(y_true, (stacked_probability >= 0.5).astype(int))),
        "calibrated_accuracy": float(
            accuracy_score(y_true, (calibrated_probability >= 0.5).astype(int))
        ),
        "stacked_log_loss": float(log_loss(y_true, stacked_probability, labels=[0, 1])),
        "calibrated_log_loss": float(log_loss(y_true, calibrated_probability, labels=[0, 1])),
        "stacked_roc_auc": _safe_roc_auc(y_true, stacked_probability),
        "calibrated_roc_auc": _safe_roc_auc(y_true, calibrated_probability),
        "stacked_reliability_diagram": stacked_reliability,
        "reliability_diagram": calibrated_reliability,
        "max_reliability_gap": max_reliability_gap,
        "quality_gates": {
            "brier_lt_0_25": calibrated_brier < TARGET_BRIER_SCORE,
            "ece_lt_0_05": calibrated_ece < TARGET_ECE,
            "reliability_gap_le_0_05": max_reliability_gap <= TARGET_RELIABILITY_GAP,
        },
    }


def _fit_probability_calibrator(
    *,
    method: str,
    probabilities: Sequence[float] | np.ndarray,
    y_true: Sequence[int] | pd.Series | np.ndarray,
) -> IdentityProbabilityCalibrator | IsotonicProbabilityCalibrator | PlattProbabilityCalibrator | BlendProbabilityCalibrator:
    resolved_method = str(method).strip().lower()
    if resolved_method not in SUPPORTED_CALIBRATION_METHODS:
        raise ValueError(
            f"Unsupported calibration_method '{method}'. Expected one of: "
            f"{', '.join(SUPPORTED_CALIBRATION_METHODS)}"
        )

    resolved_probability = _coerce_probability_array(probabilities)
    resolved_target = pd.Series(y_true).astype(int).reset_index(drop=True)
    if resolved_target.nunique() < 2 or resolved_method == "identity":
        return IdentityProbabilityCalibrator()

    if resolved_method == "isotonic":
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(resolved_probability, resolved_target)
        return IsotonicProbabilityCalibrator(model=calibrator)

    if resolved_method == "platt":
        calibrator = LogisticRegression(max_iter=1_000)
        calibrator.fit(resolved_probability.reshape(-1, 1), resolved_target)
        return PlattProbabilityCalibrator(model=calibrator)

    isotonic = _fit_probability_calibrator(
        method="isotonic",
        probabilities=resolved_probability,
        y_true=resolved_target,
    )
    platt = _fit_probability_calibrator(
        method="platt",
        probabilities=resolved_probability,
        y_true=resolved_target,
    )
    if not isinstance(isotonic, IsotonicProbabilityCalibrator) or not isinstance(
        platt,
        PlattProbabilityCalibrator,
    ):
        raise TypeError("Blend calibration requires isotonic and Platt calibrators")
    return BlendProbabilityCalibrator(isotonic=isotonic, platt=platt)


def _coerce_probability_array(probabilities: Sequence[float] | np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)


def _max_reliability_gap(reliability_diagram: Sequence[Mapping[str, Any]]) -> float:
    occupied_gaps = [
        float(entry["absolute_gap"])
        for entry in reliability_diagram
        if entry.get("count") and entry.get("absolute_gap") is not None
    ]
    if not occupied_gaps:
        return 0.0
    return float(max(occupied_gaps))


def _safe_roc_auc(y_true: Sequence[int] | pd.Series, probabilities: Sequence[float]) -> float | None:
    if pd.Series(y_true).nunique() < 2:
        return None
    return float(roc_auc_score(y_true, probabilities))


def _load_calibrated_model(model: CalibratedStackingModel | str | Path) -> CalibratedStackingModel:
    if isinstance(model, CalibratedStackingModel):
        return model

    loaded_model = joblib.load(Path(model))
    if not isinstance(loaded_model, CalibratedStackingModel):
        raise TypeError(f"Expected CalibratedStackingModel at {model}")
    return loaded_model


def _calibration_artifact_to_json_ready(artifact: CalibrationModelArtifact) -> dict[str, Any]:
    payload = asdict(artifact)
    payload["model_path"] = str(artifact.model_path)
    payload["metadata_path"] = str(artifact.metadata_path)
    payload["stacking_model_path"] = str(artifact.stacking_model_path)
    return payload


def _calibration_run_result_to_json_ready(result: CalibrationRunResult) -> dict[str, Any]:
    return {
        "model_version": result.model_version,
        "data_version_hash": result.data_version_hash,
        "holdout_season": result.holdout_season,
        "calibration_fraction": result.calibration_fraction,
        "calibration_method": result.calibration_method,
        "model_training_row_count": result.model_training_row_count,
        "calibration_row_count": result.calibration_row_count,
        "holdout_row_count": result.holdout_row_count,
        "summary_path": str(result.summary_path),
        "models": {
            name: _calibration_artifact_to_json_ready(artifact)
            for name, artifact in result.models.items()
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
