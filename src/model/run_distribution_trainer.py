from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.stats import nbinom
from xgboost import XGBClassifier, XGBRegressor

from src.model.artifact_runtime import collect_runtime_versions
from src.model.data_builder import inspect_run_count_training_data, validate_run_count_training_data
from src.model.run_distribution_metrics import (
    DEFAULT_CALIBRATION_BIN_COUNT,
    DEFAULT_SUPPORT_TAIL_PROBABILITY,
    MIN_MEAN_PREDICTION,
    MIN_PROBABILITY,
    clip_mean_predictions,
    dataclass_to_dict,
    event_probability,
    fit_negative_binomial_dispersion,
    fit_zero_adjustment,
    negative_binomial_pmf_matrix,
    summarize_distribution_metrics,
    zero_adjusted_negative_binomial_pmf_matrix,
)
from src.model.run_count_trainer import (
    DEFAULT_RUN_COUNT_FEATURE_SELECTION_MODE,
    DEFAULT_RUN_COUNT_FORCED_DELTA_FEATURE_COUNT,
    DEFAULT_RUN_COUNT_MAX_FEATURE_COUNT,
    _compute_holdout_metrics,
    _prepare_run_count_frame,
    _resolve_run_count_candidate_feature_columns,
    _select_run_count_feature_columns,
    _select_run_count_feature_columns_bucketed,
    _select_run_count_feature_columns_flat,
    create_time_series_split,
)
from src.model.xgboost_trainer import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TOP_FEATURE_COUNT,
    DEFAULT_XGBOOST_N_JOBS,
    _build_model_version,
    _extract_feature_importance_rankings,
    _load_training_dataframe,
    _resolve_data_version_hash,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CURRENT_CONTROL_PATH = Path("data/reports/run_count/registry/current_control.json")
DEFAULT_DISTRIBUTION_REPORT_DIR = Path("data/reports/run_count/distribution_eval")
DEFAULT_DISTRIBUTION_MODEL_NAME = "full_game_away_runs_distribution_model"
DEFAULT_TARGET_COLUMN = "final_away_score"
DEFAULT_DISTRIBUTION_FAMILY_NAME = "zero_adjusted_negative_binomial"
DEFAULT_DISPERSION_ALPHA_FLOOR = 1e-3
DEFAULT_DISPERSION_ALPHA_CAP = 4.0
DEFAULT_HEAD_PARAM_OVERRIDES: dict[str, float | int] = {
    "max_depth": 3,
    "n_estimators": 300,
    "learning_rate": 0.03,
    "subsample": 0.75,
    "colsample_bytree": 0.8,
    "min_child_weight": 4,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}


@dataclass(frozen=True, slots=True)
class ConstantValueRegressor:
    value: float

    def predict(self, dataframe: pd.DataFrame) -> np.ndarray:
        return np.full(len(dataframe), float(self.value), dtype=float)


@dataclass(frozen=True, slots=True)
class ConstantProbabilityClassifier:
    probability: float

    def predict_proba(self, dataframe: pd.DataFrame) -> np.ndarray:
        probability = float(np.clip(self.probability, MIN_PROBABILITY, 1.0 - MIN_PROBABILITY))
        return np.column_stack(
            [
                np.full(len(dataframe), 1.0 - probability, dtype=float),
                np.full(len(dataframe), probability, dtype=float),
            ]
        )


@dataclass(frozen=True, slots=True)
class MeanHeadReference:
    estimator: Any
    metadata: dict[str, Any]
    metadata_path: Path
    model_path: Path
    feature_columns: list[str]


@dataclass(frozen=True, slots=True)
class RunDistributionModel:
    mean_estimator: Any
    dispersion_estimator: Any
    zero_estimator: Any
    mean_feature_columns: list[str]
    dispersion_feature_columns: list[str]
    zero_feature_columns: list[str]
    global_negative_binomial_fit: dict[str, float | bool]
    global_zero_adjustment_delta: float
    dispersion_blend_weight: float
    zero_blend_weight: float
    distribution_family: str = DEFAULT_DISTRIBUTION_FAMILY_NAME

    def predict_mean(self, dataframe: pd.DataFrame) -> np.ndarray:
        features = dataframe.loc[:, self.mean_feature_columns]
        return clip_mean_predictions(self.mean_estimator.predict(features))

    def predict_components(self, dataframe: pd.DataFrame) -> dict[str, np.ndarray]:
        mean_predictions = self.predict_mean(dataframe)
        raw_alpha = _predict_dispersion_alpha(
            self.dispersion_estimator,
            dataframe.loc[:, self.dispersion_feature_columns],
            fallback_alpha=float(self.global_negative_binomial_fit["overdispersion_alpha"]),
        )
        alpha = _blend_dispersion_alpha(
            global_alpha=float(self.global_negative_binomial_fit["overdispersion_alpha"]),
            raw_alpha_predictions=raw_alpha,
            blend_weight=float(self.dispersion_blend_weight),
        )
        dispersion_size = _alpha_to_dispersion_size(alpha)
        baseline_zero_probability = _apply_zero_adjustment_delta(
            _variable_negative_binomial_zero_probability(
                mean_predictions,
                dispersion_size,
            ),
            delta=float(self.global_zero_adjustment_delta),
        )
        raw_zero_probability = _predict_zero_probability(
            self.zero_estimator,
            dataframe.loc[:, self.zero_feature_columns],
            fallback_probability=baseline_zero_probability,
        )
        adjusted_zero_probability = _blend_zero_probability(
            baseline_zero_probability=baseline_zero_probability,
            raw_zero_probability=raw_zero_probability,
            blend_weight=float(self.zero_blend_weight),
        )
        return {
            "mu": mean_predictions,
            "overdispersion_alpha": alpha,
            "dispersion": dispersion_size,
            "baseline_zero_probability": baseline_zero_probability,
            "adjusted_zero_probability": adjusted_zero_probability,
            "p_zero_extra": adjusted_zero_probability - baseline_zero_probability,
        }

    def predict_pmf(
        self,
        dataframe: pd.DataFrame,
        *,
        support: Sequence[int],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        components = self.predict_components(dataframe)
        pmf = zero_adjusted_negative_binomial_pmf_matrix_by_row(
            components["mu"],
            support,
            dispersion_size=components["dispersion"],
            adjusted_zero_probability=components["adjusted_zero_probability"],
        )
        return pmf, components


@dataclass(frozen=True, slots=True)
class RunDistributionTrainingArtifact:
    model_name: str
    target_column: str
    model_version: str
    model_path: Path
    metadata_path: Path
    summary_path: Path
    distribution_report_json_path: Path
    distribution_report_csv_path: Path
    control_comparison_json_path: Path
    holdout_season: int
    holdout_metrics: dict[str, float | None]
    distribution_metrics: dict[str, Any]
    comparison_to_control: dict[str, Any]


def train_run_distribution_model(
    *,
    training_data: pd.DataFrame | str | Path,
    output_dir: str | Path,
    mean_artifact_metadata_path: str | Path,
    holdout_season: int,
    target_column: str = DEFAULT_TARGET_COLUMN,
    model_name: str = DEFAULT_DISTRIBUTION_MODEL_NAME,
    feature_selection_mode: str = DEFAULT_RUN_COUNT_FEATURE_SELECTION_MODE,
    forced_delta_feature_count: int = DEFAULT_RUN_COUNT_FORCED_DELTA_FEATURE_COUNT,
    time_series_splits: int = 3,
    random_state: int = DEFAULT_RANDOM_STATE,
    top_feature_count: int = DEFAULT_TOP_FEATURE_COUNT,
    max_feature_count: int = DEFAULT_RUN_COUNT_MAX_FEATURE_COUNT,
    xgb_n_jobs: int = DEFAULT_XGBOOST_N_JOBS,
    calibration_bin_count: int = DEFAULT_CALIBRATION_BIN_COUNT,
    tail_probability: float = DEFAULT_SUPPORT_TAIL_PROBABILITY,
    distribution_report_dir: str | Path = DEFAULT_DISTRIBUTION_REPORT_DIR,
    head_param_overrides: Mapping[str, float | int] | None = None,
) -> RunDistributionTrainingArtifact:
    validated_training_data = validate_run_count_training_data(training_data)
    training_data_inspection = inspect_run_count_training_data(validated_training_data)
    dataset = _load_training_dataframe(validated_training_data)
    frame = _prepare_run_count_frame(dataset, target_column=target_column)
    train_frame = frame.loc[frame["season"] < int(holdout_season)].copy()
    holdout_frame = frame.loc[frame["season"] == int(holdout_season)].copy()
    if train_frame.empty:
        raise ValueError(f"No training rows found before holdout season {holdout_season}")
    if holdout_frame.empty:
        raise ValueError(f"No holdout rows found for season {holdout_season}")

    mean_head = load_mean_head_reference(mean_artifact_metadata_path)
    _validate_mean_head_reference(
        mean_head,
        target_column=target_column,
        holdout_season=holdout_season,
    )

    mean_train = clip_mean_predictions(
        mean_head.estimator.predict(train_frame.loc[:, mean_head.feature_columns])
    )
    mean_holdout = clip_mean_predictions(
        mean_head.estimator.predict(holdout_frame.loc[:, mean_head.feature_columns])
    )
    train_actual = train_frame[target_column].astype(int).to_numpy()
    holdout_actual = holdout_frame[target_column].astype(int).to_numpy()

    global_nb_fit = fit_negative_binomial_dispersion(train_actual, mean_train)
    global_train_support = build_variable_nb_support(
        actual_counts=train_actual,
        mean_predictions=mean_train,
        dispersion_size=np.full(len(mean_train), float(global_nb_fit.dispersion_size)),
        tail_probability=tail_probability,
    )
    global_train_pmf = negative_binomial_pmf_matrix(
        mean_train,
        global_train_support,
        dispersion_size=float(global_nb_fit.dispersion_size),
    )
    global_zero_adjustment_fit = fit_zero_adjustment(
        train_actual,
        event_probability(global_train_pmf, global_train_support, kind="eq", threshold=0),
    )
    candidate_resolution = _resolve_run_count_candidate_feature_columns(frame)
    if not candidate_resolution.candidate_columns:
        raise ValueError("Training data does not contain any numeric feature columns for Stage 3 heads")

    dispersion_target = _build_dispersion_training_target(
        actual_counts=train_actual,
        mean_predictions=mean_train,
        global_alpha=float(global_nb_fit.overdispersion_alpha),
    )
    zero_target = (train_actual == 0).astype(int)

    dispersion_selection = _select_head_feature_columns(
        train_frame=train_frame,
        candidate_feature_columns=candidate_resolution.candidate_columns,
        target_values=dispersion_target,
        feature_selection_mode=feature_selection_mode,
        max_feature_count=max_feature_count,
        forced_delta_feature_count=forced_delta_feature_count,
    )
    zero_selection = _select_head_feature_columns(
        train_frame=train_frame,
        candidate_feature_columns=candidate_resolution.candidate_columns,
        target_values=zero_target,
        feature_selection_mode=feature_selection_mode,
        max_feature_count=max_feature_count,
        forced_delta_feature_count=forced_delta_feature_count,
    )

    resolved_head_params = _resolve_head_params(
        mean_head.metadata,
        xgb_n_jobs=xgb_n_jobs,
        head_param_overrides=head_param_overrides,
    )
    splitter = create_time_series_split(
        row_count=len(train_frame),
        requested_splits=time_series_splits,
    )

    dispersion_oof = _generate_regression_oof_predictions(
        train_frame=train_frame,
        feature_columns=dispersion_selection.feature_columns,
        target_values=dispersion_target,
        splitter=splitter,
        random_state=random_state,
        estimator_params=resolved_head_params,
    )
    dispersion_mask = dispersion_oof.notna().to_numpy()
    dispersion_blend_weight = _optimize_dispersion_blend_weight(
        actual_counts=train_actual[dispersion_mask],
        mean_predictions=mean_train[dispersion_mask],
        global_alpha=float(global_nb_fit.overdispersion_alpha),
        raw_alpha_predictions=np.exp(dispersion_oof.loc[dispersion_oof.notna()].to_numpy()),
    )
    dispersion_estimator = _fit_dispersion_estimator(
        train_frame=train_frame,
        feature_columns=dispersion_selection.feature_columns,
        target_values=dispersion_target,
        random_state=random_state,
        estimator_params=resolved_head_params,
    )
    raw_holdout_alpha = _predict_dispersion_alpha(
        dispersion_estimator,
        holdout_frame.loc[:, dispersion_selection.feature_columns],
        fallback_alpha=float(global_nb_fit.overdispersion_alpha),
    )
    blended_holdout_alpha = _blend_dispersion_alpha(
        global_alpha=float(global_nb_fit.overdispersion_alpha),
        raw_alpha_predictions=raw_holdout_alpha,
        blend_weight=dispersion_blend_weight,
    )
    holdout_dispersion_size = _alpha_to_dispersion_size(blended_holdout_alpha)

    zero_oof = _generate_classifier_oof_predictions(
        train_frame=train_frame,
        feature_columns=zero_selection.feature_columns,
        target_values=zero_target,
        splitter=splitter,
        random_state=random_state,
        estimator_params=resolved_head_params,
    )
    common_oof_mask = dispersion_oof.notna().to_numpy() & zero_oof.notna().to_numpy()
    blended_train_alpha = _blend_dispersion_alpha(
        global_alpha=float(global_nb_fit.overdispersion_alpha),
        raw_alpha_predictions=np.exp(dispersion_oof.loc[common_oof_mask].to_numpy()),
        blend_weight=dispersion_blend_weight,
    )
    baseline_zero_oof = _apply_zero_adjustment_delta(
        _variable_negative_binomial_zero_probability(
            mean_train[common_oof_mask],
            _alpha_to_dispersion_size(blended_train_alpha),
        ),
        delta=float(global_zero_adjustment_fit.delta),
    )
    zero_blend_weight = _optimize_zero_blend_weight(
        observed_zero=zero_target[common_oof_mask],
        baseline_zero_probability=baseline_zero_oof,
        raw_zero_probability=zero_oof.loc[common_oof_mask].to_numpy(),
    )
    zero_estimator = _fit_zero_estimator(
        train_frame=train_frame,
        feature_columns=zero_selection.feature_columns,
        target_values=zero_target,
        random_state=random_state,
        estimator_params=resolved_head_params,
    )
    baseline_zero_holdout = _apply_zero_adjustment_delta(
        _variable_negative_binomial_zero_probability(
            mean_holdout,
            holdout_dispersion_size,
        ),
        delta=float(global_zero_adjustment_fit.delta),
    )
    raw_zero_holdout = _predict_zero_probability(
        zero_estimator,
        holdout_frame.loc[:, zero_selection.feature_columns],
        fallback_probability=baseline_zero_holdout,
    )
    adjusted_zero_holdout = _blend_zero_probability(
        baseline_zero_probability=baseline_zero_holdout,
        raw_zero_probability=raw_zero_holdout,
        blend_weight=zero_blend_weight,
    )

    model_version = _build_model_version(_resolve_data_version_hash(dataset))
    distribution_model = RunDistributionModel(
        mean_estimator=mean_head.estimator,
        dispersion_estimator=dispersion_estimator,
        zero_estimator=zero_estimator,
        mean_feature_columns=list(mean_head.feature_columns),
        dispersion_feature_columns=list(dispersion_selection.feature_columns),
        zero_feature_columns=list(zero_selection.feature_columns),
        global_negative_binomial_fit=dataclass_to_dict(global_nb_fit),
        global_zero_adjustment_delta=float(global_zero_adjustment_fit.delta),
        dispersion_blend_weight=float(dispersion_blend_weight),
        zero_blend_weight=float(zero_blend_weight),
    )

    support = build_variable_nb_support(
        actual_counts=holdout_actual,
        mean_predictions=mean_holdout,
        dispersion_size=holdout_dispersion_size,
        tail_probability=tail_probability,
    )
    holdout_pmf = zero_adjusted_negative_binomial_pmf_matrix_by_row(
        mean_holdout,
        support,
        dispersion_size=holdout_dispersion_size,
        adjusted_zero_probability=adjusted_zero_holdout,
    )
    distribution_metrics = summarize_distribution_metrics(
        holdout_actual,
        holdout_pmf,
        support,
        calibration_bin_count=calibration_bin_count,
    )
    mean_metrics = _compute_holdout_metrics(
        train_frame=train_frame,
        holdout_frame=holdout_frame,
        target_column=target_column,
        holdout_predictions=mean_holdout,
    )
    control_metrics = evaluate_control_distribution_baseline(
        train_actual=train_actual,
        train_mean=mean_train,
        holdout_actual=holdout_actual,
        holdout_mean=mean_holdout,
        calibration_bin_count=calibration_bin_count,
        tail_probability=tail_probability,
    )
    comparison_to_control = build_control_comparison(
        stage3_mean_metrics=mean_metrics,
        stage3_distribution_metrics=distribution_metrics,
        control_mean_metrics=mean_metrics,
        control_distribution_metrics=control_metrics["holdout_metrics"],
    )

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    distribution_report_dir_path = Path(distribution_report_dir)
    if not distribution_report_dir_path.is_absolute():
        distribution_report_dir_path = PROJECT_ROOT / distribution_report_dir_path
    distribution_report_dir_path.mkdir(parents=True, exist_ok=True)

    model_path = output_dir_path / f"{model_name}_{model_version}.joblib"
    metadata_path = output_dir_path / f"{model_name}_{model_version}.metadata.json"
    summary_path = output_dir_path / f"{model_name}_training_run_{model_version}.json"
    distribution_report_json_path = distribution_report_dir_path / f"{model_version}.distribution_eval.json"
    distribution_report_csv_path = distribution_report_dir_path / f"{model_version}.distribution_eval.csv"
    control_comparison_json_path = distribution_report_dir_path / f"{model_version}.vs_control.json"

    joblib.dump(distribution_model, model_path)

    metadata_payload = {
        "model_name": model_name,
        "target_column": target_column,
        "model_version": model_version,
        "data_version_hash": _resolve_data_version_hash(dataset),
        "holdout_season": int(holdout_season),
        "train_row_count": int(len(train_frame)),
        "holdout_row_count": int(len(holdout_frame)),
        "runtime_versions": collect_runtime_versions(),
        "model_family": "parallel_run_distribution_lane",
        "distribution_family": DEFAULT_DISTRIBUTION_FAMILY_NAME,
        "distribution_lane_stage": 3,
        "mean_head_source_artifact_path": _relative_to_project(mean_head.metadata_path),
        "mean_head_source_model_path": _relative_to_project(mean_head.model_path),
        "mean_head_model_family": mean_head.metadata.get("model_family"),
        "fitted_distribution_family_name": DEFAULT_DISTRIBUTION_FAMILY_NAME,
        "feature_selection_mode": feature_selection_mode,
        "forced_delta_feature_count": int(forced_delta_feature_count),
        "time_series_splits": int(splitter.n_splits),
        "feature_columns": list(mean_head.feature_columns),
        "feature_columns_by_head": {
            "mu": list(mean_head.feature_columns),
            "dispersion": list(dispersion_selection.feature_columns),
            "p_zero_extra": list(zero_selection.feature_columns),
        },
        "selected_features_by_head": {
            "mu": list(mean_head.feature_columns),
            "dispersion": list(dispersion_selection.feature_columns),
            "p_zero_extra": list(zero_selection.feature_columns),
        },
        "feature_selection_diagnostics_by_head": {
            "dispersion": _selection_to_payload(dispersion_selection),
            "p_zero_extra": _selection_to_payload(zero_selection),
        },
        "head_estimator_params": resolved_head_params,
        "head_feature_importance_rankings": {
            "dispersion": _extract_head_feature_importance_rankings(
                dispersion_estimator,
                dispersion_selection.feature_columns,
                top_feature_count=top_feature_count,
            ),
            "p_zero_extra": _extract_head_feature_importance_rankings(
                zero_estimator,
                zero_selection.feature_columns,
                top_feature_count=top_feature_count,
            ),
        },
        "global_negative_binomial_fit": dataclass_to_dict(global_nb_fit),
        "global_zero_adjustment_fit": dataclass_to_dict(global_zero_adjustment_fit),
        "dispersion_blend_weight": float(dispersion_blend_weight),
        "zero_blend_weight": float(zero_blend_weight),
        "mean_metrics": mean_metrics,
        "distribution_metrics": distribution_metrics,
        "zero_calibration": distribution_metrics["zero_calibration"],
        "tail_calibration": distribution_metrics["tail_calibration"],
        "dispersion_summary_statistics_holdout": summarize_array(
            holdout_dispersion_size,
            name="dispersion_size",
        ),
        "overdispersion_alpha_summary_statistics_holdout": summarize_array(
            blended_holdout_alpha,
            name="overdispersion_alpha",
        ),
        "zero_mass_summary_statistics_holdout": {
            "baseline_zero_probability": summarize_array(
                baseline_zero_holdout,
                name="baseline_zero_probability",
            ),
            "adjusted_zero_probability": summarize_array(
                adjusted_zero_holdout,
                name="adjusted_zero_probability",
            ),
            "p_zero_extra": summarize_array(
                adjusted_zero_holdout - baseline_zero_holdout,
                name="p_zero_extra",
            ),
        },
        "calibration_bins": {
            "shutout_probability": distribution_metrics["zero_calibration"]["p_0"]["bins"],
            "tail_probabilities": {
                key: value["bins"]
                for key, value in distribution_metrics["tail_calibration"].items()
            },
        },
        "control_baseline_reference": {
            "family": control_metrics["family"],
            "holdout_metrics": control_metrics["holdout_metrics"],
        },
        "comparison_to_control": comparison_to_control,
        "training_data_inspection": {
            "parquet_path": _relative_to_project(Path(str(training_data_inspection.parquet_path)))
            if training_data_inspection.parquet_path
            else None,
            "data_version_hash": training_data_inspection.data_version_hash,
            "schema_name": training_data_inspection.schema_name,
            "schema_version": training_data_inspection.schema_version,
            "row_count": training_data_inspection.row_count,
        },
        "trained_at": datetime.now(UTC).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

    summary_payload = {
        "model_version": model_version,
        "data_version_hash": metadata_payload["data_version_hash"],
        "holdout_season": int(holdout_season),
        "artifact": {
            "model_path": _relative_to_project(model_path),
            "metadata_path": _relative_to_project(metadata_path),
            "distribution_report_json_path": _relative_to_project(distribution_report_json_path),
            "distribution_report_csv_path": _relative_to_project(distribution_report_csv_path),
            "control_comparison_json_path": _relative_to_project(control_comparison_json_path),
            "mean_metrics": mean_metrics,
            "distribution_metrics": distribution_metrics,
            "comparison_to_control": comparison_to_control,
        },
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    distribution_report_payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "artifact_path": _relative_to_project(metadata_path),
        "model_path": _relative_to_project(model_path),
        "training_data_path": _relative_to_project(Path(str(training_data_inspection.parquet_path)))
        if training_data_inspection.parquet_path
        else None,
        "model_name": model_name,
        "model_version": model_version,
        "target_column": target_column,
        "holdout_season": int(holdout_season),
        "distribution_family": DEFAULT_DISTRIBUTION_FAMILY_NAME,
        "mean_metrics": mean_metrics,
        "holdout_metrics": distribution_metrics,
        "head_summaries": {
            "mu": {
                "feature_count": int(len(mean_head.feature_columns)),
                "feature_columns": list(mean_head.feature_columns),
                "source_artifact_path": _relative_to_project(mean_head.metadata_path),
            },
            "dispersion": {
                "feature_count": int(len(dispersion_selection.feature_columns)),
                "feature_columns": list(dispersion_selection.feature_columns),
            },
            "p_zero_extra": {
                "feature_count": int(len(zero_selection.feature_columns)),
                "feature_columns": list(zero_selection.feature_columns),
            },
        },
        "holdout_component_summary": {
            "dispersion": summarize_array(holdout_dispersion_size, name="dispersion_size"),
            "overdispersion_alpha": summarize_array(blended_holdout_alpha, name="overdispersion_alpha"),
            "baseline_zero_probability": summarize_array(
                baseline_zero_holdout,
                name="baseline_zero_probability",
            ),
            "adjusted_zero_probability": summarize_array(
                adjusted_zero_holdout,
                name="adjusted_zero_probability",
            ),
            "p_zero_extra": summarize_array(
                adjusted_zero_holdout - baseline_zero_holdout,
                name="p_zero_extra",
            ),
        },
    }
    distribution_report_json_path.write_text(
        json.dumps(distribution_report_payload, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            flatten_distribution_report_row(
                model_version=model_version,
                metadata_path=metadata_path,
                mean_metrics=mean_metrics,
                distribution_metrics=distribution_metrics,
                comparison_to_control=comparison_to_control,
            )
        ]
    ).to_csv(distribution_report_csv_path, index=False)
    control_comparison_json_path.write_text(
        json.dumps(comparison_to_control, indent=2),
        encoding="utf-8",
    )

    return RunDistributionTrainingArtifact(
        model_name=model_name,
        target_column=target_column,
        model_version=model_version,
        model_path=model_path,
        metadata_path=metadata_path,
        summary_path=summary_path,
        distribution_report_json_path=distribution_report_json_path,
        distribution_report_csv_path=distribution_report_csv_path,
        control_comparison_json_path=control_comparison_json_path,
        holdout_season=int(holdout_season),
        holdout_metrics=mean_metrics,
        distribution_metrics=distribution_metrics,
        comparison_to_control=comparison_to_control,
    )


def load_mean_head_reference(
    mean_artifact_metadata_path: str | Path,
) -> MeanHeadReference:
    metadata_path = Path(mean_artifact_metadata_path)
    if not metadata_path.is_absolute():
        metadata_path = (PROJECT_ROOT / metadata_path).resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_path = _metadata_path_to_model_path(metadata_path)
    estimator = joblib.load(model_path)
    feature_columns = [str(column) for column in metadata.get("feature_columns", [])]
    if not feature_columns:
        raise ValueError(f"Mean artifact metadata at {metadata_path} is missing feature_columns")
    return MeanHeadReference(
        estimator=estimator,
        metadata=metadata,
        metadata_path=metadata_path,
        model_path=model_path,
        feature_columns=feature_columns,
    )


def evaluate_control_distribution_baseline(
    *,
    train_actual: np.ndarray,
    train_mean: np.ndarray,
    holdout_actual: np.ndarray,
    holdout_mean: np.ndarray,
    calibration_bin_count: int,
    tail_probability: float,
) -> dict[str, Any]:
    negative_binomial_fit = fit_negative_binomial_dispersion(train_actual, train_mean)
    train_support = build_variable_nb_support(
        actual_counts=train_actual,
        mean_predictions=train_mean,
        dispersion_size=np.full(len(train_mean), float(negative_binomial_fit.dispersion_size)),
        tail_probability=tail_probability,
    )
    baseline_train_pmf = negative_binomial_pmf_matrix(
        train_mean,
        train_support,
        dispersion_size=float(negative_binomial_fit.dispersion_size),
    )
    zero_adjustment_fit = fit_zero_adjustment(
        train_actual,
        event_probability(baseline_train_pmf, train_support, kind="eq", threshold=0),
    )
    holdout_support = build_variable_nb_support(
        actual_counts=holdout_actual,
        mean_predictions=holdout_mean,
        dispersion_size=np.full(len(holdout_mean), float(negative_binomial_fit.dispersion_size)),
        tail_probability=tail_probability,
    )
    holdout_pmf = zero_adjusted_negative_binomial_pmf_matrix(
        holdout_mean,
        holdout_support,
        dispersion_size=float(negative_binomial_fit.dispersion_size),
        zero_adjustment_delta=float(zero_adjustment_fit.delta),
    )
    return {
        "family": DEFAULT_DISTRIBUTION_FAMILY_NAME,
        "negative_binomial_fit": dataclass_to_dict(negative_binomial_fit),
        "zero_adjustment_fit": dataclass_to_dict(zero_adjustment_fit),
        "holdout_metrics": summarize_distribution_metrics(
            holdout_actual,
            holdout_pmf,
            holdout_support,
            calibration_bin_count=calibration_bin_count,
        ),
    }


def build_control_comparison(
    *,
    stage3_mean_metrics: Mapping[str, Any],
    stage3_distribution_metrics: Mapping[str, Any],
    control_mean_metrics: Mapping[str, Any],
    control_distribution_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    stage3_bias = float(stage3_mean_metrics["predicted_mean"] - stage3_mean_metrics["actual_mean"])
    control_bias = float(control_mean_metrics["predicted_mean"] - control_mean_metrics["actual_mean"])
    stage3_tail_errors = [
        float(item["absolute_error"])
        for item in stage3_distribution_metrics["tail_calibration"].values()
    ]
    control_tail_errors = [
        float(item["absolute_error"])
        for item in control_distribution_metrics["tail_calibration"].values()
    ]
    rmse_regression_pct = (
        ((float(stage3_mean_metrics["rmse"]) - float(control_mean_metrics["rmse"])) / float(control_mean_metrics["rmse"]))
        * 100.0
        if float(control_mean_metrics["rmse"]) > 0.0
        else 0.0
    )
    bias_regression = abs(stage3_bias) - abs(control_bias)
    tail_error_change = max(stage3_tail_errors) - max(control_tail_errors)
    guardrails = {
        "rmse_within_2pct": rmse_regression_pct <= 2.0,
        "mean_bias_within_0_15_runs": abs(stage3_bias) <= 0.15,
        "tail_calibration_stable": tail_error_change <= 0.05,
    }
    return {
        "stage3_mean_metrics": dict(stage3_mean_metrics),
        "control_mean_metrics": dict(control_mean_metrics),
        "stage3_distribution_metrics": dict(stage3_distribution_metrics),
        "control_distribution_metrics": dict(control_distribution_metrics),
        "deltas": {
            "mean_crps": float(stage3_distribution_metrics["mean_crps"])
            - float(control_distribution_metrics["mean_crps"]),
            "mean_negative_log_score": float(stage3_distribution_metrics["mean_negative_log_score"])
            - float(control_distribution_metrics["mean_negative_log_score"]),
            "rmse": float(stage3_mean_metrics["rmse"]) - float(control_mean_metrics["rmse"]),
            "rmse_regression_pct": rmse_regression_pct,
            "mean_bias": stage3_bias - control_bias,
            "mean_bias_abs_regression": bias_regression,
            "zero_calibration_abs_error": float(stage3_distribution_metrics["zero_calibration"]["p_0"]["absolute_error"])
            - float(control_distribution_metrics["zero_calibration"]["p_0"]["absolute_error"]),
            "max_tail_abs_error": max(stage3_tail_errors) - max(control_tail_errors),
        },
        "improvement_flags": {
            "beats_control_on_crps": float(stage3_distribution_metrics["mean_crps"])
            < float(control_distribution_metrics["mean_crps"]),
            "beats_control_on_negative_log_score": float(stage3_distribution_metrics["mean_negative_log_score"])
            < float(control_distribution_metrics["mean_negative_log_score"]),
            "improves_zero_calibration": float(stage3_distribution_metrics["zero_calibration"]["p_0"]["absolute_error"])
            < float(control_distribution_metrics["zero_calibration"]["p_0"]["absolute_error"]),
            "improves_max_tail_calibration": max(stage3_tail_errors) < max(control_tail_errors),
        },
        "guardrails": guardrails,
        "catastrophic_regression": not all(guardrails.values()),
    }


def build_variable_nb_support(
    *,
    actual_counts: Sequence[int] | np.ndarray,
    mean_predictions: Sequence[float] | np.ndarray,
    dispersion_size: Sequence[float] | np.ndarray,
    tail_probability: float,
) -> list[int]:
    counts = np.asarray(actual_counts, dtype=int)
    means = clip_mean_predictions(np.asarray(mean_predictions, dtype=float))
    size = np.clip(np.asarray(dispersion_size, dtype=float), MIN_MEAN_PREDICTION, None)
    probability = np.clip(size / (size + means), MIN_PROBABILITY, 1.0 - MIN_PROBABILITY)
    quantiles = nbinom.ppf(
        1.0 - min(max(float(tail_probability), 1e-12), 1e-3),
        size,
        probability,
    )
    if not np.all(np.isfinite(quantiles)):
        variance = means + ((means**2) / size)
        quantiles = means + (10.0 * np.sqrt(variance))
    support_max = int(max(np.max(counts, initial=0), math.ceil(float(np.max(quantiles))), 15))
    return list(range(support_max + 1))


def zero_adjusted_negative_binomial_pmf_matrix_by_row(
    mean_predictions: Sequence[float] | np.ndarray,
    support: Sequence[int],
    *,
    dispersion_size: Sequence[float] | np.ndarray,
    adjusted_zero_probability: Sequence[float] | np.ndarray,
) -> np.ndarray:
    baseline_pmf = negative_binomial_pmf_matrix_by_row(
        mean_predictions,
        support,
        dispersion_size=dispersion_size,
    )
    baseline_zero_probability = np.clip(
        baseline_pmf[:, 0],
        MIN_PROBABILITY,
        1.0 - MIN_PROBABILITY,
    )
    adjusted_zero = np.clip(
        np.asarray(adjusted_zero_probability, dtype=float),
        MIN_PROBABILITY,
        1.0 - MIN_PROBABILITY,
    )
    adjusted = baseline_pmf.copy()
    adjusted[:, 0] = adjusted_zero
    positive_mass = np.clip(1.0 - baseline_zero_probability, MIN_PROBABILITY, 1.0)
    adjusted[:, 1:] = baseline_pmf[:, 1:] * (((1.0 - adjusted_zero) / positive_mass)[:, None])
    row_sums = adjusted.sum(axis=1, keepdims=True)
    return adjusted / np.where(row_sums <= 0.0, 1.0, row_sums)


def negative_binomial_pmf_matrix_by_row(
    mean_predictions: Sequence[float] | np.ndarray,
    support: Sequence[int],
    *,
    dispersion_size: Sequence[float] | np.ndarray,
) -> np.ndarray:
    means = clip_mean_predictions(np.asarray(mean_predictions, dtype=float))
    resolved_support = np.asarray(list(support), dtype=int)
    size = np.clip(np.asarray(dispersion_size, dtype=float), MIN_MEAN_PREDICTION, None)
    probability = np.clip(size / (size + means), MIN_PROBABILITY, 1.0 - MIN_PROBABILITY)
    pmf = nbinom.pmf(resolved_support[None, :], size[:, None], probability[:, None])
    row_sums = pmf.sum(axis=1, keepdims=True)
    return pmf / np.where(row_sums <= 0.0, 1.0, row_sums)


def summarize_array(values: Sequence[float] | np.ndarray, *, name: str) -> dict[str, float | str]:
    array = np.asarray(values, dtype=float)
    return {
        "name": name,
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }


def flatten_distribution_report_row(
    *,
    model_version: str,
    metadata_path: Path,
    mean_metrics: Mapping[str, Any],
    distribution_metrics: Mapping[str, Any],
    comparison_to_control: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model_version": model_version,
        "artifact_path": _relative_to_project(metadata_path),
        "family": DEFAULT_DISTRIBUTION_FAMILY_NAME,
        "rmse": mean_metrics["rmse"],
        "mean_bias": float(mean_metrics["predicted_mean"] - mean_metrics["actual_mean"]),
        "mean_crps": distribution_metrics["mean_crps"],
        "mean_negative_log_score": distribution_metrics["mean_negative_log_score"],
        "zero_abs_error": distribution_metrics["zero_calibration"]["p_0"]["absolute_error"],
        "ge_3_abs_error": distribution_metrics["tail_calibration"]["p_ge_3"]["absolute_error"],
        "ge_5_abs_error": distribution_metrics["tail_calibration"]["p_ge_5"]["absolute_error"],
        "ge_10_abs_error": distribution_metrics["tail_calibration"]["p_ge_10"]["absolute_error"],
        "beats_control_on_crps": comparison_to_control["improvement_flags"]["beats_control_on_crps"],
        "beats_control_on_negative_log_score": comparison_to_control["improvement_flags"][
            "beats_control_on_negative_log_score"
        ],
        "rmse_within_2pct": comparison_to_control["guardrails"]["rmse_within_2pct"],
        "mean_bias_within_0_15_runs": comparison_to_control["guardrails"]["mean_bias_within_0_15_runs"],
        "tail_calibration_stable": comparison_to_control["guardrails"]["tail_calibration_stable"],
    }


def resolve_mean_artifact_metadata_path(
    *,
    current_control_path: str | Path = DEFAULT_CURRENT_CONTROL_PATH,
    explicit_mean_artifact_metadata_path: str | Path | None = None,
) -> Path:
    if explicit_mean_artifact_metadata_path is not None:
        path = Path(explicit_mean_artifact_metadata_path)
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    current_control_resolved = Path(current_control_path)
    if not current_control_resolved.is_absolute():
        current_control_resolved = (PROJECT_ROOT / current_control_resolved).resolve()
    payload = json.loads(current_control_resolved.read_text(encoding="utf-8"))
    selected_artifact_path = payload.get("selected_artifact_path")
    if not selected_artifact_path:
        raise ValueError(f"Current control payload at {current_control_resolved} is missing selected_artifact_path")
    selected = Path(str(selected_artifact_path))
    return selected if selected.is_absolute() else (PROJECT_ROOT / selected).resolve()


def _resolve_head_params(
    mean_metadata: Mapping[str, Any],
    *,
    xgb_n_jobs: int,
    head_param_overrides: Mapping[str, float | int] | None,
) -> dict[str, float | int]:
    params = dict(DEFAULT_HEAD_PARAM_OVERRIDES)
    best_params = mean_metadata.get("best_params", {})
    if isinstance(best_params, Mapping):
        for key in (
            "max_depth",
            "n_estimators",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "gamma",
            "reg_alpha",
            "reg_lambda",
        ):
            if key in best_params:
                params[key] = best_params[key]
    if head_param_overrides is not None:
        params.update(dict(head_param_overrides))
    params["n_jobs"] = max(1, int(xgb_n_jobs))
    return {
        str(key): int(value) if key in {"max_depth", "n_estimators", "min_child_weight", "n_jobs"} else float(value)
        for key, value in params.items()
    }


def _build_regression_estimator(
    *,
    random_state: int,
    estimator_params: Mapping[str, float | int],
) -> XGBRegressor:
    params = dict(estimator_params)
    n_jobs = int(params.pop("n_jobs", DEFAULT_XGBOOST_N_JOBS))
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=random_state,
        tree_method="hist",
        n_jobs=n_jobs,
        verbosity=0,
        **params,
    )


def _build_classifier_estimator(
    *,
    random_state: int,
    estimator_params: Mapping[str, float | int],
) -> XGBClassifier:
    params = dict(estimator_params)
    n_jobs = int(params.pop("n_jobs", DEFAULT_XGBOOST_N_JOBS))
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        tree_method="hist",
        n_jobs=n_jobs,
        verbosity=0,
        **params,
    )


def _fit_dispersion_estimator(
    *,
    train_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_values: Sequence[float],
    random_state: int,
    estimator_params: Mapping[str, float | int],
) -> Any:
    target = np.asarray(target_values, dtype=float)
    if np.allclose(target, target[0]):
        return ConstantValueRegressor(float(target[0]))
    estimator = _build_regression_estimator(
        random_state=random_state,
        estimator_params=estimator_params,
    )
    estimator.fit(train_frame.loc[:, feature_columns], target)
    return estimator


def _fit_zero_estimator(
    *,
    train_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_values: Sequence[int],
    random_state: int,
    estimator_params: Mapping[str, float | int],
) -> Any:
    target = np.asarray(target_values, dtype=int)
    unique = np.unique(target)
    if len(unique) == 1:
        return ConstantProbabilityClassifier(float(unique[0]))
    estimator = _build_classifier_estimator(
        random_state=random_state,
        estimator_params=estimator_params,
    )
    estimator.fit(train_frame.loc[:, feature_columns], target)
    return estimator


def _generate_regression_oof_predictions(
    *,
    train_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_values: Sequence[float],
    splitter: Any,
    random_state: int,
    estimator_params: Mapping[str, float | int],
) -> pd.Series:
    predictions = pd.Series(index=train_frame.index, dtype=float)
    target = np.asarray(target_values, dtype=float)
    for fold_index, (train_indices, test_indices) in enumerate(splitter.split(train_frame), start=1):
        train_target = target[train_indices]
        if np.allclose(train_target, train_target[0]):
            fold_model: Any = ConstantValueRegressor(float(train_target[0]))
        else:
            fold_model = _build_regression_estimator(
                random_state=random_state + fold_index,
                estimator_params=estimator_params,
            )
            fold_model.fit(
                train_frame.iloc[train_indices].loc[:, feature_columns],
                train_target,
            )
        predictions.iloc[test_indices] = fold_model.predict(
            train_frame.iloc[test_indices].loc[:, feature_columns]
        )
    return predictions


def _generate_classifier_oof_predictions(
    *,
    train_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_values: Sequence[int],
    splitter: Any,
    random_state: int,
    estimator_params: Mapping[str, float | int],
) -> pd.Series:
    predictions = pd.Series(index=train_frame.index, dtype=float)
    target = np.asarray(target_values, dtype=int)
    for fold_index, (train_indices, test_indices) in enumerate(splitter.split(train_frame), start=1):
        train_target = target[train_indices]
        unique = np.unique(train_target)
        if len(unique) == 1:
            fold_model: Any = ConstantProbabilityClassifier(float(unique[0]))
        else:
            fold_model = _build_classifier_estimator(
                random_state=random_state + fold_index,
                estimator_params=estimator_params,
            )
            fold_model.fit(
                train_frame.iloc[train_indices].loc[:, feature_columns],
                train_target,
            )
        predictions.iloc[test_indices] = _predict_zero_probability(
            fold_model,
            train_frame.iloc[test_indices].loc[:, feature_columns],
            fallback_probability=np.full(len(test_indices), np.mean(train_target), dtype=float),
        )
    return predictions


def _build_dispersion_training_target(
    *,
    actual_counts: Sequence[int] | np.ndarray,
    mean_predictions: Sequence[float] | np.ndarray,
    global_alpha: float,
) -> np.ndarray:
    actual = np.asarray(actual_counts, dtype=float)
    mean = clip_mean_predictions(np.asarray(mean_predictions, dtype=float))
    raw_alpha = ((actual - mean) ** 2 - actual) / np.maximum(mean**2, MIN_MEAN_PREDICTION)
    alpha_floor = max(DEFAULT_DISPERSION_ALPHA_FLOOR, float(global_alpha) * 0.25)
    alpha_cap = max(DEFAULT_DISPERSION_ALPHA_CAP, float(global_alpha) * 8.0)
    resolved = np.where(np.isfinite(raw_alpha) & (raw_alpha > 0.0), raw_alpha, float(global_alpha))
    clipped = np.clip(resolved, alpha_floor, alpha_cap)
    return np.log(clipped)


def _blend_dispersion_alpha(
    *,
    global_alpha: float,
    raw_alpha_predictions: Sequence[float] | np.ndarray,
    blend_weight: float,
) -> np.ndarray:
    global_alpha_array = np.full(
        len(np.asarray(raw_alpha_predictions, dtype=float)),
        max(float(global_alpha), DEFAULT_DISPERSION_ALPHA_FLOOR),
        dtype=float,
    )
    raw = np.clip(np.asarray(raw_alpha_predictions, dtype=float), DEFAULT_DISPERSION_ALPHA_FLOOR, None)
    weight = float(np.clip(blend_weight, 0.0, 1.0))
    return np.exp(((1.0 - weight) * np.log(global_alpha_array)) + (weight * np.log(raw)))


def _blend_zero_probability(
    *,
    baseline_zero_probability: Sequence[float] | np.ndarray,
    raw_zero_probability: Sequence[float] | np.ndarray,
    blend_weight: float,
) -> np.ndarray:
    baseline = np.clip(np.asarray(baseline_zero_probability, dtype=float), MIN_PROBABILITY, 1.0 - MIN_PROBABILITY)
    raw = np.clip(np.asarray(raw_zero_probability, dtype=float), MIN_PROBABILITY, 1.0 - MIN_PROBABILITY)
    weight = float(np.clip(blend_weight, 0.0, 1.0))
    return expit(logit(baseline) + (weight * (logit(raw) - logit(baseline))))


def _apply_zero_adjustment_delta(
    baseline_zero_probability: Sequence[float] | np.ndarray,
    *,
    delta: float,
) -> np.ndarray:
    baseline = np.clip(np.asarray(baseline_zero_probability, dtype=float), MIN_PROBABILITY, 1.0 - MIN_PROBABILITY)
    return expit(logit(baseline) + float(delta))


def _optimize_dispersion_blend_weight(
    *,
    actual_counts: Sequence[int] | np.ndarray,
    mean_predictions: Sequence[float] | np.ndarray,
    global_alpha: float,
    raw_alpha_predictions: Sequence[float] | np.ndarray,
) -> float:
    actual = np.asarray(actual_counts, dtype=int)
    mean = clip_mean_predictions(np.asarray(mean_predictions, dtype=float))
    raw_alpha = np.clip(np.asarray(raw_alpha_predictions, dtype=float), DEFAULT_DISPERSION_ALPHA_FLOOR, None)

    def objective(weight: float) -> float:
        blended_alpha = _blend_dispersion_alpha(
            global_alpha=global_alpha,
            raw_alpha_predictions=raw_alpha,
            blend_weight=float(weight),
        )
        dispersion_size = _alpha_to_dispersion_size(blended_alpha)
        probability = np.clip(dispersion_size / (dispersion_size + mean), MIN_PROBABILITY, 1.0 - MIN_PROBABILITY)
        log_likelihood = nbinom.logpmf(actual, dispersion_size, probability)
        return float(-np.sum(np.nan_to_num(log_likelihood, nan=-1e12, neginf=-1e12)))

    result = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")
    return float(result.x if result.success else 0.0)


def _optimize_zero_blend_weight(
    *,
    observed_zero: Sequence[int] | np.ndarray,
    baseline_zero_probability: Sequence[float] | np.ndarray,
    raw_zero_probability: Sequence[float] | np.ndarray,
) -> float:
    observed = np.asarray(observed_zero, dtype=float)
    baseline = np.clip(np.asarray(baseline_zero_probability, dtype=float), MIN_PROBABILITY, 1.0 - MIN_PROBABILITY)
    raw = np.clip(np.asarray(raw_zero_probability, dtype=float), MIN_PROBABILITY, 1.0 - MIN_PROBABILITY)

    def objective(weight: float) -> float:
        adjusted = _blend_zero_probability(
            baseline_zero_probability=baseline,
            raw_zero_probability=raw,
            blend_weight=float(weight),
        )
        log_likelihood = (
            observed * np.log(adjusted)
            + (1.0 - observed) * np.log(np.clip(1.0 - adjusted, MIN_PROBABILITY, 1.0))
        )
        return float(-np.sum(log_likelihood))

    result = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")
    return float(result.x if result.success else 0.0)


def _predict_dispersion_alpha(
    estimator: Any,
    feature_frame: pd.DataFrame,
    *,
    fallback_alpha: float,
) -> np.ndarray:
    if hasattr(estimator, "predict"):
        raw = np.asarray(estimator.predict(feature_frame), dtype=float)
    else:
        raw = np.full(len(feature_frame), math.log(max(fallback_alpha, DEFAULT_DISPERSION_ALPHA_FLOOR)), dtype=float)
    alpha = np.exp(np.clip(raw, math.log(DEFAULT_DISPERSION_ALPHA_FLOOR), math.log(32.0)))
    return np.clip(alpha, DEFAULT_DISPERSION_ALPHA_FLOOR, None)


def _predict_zero_probability(
    estimator: Any,
    feature_frame: pd.DataFrame,
    *,
    fallback_probability: Sequence[float] | np.ndarray,
) -> np.ndarray:
    fallback = np.clip(np.asarray(fallback_probability, dtype=float), MIN_PROBABILITY, 1.0 - MIN_PROBABILITY)
    if hasattr(estimator, "predict_proba"):
        probability = np.asarray(estimator.predict_proba(feature_frame), dtype=float)
        if probability.ndim == 2 and probability.shape[1] >= 2:
            return np.clip(probability[:, 1], MIN_PROBABILITY, 1.0 - MIN_PROBABILITY)
    return fallback


def _variable_negative_binomial_zero_probability(
    mean_predictions: Sequence[float] | np.ndarray,
    dispersion_size: Sequence[float] | np.ndarray,
) -> np.ndarray:
    mean = clip_mean_predictions(np.asarray(mean_predictions, dtype=float))
    size = np.clip(np.asarray(dispersion_size, dtype=float), MIN_MEAN_PREDICTION, None)
    probability = np.clip(size / (size + mean), MIN_PROBABILITY, 1.0 - MIN_PROBABILITY)
    return np.asarray(nbinom.pmf(0, size, probability), dtype=float)


def _alpha_to_dispersion_size(alpha: Sequence[float] | np.ndarray) -> np.ndarray:
    resolved = np.clip(np.asarray(alpha, dtype=float), DEFAULT_DISPERSION_ALPHA_FLOOR, None)
    return 1.0 / resolved


def _select_head_feature_columns(
    *,
    train_frame: pd.DataFrame,
    candidate_feature_columns: Sequence[str],
    target_values: Sequence[float] | np.ndarray,
    feature_selection_mode: str,
    max_feature_count: int,
    forced_delta_feature_count: int,
) -> Any:
    frame = train_frame.copy()
    target_column = "__stage3_head_target__"
    frame[target_column] = np.asarray(target_values, dtype=float)
    if feature_selection_mode == "flat":
        return _select_run_count_feature_columns_flat(
            frame,
            target_column=target_column,
            candidate_feature_columns=candidate_feature_columns,
            max_feature_count=max_feature_count,
            forced_delta_count=forced_delta_feature_count,
        )
    if feature_selection_mode == "bucketed":
        if forced_delta_feature_count > 0:
            raise ValueError("forced_delta_feature_count is currently supported only with flat selection mode")
        return _select_run_count_feature_columns_bucketed(
            frame,
            target_column=target_column,
            candidate_feature_columns=candidate_feature_columns,
            max_feature_count=max_feature_count,
        )
    if forced_delta_feature_count > 0:
        raise ValueError("forced_delta_feature_count is currently supported only with flat selection mode")
    return _select_run_count_feature_columns(
        frame,
        target_column=target_column,
        candidate_feature_columns=candidate_feature_columns,
        max_feature_count=max_feature_count,
    )


def _extract_head_feature_importance_rankings(
    estimator: Any,
    feature_columns: Sequence[str],
    *,
    top_feature_count: int,
) -> list[dict[str, float | str]]:
    if isinstance(estimator, (ConstantValueRegressor, ConstantProbabilityClassifier)):
        return []
    return _extract_feature_importance_rankings(
        estimator,
        feature_columns,
        top_feature_count=min(int(top_feature_count), len(feature_columns)),
    )


def _selection_to_payload(selection: Any) -> dict[str, Any]:
    return {
        "feature_columns": list(selection.feature_columns),
        "bucket_counts": dict(selection.bucket_counts),
        "bucket_targets": dict(selection.bucket_targets),
        "selected_features_by_bucket": {
            key: list(value) for key, value in selection.selected_features_by_bucket.items()
        },
        "forced_delta_features": list(selection.forced_delta_features),
        "omitted_top_features_by_bucket": {
            key: [dict(item) for item in value]
            for key, value in selection.omitted_top_features_by_bucket.items()
        },
        "family_decisions": [dict(item) for item in selection.family_decisions],
    }


def _validate_mean_head_reference(
    mean_head: MeanHeadReference,
    *,
    target_column: str,
    holdout_season: int,
) -> None:
    if str(mean_head.metadata.get("target_column")) != target_column:
        raise ValueError(
            f"Mean head target mismatch: expected {target_column}, got {mean_head.metadata.get('target_column')}"
        )
    metadata_holdout = mean_head.metadata.get("holdout_season")
    if metadata_holdout is not None and int(metadata_holdout) != int(holdout_season):
        raise ValueError(
            f"Mean head holdout mismatch: expected {holdout_season}, got {metadata_holdout}"
        )


def _metadata_path_to_model_path(metadata_path: Path) -> Path:
    suffix = ".metadata.json"
    if not metadata_path.name.endswith(suffix):
        raise ValueError(f"Expected metadata artifact path, got {metadata_path}")
    return metadata_path.with_name(metadata_path.name[: -len(suffix)] + ".joblib")


def _relative_to_project(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


__all__ = [
    "DEFAULT_CURRENT_CONTROL_PATH",
    "DEFAULT_DISTRIBUTION_FAMILY_NAME",
    "DEFAULT_DISTRIBUTION_MODEL_NAME",
    "DEFAULT_DISTRIBUTION_REPORT_DIR",
    "DEFAULT_TARGET_COLUMN",
    "RunDistributionModel",
    "RunDistributionTrainingArtifact",
    "build_control_comparison",
    "build_variable_nb_support",
    "evaluate_control_distribution_baseline",
    "flatten_distribution_report_row",
    "load_mean_head_reference",
    "negative_binomial_pmf_matrix_by_row",
    "resolve_mean_artifact_metadata_path",
    "summarize_array",
    "train_run_distribution_model",
    "zero_adjusted_negative_binomial_pmf_matrix_by_row",
]
