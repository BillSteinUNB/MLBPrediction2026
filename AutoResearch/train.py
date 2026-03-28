from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import fnmatch
import json
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterator, Sequence

import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error

AUTORESEARCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = AUTORESEARCH_ROOT.parent
for path in (AUTORESEARCH_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.model.data_builder import inspect_run_count_training_data, validate_run_count_training_data  # noqa: E402
import src.model.run_count_trainer as rct  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logging.getLogger("src.model.run_count_trainer").setLevel(logging.INFO)
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_NAME = "full_game_away_runs_model"
TARGET_COLUMN = "final_away_score"
TRAINING_DATA_PATH = REPO_ROOT / "data" / "training" / "ParquetDefault.parquet"
MODEL_OUTPUT_ROOT = REPO_ROOT / "data" / "models"
HOLDOUT_SEASON = 2025
OPTUNA_WORKERS = max(1, int(os.getenv("MLB_OPTUNA_N_JOBS", "2")))
BLEND_MODE = "xgb_only"
CV_AGGREGATION_MODE = "mean"
LIGHTGBM_PARAM_MODE = "derived"
EXPERIMENT_PREFIX = "autoresearch-away-runs"

# AGENT_CONFIG_START
MAX_FEATURES = 80
SELECTOR_TYPE = "pearson"
BUCKET_QUOTAS = [80, 0, 0, 0]
EXCLUDE_PATTERNS: list[str] = []
FORCE_INCLUDE_PATTERNS: list[str] = ["*_7g", "*_7s", "*_delta_7v30g", "*_delta_7v30s"]
FORCED_DELTA_COUNT = 8
TRIALS = 120
FOLDS = 3
# AGENT_CONFIG_END

FULL_TRIALS = 300
FULL_FOLDS = 3
FAST_EARLY_STOPPING_ROUNDS = 30
FULL_EARLY_STOPPING_ROUNDS = 40
SUPPORTED_SELECTOR_TYPES = {"pearson", "bucketed", "ablation"}
_BUCKET_ORDER = ("short_form", "medium_form", "delta", "context")


@dataclass(frozen=True, slots=True)
class EffectiveTrainingConfig:
    mode: str
    max_features: int
    selector_type: str
    feature_selection_mode: str
    bucket_targets: dict[str, int]
    exclude_patterns: list[str]
    force_include_patterns: list[str]
    forced_delta_count: int
    search_iterations: int
    time_series_splits: int
    early_stopping_rounds: int
    blend_mode: str
    cv_aggregation_mode: str
    lightgbm_param_mode: str


def resolve_effective_config(mode: str) -> EffectiveTrainingConfig:
    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"fast", "full"}:
        raise ValueError(f"Unsupported mode: {mode}")

    selector_type = SELECTOR_TYPE.strip().lower()
    if selector_type not in SUPPORTED_SELECTOR_TYPES:
        raise ValueError(
            f"SELECTOR_TYPE must be one of {sorted(SUPPORTED_SELECTOR_TYPES)}; got {SELECTOR_TYPE!r}"
        )

    if MAX_FEATURES <= 0:
        raise ValueError("MAX_FEATURES must be positive")

    bucket_targets = resolve_bucket_targets(max_features=MAX_FEATURES, bucket_quotas=BUCKET_QUOTAS)
    forced_delta_count = int(FORCED_DELTA_COUNT)
    return EffectiveTrainingConfig(
        mode=normalized_mode,
        max_features=int(MAX_FEATURES),
        selector_type=selector_type,
        feature_selection_mode="flat" if selector_type == "pearson" else "bucketed",
        bucket_targets=bucket_targets,
        exclude_patterns=[pattern for pattern in EXCLUDE_PATTERNS if str(pattern).strip()],
        force_include_patterns=[pattern for pattern in FORCE_INCLUDE_PATTERNS if str(pattern).strip()],
        forced_delta_count=forced_delta_count,
        search_iterations=int(TRIALS if normalized_mode == "fast" else FULL_TRIALS),
        time_series_splits=int(FOLDS if normalized_mode == "fast" else FULL_FOLDS),
        early_stopping_rounds=int(
            FAST_EARLY_STOPPING_ROUNDS if normalized_mode == "fast" else FULL_EARLY_STOPPING_ROUNDS
        ),
        blend_mode=BLEND_MODE,
        cv_aggregation_mode=CV_AGGREGATION_MODE,
        lightgbm_param_mode=LIGHTGBM_PARAM_MODE,
    )


def resolve_bucket_targets(*, max_features: int, bucket_quotas: Sequence[int]) -> dict[str, int]:
    quotas = [int(value) for value in bucket_quotas]
    if len(quotas) == 3:
        short_form, medium_form, context = quotas
        delta = 0
    elif len(quotas) == 4:
        short_form, medium_form, delta, context = quotas
    else:
        raise ValueError("BUCKET_QUOTAS must contain 3 or 4 integers")

    if any(value < 0 for value in (short_form, medium_form, delta, context)):
        raise ValueError("BUCKET_QUOTAS values must be non-negative")

    base_total = short_form + medium_form + delta + context
    if base_total > max_features:
        raise ValueError(
            f"BUCKET_QUOTAS total {base_total} cannot exceed MAX_FEATURES {max_features}"
        )

    return {
        "short_form": short_form,
        "medium_form": medium_form,
        "delta": delta,
        "context": context + (max_features - base_total),
    }


def build_experiment_name(mode: str, config: EffectiveTrainingConfig) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    selector_tag = config.selector_type
    feature_tag = f"{config.max_features}f"
    search_tag = f"{config.search_iterations}x{config.time_series_splits}"
    ablation_tag = _slugify("-".join(config.exclude_patterns + config.force_include_patterns) or "baseline")
    return f"{EXPERIMENT_PREFIX}-{mode}-{selector_tag}-{feature_tag}-{search_tag}-{ablation_tag}-{timestamp}"


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return normalized or "baseline"


def _matches_pattern(feature_name: str, pattern: str) -> bool:
    normalized = pattern.strip()
    if not normalized:
        return False
    if any(token in normalized for token in "*?[]"):
        return fnmatch.fnmatch(feature_name, normalized)
    return normalized in feature_name


def _matches_any_pattern(feature_name: str, patterns: Sequence[str]) -> bool:
    return any(_matches_pattern(feature_name, pattern) for pattern in patterns)


def _ordered_unique(columns: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for column in columns:
        if column in seen:
            continue
        seen.add(column)
        ordered.append(column)
    return ordered


@contextmanager
def apply_training_overrides(config: EffectiveTrainingConfig) -> Iterator[None]:
    original_max_features = rct.DEFAULT_RUN_COUNT_MAX_FEATURE_COUNT
    original_short_form = rct.DEFAULT_RUN_COUNT_SHORT_FORM_FEATURE_COUNT
    original_medium_form = rct.DEFAULT_RUN_COUNT_MEDIUM_FORM_FEATURE_COUNT
    original_delta = rct.DEFAULT_RUN_COUNT_DELTA_FEATURE_COUNT
    original_context = rct.DEFAULT_RUN_COUNT_CONTEXT_FEATURE_COUNT
    original_candidate_resolver = rct._resolve_run_count_candidate_feature_columns
    original_flat_selector = rct._select_run_count_feature_columns_flat
    original_bucketed_selector = rct._select_run_count_feature_columns_bucketed

    rct.DEFAULT_RUN_COUNT_MAX_FEATURE_COUNT = config.max_features
    rct.DEFAULT_RUN_COUNT_SHORT_FORM_FEATURE_COUNT = config.bucket_targets["short_form"]
    rct.DEFAULT_RUN_COUNT_MEDIUM_FORM_FEATURE_COUNT = config.bucket_targets["medium_form"]
    rct.DEFAULT_RUN_COUNT_DELTA_FEATURE_COUNT = config.bucket_targets["delta"]
    rct.DEFAULT_RUN_COUNT_CONTEXT_FEATURE_COUNT = config.bucket_targets["context"]

    def _resolve_candidates(dataframe: pd.DataFrame) -> rct.RunCountCandidateResolution:
        base_resolution = original_candidate_resolver(dataframe)
        raw_numeric_columns = rct._resolve_numeric_feature_columns(dataframe)
        forced_columns = [
            column
            for column in raw_numeric_columns
            if _matches_any_pattern(column, config.force_include_patterns)
        ]
        merged_candidates = _ordered_unique([*base_resolution.candidate_columns, *forced_columns])
        filtered_candidates: list[str] = []
        pattern_excluded_count = 0
        forced_candidate_count = 0

        for column in merged_candidates:
            forced = _matches_any_pattern(column, config.force_include_patterns)
            if forced and column not in base_resolution.candidate_columns:
                forced_candidate_count += 1
            if not forced and _matches_any_pattern(column, config.exclude_patterns):
                pattern_excluded_count += 1
                continue
            filtered_candidates.append(column)

        return rct.RunCountCandidateResolution(
            candidate_columns=filtered_candidates,
            excluded_candidate_counts={
                **base_resolution.excluded_candidate_counts,
                "pattern_excluded": pattern_excluded_count,
                "forced_candidate_includes": forced_candidate_count,
            },
        )

    def _wrap_flat_selector(selector):
        def _wrapped(
            dataframe: pd.DataFrame,
            *,
            target_column: str,
            candidate_feature_columns: Sequence[str],
            max_feature_count: int = rct.DEFAULT_RUN_COUNT_MAX_FEATURE_COUNT,
            forced_delta_count: int = rct.DEFAULT_RUN_COUNT_FORCED_DELTA_FEATURE_COUNT,
        ) -> rct.RunCountFeatureSelectionResult:
            if not config.force_include_patterns:
                return selector(
                    dataframe,
                    target_column=target_column,
                    candidate_feature_columns=candidate_feature_columns,
                    max_feature_count=max_feature_count,
                    forced_delta_count=forced_delta_count,
                )
            return _select_flat_with_force_includes(
                dataframe,
                target_column=target_column,
                candidate_feature_columns=candidate_feature_columns,
                max_feature_count=max_feature_count,
                forced_delta_count=forced_delta_count,
                base_selector=selector,
                force_include_patterns=config.force_include_patterns,
            )

        return _wrapped

    def _wrap_bucketed_selector(selector):
        def _wrapped(
            dataframe: pd.DataFrame,
            *,
            target_column: str,
            candidate_feature_columns: Sequence[str],
            max_feature_count: int = rct.DEFAULT_RUN_COUNT_MAX_FEATURE_COUNT,
        ) -> rct.RunCountFeatureSelectionResult:
            if not config.force_include_patterns:
                return selector(
                    dataframe,
                    target_column=target_column,
                    candidate_feature_columns=candidate_feature_columns,
                    max_feature_count=max_feature_count,
                )
            bucket_targets = resolve_bucket_targets(
                max_features=max_feature_count,
                bucket_quotas=[
                    config.bucket_targets["short_form"],
                    config.bucket_targets["medium_form"],
                    config.bucket_targets["delta"],
                    config.bucket_targets["context"],
                ],
            )
            return _select_bucketed_with_force_includes(
                dataframe,
                target_column=target_column,
                candidate_feature_columns=candidate_feature_columns,
                bucket_targets=bucket_targets,
                force_include_patterns=config.force_include_patterns,
            )

        return _wrapped

    rct._resolve_run_count_candidate_feature_columns = _resolve_candidates
    rct._select_run_count_feature_columns_flat = _wrap_flat_selector(original_flat_selector)
    rct._select_run_count_feature_columns_bucketed = _wrap_bucketed_selector(
        original_bucketed_selector
    )

    try:
        yield
    finally:
        rct.DEFAULT_RUN_COUNT_MAX_FEATURE_COUNT = original_max_features
        rct.DEFAULT_RUN_COUNT_SHORT_FORM_FEATURE_COUNT = original_short_form
        rct.DEFAULT_RUN_COUNT_MEDIUM_FORM_FEATURE_COUNT = original_medium_form
        rct.DEFAULT_RUN_COUNT_DELTA_FEATURE_COUNT = original_delta
        rct.DEFAULT_RUN_COUNT_CONTEXT_FEATURE_COUNT = original_context
        rct._resolve_run_count_candidate_feature_columns = original_candidate_resolver
        rct._select_run_count_feature_columns_flat = original_flat_selector
        rct._select_run_count_feature_columns_bucketed = original_bucketed_selector


def _select_flat_with_force_includes(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    candidate_feature_columns: Sequence[str],
    max_feature_count: int,
    forced_delta_count: int,
    base_selector,
    force_include_patterns: Sequence[str],
) -> rct.RunCountFeatureSelectionResult:
    base_result = base_selector(
        dataframe,
        target_column=target_column,
        candidate_feature_columns=candidate_feature_columns,
        max_feature_count=max_feature_count,
        forced_delta_count=forced_delta_count,
    )
    scored = rct._score_run_count_candidate_features(
        dataframe,
        target_column=target_column,
        candidate_feature_columns=candidate_feature_columns,
    )
    forced_pattern_features = [
        feature_name
        for _, feature_name in scored
        if _matches_any_pattern(feature_name, force_include_patterns)
    ]
    selected = _ordered_unique([*forced_pattern_features, *base_result.feature_columns])[:max_feature_count]
    forced_delta_features = [
        feature_name for feature_name in selected if rct._resolve_run_count_feature_bucket(feature_name) == "delta"
    ]
    selected_set = set(selected)
    omitted = [
        {"feature": feature_name, "score": score}
        for score, feature_name in scored
        if feature_name not in selected_set
    ][:10]
    return rct.RunCountFeatureSelectionResult(
        feature_columns=sorted(selected),
        bucket_counts={"flat": len(selected)},
        bucket_targets={"flat": max_feature_count},
        selected_features_by_bucket={"flat": sorted(selected)},
        forced_delta_features=sorted(forced_delta_features),
        omitted_top_features_by_bucket={"flat": omitted},
        family_decisions=[],
    )


def _select_bucketed_with_force_includes(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    candidate_feature_columns: Sequence[str],
    bucket_targets: dict[str, int],
    force_include_patterns: Sequence[str],
) -> rct.RunCountFeatureSelectionResult:
    rankings_by_bucket = rct._rank_run_count_candidate_features(
        dataframe,
        target_column=target_column,
        candidate_feature_columns=candidate_feature_columns,
    )
    selected_by_bucket: dict[str, list[str]] = {bucket: [] for bucket in bucket_targets}
    selected_feature_set: set[str] = set()
    max_feature_count = sum(bucket_targets.values())

    for bucket_name in _BUCKET_ORDER:
        for _, feature_name in rankings_by_bucket[bucket_name]:
            if len(selected_feature_set) >= max_feature_count:
                break
            if not _matches_any_pattern(feature_name, force_include_patterns):
                continue
            if feature_name in selected_feature_set:
                continue
            selected_by_bucket[bucket_name].append(feature_name)
            selected_feature_set.add(feature_name)

    for bucket_name in _BUCKET_ORDER:
        for _, feature_name in rankings_by_bucket[bucket_name]:
            if len(selected_by_bucket[bucket_name]) >= bucket_targets[bucket_name]:
                break
            if feature_name in selected_feature_set:
                continue
            selected_by_bucket[bucket_name].append(feature_name)
            selected_feature_set.add(feature_name)

    if len(selected_feature_set) < max_feature_count:
        for bucket_name in _BUCKET_ORDER:
            for _, feature_name in rankings_by_bucket[bucket_name]:
                if len(selected_feature_set) >= max_feature_count:
                    break
                if feature_name in selected_feature_set:
                    continue
                selected_by_bucket[bucket_name].append(feature_name)
                selected_feature_set.add(feature_name)

    return rct.RunCountFeatureSelectionResult(
        feature_columns=sorted(selected_feature_set),
        bucket_counts={bucket: len(columns) for bucket, columns in selected_by_bucket.items()},
        bucket_targets=dict(bucket_targets),
        selected_features_by_bucket={
            bucket: sorted(columns) for bucket, columns in selected_by_bucket.items()
        },
        omitted_top_features_by_bucket=rct._build_omitted_top_features_by_bucket(
            ranked_features_by_bucket=rankings_by_bucket,
            selected_feature_set=selected_feature_set,
        ),
        family_decisions=[],
    )


def _print_training_data_summary(inspection) -> None:
    print("\nTraining data")
    print(f"  parquet_path={inspection.parquet_path or '<in-memory>'}")
    print(f"  data_version_hash={inspection.data_version_hash}")
    print(f"  schema={inspection.schema_name}")
    print(f"  schema_version={inspection.schema_version}")
    print(f"  row_count={inspection.row_count}")
    print(f"  feature_column_count={inspection.feature_column_count}")
    if inspection.metadata_path is not None:
        print(f"  metadata_path={inspection.metadata_path}")
    print(
        "  temporal_delta_features="
        + ("present" if inspection.has_temporal_delta_features else "missing")
    )


def _compute_cv_rmse(
    *,
    training_data: pd.DataFrame | str | Path,
    artifact,
    time_series_splits: int,
    blend_mode: str,
) -> float:
    validated_training_data = validate_run_count_training_data(training_data)
    dataset = rct._load_training_dataframe(validated_training_data)
    frame = rct._prepare_run_count_frame(dataset, target_column=artifact.target_column)
    train_frame = frame.loc[frame["season"] < artifact.holdout_season].copy()
    oof_predictions = rct._generate_run_count_oof_predictions(
        best_params=artifact.best_params,
        training_frame=train_frame,
        feature_columns=artifact.feature_columns,
        target_column=artifact.target_column,
        random_state=rct.DEFAULT_RANDOM_STATE,
        time_series_splits=time_series_splits,
        lightgbm_param_mode=rct.DEFAULT_RUN_COUNT_LIGHTGBM_PARAM_MODE,
    )
    blend_selection = rct._resolve_run_count_blend_selection(
        actual=oof_predictions["actual"],
        xgboost_predictions=oof_predictions["xgboost"],
        lightgbm_predictions=oof_predictions["lightgbm"],
        blend_mode=blend_mode,
    )
    blended_predictions = rct._blend_run_count_predictions(
        oof_predictions["xgboost"],
        oof_predictions["lightgbm"],
        xgboost_weight=blend_selection.xgboost_weight,
        lightgbm_weight=blend_selection.lightgbm_weight,
    )
    return float(mean_squared_error(oof_predictions["actual"], blended_predictions) ** 0.5)


def run_training(
    *,
    mode: str,
    experiment_name: str | None = None,
    training_data: str | Path = TRAINING_DATA_PATH,
    show_training_data_summary: bool = True,
) -> dict[str, Any]:
    config = resolve_effective_config(mode)
    training_path = Path(training_data)
    inspection = inspect_run_count_training_data(training_path)
    if show_training_data_summary:
        _print_training_data_summary(inspection)

    started_at = datetime.now(UTC)
    effective_experiment_name = experiment_name or build_experiment_name(mode, config)
    output_dir = MODEL_OUTPUT_ROOT / effective_experiment_name

    rct.DEFAULT_RUN_COUNT_MODEL_SPECS = (
        {"model_name": MODEL_NAME, "target_column": TARGET_COLUMN},
    )
    with apply_training_overrides(config):
        result = rct.train_run_count_models(
            training_data=training_path,
            output_dir=output_dir,
            holdout_season=HOLDOUT_SEASON,
            search_iterations=config.search_iterations,
            time_series_splits=config.time_series_splits,
            optuna_workers=OPTUNA_WORKERS,
            early_stopping_rounds=config.early_stopping_rounds,
            feature_selection_mode=config.feature_selection_mode,
            cv_aggregation_mode=config.cv_aggregation_mode,
            lightgbm_param_mode=config.lightgbm_param_mode,
            blend_mode=config.blend_mode,
        )

    artifact = result.models[MODEL_NAME]
    cv_rmse = _compute_cv_rmse(
        training_data=training_path,
        artifact=artifact,
        time_series_splits=config.time_series_splits,
        blend_mode=config.blend_mode,
    )
    completed_at = datetime.now(UTC)
    payload = {
        "mode": config.mode,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_seconds": (completed_at - started_at).total_seconds(),
        "experiment_name": effective_experiment_name,
        "output_dir": str(output_dir),
        "summary_path": str(result.summary_path),
        "config": asdict(config),
        "metrics": {
            "holdout_r2": float(artifact.holdout_metrics["r2"]),
            "holdout_rmse": float(artifact.holdout_metrics["rmse"]),
            "holdout_mae": float(artifact.holdout_metrics["mae"]),
            "cv_rmse": cv_rmse,
            "cv_metric_name": artifact.cv_metric_name,
            "cv_metric_value": float(artifact.cv_best_score),
            "holdout_poisson_deviance": float(artifact.holdout_metrics["poisson_deviance"]),
            "rmse_improvement_vs_naive_pct": float(
                artifact.holdout_metrics["rmse_improvement_vs_naive_pct"]
            ),
        },
        "model": {
            "model_name": artifact.model_name,
            "target_column": artifact.target_column,
            "model_version": artifact.model_version,
            "model_path": str(artifact.model_path),
            "metadata_path": str(artifact.metadata_path),
            "final_n_estimators": int(artifact.final_n_estimators),
            "best_params": dict(artifact.best_params),
            "feature_columns": list(artifact.feature_columns),
            "feature_selection_bucket_targets": dict(artifact.feature_selection_bucket_targets),
            "feature_selection_bucket_counts": dict(artifact.feature_selection_bucket_counts),
            "excluded_candidate_counts": dict(artifact.excluded_candidate_counts),
        },
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run autoresearch training experiments")
    parser.add_argument("--mode", choices=("fast", "full"), required=True)
    parser.add_argument("--experiment-name")
    parser.add_argument("--training-data", default=str(TRAINING_DATA_PATH))
    parser.add_argument("--json-output", action="store_true")
    args = parser.parse_args(argv)

    payload = run_training(
        mode=args.mode,
        experiment_name=args.experiment_name,
        training_data=args.training_data,
        show_training_data_summary=not args.json_output,
    )
    if args.json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    metrics = payload["metrics"]
    print(f"\nDone. Summary: {payload['summary_path']}")
    print(
        "  "
        f"mode={payload['mode']} "
        f"holdout_r2={metrics['holdout_r2'] * 100:.2f}% "
        f"holdout_rmse={metrics['holdout_rmse']:.4f} "
        f"cv_rmse={metrics['cv_rmse']:.4f} "
        f"{metrics['cv_metric_name']}={metrics['cv_metric_value']:.4f}"
    )
    print(
        "  "
        f"selector={payload['config']['selector_type']} "
        f"features={payload['config']['max_features']} "
        f"bucket_targets={payload['config']['bucket_targets']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
