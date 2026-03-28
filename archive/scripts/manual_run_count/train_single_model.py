r"""
Train a single run-count model in isolation.

This script trains from an existing parquet only. For feature-engineering experiments,
the safer path is `scripts/rebuild_and_train_single_model.py`, which rebuilds the parquet
before training and avoids stale-feature failures.

Usage:
    .\.venv\Scripts\python.exe scripts\train_single_model.py
    .\.venv\Scripts\python.exe scripts\train_single_model.py --profile fast
    .\.venv\Scripts\python.exe scripts\train_single_model.py --profile fast --feature-selection-mode grouped
    .\.venv\Scripts\python.exe scripts\train_single_model.py --profile fast --cv-aggregation-mode recent_weighted
"""

import argparse
from pathlib import Path
import logging
import os
import re
import sys
import optuna

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logging.getLogger("src.model.run_count_trainer").setLevel(logging.INFO)
logging.getLogger("src.model.xgboost_trainer").setLevel(logging.INFO)

optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.model.data_builder import (  # noqa: E402
    build_training_dataset,
    inspect_run_count_training_data,
)
import src.model.run_count_trainer as rct  # noqa: E402
from src.model.single_model_profiles import resolve_single_model_experiment_profile  # noqa: E402

# Configuration
MODEL_NAME = "full_game_away_runs_model"
TARGET_COLUMN = "final_away_score"
OPTUNA_WORKERS = max(1, int(os.getenv("MLB_OPTUNA_N_JOBS", "2")))
START_YEAR = 2018
END_YEAR = 2025


def _is_delta_experiment_name(experiment_name: str) -> bool:
    return bool(re.search(r"delta", experiment_name, flags=re.IGNORECASE))


def _print_training_data_summary(*, rebuilt: bool, refresh_raw_data: bool, inspection) -> None:
    print("\nTraining data")
    print(f"  parquet_path={inspection.parquet_path or '<in-memory>'}")
    print(f"  rebuilt={'yes' if rebuilt else 'no'}")
    print(f"  refresh_raw_data={'yes' if refresh_raw_data else 'no'}")
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


def _maybe_rebuild_training_data(
    *,
    training_path: Path,
    force_rebuild: bool,
    refresh_raw_data: bool,
) -> bool:
    if not force_rebuild:
        return False
    build_training_dataset(
        start_year=START_YEAR,
        end_year=END_YEAR,
        output_path=training_path,
        refresh=refresh_raw_data,
        refresh_raw_data=refresh_raw_data,
    )
    return True


def _inspect_and_validate_training_data(*, experiment_name: str, training_path: Path):
    inspection = inspect_run_count_training_data(training_path)
    if _is_delta_experiment_name(experiment_name) and not inspection.has_temporal_delta_features:
        missing_columns = ", ".join(inspection.missing_temporal_delta_columns[:6])
        extra_missing = len(inspection.missing_temporal_delta_columns) - min(
            len(inspection.missing_temporal_delta_columns),
            6,
        )
        if extra_missing > 0:
            missing_columns = f"{missing_columns}, ... (+{extra_missing} more)"
        raise ValueError(
            "Refusing to run delta experiment "
            f"{experiment_name!r} with stale parquet {training_path}. "
            "Required temporal delta features are missing: "
            f"{missing_columns}. Rebuild first with --force-rebuild or use "
            "scripts/rebuild_and_train_single_model.py."
        )
    return inspection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train a single run-count model from an existing parquet. "
            "Recommended safe path for feature work: scripts/rebuild_and_train_single_model.py"
        ),
    )
    parser.add_argument(
        "--profile", choices=("smoke", "fast", "full", "flat-fast", "flat-full"), default="fast"
    )
    parser.add_argument("--experiment")
    parser.add_argument("--holdout-season", type=int, default=2025)
    parser.add_argument("--training-data", default="data/training/training_data_2018_2025.parquet")
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild the parquet before training so research runs cannot use stale data.",
    )
    parser.add_argument(
        "--refresh-data",
        "--refresh-raw-data",
        dest="refresh_raw_data",
        action="store_true",
        help="When rebuilding, bypass cached upstream raw/derived data instead of reusing cached inputs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--feature-selection-mode",
        choices=("grouped", "bucketed", "flat"),
        default=rct.DEFAULT_RUN_COUNT_FEATURE_SELECTION_MODE,
    )
    parser.add_argument(
        "--cv-aggregation-mode",
        choices=("mean", "recent_weighted"),
        default=rct.DEFAULT_RUN_COUNT_CV_AGGREGATION_MODE,
    )
    parser.add_argument(
        "--lightgbm-param-mode",
        choices=("independent", "derived"),
        default=rct.DEFAULT_RUN_COUNT_LIGHTGBM_PARAM_MODE,
    )
    parser.add_argument(
        "--blend-mode",
        choices=rct.RUN_COUNT_BLEND_MODES,
        default=rct.DEFAULT_RUN_COUNT_BLEND_MODE,
    )
    args = parser.parse_args(argv)

    profile = resolve_single_model_experiment_profile(
        args.profile,
        experiment_name_override=args.experiment,
    )
    feature_selection_mode = (
        "flat" if profile.profile_name.startswith("flat-") else args.feature_selection_mode
    )
    rct.DEFAULT_RUN_COUNT_MODEL_SPECS = (
        {"model_name": MODEL_NAME, "target_column": TARGET_COLUMN},
    )

    if args.dry_run:
        print(
            f"profile={profile.profile_name} experiment={profile.experiment_name} "
            f"search_iters={profile.search_iterations} splits={profile.time_series_splits} "
            f"early_stop={profile.early_stopping_rounds} optuna_workers={OPTUNA_WORKERS} "
            f"feature_selection_mode={feature_selection_mode} "
            f"cv_aggregation_mode={args.cv_aggregation_mode} "
            f"lightgbm_param_mode={args.lightgbm_param_mode} "
            f"blend_mode={args.blend_mode} "
            f"force_rebuild={args.force_rebuild} "
            f"refresh_raw_data={args.refresh_raw_data} "
            "safe_path=scripts/rebuild_and_train_single_model.py"
        )
        return 0

    training_path = Path(args.training_data)
    rebuilt = _maybe_rebuild_training_data(
        training_path=training_path,
        force_rebuild=args.force_rebuild,
        refresh_raw_data=args.refresh_raw_data,
    )
    inspection = _inspect_and_validate_training_data(
        experiment_name=profile.experiment_name,
        training_path=training_path,
    )
    _print_training_data_summary(
        rebuilt=rebuilt,
        refresh_raw_data=args.refresh_raw_data,
        inspection=inspection,
    )

    result = rct.train_run_count_models(
        training_data=training_path,
        output_dir=Path(f"data/models/{profile.experiment_name}"),
        holdout_season=args.holdout_season,
        search_space=profile.search_space,
        search_iterations=profile.search_iterations,
        time_series_splits=profile.time_series_splits,
        optuna_workers=OPTUNA_WORKERS,
        early_stopping_rounds=profile.early_stopping_rounds,
        feature_selection_mode=feature_selection_mode,
        cv_aggregation_mode=args.cv_aggregation_mode,
        lightgbm_param_mode=args.lightgbm_param_mode,
        blend_mode=args.blend_mode,
    )

    print(f"\nDone. Summary: {result.summary_path}")
    model = result.models[MODEL_NAME]
    r2 = model.holdout_metrics["r2"] * 100
    rmse = model.holdout_metrics["rmse"]
    improvement = model.holdout_metrics["rmse_improvement_vs_naive_pct"]
    print(
        f"profile={profile.profile_name}  R2={r2:.2f}%  RMSE={rmse:.4f}  "
        f"RMSE_impr={improvement:.2f}%"
    )
    print(
        f"n_est={model.final_n_estimators}  optuna_workers={model.optuna_parallel_workers}  "
        f"feature_selection_mode={feature_selection_mode}  "
        f"cv_aggregation_mode={args.cv_aggregation_mode}  "
        f"lightgbm_param_mode={args.lightgbm_param_mode}  "
        f"blend_mode={args.blend_mode}  best_params={model.best_params}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
