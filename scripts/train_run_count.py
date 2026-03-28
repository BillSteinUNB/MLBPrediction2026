from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import optuna
import pandas as pd
from rich.console import Console


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.data_builder import inspect_run_count_training_data, validate_run_count_training_data  # noqa: E402
import src.model.run_count_trainer as rct  # noqa: E402
from src.model.single_model_profiles import resolve_single_model_experiment_profile  # noqa: E402


console = Console()
MODEL_NAME = "full_game_away_runs_model"
TARGET_COLUMN = "final_away_score"
DEFAULT_TRAINING_DATA = Path("data/training/ParquetDefault.parquet")


def _print_training_data_summary(inspection) -> None:
    console.print("[bold green]Training parquet[/bold green]")
    console.print(f"  parquet_path={inspection.parquet_path or '<in-memory>'}")
    console.print(f"  data_version_hash={inspection.data_version_hash}")
    console.print(f"  schema={inspection.schema_name} v{inspection.schema_version}")
    console.print(f"  row_count={inspection.row_count}")
    console.print(f"  feature_column_count={inspection.feature_column_count}")
    console.print(
        "  temporal_delta_features="
        + ("present" if inspection.has_temporal_delta_features else "missing")
    )


def _filter_training_data_by_season(
    dataset: pd.DataFrame,
    *,
    start_year: int | None,
    end_year: int | None,
) -> pd.DataFrame:
    if start_year is None and end_year is None:
        return dataset

    filtered = dataset.copy()
    if "season" in filtered.columns:
        season_values = pd.to_numeric(filtered["season"], errors="coerce")
    else:
        season_values = pd.to_datetime(filtered["game_date"], errors="coerce").dt.year

    if start_year is not None:
        filtered = filtered.loc[season_values >= int(start_year)].copy()
        season_values = season_values.loc[filtered.index]
    if end_year is not None:
        filtered = filtered.loc[season_values <= int(end_year)].copy()

    filtered.attrs.update(dataset.attrs)
    return filtered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the canonical manual run-count model from an existing parquet only.",
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "fast", "full", "flat-fast", "flat-full"),
        default="fast",
    )
    parser.add_argument("--experiment")
    parser.add_argument("--training-data", default=str(DEFAULT_TRAINING_DATA))
    parser.add_argument("--start", "--start-year", dest="start_year", type=int, default=None)
    parser.add_argument("--end", "--end-year", dest="end_year", type=int, default=None)
    parser.add_argument("--holdout", "--holdout-season", dest="holdout_season", type=int, default=2025)
    parser.add_argument("--XGBWork", "--xgb-workers", dest="xgb_workers", type=int, default=None)
    parser.add_argument(
        "--OptunaWork",
        "--optuna-workers",
        dest="optuna_workers",
        type=int,
        default=max(1, int(os.getenv("MLB_OPTUNA_N_JOBS", "2"))),
    )
    parser.add_argument("--Iterations", "--iterations", dest="iterations", type=int, default=None)
    parser.add_argument("--Folds", "--folds", dest="folds", type=int, default=None)
    parser.add_argument(
        "--feature-selection-mode",
        choices=("grouped", "bucketed", "flat"),
        default=rct.DEFAULT_RUN_COUNT_FEATURE_SELECTION_MODE,
    )
    parser.add_argument(
        "--forced-delta-count",
        type=int,
        default=rct.DEFAULT_RUN_COUNT_FORCED_DELTA_FEATURE_COUNT,
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    profile = resolve_single_model_experiment_profile(
        args.profile,
        experiment_name_override=args.experiment,
    )
    feature_selection_mode = (
        "flat" if profile.profile_name.startswith("flat-") else args.feature_selection_mode
    )
    resolved_iterations = int(args.iterations or profile.search_iterations)
    resolved_folds = int(args.folds or profile.time_series_splits)
    resolved_early_stopping_rounds = int(profile.early_stopping_rounds)
    resolved_xgb_workers = None
    if args.xgb_workers is not None:
        resolved_xgb_workers = max(1, int(args.xgb_workers))
        os.environ["MLB_XGBOOST_N_JOBS"] = str(resolved_xgb_workers)
    resolved_optuna_workers = max(1, int(args.optuna_workers))
    if args.dry_run:
        console.print(
            f"profile={profile.profile_name} experiment={profile.experiment_name} "
            f"search_iters={resolved_iterations} splits={resolved_folds} "
            f"early_stop={resolved_early_stopping_rounds} optuna_workers={resolved_optuna_workers} "
            f"xgb_workers={resolved_xgb_workers if resolved_xgb_workers is not None else 'env/default'} "
            f"season_range={args.start_year if args.start_year is not None else 'min'}-"
            f"{args.end_year if args.end_year is not None else 'max'} "
            f"holdout={args.holdout_season} "
            f"feature_selection_mode={feature_selection_mode} "
            f"forced_delta_count={args.forced_delta_count} "
            f"cv_aggregation_mode={args.cv_aggregation_mode} "
            f"lightgbm_param_mode={args.lightgbm_param_mode} "
            f"blend_mode={args.blend_mode}"
        )
        return 0

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    training_path = Path(args.training_data)
    validated_training_data = validate_run_count_training_data(training_path)
    filtered_training_data = _filter_training_data_by_season(
        validated_training_data,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    if filtered_training_data.empty:
        raise ValueError(
            f"No training rows remain after applying season filter start={args.start_year} end={args.end_year}"
        )
    inspection = inspect_run_count_training_data(filtered_training_data)
    _print_training_data_summary(inspection)
    if args.start_year is not None or args.end_year is not None:
        console.print(
            "  season_filter="
            f"{args.start_year if args.start_year is not None else 'min'}-"
            f"{args.end_year if args.end_year is not None else 'max'}"
        )

    console.print(
        f"[bold green]Training run-count model[/bold green] profile={profile.profile_name} "
        f"experiment={profile.experiment_name} "
        f"season_range={args.start_year if args.start_year is not None else 'min'}-"
        f"{args.end_year if args.end_year is not None else 'max'} "
        f"holdout={args.holdout_season} "
        f"xgb_workers={resolved_xgb_workers if resolved_xgb_workers is not None else 'env/default'} "
        f"optuna_workers={resolved_optuna_workers} iterations={resolved_iterations} folds={resolved_folds} "
        f"forced_delta_count={args.forced_delta_count}"
    )
    result = rct.train_run_count_models(
        training_data=filtered_training_data,
        output_dir=Path(f"data/models/{profile.experiment_name}"),
        holdout_season=args.holdout_season,
        search_space=profile.search_space,
        search_iterations=resolved_iterations,
        time_series_splits=resolved_folds,
        optuna_workers=resolved_optuna_workers,
        early_stopping_rounds=resolved_early_stopping_rounds,
        feature_selection_mode=feature_selection_mode,
        forced_delta_feature_count=args.forced_delta_count,
        cv_aggregation_mode=args.cv_aggregation_mode,
        lightgbm_param_mode=args.lightgbm_param_mode,
        blend_mode=args.blend_mode,
        model_specs=(
            {"model_name": MODEL_NAME, "target_column": TARGET_COLUMN},
        ),
    )

    model = result.models[MODEL_NAME]
    console.print(f"[bold green]Done[/bold green] summary={result.summary_path}")
    console.print(
        f"R2={model.holdout_metrics['r2'] * 100:.2f}% "
        f"RMSE={model.holdout_metrics['rmse']:.4f} "
        f"poisson_deviance={model.holdout_metrics['poisson_deviance']:.6f}"
    )
    console.print(
        f"n_est={model.final_n_estimators} "
        f"xgb_workers={resolved_xgb_workers if resolved_xgb_workers is not None else 'env/default'} "
        f"optuna_workers={resolved_optuna_workers} "
        f"iterations={resolved_iterations} folds={resolved_folds} "
        f"feature_selection_mode={feature_selection_mode} "
        f"forced_delta_count={args.forced_delta_count} "
        f"cv_aggregation_mode={args.cv_aggregation_mode} "
        f"lightgbm_param_mode={args.lightgbm_param_mode} "
        f"blend_mode={args.blend_mode}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
