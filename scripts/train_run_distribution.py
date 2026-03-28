from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.data_builder import inspect_run_count_training_data, validate_run_count_training_data  # noqa: E402
from src.model.run_distribution_trainer import (  # noqa: E402
    DEFAULT_CURRENT_CONTROL_PATH,
    DEFAULT_DISTRIBUTION_MODEL_NAME,
    DEFAULT_DISTRIBUTION_REPORT_DIR,
    DEFAULT_TARGET_COLUMN,
    resolve_mean_artifact_metadata_path,
    train_run_distribution_model,
)


console = Console()
DEFAULT_TRAINING_DATA = Path("data/training/ParquetDefault.parquet")


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


def _default_experiment_name() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"2026-away-dist-zanb-v1-controlmu-flat-3f-{timestamp}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the Stage 3 away-run distribution lane without replacing the control lane.",
    )
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--training-data", default=str(DEFAULT_TRAINING_DATA))
    parser.add_argument("--start", "--start-year", dest="start_year", type=int, default=2018)
    parser.add_argument("--end", "--end-year", dest="end_year", type=int, default=2025)
    parser.add_argument("--holdout", "--holdout-season", dest="holdout_season", type=int, default=2025)
    parser.add_argument("--Folds", "--folds", dest="folds", type=int, default=3)
    parser.add_argument(
        "--feature-selection-mode",
        choices=("grouped", "bucketed", "flat"),
        default="flat",
    )
    parser.add_argument("--forced-delta-count", type=int, default=0)
    parser.add_argument("--XGBWork", "--xgb-workers", dest="xgb_workers", type=int, default=None)
    parser.add_argument(
        "--current-control",
        default=str(DEFAULT_CURRENT_CONTROL_PATH),
        help="Path to Stage 1 current_control.json. Ignored when --mean-artifact-metadata is supplied.",
    )
    parser.add_argument(
        "--mean-artifact-metadata",
        default=None,
        help="Optional explicit metadata path for the frozen mu head. Defaults to current_control.json selection.",
    )
    parser.add_argument("--distribution-report-dir", default=str(DEFAULT_DISTRIBUTION_REPORT_DIR))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    experiment_name = args.experiment or _default_experiment_name()
    output_dir = Path("data/models") / experiment_name
    mean_artifact_metadata_path = resolve_mean_artifact_metadata_path(
        current_control_path=args.current_control,
        explicit_mean_artifact_metadata_path=args.mean_artifact_metadata,
    )

    resolved_xgb_workers = args.xgb_workers
    if resolved_xgb_workers is not None:
        resolved_xgb_workers = max(1, int(resolved_xgb_workers))
        os.environ["MLB_XGBOOST_N_JOBS"] = str(resolved_xgb_workers)

    if args.dry_run:
        console.print(
            f"experiment={experiment_name} training_data={args.training_data} "
            f"season_range={args.start_year}-{args.end_year} holdout={args.holdout_season} "
            f"folds={args.folds} feature_selection_mode={args.feature_selection_mode} "
            f"forced_delta_count={args.forced_delta_count} "
            f"mean_artifact_metadata={mean_artifact_metadata_path}"
        )
        return 0

    validated_training_data = validate_run_count_training_data(Path(args.training_data))
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
    console.print("[bold green]Training parquet[/bold green]")
    console.print(f"  parquet_path={inspection.parquet_path or '<in-memory>'}")
    console.print(f"  data_version_hash={inspection.data_version_hash}")
    console.print(f"  schema={inspection.schema_name} v{inspection.schema_version}")
    console.print(f"  row_count={inspection.row_count}")
    console.print(f"  mean_head_artifact={mean_artifact_metadata_path}")
    console.print(
        f"[bold green]Training Stage 3 distribution lane[/bold green] "
        f"experiment={experiment_name} season_range={args.start_year}-{args.end_year} "
        f"holdout={args.holdout_season} folds={args.folds} "
        f"feature_selection_mode={args.feature_selection_mode}"
    )

    artifact = train_run_distribution_model(
        training_data=filtered_training_data,
        output_dir=output_dir,
        mean_artifact_metadata_path=mean_artifact_metadata_path,
        holdout_season=args.holdout_season,
        target_column=DEFAULT_TARGET_COLUMN,
        model_name=DEFAULT_DISTRIBUTION_MODEL_NAME,
        feature_selection_mode=args.feature_selection_mode,
        forced_delta_feature_count=args.forced_delta_count,
        time_series_splits=args.folds,
        xgb_n_jobs=resolved_xgb_workers or int(os.getenv("MLB_XGBOOST_N_JOBS", "1")),
        distribution_report_dir=args.distribution_report_dir,
    )

    comparison = artifact.comparison_to_control
    console.print(f"[bold green]Done[/bold green] metadata={artifact.metadata_path}")
    console.print(
        f"RMSE={artifact.holdout_metrics['rmse']:.4f} "
        f"CRPS={artifact.distribution_metrics['mean_crps']:.6f} "
        f"NegLogScore={artifact.distribution_metrics['mean_negative_log_score']:.6f}"
    )
    console.print(
        f"beats_control_on_crps={comparison['improvement_flags']['beats_control_on_crps']} "
        f"beats_control_on_negative_log_score={comparison['improvement_flags']['beats_control_on_negative_log_score']} "
        f"catastrophic_regression={comparison['catastrophic_regression']}"
    )
    console.print(
        json.dumps(
            {
                "model_version": artifact.model_version,
                "model_path": str(artifact.model_path),
                "metadata_path": str(artifact.metadata_path),
                "distribution_report_json_path": str(artifact.distribution_report_json_path),
                "distribution_report_csv_path": str(artifact.distribution_report_csv_path),
                "control_comparison_json_path": str(artifact.control_comparison_json_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
