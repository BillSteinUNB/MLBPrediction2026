from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from rich.console import Console


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.data_builder import inspect_run_count_training_data, validate_run_count_training_data  # noqa: E402
from src.model.mcmc_engine import DEFAULT_SIMULATION_COUNT, pad_probability_vector, simulate_away_game_distribution  # noqa: E402
from src.model.mcmc_feature_builder import build_mcmc_feature_bundle  # noqa: E402
from src.model.mcmc_pricing import (  # noqa: E402
    build_distribution_comparison,
    flatten_mcmc_report_row,
    summarize_away_run_distribution,
)
from src.model.run_count_trainer import _compute_holdout_metrics, _prepare_run_count_frame  # noqa: E402
from src.model.run_distribution_metrics import summarize_distribution_metrics  # noqa: E402
from src.model.run_distribution_trainer import (  # noqa: E402
    DEFAULT_CURRENT_CONTROL_PATH,
    DEFAULT_DISTRIBUTION_REPORT_DIR,
    DEFAULT_TARGET_COLUMN,
    evaluate_control_distribution_baseline,
    load_mean_head_reference,
    resolve_mean_artifact_metadata_path,
)
from src.model.xgboost_trainer import _build_model_version, _load_training_dataframe, _resolve_data_version_hash  # noqa: E402


console = Console()
DEFAULT_TRAINING_DATA = Path("data/training/ParquetDefault.parquet")
DEFAULT_MCMC_REPORT_DIR = Path("data/reports/run_count/mcmc")
DEFAULT_MODEL_NAME = "full_game_away_runs_mcmc_model"


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
    return f"2026-away-mcmc-markov-v1-controlmu-{timestamp}"


def _resolve_stage3_distribution_report_path(
    *,
    explicit_path: str | Path | None,
    report_dir: str | Path,
) -> Path | None:
    if explicit_path is not None:
        path = Path(explicit_path)
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    report_dir_path = Path(report_dir)
    if not report_dir_path.is_absolute():
        report_dir_path = (PROJECT_ROOT / report_dir_path).resolve()
    candidates = sorted(report_dir_path.glob("*.distribution_eval.json"), reverse=True)
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        artifact_path = str(payload.get("artifact_path") or "")
        if "away_runs_distribution_model" in artifact_path:
            return candidate
    return None


def _relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _row_seed(base_seed: int, row: pd.Series) -> int:
    game_pk = pd.to_numeric(pd.Series([row.get("game_pk")]), errors="coerce").iloc[0]
    if pd.notna(game_pk):
        return int(base_seed + int(game_pk))
    return int(base_seed + (int(row.name) * 17))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Stage 4 away-run Markov / Monte Carlo distribution lane.",
    )
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--training-data", default=str(DEFAULT_TRAINING_DATA))
    parser.add_argument("--start", "--start-year", dest="start_year", type=int, default=2018)
    parser.add_argument("--end", "--end-year", dest="end_year", type=int, default=2025)
    parser.add_argument("--holdout", "--holdout-season", dest="holdout_season", type=int, default=2025)
    parser.add_argument(
        "--current-control",
        default=str(DEFAULT_CURRENT_CONTROL_PATH),
        help="Path to Stage 1 current_control.json. Ignored when --mean-artifact-metadata is supplied.",
    )
    parser.add_argument("--mean-artifact-metadata", default=None)
    parser.add_argument("--stage3-report-json", default=None)
    parser.add_argument("--distribution-report-dir", default=str(DEFAULT_DISTRIBUTION_REPORT_DIR))
    parser.add_argument("--mcmc-report-dir", default=str(DEFAULT_MCMC_REPORT_DIR))
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATION_COUNT)
    parser.add_argument("--starter-innings", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260328)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    experiment_name = args.experiment or _default_experiment_name()
    output_dir = PROJECT_ROOT / "data" / "models" / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    stage3_report_path = _resolve_stage3_distribution_report_path(
        explicit_path=args.stage3_report_json,
        report_dir=args.distribution_report_dir,
    )
    mean_artifact_metadata_path = resolve_mean_artifact_metadata_path(
        current_control_path=args.current_control,
        explicit_mean_artifact_metadata_path=args.mean_artifact_metadata,
    )

    if args.dry_run:
        console.print(
            f"experiment={experiment_name} training_data={args.training_data} "
            f"season_range={args.start_year}-{args.end_year} holdout={args.holdout_season} "
            f"simulations={args.simulations} starter_innings={args.starter_innings} "
            f"mean_artifact_metadata={mean_artifact_metadata_path} stage3_report={stage3_report_path}"
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
    dataset = _load_training_dataframe(filtered_training_data)
    frame = _prepare_run_count_frame(dataset, target_column=DEFAULT_TARGET_COLUMN)
    train_frame = frame.loc[frame["season"] < int(args.holdout_season)].copy()
    holdout_frame = frame.loc[frame["season"] == int(args.holdout_season)].copy()
    if train_frame.empty:
        raise ValueError(f"No training rows found before holdout season {args.holdout_season}")
    if holdout_frame.empty:
        raise ValueError(f"No holdout rows found for season {args.holdout_season}")

    mean_head = load_mean_head_reference(mean_artifact_metadata_path)
    train_mean = np.clip(
        np.asarray(mean_head.estimator.predict(train_frame.loc[:, mean_head.feature_columns]), dtype=float),
        1e-9,
        None,
    )
    holdout_control_mean = np.clip(
        np.asarray(mean_head.estimator.predict(holdout_frame.loc[:, mean_head.feature_columns]), dtype=float),
        1e-9,
        None,
    )
    train_actual = train_frame[DEFAULT_TARGET_COLUMN].astype(int).to_numpy()
    holdout_actual = holdout_frame[DEFAULT_TARGET_COLUMN].astype(int).to_numpy()

    console.print("[bold green]Stage 4 MCMC lane[/bold green]")
    console.print(f"  parquet_path={inspection.parquet_path or '<in-memory>'}")
    console.print(f"  data_version_hash={inspection.data_version_hash}")
    console.print(f"  row_count={inspection.row_count}")
    console.print(f"  holdout_rows={len(holdout_frame)}")
    console.print(f"  mean_head_artifact={mean_artifact_metadata_path}")
    console.print(f"  stage3_report={stage3_report_path or '<none found>'}")
    console.print(
        f"  simulations={int(args.simulations)} starter_innings={int(args.starter_innings)} seed={int(args.seed)}"
    )

    per_row_probabilities: list[np.ndarray] = []
    per_row_summaries: list[dict[str, Any]] = []
    support_max = int(np.max(holdout_actual, initial=0))
    total_event_counts: dict[str, int] | None = None
    mean_runs_by_inning_accumulator: np.ndarray | None = None
    mean_plate_appearances_by_inning_accumulator: np.ndarray | None = None
    truncated_half_innings = 0

    for position, (_, row) in enumerate(holdout_frame.iterrows(), start=1):
        feature_bundle = build_mcmc_feature_bundle(row, target_mean_runs=float(holdout_control_mean[position - 1]))
        simulation = simulate_away_game_distribution(
            starter_profile=feature_bundle.starter_profile,
            bullpen_profile=feature_bundle.bullpen_profile,
            simulations=args.simulations,
            starter_innings=args.starter_innings,
            seed=_row_seed(int(args.seed), row),
        )
        support_max = max(support_max, int(simulation.support[-1]) if len(simulation.support) > 0 else 0)
        per_row_probabilities.append(simulation.pmf)
        summary = summarize_away_run_distribution(
            support=simulation.support,
            probabilities=simulation.pmf,
        )
        per_row_summaries.append(
            {
                "game_pk": int(row["game_pk"]),
                "game_date": str(row["game_date"]),
                "away_team": str(row["away_team"]),
                "home_team": str(row["home_team"]),
                "actual_away_runs": int(row[DEFAULT_TARGET_COLUMN]),
                "control_mean_runs": float(holdout_control_mean[position - 1]),
                "mcmc_expected_away_runs": float(summary.expected_away_runs),
                "shutout_probability": float(summary.shutout_probability),
                "p_ge_3": float(summary.tail_probabilities["p_ge_3"]),
                "p_ge_5": float(summary.tail_probabilities["p_ge_5"]),
                "p_ge_10": float(summary.tail_probabilities["p_ge_10"]),
                "away_run_pmf_json": json.dumps(summary.away_run_pmf, separators=(",", ":")),
                "diagnostics_json": json.dumps(simulation.diagnostics, separators=(",", ":")),
            }
        )

        diagnostics = simulation.diagnostics
        if total_event_counts is None:
            total_event_counts = {name: int(count) for name, count in diagnostics["event_counts"].items()}
        else:
            for event_name, count in diagnostics["event_counts"].items():
                total_event_counts[event_name] += int(count)

        inning_run_values = np.asarray(diagnostics["mean_runs_by_inning"], dtype=float)
        inning_pa_values = np.asarray(diagnostics["mean_plate_appearances_by_inning"], dtype=float)
        if mean_runs_by_inning_accumulator is None:
            mean_runs_by_inning_accumulator = inning_run_values
            mean_plate_appearances_by_inning_accumulator = inning_pa_values
        else:
            mean_runs_by_inning_accumulator = mean_runs_by_inning_accumulator + inning_run_values
            mean_plate_appearances_by_inning_accumulator = (
                mean_plate_appearances_by_inning_accumulator + inning_pa_values
            )
        truncated_half_innings += int(diagnostics["truncated_half_innings"])

        if position % 250 == 0 or position == len(holdout_frame):
            console.print(f"  simulated {position}/{len(holdout_frame)} holdout games")

    pmf_matrix = np.vstack(
        [pad_probability_vector(probabilities, support_max=support_max) for probabilities in per_row_probabilities]
    )
    support = np.arange(support_max + 1, dtype=int)
    distribution_metrics = summarize_distribution_metrics(
        holdout_actual,
        pmf_matrix,
        support,
    )
    pmf_expected_runs = np.sum(pmf_matrix * support[None, :], axis=1)
    mean_metrics = _compute_holdout_metrics(
        train_frame=train_frame,
        holdout_frame=holdout_frame,
        target_column=DEFAULT_TARGET_COLUMN,
        holdout_predictions=pmf_expected_runs,
    )
    control_baseline = evaluate_control_distribution_baseline(
        train_actual=train_actual,
        train_mean=train_mean,
        holdout_actual=holdout_actual,
        holdout_mean=holdout_control_mean,
        calibration_bin_count=10,
        tail_probability=1e-8,
    )
    control_mean_metrics = _compute_holdout_metrics(
        train_frame=train_frame,
        holdout_frame=holdout_frame,
        target_column=DEFAULT_TARGET_COLUMN,
        holdout_predictions=holdout_control_mean,
    )
    comparison_to_control = build_distribution_comparison(
        challenger_label="stage4_mcmc",
        challenger_mean_metrics=mean_metrics,
        challenger_distribution_metrics=distribution_metrics,
        baseline_label="control",
        baseline_mean_metrics=control_mean_metrics,
        baseline_distribution_metrics=control_baseline["holdout_metrics"],
    )

    stage3_report_payload: dict[str, Any] | None = None
    comparison_to_stage3: dict[str, Any] | None = None
    if stage3_report_path is not None and stage3_report_path.exists():
        stage3_report_payload = json.loads(stage3_report_path.read_text(encoding="utf-8"))
        comparison_to_stage3 = build_distribution_comparison(
            challenger_label="stage4_mcmc",
            challenger_mean_metrics=mean_metrics,
            challenger_distribution_metrics=distribution_metrics,
            baseline_label="stage3_distribution",
            baseline_mean_metrics=stage3_report_payload["mean_metrics"],
            baseline_distribution_metrics=stage3_report_payload["holdout_metrics"],
        )

    model_version = _build_model_version(_resolve_data_version_hash(dataset))
    report_dir = Path(args.mcmc_report_dir)
    if not report_dir.is_absolute():
        report_dir = (PROJECT_ROOT / report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{DEFAULT_MODEL_NAME}_{model_version}.joblib"
    metadata_path = output_dir / f"{DEFAULT_MODEL_NAME}_{model_version}.metadata.json"
    summary_path = output_dir / f"{DEFAULT_MODEL_NAME}_run_{model_version}.json"
    report_json_path = report_dir / f"{model_version}.mcmc_eval.json"
    report_csv_path = report_dir / f"{model_version}.mcmc_eval.csv"
    control_comparison_path = report_dir / f"{model_version}.vs_control.json"
    stage3_comparison_path = report_dir / f"{model_version}.vs_stage3.json"
    predictions_csv_path = report_dir / f"{model_version}.holdout_predictions.csv"

    aggregate_event_counts = total_event_counts or {}
    total_plate_appearances = sum(aggregate_event_counts.values())
    simulation_diagnostics = {
        "holdout_game_count": int(len(holdout_frame)),
        "total_truncated_half_innings": int(truncated_half_innings),
        "mean_runs_by_inning": (
            (mean_runs_by_inning_accumulator / len(holdout_frame)).tolist()
            if mean_runs_by_inning_accumulator is not None
            else []
        ),
        "mean_plate_appearances_by_inning": (
            (mean_plate_appearances_by_inning_accumulator / len(holdout_frame)).tolist()
            if mean_plate_appearances_by_inning_accumulator is not None
            else []
        ),
        "aggregate_event_counts": aggregate_event_counts,
        "aggregate_event_share_by_plate_appearance": {
            name: (float(count) / float(total_plate_appearances) if total_plate_appearances > 0 else 0.0)
            for name, count in aggregate_event_counts.items()
        },
        "mean_control_expected_runs": float(np.mean(holdout_control_mean)),
        "mean_mcmc_expected_runs": float(np.mean(pmf_expected_runs)),
    }
    metadata_payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model_name": DEFAULT_MODEL_NAME,
        "model_version": model_version,
        "target_column": DEFAULT_TARGET_COLUMN,
        "holdout_season": int(args.holdout_season),
        "distribution_family": "markov_monte_carlo",
        "distribution_lane_stage": 4,
        "simulation_count": int(args.simulations),
        "starter_innings": int(args.starter_innings),
        "seed": int(args.seed),
        "data_version_hash": _resolve_data_version_hash(dataset),
        "training_data_path": str(args.training_data),
        "mean_head_source_artifact_path": _relative_to_project(Path(mean_artifact_metadata_path)),
        "stage3_report_path": None if stage3_report_path is None else _relative_to_project(stage3_report_path),
        "mean_metrics": mean_metrics,
        "distribution_metrics": distribution_metrics,
        "prediction_summary": distribution_metrics["prediction_summary"],
        "simulation_diagnostics": simulation_diagnostics,
        "comparison_to_control": comparison_to_control,
        "comparison_to_stage3": comparison_to_stage3,
        "output_paths": {
            "report_json": _relative_to_project(report_json_path),
            "report_csv": _relative_to_project(report_csv_path),
            "predictions_csv": _relative_to_project(predictions_csv_path),
            "vs_control_json": _relative_to_project(control_comparison_path),
            "vs_stage3_json": _relative_to_project(stage3_comparison_path),
        },
    }

    joblib.dump(
        {
            "model_name": DEFAULT_MODEL_NAME,
            "model_version": model_version,
            "simulation_count": int(args.simulations),
            "starter_innings": int(args.starter_innings),
            "seed": int(args.seed),
            "mean_head_source_artifact_path": str(mean_artifact_metadata_path),
        },
        model_path,
    )
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

    report_payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "artifact_path": _relative_to_project(metadata_path),
        "model_path": _relative_to_project(model_path),
        "model_version": model_version,
        "model_name": DEFAULT_MODEL_NAME,
        "mean_metrics": mean_metrics,
        "holdout_metrics": distribution_metrics,
        "simulation_diagnostics": simulation_diagnostics,
        "comparison_to_control": comparison_to_control,
        "comparison_to_stage3": comparison_to_stage3,
    }
    report_json_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            flatten_mcmc_report_row(
                model_version=model_version,
                metadata_path=metadata_path,
                mean_metrics=mean_metrics,
                distribution_metrics=distribution_metrics,
                control_comparison=comparison_to_control,
                stage3_comparison=comparison_to_stage3,
            )
        ]
    ).to_csv(report_csv_path, index=False)
    pd.DataFrame(per_row_summaries).to_csv(predictions_csv_path, index=False)
    control_comparison_path.write_text(json.dumps(comparison_to_control, indent=2), encoding="utf-8")
    stage3_comparison_path.write_text(
        json.dumps(comparison_to_stage3, indent=2) if comparison_to_stage3 is not None else "null",
        encoding="utf-8",
    )

    console.print(f"[bold green]Done[/bold green] metadata={metadata_path}")
    console.print(
        f"RMSE={mean_metrics['rmse']:.4f} "
        f"CRPS={distribution_metrics['mean_crps']:.6f} "
        f"NegLogScore={distribution_metrics['mean_negative_log_score']:.6f}"
    )
    console.print(
        f"beats_control_on_crps={comparison_to_control['improvement_flags']['beats_baseline_on_crps']} "
        f"beats_control_on_negative_log_score={comparison_to_control['improvement_flags']['beats_baseline_on_negative_log_score']} "
        f"catastrophic_regression={comparison_to_control['catastrophic_regression']}"
    )
    if comparison_to_stage3 is not None:
        console.print(
            f"beats_stage3_on_crps={comparison_to_stage3['improvement_flags']['beats_baseline_on_crps']} "
            f"beats_stage3_on_negative_log_score={comparison_to_stage3['improvement_flags']['beats_baseline_on_negative_log_score']}"
        )
    console.print(
        json.dumps(
            {
                "model_version": model_version,
                "model_path": str(model_path),
                "metadata_path": str(metadata_path),
                "report_json_path": str(report_json_path),
                "report_csv_path": str(report_csv_path),
                "predictions_csv_path": str(predictions_csv_path),
                "vs_control_json_path": str(control_comparison_path),
                "vs_stage3_json_path": str(stage3_comparison_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
