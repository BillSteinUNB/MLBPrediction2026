from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from rich.console import Console


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ops.run_count_dual_view import resolve_stage5_inputs  # noqa: E402


console = Console()
DEFAULT_TRAINING_DATA = Path("data/training/ParquetDefault.parquet")
DEFAULT_CURRENT_CONTROL_PATH = Path("data/reports/run_count/registry/current_control.json")
DEFAULT_DISTRIBUTION_REPORT_DIR = Path("data/reports/run_count/distribution_eval")
DEFAULT_MCMC_REPORT_DIR = Path("data/reports/run_count/mcmc")
DEFAULT_WALK_FORWARD_OUTPUT_DIR = Path("data/reports/run_count/walk_forward")
DEFAULT_DUAL_VIEW_OUTPUT_DIR = Path("data/reports/run_count/dual_view")


def _default_stage3_experiment_name() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"2026-away-dist-zanb-v1-controlmu-flat-3f-{timestamp}"


def _default_stage4_experiment_name() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"2026-away-mcmc-markov-v1-controlmu-{timestamp}"


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _snapshot_paths(directory: Path, pattern: str) -> set[Path]:
    if not directory.exists():
        return set()
    return {path.resolve() for path in directory.glob(pattern)}


def _find_newest_path_since(directory: Path, pattern: str, before: Iterable[Path]) -> Path:
    before_resolved = {path.resolve() for path in before}
    candidates = [path.resolve() for path in directory.glob(pattern) if path.resolve() not in before_resolved]
    if not candidates:
        raise FileNotFoundError(f"No new artifact found in {directory} matching {pattern}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _run_command(command: list[str]) -> None:
    console.print(f"[bold green]Running[/bold green] {' '.join(command)}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Stage 3, Stage 4, and walk-forward evaluation in one command.",
    )
    parser.add_argument("--training-data", default=str(DEFAULT_TRAINING_DATA))
    parser.add_argument("--start", "--start-year", dest="start_year", type=int, default=2021)
    parser.add_argument("--end", "--end-year", dest="end_year", type=int, default=2025)
    parser.add_argument("--holdout", "--holdout-season", dest="holdout_season", type=int, default=2025)
    parser.add_argument("--Folds", "--folds", dest="folds", type=int, default=3)
    parser.add_argument(
        "--feature-selection-mode",
        choices=("grouped", "bucketed", "flat"),
        default="flat",
    )
    parser.add_argument("--forced-delta-count", type=int, default=0)
    parser.add_argument("--XGBWork", "--xgb-workers", dest="xgb_workers", type=int, default=4)
    parser.add_argument("--current-control", default=str(DEFAULT_CURRENT_CONTROL_PATH))
    parser.add_argument("--mean-artifact-metadata", default=None)
    parser.add_argument("--distribution-report-dir", default=str(DEFAULT_DISTRIBUTION_REPORT_DIR))
    parser.add_argument("--mcmc-report-dir", default=str(DEFAULT_MCMC_REPORT_DIR))
    parser.add_argument("--walk-forward-output-dir", default=str(DEFAULT_WALK_FORWARD_OUTPUT_DIR))
    parser.add_argument("--dual-view-output-dir", default=str(DEFAULT_DUAL_VIEW_OUTPUT_DIR))
    parser.add_argument("--simulations", type=int, default=1000)
    parser.add_argument("--starter-innings", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260328)
    parser.add_argument("--enable-market-priors", action="store_true")
    parser.add_argument("--historical-odds-db", default=None)
    parser.add_argument("--historical-market-book", default=None)
    parser.add_argument("--stage3-experiment", default=None)
    parser.add_argument("--stage4-experiment", default=None)
    parser.add_argument("--stage3-research-lane-name", default="distribution_market_priors_adv_v1")
    parser.add_argument("--stage4-research-lane-name", default="mcmc_market_priors_adv_v1")
    parser.add_argument(
        "--mu-delta-mode",
        choices=("off", "gap_only", "gap_linear", "anchor_bundle"),
        default="off",
    )
    parser.add_argument("--skip-dual-view", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    stage3_experiment = args.stage3_experiment or _default_stage3_experiment_name()
    stage4_experiment = args.stage4_experiment or _default_stage4_experiment_name()
    distribution_report_dir = _resolve_path(args.distribution_report_dir)
    mcmc_report_dir = _resolve_path(args.mcmc_report_dir)
    walk_forward_output_dir = _resolve_path(args.walk_forward_output_dir)

    stage3_command = [
        sys.executable,
        "scripts/train_run_distribution.py",
        "--experiment",
        stage3_experiment,
        "--training-data",
        str(args.training_data),
        "--start",
        str(args.start_year),
        "--end",
        str(args.end_year),
        "--holdout",
        str(args.holdout_season),
        "--Folds",
        str(args.folds),
        "--feature-selection-mode",
        str(args.feature_selection_mode),
        "--forced-delta-count",
        str(args.forced_delta_count),
        "--XGBWork",
        str(args.xgb_workers),
        "--current-control",
        str(args.current_control),
        "--distribution-report-dir",
        str(args.distribution_report_dir),
        "--research-lane-name",
        str(args.stage3_research_lane_name),
        "--mu-delta-mode",
        str(args.mu_delta_mode),
    ]
    if args.mean_artifact_metadata:
        stage3_command.extend(["--mean-artifact-metadata", str(args.mean_artifact_metadata)])
    if args.enable_market_priors:
        stage3_command.append("--enable-market-priors")
    if args.historical_odds_db:
        stage3_command.extend(["--historical-odds-db", str(args.historical_odds_db)])
    if args.historical_market_book:
        stage3_command.extend(["--historical-market-book", str(args.historical_market_book)])

    stage4_command_base = [
        sys.executable,
        "scripts/run_mcmc_distribution.py",
        "--experiment",
        stage4_experiment,
        "--training-data",
        str(args.training_data),
        "--start",
        str(args.start_year),
        "--end",
        str(args.end_year),
        "--holdout",
        str(args.holdout_season),
        "--current-control",
        str(args.current_control),
        "--distribution-report-dir",
        str(args.distribution_report_dir),
        "--mcmc-report-dir",
        str(args.mcmc_report_dir),
        "--simulations",
        str(args.simulations),
        "--starter-innings",
        str(args.starter_innings),
        "--seed",
        str(args.seed),
        "--research-lane-name",
        str(args.stage4_research_lane_name),
    ]
    if args.enable_market_priors:
        stage4_command_base.append("--enable-market-priors")
    if args.historical_odds_db:
        stage4_command_base.extend(["--historical-odds-db", str(args.historical_odds_db)])
    if args.historical_market_book:
        stage4_command_base.extend(["--historical-market-book", str(args.historical_market_book)])

    if args.dry_run:
        console.print("[bold green]Stage 3[/bold green]")
        console.print(" ".join(stage3_command))
        console.print("[bold green]Stage 4[/bold green]")
        console.print(" ".join(stage4_command_base + ["--stage3-report-json", "<resolved after stage3>"]))
        console.print("[bold green]Walk-forward[/bold green]")
        console.print(
            " ".join(
                [
                    sys.executable,
                    "scripts/evaluate_run_count_walk_forward.py",
                    "--training-data",
                    str(args.training_data),
                    "--stage3-metadata",
                    "<resolved after stage3>",
                    "--mcmc-metadata",
                    "<resolved after stage4>",
                    "--holdout",
                    str(args.holdout_season),
                    "--output-dir",
                    str(args.walk_forward_output_dir),
                ]
            )
        )
        return 0

    stage3_before = _snapshot_paths(distribution_report_dir, "*.distribution_eval.json")
    _run_command(stage3_command)
    stage3_report_json = _find_newest_path_since(distribution_report_dir, "*.distribution_eval.json", stage3_before)
    stage3_report_payload = json.loads(stage3_report_json.read_text(encoding="utf-8"))
    stage3_metadata_path = _resolve_path(stage3_report_payload["artifact_path"])

    stage4_before = _snapshot_paths(mcmc_report_dir, "*.mcmc_eval.json")
    stage4_command = stage4_command_base + ["--stage3-report-json", str(stage3_report_json)]
    _run_command(stage4_command)
    stage4_report_json = _find_newest_path_since(mcmc_report_dir, "*.mcmc_eval.json", stage4_before)
    stage4_report_payload = json.loads(stage4_report_json.read_text(encoding="utf-8"))
    stage4_metadata_path = _resolve_path(stage4_report_payload["artifact_path"])

    walk_forward_before = _snapshot_paths(walk_forward_output_dir, "*.json")
    walk_forward_command = [
        sys.executable,
        "scripts/evaluate_run_count_walk_forward.py",
        "--training-data",
        str(args.training_data),
        "--stage3-metadata",
        str(stage3_metadata_path),
        "--mcmc-metadata",
        str(stage4_metadata_path),
        "--holdout",
        str(args.holdout_season),
        "--output-dir",
        str(args.walk_forward_output_dir),
    ]
    if args.enable_market_priors:
        walk_forward_command.append("--enable-market-priors")
    if args.historical_odds_db:
        walk_forward_command.extend(["--historical-odds-db", str(args.historical_odds_db)])
    if args.historical_market_book:
        walk_forward_command.extend(["--historical-market-book", str(args.historical_market_book)])
    _run_command(walk_forward_command)
    stage3_walk_forward_json = _find_newest_path_since(
        walk_forward_output_dir,
        "*.stage3_walk_forward.json",
        walk_forward_before,
    )
    stage4_walk_forward_json = _find_newest_path_since(
        walk_forward_output_dir,
        "*.mcmc_walk_forward.json",
        walk_forward_before,
    )

    dual_view_result: dict[str, object] | None = None
    if not args.skip_dual_view:
        dual_view_command = [
            sys.executable,
            "scripts/report_run_count_dual_view.py",
            "--current-control",
            str(args.current_control),
            "--distribution-report-dir",
            str(args.distribution_report_dir),
            "--stage3-report-json",
            str(stage3_report_json),
            "--stage3-vs-control-json",
            str(stage3_report_json).replace(".distribution_eval.json", ".vs_control.json"),
            "--mcmc-report-dir",
            str(args.mcmc_report_dir),
            "--mcmc-report-json",
            str(stage4_report_json),
            "--mcmc-vs-control-json",
            str(stage4_report_json).replace(".mcmc_eval.json", ".vs_control.json"),
            "--mcmc-vs-stage3-json",
            str(stage4_report_json).replace(".mcmc_eval.json", ".vs_stage3.json"),
            "--stage3-walk-forward-report-json",
            str(stage3_walk_forward_json),
            "--mcmc-walk-forward-report-json",
            str(stage4_walk_forward_json),
            "--output-dir",
            str(args.dual_view_output_dir),
        ]
        _run_command(dual_view_command)
        resolved_dual_view_inputs = resolve_stage5_inputs(
            current_control_path=args.current_control,
            stage3_report_json=stage3_report_json,
            stage3_vs_control_json=str(stage3_report_json).replace(".distribution_eval.json", ".vs_control.json"),
            distribution_report_dir=args.distribution_report_dir,
            mcmc_report_json=stage4_report_json,
            mcmc_vs_control_json=str(stage4_report_json).replace(".mcmc_eval.json", ".vs_control.json"),
            mcmc_vs_stage3_json=str(stage4_report_json).replace(".mcmc_eval.json", ".vs_stage3.json"),
            mcmc_report_dir=args.mcmc_report_dir,
            stage3_walk_forward_report_json=stage3_walk_forward_json,
            mcmc_walk_forward_report_json=stage4_walk_forward_json,
            walk_forward_report_dir=args.walk_forward_output_dir,
        )
        dual_view_result = {
            "resolved_inputs": {
                key: (None if value is None else str(value))
                for key, value in asdict(resolved_dual_view_inputs).items()
            },
            "output_dir": str(_resolve_path(args.dual_view_output_dir)),
        }

    payload = {
        "stage3": {
            "report_json": str(stage3_report_json),
            "metadata_path": str(stage3_metadata_path),
            "vs_control_json": str(stage3_report_json).replace(".distribution_eval.json", ".vs_control.json"),
        },
        "stage4": {
            "report_json": str(stage4_report_json),
            "metadata_path": str(stage4_metadata_path),
            "vs_control_json": str(stage4_report_json).replace(".mcmc_eval.json", ".vs_control.json"),
            "vs_stage3_json": str(stage4_report_json).replace(".mcmc_eval.json", ".vs_stage3.json"),
        },
        "walk_forward": {
            "stage3_json": str(stage3_walk_forward_json),
            "stage4_json": str(stage4_walk_forward_json),
            "output_dir": str(walk_forward_output_dir),
        },
        "dual_view": dual_view_result,
    }
    console.print("[bold green]Workflow complete[/bold green]")
    console.print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
