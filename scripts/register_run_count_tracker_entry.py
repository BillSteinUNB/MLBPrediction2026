from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ops.run_count_dual_view import DEFAULT_DUAL_VIEW_OUTPUT_DIR, resolve_stage5_inputs  # noqa: E402
from src.ops.run_count_tracker import (  # noqa: E402
    DEFAULT_RUN_COUNT_TRACKING_DIR,
    record_run_count_workflow_run,
    write_frontend_connection_contract,
)


console = Console()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Register the latest run-count workflow artifacts in the tracker.")
    parser.add_argument("--current-control", default="data/reports/run_count/registry/current_control.json")
    parser.add_argument("--distribution-report-dir", default="data/reports/run_count/distribution_eval")
    parser.add_argument("--mcmc-report-dir", default="data/reports/run_count/mcmc")
    parser.add_argument("--walk-forward-report-dir", default="data/reports/run_count/walk_forward")
    parser.add_argument("--dual-view-output-dir", default=str(DEFAULT_DUAL_VIEW_OUTPUT_DIR))
    parser.add_argument("--training-data", default="data/training/ParquetDefault.parquet")
    parser.add_argument("--start", dest="start_year", type=int, default=2021)
    parser.add_argument("--end", dest="end_year", type=int, default=2025)
    parser.add_argument("--holdout", dest="holdout_season", type=int, default=2025)
    parser.add_argument("--Folds", dest="folds", type=int, default=3)
    parser.add_argument("--feature-selection-mode", default="flat")
    parser.add_argument("--forced-delta-count", type=int, default=0)
    parser.add_argument("--XGBWork", dest="xgb_workers", type=int, default=4)
    parser.add_argument("--enable-market-priors", action="store_true")
    parser.add_argument("--historical-odds-db", default=None)
    parser.add_argument("--historical-market-book", default=None)
    parser.add_argument("--mu-delta-mode", default="off")
    parser.add_argument("--stage3-experiment", default="registered-latest-stage3")
    parser.add_argument("--stage4-experiment", default="registered-latest-stage4")
    parser.add_argument("--stage3-research-lane-name", default="distribution_market_priors_adv_v1")
    parser.add_argument("--stage4-research-lane-name", default="mcmc_market_priors_adv_v1")
    parser.add_argument("--tracker-dir", default=str(DEFAULT_RUN_COUNT_TRACKING_DIR))
    parser.add_argument("--tracker-hypothesis", default="Backfilled benchmark registration from existing artifacts.")
    parser.add_argument("--tracker-run-label", default="latest_registered_run")
    parser.add_argument("--set-as-benchmark", action="store_true")
    parser.add_argument("--benchmark-label", default=None)
    args = parser.parse_args(argv)

    resolved = resolve_stage5_inputs(
        current_control_path=args.current_control,
        distribution_report_dir=args.distribution_report_dir,
        mcmc_report_dir=args.mcmc_report_dir,
        walk_forward_report_dir=args.walk_forward_report_dir,
    )
    dual_view_path = Path(args.dual_view_output_dir) / "current_dual_view.json"

    record = record_run_count_workflow_run(
        stage3_report_path=resolved.stage3_report_path,
        stage3_vs_control_path=resolved.stage3_vs_control_path,
        stage4_report_path=resolved.mcmc_report_path,
        stage4_vs_control_path=resolved.mcmc_vs_control_path,
        stage4_vs_stage3_path=resolved.mcmc_vs_stage3_path,
        stage3_walk_forward_path=resolved.stage3_walk_forward_report_path,
        stage4_walk_forward_path=resolved.mcmc_walk_forward_report_path,
        dual_view_path=dual_view_path,
        training_data_path=args.training_data,
        start_year=args.start_year,
        end_year=args.end_year,
        holdout_season=args.holdout_season,
        folds=args.folds,
        feature_selection_mode=args.feature_selection_mode,
        forced_delta_count=args.forced_delta_count,
        xgb_workers=args.xgb_workers,
        enable_market_priors=args.enable_market_priors,
        historical_odds_db=args.historical_odds_db,
        historical_market_book=args.historical_market_book,
        mu_delta_mode=args.mu_delta_mode,
        stage3_experiment=args.stage3_experiment,
        stage4_experiment=args.stage4_experiment,
        stage3_research_lane_name=args.stage3_research_lane_name,
        stage4_research_lane_name=args.stage4_research_lane_name,
        tracking_dir=args.tracker_dir,
        hypothesis=args.tracker_hypothesis,
        run_label=args.tracker_run_label,
        set_as_benchmark=args.set_as_benchmark,
        benchmark_label=args.benchmark_label,
    )
    contract = write_frontend_connection_contract(tracking_dir=args.tracker_dir)
    console.print(json.dumps({"run_id": record["run_id"], "benchmark_status": record["benchmark_status"], "connection_contract": contract["primary_data_sources"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
