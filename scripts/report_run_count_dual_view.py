from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ops.run_count_dual_view import (  # noqa: E402
    DEFAULT_CURRENT_CONTROL_PATH,
    DEFAULT_DISTRIBUTION_REPORT_DIR,
    DEFAULT_DUAL_VIEW_OUTPUT_DIR,
    DEFAULT_MCMC_REPORT_DIR,
    DEFAULT_WALK_FORWARD_REPORT_DIR,
    build_dual_view_payload,
    load_dual_view_inputs,
    resolve_stage5_inputs,
    write_dual_view_outputs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the Stage 5 away-run dual-view and promotion summary from existing artifacts.",
    )
    parser.add_argument("--current-control", default=str(DEFAULT_CURRENT_CONTROL_PATH))
    parser.add_argument("--distribution-report-dir", default=str(DEFAULT_DISTRIBUTION_REPORT_DIR))
    parser.add_argument("--stage3-report-json", default=None)
    parser.add_argument("--stage3-vs-control-json", default=None)
    parser.add_argument("--mcmc-report-dir", default=str(DEFAULT_MCMC_REPORT_DIR))
    parser.add_argument("--mcmc-report-json", default=None)
    parser.add_argument("--mcmc-vs-control-json", default=None)
    parser.add_argument("--mcmc-vs-stage3-json", default=None)
    parser.add_argument("--walk-forward-report-dir", default=str(DEFAULT_WALK_FORWARD_REPORT_DIR))
    parser.add_argument("--stage3-walk-forward-report-json", default=None)
    parser.add_argument("--mcmc-walk-forward-report-json", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_DUAL_VIEW_OUTPUT_DIR))
    args = parser.parse_args(argv)

    resolved_inputs = resolve_stage5_inputs(
        current_control_path=args.current_control,
        stage3_report_json=args.stage3_report_json,
        stage3_vs_control_json=args.stage3_vs_control_json,
        distribution_report_dir=args.distribution_report_dir,
        mcmc_report_json=args.mcmc_report_json,
        mcmc_vs_control_json=args.mcmc_vs_control_json,
        mcmc_vs_stage3_json=args.mcmc_vs_stage3_json,
        mcmc_report_dir=args.mcmc_report_dir,
        stage3_walk_forward_report_json=args.stage3_walk_forward_report_json,
        mcmc_walk_forward_report_json=args.mcmc_walk_forward_report_json,
        walk_forward_report_dir=args.walk_forward_report_dir,
    )
    loaded_inputs = load_dual_view_inputs(resolved_inputs)
    payload = build_dual_view_payload(
        current_control_payload=loaded_inputs["current_control_payload"],
        stage3_report=loaded_inputs["stage3_report"],
        stage3_vs_control=loaded_inputs["stage3_vs_control"],
        mcmc_report=loaded_inputs["mcmc_report"],
        mcmc_vs_control=loaded_inputs["mcmc_vs_control"],
        mcmc_vs_stage3=loaded_inputs["mcmc_vs_stage3"],
        stage3_walk_forward_report=loaded_inputs["stage3_walk_forward_report"],
        mcmc_walk_forward_report=loaded_inputs["mcmc_walk_forward_report"],
        source_paths=loaded_inputs["source_paths"],
    )
    output_paths = write_dual_view_outputs(
        payload=payload,
        output_dir=args.output_dir,
    )

    print(
        json.dumps(
            {
                "best_research_lane": payload["promotion_summary"]["best_research_lane_key"],
                "promoted_second_opinion_lane": payload["promotion_summary"]["promoted_second_opinion_lane_key"],
                "production_promotable_lane": payload["promotion_summary"]["production_promotable_lane_key"],
                "output_paths": output_paths,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
