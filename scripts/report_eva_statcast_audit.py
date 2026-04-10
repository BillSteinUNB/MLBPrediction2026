from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clients.eva_analytics_client import (  # noqa: E402
    HTTP_TIMEOUT,
    fetch_eva_statcast_correlations,
)
from src.db import DEFAULT_DB_PATH  # noqa: E402
from src.display import Console  # noqa: E402
from src.ops.eva_statcast_audit import (  # noqa: E402
    DEFAULT_EVA_STATCAST_AUDIT_DIR,
    build_eva_statcast_audit_summary,
    build_eva_statcast_metric_summary,
    build_eva_statcast_starter_audit_summary,
    build_eva_statcast_starter_metric_summary,
    build_eva_statcast_starter_start_audit_frame,
    build_eva_statcast_team_game_audit_frame,
    write_eva_statcast_audit_outputs,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an EVA Analytics informed Statcast audit against repo-local game results."
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--rolling-window", type=int, default=30)
    parser.add_argument("--starter-window", type=int, default=5)
    parser.add_argument("--output-dir", default=str(DEFAULT_EVA_STATCAST_AUDIT_DIR))
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=HTTP_TIMEOUT)
    parser.add_argument(
        "--skip-eva-fetch",
        action="store_true",
        help="Run the repo-native Statcast audit without fetching the EVA snapshot.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    console = Console(force_terminal=True)

    eva_correlations = None
    if not args.skip_eva_fetch:
        eva_correlations = fetch_eva_statcast_correlations(timeout=args.timeout)

    audit_frame = build_eva_statcast_team_game_audit_frame(
        season=args.season,
        db_path=args.db_path,
        rolling_window=args.rolling_window,
    )
    metric_summary = build_eva_statcast_metric_summary(
        audit_frame,
        rolling_window=args.rolling_window,
        eva_correlations=eva_correlations,
    )
    starter_audit_frame = build_eva_statcast_starter_start_audit_frame(
        season=args.season,
        db_path=args.db_path,
        rolling_window=args.starter_window,
    )
    starter_metric_summary = build_eva_statcast_starter_metric_summary(
        starter_audit_frame,
        rolling_window=args.starter_window,
        eva_correlations=eva_correlations,
    )
    summary = build_eva_statcast_audit_summary(
        audit_frame,
        metric_summary=metric_summary,
        rolling_window=args.rolling_window,
        eva_correlations=eva_correlations,
        top_n=args.top_n,
    )
    starter_summary = build_eva_statcast_starter_audit_summary(
        starter_audit_frame,
        metric_summary=starter_metric_summary,
        rolling_window=args.starter_window,
        eva_correlations=eva_correlations,
        top_n=args.top_n,
    )
    outputs = write_eva_statcast_audit_outputs(
        audit_frame=audit_frame,
        metric_summary=metric_summary,
        summary=summary,
        output_dir=args.output_dir,
        eva_correlations=eva_correlations,
        starter_audit_frame=starter_audit_frame,
        starter_metric_summary=starter_metric_summary,
        starter_summary=starter_summary,
    )

    console.print(
        json.dumps(
            {
                "season": args.season,
                "rolling_window": args.rolling_window,
                "starter_window": args.starter_window,
                "eva_snapshot_available": summary["eva_snapshot_available"],
                "team_recommended_feature_targets": summary["recommended_feature_targets"],
                "starter_recommended_feature_targets": starter_summary["recommended_feature_targets"],
                "outputs": {key: str(value) for key, value in outputs.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
