from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clients.fangraphs_baseruns_client import (  # noqa: E402
    HTTP_TIMEOUT,
    load_fangraphs_baseruns_standings,
)
from src.db import DEFAULT_DB_PATH  # noqa: E402
from src.display import Console  # noqa: E402
from src.ops.fangraphs_baseruns_audit import (  # noqa: E402
    DEFAULT_FANGRAPHS_BASERUNS_AUDIT_DIR,
    build_fangraphs_baseruns_audit_frame,
    build_fangraphs_baseruns_audit_summary,
    write_fangraphs_baseruns_audit_outputs,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a FanGraphs BaseRuns sanity audit against repo-local team results."
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_FANGRAPHS_BASERUNS_AUDIT_DIR))
    parser.add_argument(
        "--html-input",
        default=None,
        help="Optional browser-saved FanGraphs standings page to parse instead of fetching.",
    )
    parser.add_argument(
        "--save-html",
        default=None,
        help="Optional path where fetched HTML will be written for reproducibility.",
    )
    parser.add_argument("--timeout", type=float, default=HTTP_TIMEOUT)
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    console = Console(force_terminal=True)

    standings = load_fangraphs_baseruns_standings(
        season=args.season,
        html_path=args.html_input,
        save_html_path=args.save_html,
        timeout=args.timeout,
    )
    audit_frame = build_fangraphs_baseruns_audit_frame(
        standings,
        db_path=args.db_path,
    )
    summary = build_fangraphs_baseruns_audit_summary(
        audit_frame,
        top_n=args.top_n,
    )
    outputs = write_fangraphs_baseruns_audit_outputs(
        audit_frame=audit_frame,
        summary=summary,
        output_dir=args.output_dir,
    )

    console.print(
        json.dumps(
            {
                "season": args.season,
                "team_count": int(len(audit_frame)),
                "missing_repo_teams": summary["missing_repo_teams"],
                "missing_fangraphs_teams": summary["missing_fangraphs_teams"],
                "outputs": {key: str(value) for key, value in outputs.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
