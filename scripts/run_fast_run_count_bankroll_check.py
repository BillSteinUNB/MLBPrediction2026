from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ops.fast_run_count_bankroll_check import (  # noqa: E402
    DEFAULT_FAST_BANKROLL_OUTPUT_DIR,
    DEFAULT_TRACKER_LATEST_JSON,
    run_fast_bankroll_check,
)


console = Console()


def _parse_csv_strings(value: str) -> list[str]:
    return [token.strip() for token in value.split(",") if token.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a fast bankroll check using the latest saved run-count artifacts.",
    )
    parser.add_argument(
        "--latest-tracker-json",
        default=str(DEFAULT_TRACKER_LATEST_JSON),
        help="Tracker JSON to resolve the latest saved Stage 4 artifact from.",
    )
    parser.add_argument(
        "--historical-odds-db",
        default=None,
        help="Historical odds DB to price against. Defaults to the tracker workflow value.",
    )
    parser.add_argument("--full-game-home-metadata", default=None)
    parser.add_argument("--f5-home-metadata", default=None)
    parser.add_argument("--f5-away-metadata", default=None)
    parser.add_argument(
        "--markets",
        default=None,
        help="Comma-separated market types to include. Defaults to auto-supported trusted markets.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_FAST_BANKROLL_OUTPUT_DIR))
    parser.add_argument("--starting-bankroll-units", type=float, default=100.0)
    parser.add_argument("--flat-bet-size-units", type=float, default=1.0)
    parser.add_argument("--max-games", type=int, default=None)
    args = parser.parse_args(argv)

    console.print(
        "[bold green]Starting fast bankroll check[/bold green] "
        f"tracker={args.latest_tracker_json}"
    )
    result = run_fast_bankroll_check(
        latest_tracker_json=args.latest_tracker_json,
        historical_odds_db=args.historical_odds_db,
        full_game_home_metadata=args.full_game_home_metadata,
        f5_home_metadata=args.f5_home_metadata,
        f5_away_metadata=args.f5_away_metadata,
        markets=_parse_csv_strings(args.markets) if args.markets else None,
        output_dir=args.output_dir,
        starting_bankroll_units=args.starting_bankroll_units,
        flat_bet_size_units=args.flat_bet_size_units,
        max_games=args.max_games,
    )

    console.print(
        "[bold green]Fast bankroll check complete[/bold green] "
        f"-> {result.summary_json_path}"
    )
    console.print(json.dumps(result.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
