from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ops.fast_bankroll_strategy_grid import (  # noqa: E402
    DEFAULT_FAST_BANKROLL_GRID_DIR,
    DEFAULT_FULL_GAME_MARKETS,
    DEFAULT_ODDS_WINDOWS,
    DEFAULT_STANDARD_EDGE_PCTS,
    run_fast_bankroll_strategy_grid,
)


console = Console()


def _parse_csv_floats(value: str) -> list[float]:
    return [float(token.strip()) for token in value.split(",") if token.strip()]


def _parse_odds_windows(value: str) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        minimum, maximum = token.split(":")
        windows.append((int(minimum), int(maximum)))
    return windows


def _parse_csv_strings(value: str) -> list[str]:
    return [token.strip() for token in value.split(",") if token.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a fast bankroll sizing and odds-window strategy grid on the latest bet ledger.",
    )
    parser.add_argument("--bets-csv", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_FAST_BANKROLL_GRID_DIR))
    parser.add_argument("--starting-bankroll-units", type=float, default=100.0)
    parser.add_argument(
        "--standard-edges",
        default=",".join(f"{value:.2f}" for value in DEFAULT_STANDARD_EDGE_PCTS),
        help="Comma-separated edge percentages that map to a 1U baseline stake.",
    )
    parser.add_argument(
        "--odds-windows",
        default=",".join(f"{minimum}:{maximum}" for minimum, maximum in DEFAULT_ODDS_WINDOWS),
        help="Comma-separated min:max American odds windows.",
    )
    parser.add_argument(
        "--markets",
        default=",".join(DEFAULT_FULL_GAME_MARKETS),
        help="Comma-separated market types to keep before running the grid.",
    )
    parser.add_argument("--rl-coinflip-min-odds", type=int, default=-140)
    parser.add_argument("--rl-coinflip-max-odds", type=int, default=105)
    args = parser.parse_args(argv)

    console.print("[bold green]Starting fast bankroll strategy grid[/bold green]")
    result = run_fast_bankroll_strategy_grid(
        bets_csv_path=args.bets_csv,
        output_dir=args.output_dir,
        starting_bankroll_units=args.starting_bankroll_units,
        standard_edge_pcts=_parse_csv_floats(args.standard_edges),
        odds_windows=_parse_odds_windows(args.odds_windows),
        include_markets=_parse_csv_strings(args.markets),
        rl_coinflip_min_odds=args.rl_coinflip_min_odds,
        rl_coinflip_max_odds=args.rl_coinflip_max_odds,
    )
    console.print(
        "[bold green]Fast bankroll strategy grid complete[/bold green] "
        f"-> {result.summary_csv_path}"
    )
    payload = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    console.print(json.dumps(payload["top_by_roi"][:5], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
