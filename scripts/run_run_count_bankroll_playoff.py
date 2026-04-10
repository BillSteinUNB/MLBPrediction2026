from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ops.run_count_bankroll_playoff import (  # noqa: E402
    DEFAULT_BANKROLL_CANDIDATE_LABELS,
    DEFAULT_BANKROLL_HOLDOUT_SEASONS,
    DEFAULT_BANKROLL_PLAYOFF_DIR,
    DEFAULT_FLAT_BET_SIZE_UNITS,
    DEFAULT_STARTING_BANKROLL_UNITS,
    run_bankroll_playoff,
)


console = Console()


def _parse_csv_ints(value: str) -> list[int]:
    return [int(token.strip()) for token in value.split(",") if token.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    return [token.strip() for token in value.split(",") if token.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the run-count bankroll playoff for a fixed set of candidate models.",
    )
    parser.add_argument(
        "--candidates",
        default=",".join(DEFAULT_BANKROLL_CANDIDATE_LABELS),
        help="Comma-separated bankroll candidate labels.",
    )
    parser.add_argument(
        "--holdout-seasons",
        default=",".join(str(season) for season in DEFAULT_BANKROLL_HOLDOUT_SEASONS),
        help="Comma-separated holdout seasons to simulate in order.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_BANKROLL_PLAYOFF_DIR))
    parser.add_argument("--starting-bankroll-units", type=float, default=DEFAULT_STARTING_BANKROLL_UNITS)
    parser.add_argument("--flat-bet-size-units", type=float, default=DEFAULT_FLAT_BET_SIZE_UNITS)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--companion-search-iterations", type=int, default=25)
    parser.add_argument("--companion-optuna-workers", type=int, default=3)
    parser.add_argument("--companion-early-stopping-rounds", type=int, default=50)
    args = parser.parse_args(argv)

    console.print(
        "[bold green]Starting bankroll playoff[/bold green] "
        f"candidates={args.candidates} holdout_seasons={args.holdout_seasons}"
    )

    result = run_bankroll_playoff(
        candidate_labels=_parse_csv_strings(args.candidates),
        holdout_seasons=_parse_csv_ints(args.holdout_seasons),
        output_dir=args.output_dir,
        starting_bankroll_units=args.starting_bankroll_units,
        flat_bet_size_units=args.flat_bet_size_units,
        max_games=args.max_games,
        companion_search_iterations=args.companion_search_iterations,
        companion_optuna_workers=args.companion_optuna_workers,
        companion_early_stopping_rounds=args.companion_early_stopping_rounds,
    )

    console.print(
        "[bold green]Bankroll playoff complete[/bold green] "
        f"-> {result.summary_json_path}"
    )
    summary_payload = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    for candidate_summary in summary_payload.get("candidate_summaries", []):
        console.print(
            f"{candidate_summary['candidate_label']}: "
            f"ending_bankroll={candidate_summary['ending_bankroll_units']:.3f}, "
            f"roi={candidate_summary['roi'] if candidate_summary['roi'] is not None else 'n/a'}, "
            f"bets={candidate_summary['total_bets']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
