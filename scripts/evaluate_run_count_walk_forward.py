from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ops.run_count_walk_forward import (  # noqa: E402
    DEFAULT_RUN_COUNT_WALK_FORWARD_DIR,
    evaluate_mcmc_holdout_walk_forward,
    evaluate_stage3_holdout_walk_forward,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Stage 3 and Stage 4 away-run lanes in holdout-month walk-forward form.",
    )
    parser.add_argument("--training-data", default="data/training/ParquetDefault.parquet")
    parser.add_argument("--stage3-metadata", default=None)
    parser.add_argument("--mcmc-metadata", default=None)
    parser.add_argument("--holdout", "--holdout-season", dest="holdout_season", type=int, default=2025)
    parser.add_argument("--output-dir", default=str(DEFAULT_RUN_COUNT_WALK_FORWARD_DIR))
    parser.add_argument("--enable-market-priors", action="store_true")
    parser.add_argument("--historical-odds-db", default=None)
    parser.add_argument("--historical-market-book", default=None)
    args = parser.parse_args(argv)

    outputs: dict[str, object] = {}
    if args.stage3_metadata:
        report, paths = evaluate_stage3_holdout_walk_forward(
            training_data=args.training_data,
            model_metadata_path=args.stage3_metadata,
            output_dir=args.output_dir,
            holdout_season=args.holdout_season,
            enable_market_priors=args.enable_market_priors,
            historical_odds_db_path=args.historical_odds_db,
            historical_market_book_name=args.historical_market_book,
        )
        outputs["stage3"] = {
            "artifact_path": report["artifact_path"],
            "json_path": str(paths.json_path),
            "csv_path": str(paths.csv_path),
        }

    if args.mcmc_metadata:
        report, paths = evaluate_mcmc_holdout_walk_forward(
            mcmc_metadata_path=args.mcmc_metadata,
            output_dir=args.output_dir,
            historical_odds_db_path=args.historical_odds_db,
            historical_market_book_name=args.historical_market_book,
        )
        outputs["mcmc"] = {
            "artifact_path": report["artifact_path"],
            "json_path": str(paths.json_path),
            "csv_path": str(paths.csv_path),
        }

    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
