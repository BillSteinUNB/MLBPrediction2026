from __future__ import annotations

import argparse
import os
import sys
from functools import partial
from pathlib import Path

from rich.console import Console


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.data_builder import build_training_dataset  # noqa: E402
from src.clients.retrosheet_client import fetch_retrosheet_umpires  # noqa: E402
from src.clients.weather_client import fetch_game_weather, fetch_game_weather_local_only  # noqa: E402
from src.db import DEFAULT_DB_PATH  # noqa: E402


console = Console()
DEFAULT_TRAINING_DATA = Path("data/training/ParquetDefault.parquet")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the canonical manual-training parquet without running any training.",
    )
    parser.add_argument("--training-data", default=str(DEFAULT_TRAINING_DATA))
    parser.add_argument("--start", "--start-year", dest="start_year", type=int, default=2018)
    parser.add_argument("--end", "--end-year", dest="end_year", type=int, default=2025)
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--allow-backfill-years", action="store_true")
    parser.add_argument("--FeatureWorker", "--feature-workers", dest="feature_workers", type=int, default=None)
    args = parser.parse_args(argv)

    if args.feature_workers is not None:
        feature_workers = max(1, min(12, int(args.feature_workers)))
        os.environ["MLB_FEATURE_BUILD_WORKERS"] = str(feature_workers)
    else:
        feature_workers = None

    training_path = Path(args.training_data)
    console.print(
        f"[bold green]Building parquet[/bold green] path={training_path} "
        f"years={args.start_year}-{args.end_year} refresh_raw_data={'yes' if args.refresh_data else 'no'} "
        f"feature_workers={feature_workers if feature_workers is not None else 'default'}"
    )
    weather_fetcher = (
        fetch_game_weather
        if args.refresh_data
        else partial(fetch_game_weather_local_only, db_path=DEFAULT_DB_PATH)
    )
    umpire_fetcher = partial(fetch_retrosheet_umpires, db_path=DEFAULT_DB_PATH)
    console.print(
        f"weather_mode={'live_refresh' if args.refresh_data else 'local_only'}"
    )
    console.print(f"umpire_db={DEFAULT_DB_PATH}")
    result = build_training_dataset(
        start_year=args.start_year,
        end_year=args.end_year,
        output_path=training_path,
        allow_backfill_years=args.allow_backfill_years,
        refresh=args.refresh_data,
        refresh_raw_data=args.refresh_data,
        weather_fetcher=weather_fetcher,
        umpire_fetcher=umpire_fetcher,
    )
    console.print(
        f"[bold green]Done[/bold green] rows={len(result.dataframe)} parquet={result.output_path}"
    )
    console.print(f"metadata={result.metadata_path}")
    console.print(f"hash={result.data_version_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
