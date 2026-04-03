"""
Full historical scrape: 2026 backwards to 2015, writing to SQLite.

Supports resume — if the DB already has data, picks up where it left off.
Run as many times as needed; it won't re-scrape dates already in the DB.

Usage:
    python run_full.py              # 2026 -> 2015, default settings
    python run_full.py --export     # also dump final CSV after scraping
"""

import argparse
import asyncio
import shutil
from datetime import datetime
from pathlib import Path

from playwright.async_api import async_playwright
from scraper import MLBOddsScraper, SQLiteStore, SEASON_DATES


DB_PATH = Path("data/mlb_odds.db")
BACKUP_DB_PATH = Path("data/mlb_odds_backup.db")


def _database_coverage(db_path: Path) -> tuple[str | None, str | None, int]:
    if not db_path.exists():
        return None, None, 0

    db = SQLiteStore(db_path)
    try:
        earliest, latest = db.date_range()
        return earliest, latest, db.count()
    finally:
        db.close()


def _seed_active_db_from_backup() -> Path:
    active_earliest, active_latest, active_rows = _database_coverage(DB_PATH)
    backup_earliest, backup_latest, backup_rows = _database_coverage(BACKUP_DB_PATH)

    if backup_rows == 0:
        return DB_PATH

    should_promote_backup = False
    if active_rows == 0:
        should_promote_backup = True
    elif backup_earliest and (active_earliest is None or backup_earliest < active_earliest):
        should_promote_backup = True
    elif backup_rows > active_rows and backup_earliest == active_earliest:
        should_promote_backup = True

    if should_promote_backup:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BACKUP_DB_PATH, DB_PATH)
        print(
            f"Seeded active DB from backup: {BACKUP_DB_PATH} -> {DB_PATH} "
            f"(backup rows={backup_rows:,}, range={backup_earliest} to {backup_latest})"
        )
    else:
        print(
            f"Using existing active DB: {DB_PATH} "
            f"(rows={active_rows:,}, range={active_earliest} to {active_latest})"
        )

    return DB_PATH


def _parse_cli_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


async def main(
    export_csv: bool = False,
    *,
    start_date_override: str | None = None,
    end_date_override: str | None = None,
    no_resume: bool = False,
):
    scraper = MLBOddsScraper(output_dir="data", base_delay=2.0, max_parallel=3)
    active_db_path = _seed_active_db_from_backup()
    db = SQLiteStore(active_db_path)

    # Today backwards to the start of 2015 spring training
    start_date = (
        _parse_cli_date(start_date_override)
        if start_date_override is not None
        else datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    )
    end_date = (
        _parse_cli_date(end_date_override)
        if end_date_override is not None
        else SEASON_DATES[2015]["spring_training_start"]
    )
    if end_date > start_date:
        raise ValueError(
            f"end_date must be on or before start_date; got start={start_date.date()} end={end_date.date()}"
        )

    # Show existing DB state
    existing = db.count()
    date_range = db.date_range()

    print(f"{'=' * 60}")
    print(f"MLB Odds Scraper -> SQLite")
    print(f"  {start_date.date()} backwards to {end_date.date()}")
    print(f"  DB: {active_db_path}")
    if existing:
        print(f"  Existing rows: {existing:,}")
        print(f"  Existing range: {date_range[0]} to {date_range[1]}")
        if no_resume:
            print(f"  Resume disabled for this run")
        else:
            print(f"  Will resume from earliest date in DB")
    else:
        print(f"  Fresh database — starting from scratch")
    print(f"{'=' * 60}")
    print()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )

        try:
            inserted = await scraper.scrape_date_range_backwards_db(
                context,
                start_date=start_date,
                end_date=end_date,
                db=db,
                resume=not no_resume,
            )
        except KeyboardInterrupt:
            print("\nInterrupted! Data already saved to DB — resume anytime.")
            inserted = 0
        finally:
            await browser.close()

    # Summary
    summary = db.summary()
    print(f"\n{'=' * 60}")
    print(f"SCRAPING COMPLETE")
    print(f"{'=' * 60}")
    print(f"New rows this run: {inserted:,}")
    print(f"Total rows in DB:  {summary['total_rows']:,}")
    print(f"Date range:        {summary.get('date_range', 'N/A')}")
    print(f"Unique dates:      {summary.get('unique_dates', 0):,}")
    print(f"Unique games:      {summary.get('unique_games', 0):,}")

    if summary.get("by_game_type"):
        print(f"\nBy game_type:")
        for k, v in sorted(summary["by_game_type"].items()):
            print(f"  {k}: {v:,}")

    if summary.get("by_market_type"):
        print(f"\nBy market_type:")
        for k, v in sorted(summary["by_market_type"].items()):
            print(f"  {k}: {v:,}")

    if summary.get("by_bookmaker"):
        print(f"\nBy bookmaker:")
        for k, v in sorted(summary["by_bookmaker"].items()):
            print(f"  {k}: {v:,}")

    # Optional CSV export
    if export_csv:
        csv_path = "data/mlb_odds_full_export.csv"
        print(f"\nExporting to {csv_path}...")
        db.export_csv(csv_path)

    db.close()
    print(f"\nResume anytime with: python run_full.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full historical MLB odds scrape to SQLite")
    parser.add_argument("--export", action="store_true", help="Export CSV after scraping")
    parser.add_argument("--start-date", help="Optional YYYY-MM-DD start date override")
    parser.add_argument("--end-date", help="Optional YYYY-MM-DD end date override")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore earliest-date resume logic and scrape only the requested window",
    )
    args = parser.parse_args()
    asyncio.run(
        main(
            export_csv=args.export,
            start_date_override=args.start_date,
            end_date_override=args.end_date,
            no_resume=args.no_resume,
        )
    )
