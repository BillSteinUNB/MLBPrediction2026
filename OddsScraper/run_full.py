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
from datetime import datetime

from playwright.async_api import async_playwright
from scraper import MLBOddsScraper, SQLiteStore, SEASON_DATES


DB_PATH = "data/mlb_odds.db"


async def main(export_csv: bool = False):
    scraper = MLBOddsScraper(output_dir="data", base_delay=2.0, max_parallel=3)
    db = SQLiteStore(DB_PATH)

    # Today backwards to the start of 2015 spring training
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = SEASON_DATES[2015]["spring_training_start"]

    # Show existing DB state
    existing = db.count()
    date_range = db.date_range()

    print(f"{'=' * 60}")
    print(f"MLB Odds Scraper -> SQLite")
    print(f"  {start_date.date()} backwards to {end_date.date()}")
    print(f"  DB: {DB_PATH}")
    if existing:
        print(f"  Existing rows: {existing:,}")
        print(f"  Existing range: {date_range[0]} to {date_range[1]}")
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
    args = parser.parse_args()
    asyncio.run(main(export_csv=args.export))
