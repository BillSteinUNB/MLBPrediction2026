"""
Scrape current BetUS MLB board team totals into the shared SQLite warehouse.

This collector captures:
  - full_game_team_total_away
  - full_game_team_total_home

It preserves "opening" semantics by marking the first seen snapshot for a
given event/book/market/side/point as is_opening=1 and keeps later runs as
newer snapshots with is_opening=0.
"""

import argparse
import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

from scraper import MLBOddsScraper, SQLiteStore


DEFAULT_DB_PATH = Path("data/mlb_odds.db")


async def main(*, db_path: str | Path = DEFAULT_DB_PATH, export_csv: bool = False) -> None:
    scraper = MLBOddsScraper(output_dir="data", base_delay=2.0, max_parallel=3)
    db = SQLiteStore(db_path)

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
            rows = await scraper.scrape_betus_team_totals(context)
        finally:
            await browser.close()

    db.flag_first_seen_rows_as_opening(rows)
    db.insert_rows(rows)
    summary = db.summary()

    print(f"{'=' * 60}")
    print("BetUS MLB Team Totals Collector")
    print(f"{'=' * 60}")
    print(f"Rows inserted this run: {len(rows):,}")
    print(f"Total rows in DB:      {summary.get('total_rows', 0):,}")
    print(f"Date range:            {summary.get('date_range', 'N/A')}")

    by_market = summary.get("by_market_type", {})
    if by_market:
        print("\nBy market_type:")
        for market_type, count in sorted(by_market.items()):
            print(f"  {market_type}: {count:,}")

    if export_csv:
        csv_path = Path("data/mlb_odds_full_export.csv")
        db.export_csv(csv_path)
        print(f"\nExported CSV: {csv_path}")

    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape BetUS MLB team totals into SQLite")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="SQLite database path")
    parser.add_argument("--export", action="store_true", help="Export CSV after scraping")
    args = parser.parse_args()
    asyncio.run(main(db_path=args.db_path, export_csv=args.export))
