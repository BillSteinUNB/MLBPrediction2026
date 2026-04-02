from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from pathlib import Path

from playwright.async_api import Error as PlaywrightError, async_playwright

from oddsportal_scraper import OddsPortalScraper
from scraper import SQLiteStore


DEFAULT_DB_PATH = Path("data/mlb_odds_oddsportal.db")


def _parse_optional_date(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.strptime(value, "%Y-%m-%d")


def _coerce_optional_date(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d")


def _season_range_descending(start_season: int, end_date: datetime | None) -> list[int]:
    if end_date is None or end_date.year >= start_season:
        return [start_season]
    return list(range(start_season, end_date.year - 1, -1))


def _season_bounds(
    *,
    season: int,
    explicit_start_date: datetime | None,
    explicit_end_date: datetime | None,
    resume_oldest_date: datetime | None,
) -> tuple[datetime | None, datetime | None]:
    lower_bound = explicit_start_date if explicit_start_date is not None and explicit_start_date.year == season else None
    upper_bound = None
    if explicit_end_date is not None and explicit_end_date.year == season:
        upper_bound = explicit_end_date
    elif resume_oldest_date is not None:
        upper_bound = resume_oldest_date
    return lower_bound, upper_bound


async def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill MLB standard markets from OddsPortal")
    parser.add_argument("--season", type=int, required=True, help="Season year, e.g. 2023")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="SQLite database path")
    parser.add_argument("--start-date", help="Optional YYYY-MM-DD lower bound")
    parser.add_argument("--end-date", help="Optional YYYY-MM-DD upper bound")
    parser.add_argument("--max-pages", type=int, help="Optional max results pages to scan")
    parser.add_argument("--max-events", type=int, help="Optional max events to scrape")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent event workers (default: 4)")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable automatic resume from the oldest existing OddsPortal date in the target DB",
    )
    args = parser.parse_args()

    db = SQLiteStore(Path(args.db_path))
    scraper = OddsPortalScraper()

    explicit_start_date = _parse_optional_date(args.start_date)
    explicit_end_date = _parse_optional_date(args.end_date)
    seasons_to_process = _season_range_descending(args.season, explicit_end_date)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        )
        processed_events = 0
        inserted_rows = 0
        season_summaries: list[tuple[int, int, int, datetime | None]] = []
        try:
            try:
                remaining_max_events = args.max_events
                for season in seasons_to_process:
                    print(f"Starting season {season}...")
                    existing_event_ids = db.get_existing_event_ids_for_bookmaker_prefix(
                        bookmaker_prefix="OddsPortal:",
                        season=season,
                    )
                    resume_oldest_date = None
                    if not args.no_resume and explicit_start_date is None:
                        oldest_existing = db.get_oldest_game_date_for_bookmaker_prefix(
                            bookmaker_prefix="OddsPortal:",
                            season=season,
                        )
                        resume_oldest_date = _coerce_optional_date(oldest_existing)

                    effective_start_date, effective_end_date = _season_bounds(
                        season=season,
                        explicit_start_date=explicit_start_date,
                        explicit_end_date=explicit_end_date,
                        resume_oldest_date=resume_oldest_date,
                    )

                    season_processed, season_inserted = await scraper.backfill_events_to_db(
                        context,
                        season=season,
                        db=db,
                        start_date=effective_start_date,
                        end_date=effective_end_date,
                        max_pages=args.max_pages,
                        max_events=remaining_max_events,
                        existing_event_ids=existing_event_ids,
                        progress_callback=lambda label, current_season=season: print(f"{current_season} {label} complete"),
                        concurrency=args.concurrency,
                    )
                    processed_events += season_processed
                    inserted_rows += season_inserted
                    season_summaries.append((season, season_processed, season_inserted, resume_oldest_date))
                    if remaining_max_events is not None:
                        remaining_max_events = max(0, remaining_max_events - season_processed)
                        if remaining_max_events == 0:
                            break
            except KeyboardInterrupt:
                print("\nInterrupted. Rerun the same command to continue from the oldest OddsPortal date already in the DB.")
        finally:
            try:
                await context.close()
            except PlaywrightError:
                pass
            try:
                await browser.close()
            except PlaywrightError:
                pass

    summary = db.summary()
    print("=" * 60)
    print("OddsPortal MLB Backfill")
    print("=" * 60)
    if len(seasons_to_process) == 1:
        print(f"Season:               {args.season}")
    else:
        print(f"Season range:         {seasons_to_process[0]} -> {seasons_to_process[-1]}")
    print(f"Processed events:     {processed_events:,}")
    print(f"Inserted rows:        {inserted_rows:,}")
    print(f"Total rows in DB:     {summary.get('total_rows', 0):,}")
    print(f"Date range:           {summary.get('date_range', 'N/A')}")
    if season_summaries:
        print("\nPer season:")
        for season, season_processed, season_inserted, season_resume_oldest in season_summaries:
            print(f"  {season}: events={season_processed:,}, inserted={season_inserted:,}")
            if season_resume_oldest is not None:
                print(f"    resume oldest date: {season_resume_oldest.date().isoformat()}")
    if summary.get("by_market_type"):
        print("\nBy market_type:")
        for market_type, count in sorted(summary["by_market_type"].items()):
            print(f"  {market_type}: {count:,}")
    if summary.get("by_bookmaker"):
        print("\nBy bookmaker:")
        for bookmaker, count in sorted(summary["by_bookmaker"].items()):
            if bookmaker.startswith("OddsPortal:"):
                print(f"  {bookmaker}: {count:,}")
    db.close()


if __name__ == "__main__":
    asyncio.run(main())
