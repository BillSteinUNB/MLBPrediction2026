"""
Run scraper: today backwards through 2025 spring training.

Covers:
  - 2026 regular season (from today)
  - 2026 spring training
  - 2025 postseason
  - 2025 regular season
  - 2025 spring training

Normalized output: one row per game × bookmaker × market × side.
All rows labeled with game_type.
"""

import asyncio
from collections import Counter
from datetime import datetime
from scraper import MLBOddsScraper, SEASON_DATES
from playwright.async_api import async_playwright


async def main():
    scraper = MLBOddsScraper(output_dir="data", base_delay=2.0, max_parallel=3)

    # Today backwards to the start of 2025 spring training
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = SEASON_DATES[2025]["spring_training_start"]

    print(f"{'=' * 60}")
    print(f"MLB Odds Scraper (normalized)")
    print(f"  {start_date.date()} backwards to {end_date.date()}")
    print(f"{'=' * 60}")
    print(f"Output: one row per game × bookmaker × market × side")
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

        rows = []
        try:
            rows = await scraper.scrape_date_range_backwards(
                context,
                start_date=start_date,
                end_date=end_date,
                progress_filename="mlb_odds_2025_2026_progress.csv",
            )

            scraper.save_to_csv(rows, "mlb_odds_2025_2026.csv")
            scraper.save_to_json(rows, "mlb_odds_2025_2026.json")

        except KeyboardInterrupt:
            print("\nInterrupted! Saving progress...")
            if rows:
                scraper.save_to_csv(rows, "mlb_odds_2025_2026_interrupted.csv")

        finally:
            await browser.close()

    # ---- Summary ---------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"SCRAPING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total rows: {len(rows)}")

    if not rows:
        return

    by_type = Counter(r.game_type for r in rows)
    print(f"\nBy game_type:")
    for gt in sorted(by_type):
        print(f"  {gt}: {by_type[gt]:,}")

    by_mkt = Counter(r.market_type for r in rows)
    print(f"\nBy market_type:")
    for mt in sorted(by_mkt):
        print(f"  {mt}: {by_mkt[mt]:,}")

    by_bk = Counter(r.bookmaker for r in rows)
    print(f"\nBy bookmaker:")
    for bk in sorted(by_bk):
        print(f"  {bk}: {by_bk[bk]:,}")

    dates = sorted(set(r.game_date for r in rows))
    games = set((r.game_date, r.away_team, r.home_team, r.commence_time_utc) for r in rows)
    print(f"\nDate range: {dates[0]} to {dates[-1]}")
    print(f"Unique dates: {len(dates)}")
    print(f"Unique games: {len(games)}")

    print(f"\nSample rows (first 3):")
    for r in rows[:3]:
        print(
            f"  {r.game_date} [{r.game_type}] {r.away_team}@{r.home_team} "
            f"| {r.bookmaker} {r.market_type} {r.side} "
            f"pt={r.point or '-'} px={r.price}"
        )


if __name__ == "__main__":
    asyncio.run(main())
