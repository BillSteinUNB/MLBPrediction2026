"""
Run scraper for 2026 and 2025 seasons only
"""

import asyncio
import sys
from datetime import datetime
from scraper import MLBOddsScraper


async def main():
    """Run scraper for 2026-2025"""
    scraper = MLBOddsScraper(output_dir="data")

    # Scrape from today back to January 1, 2025
    start_date = datetime.now()
    end_date = datetime(2025, 1, 1)

    print(f"Starting scrape from {start_date.date()} to {end_date.date()}")
    print(f"This will scrape MLB seasons 2026 and 2025")

    odds = await scraper.scrape_date_range(start_date=start_date, end_date=end_date, headless=True)

    # Save final results
    scraper.save_to_csv(odds, "mlb_odds_2025_2026.csv")
    scraper.save_to_json(odds, "mlb_odds_2025_2026.json")

    print(f"\nScraping complete! Total games: {len(odds)}")

    # Summary statistics
    dates = set(o.date for o in odds)
    teams = set(o.away_team for o in odds) | set(o.home_team for o in odds)

    print(f"Summary:")
    print(f"  - Total games: {len(odds)}")
    print(f"  - Unique dates: {len(dates)}")
    print(f"  - Teams: {sorted(teams)}")


if __name__ == "__main__":
    asyncio.run(main())
