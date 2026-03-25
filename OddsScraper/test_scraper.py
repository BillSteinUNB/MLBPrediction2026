"""
Test script to verify the MLB odds scraper works correctly
Tests on a few dates before running the full scrape
"""

import asyncio
import sys
from datetime import datetime, timedelta
from scraper import MLBOddsScraper


async def test_scraper():
    """Test the scraper on a few dates"""
    scraper = MLBOddsScraper(output_dir="data")

    # Test on 3 specific dates across different years
    test_dates = [
        datetime(2021, 4, 4),  # Your original target
        datetime(2019, 6, 15),  # Mid-season 2019
        datetime(2016, 7, 4),  # 2016 season
    ]

    print("=" * 60)
    print("MLB Odds Scraper - Test Run")
    print("=" * 60)

    all_odds = []

    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        page = await context.new_page()

        for test_date in test_dates:
            print(f"\nTesting {test_date.strftime('%Y-%m-%d')}...")
            odds = await scraper.scrape_date(page, test_date)

            if odds:
                print(f"  [OK] Found {len(odds)} games")
                print(f"    Sample: {odds[0].away_team} @ {odds[0].home_team}")
                all_odds.extend(odds)
            else:
                print(f"  [X] No games found")

            await asyncio.sleep(1)

        await browser.close()

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    print(f"Total games extracted: {len(all_odds)}")

    if all_odds:
        print("\nSample data:")
        for o in all_odds[:3]:
            print(f"  {o.date}: {o.away_team} @ {o.home_team} ({o.game_time})")
            if o.away_ml_odds:
                print(f"    Away ML: {o.away_ml_odds[:50]}...")
            if o.home_ml_odds:
                print(f"    Home ML: {o.home_ml_odds[:50]}...")

        # Save test results
        scraper.save_to_csv(all_odds, "test_results.csv")
        print(f"\n[OK] Test results saved to data/test_results.csv")
        print("\nThe scraper is working! Ready to run full scrape.")
        return True
    else:
        print("\n[X] No data extracted. Check the scraper logic.")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_scraper())
    sys.exit(0 if success else 1)
