from __future__ import annotations

import argparse

from oddsportal_scraper import OddsPortalScraper


DEFAULT_EVENT_URL = (
    "https://www.oddsportal.com/baseball/usa/mlb-2023/"
    "new-york-yankees-houston-astros-thHHjUZt/"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect OddsPortal MLB event market inventory")
    parser.add_argument("--url", default=DEFAULT_EVENT_URL, help="OddsPortal event URL")
    args = parser.parse_args()

    scraper = OddsPortalScraper()
    inventory = scraper.scan_event(args.url)

    print("=" * 60)
    print("OddsPortal Event Inventory")
    print("=" * 60)
    print(f"Event:        {inventory.event.name}")
    print(f"Start date:   {inventory.event.start_date}")
    print(f"Home team:    {inventory.event.home_team}")
    print(f"Away team:    {inventory.event.away_team}")
    print(f"Source URL:   {inventory.event.url or args.url}")
    print(f"Default bet:  {inventory.default_betting_type_id}")
    print(f"Default scope:{inventory.default_scope_id}")

    print("\nMarket scopes:")
    for scope in inventory.scopes:
        print(
            f"  bt={scope.betting_type_id} {scope.betting_type_name} | "
            f"scope={scope.scope_id} {scope.scope_name} | "
            f"params={', '.join(scope.parameter_ids)}"
        )


if __name__ == "__main__":
    main()
