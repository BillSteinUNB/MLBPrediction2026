from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


MARKETS = (
    "full_game_ml",
    "full_game_total",
    "full_game_rl",
    "f5_ml",
    "f5_total",
    "f5_rl",
)


def _season_summary(db_path: Path, season: int) -> dict[str, object]:
    with sqlite3.connect(db_path) as connection:
        payload: dict[str, object] = {}
        for market in MARKETS:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS row_count,
                    COUNT(DISTINCT event_id) AS event_count,
                    MIN(game_date) AS min_game_date,
                    MAX(game_date) AS max_game_date
                FROM odds
                WHERE market_type = ?
                  AND game_date >= ?
                  AND game_date < ?
                """,
                (market, f"{season:04d}-01-01", f"{season + 1:04d}-01-01"),
            ).fetchone()
            payload[market] = {
                "row_count": int(row[0] or 0),
                "event_count": int(row[1] or 0),
                "min_game_date": row[2],
                "max_game_date": row[3],
            }
        return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit legacy vs OddsPortal historical market coverage")
    parser.add_argument("--legacy-db", required=True)
    parser.add_argument("--oddsportal-db", required=True)
    parser.add_argument("--start-season", type=int, default=2021)
    parser.add_argument("--end-season", type=int, default=2025)
    args = parser.parse_args()

    legacy_db = Path(args.legacy_db)
    oddsportal_db = Path(args.oddsportal_db)
    report: dict[str, object] = {
        "legacy_db": str(legacy_db),
        "oddsportal_db": str(oddsportal_db),
        "seasons": {},
    }
    for season in range(int(args.end_season), int(args.start_season) - 1, -1):
        report["seasons"][str(season)] = {
            "legacy": _season_summary(legacy_db, season),
            "oddsportal": _season_summary(oddsportal_db, season),
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
