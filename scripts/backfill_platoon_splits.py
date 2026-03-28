from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from src.db import DEFAULT_DB_PATH, init_db, sqlite_connection
from src.features.offense import TEAM_PLATOON_SPLITS_TABLE, _normalize_team_platoon_abbr


FANGRAPHS_SPLITS_URL = "https://www.fangraphs.com/api/leaders/splits/splits-leaders"
FANGRAPHS_REFERER = "https://www.fangraphs.com/leaders/splits-leaderboards"
FANGRAPHS_HEADERS = {
    "origin": "https://www.fangraphs.com",
    "referer": FANGRAPHS_REFERER,
}
_FANGRAPHS_SPLIT_IDS = {"L": 1, "R": 2}


def _build_payload(*, season: int, vs_hand: str) -> dict[str, Any]:
    return {
        "strPlayerId": "all",
        "strSplitArr": [_FANGRAPHS_SPLIT_IDS[vs_hand]],
        "strSplitArrPitch": [],
        "strGroup": "season",
        "strPosition": "B",
        "strType": 2,
        "strStartDate": f"{season}-01-01",
        "strEndDate": f"{season}-12-31",
        "strSplitTeams": False,
        "dctFilters": [],
        "strStatType": "team",
        "strAutoPt": "false",
        "arrPlayerId": [],
    }


def _fetch_fangraphs_team_splits(
    client: httpx.Client,
    *,
    season: int,
    vs_hand: str,
) -> list[dict[str, Any]]:
    response = client.post(
        FANGRAPHS_SPLITS_URL,
        json=_build_payload(season=season, vs_hand=vs_hand),
        timeout=60.0,
    )
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data", [])
    if not isinstance(rows, list):
        raise ValueError(f"Unexpected FanGraphs response for season={season}, vs_hand={vs_hand}")
    return rows


def _normalize_split_row(row: dict[str, Any], *, season: int, vs_hand: str) -> tuple[Any, ...] | None:
    raw_team = row.get("TeamNameAbb") or row.get("Team") or row.get("Name")
    team_abbr = _normalize_team_platoon_abbr(str(raw_team or ""))
    if team_abbr is None:
        return None

    woba = pd.to_numeric(row.get("wOBA"), errors="coerce")
    xwoba = pd.to_numeric(row.get("xwOBA"), errors="coerce")
    k_pct = pd.to_numeric(row.get("K%"), errors="coerce")
    bb_pct = pd.to_numeric(row.get("BB%"), errors="coerce")
    pa = pd.to_numeric(row.get("PA"), errors="coerce")

    return (
        team_abbr,
        int(season),
        str(vs_hand).upper(),
        float(woba) if pd.notna(woba) else None,
        float(xwoba) if pd.notna(xwoba) else None,
        float(k_pct) if pd.notna(k_pct) else None,
        float(bb_pct) if pd.notna(bb_pct) else None,
        int(pa) if pd.notna(pa) else None,
    )


def backfill_platoon_splits(
    *,
    start_season: int,
    end_season: int,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    if end_season < start_season:
        raise ValueError("end_season must be greater than or equal to start_season")

    database_path = init_db(db_path)
    rows_to_upsert: list[tuple[Any, ...]] = []

    with httpx.Client(headers=FANGRAPHS_HEADERS, follow_redirects=True) as client:
        client.get(FANGRAPHS_REFERER, timeout=60.0)
        for season in range(start_season, end_season + 1):
            for vs_hand in ("L", "R"):
                print(f"[platoon] Fetching FanGraphs team splits for {season} vs {vs_hand}HP")
                raw_rows = _fetch_fangraphs_team_splits(client, season=season, vs_hand=vs_hand)
                normalized_rows = [
                    normalized
                    for normalized in (
                        _normalize_split_row(row, season=season, vs_hand=vs_hand)
                        for row in raw_rows
                    )
                    if normalized is not None
                ]
                rows_to_upsert.extend(normalized_rows)
                print(
                    f"[platoon] Normalized {len(normalized_rows)} rows for {season} vs {vs_hand}HP"
                )

    with sqlite_connection(database_path, builder_optimized=True) as connection:
        connection.executemany(
            f"""
            INSERT INTO {TEAM_PLATOON_SPLITS_TABLE}
                (team_abbr, season, vs_hand, woba, xwoba, k_pct, bb_pct, pa)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(team_abbr, season, vs_hand)
            DO UPDATE SET
                woba = excluded.woba,
                xwoba = excluded.xwoba,
                k_pct = excluded.k_pct,
                bb_pct = excluded.bb_pct,
                pa = excluded.pa
            """,
            rows_to_upsert,
        )
        total_rows = connection.execute(
            f"SELECT COUNT(*) FROM {TEAM_PLATOON_SPLITS_TABLE}"
        ).fetchone()[0]
        connection.commit()

    print(
        "[platoon] FanGraphs advanced team splits do not currently expose xwOBA on the public "
        "splits-leaderboard endpoint, so `xwoba` is stored as NULL when unavailable."
    )
    print(
        f"[platoon] Summary: upserted {len(rows_to_upsert)} rows for seasons "
        f"{start_season}-{end_season}; table now contains {int(total_rows)} rows"
    )
    return int(total_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill FanGraphs team platoon splits")
    parser.add_argument("--start-season", type=int, default=2018)
    parser.add_argument("--end-season", type=int, default=2024)
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    args = parser.parse_args()

    backfill_platoon_splits(
        start_season=args.start_season,
        end_season=args.end_season,
        db_path=args.db_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
