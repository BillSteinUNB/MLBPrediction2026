from __future__ import annotations

from pathlib import Path
import sqlite3

import pandas as pd

from src.clients.historical_odds_client import import_historical_odds, load_historical_odds_for_games


def test_import_historical_odds_and_load_latest_snapshot(tmp_path: Path) -> None:
    source_path = tmp_path / "historical_odds.csv"
    pd.DataFrame(
        [
            {
                "commence_time": "2024-04-01T19:05:00Z",
                "home_team": "New York Yankees",
                "away_team": "Boston Red Sox",
                "home_odds": -120,
                "away_odds": 110,
                "book_name": "archive-book",
            },
            {
                "commence_time": "2024-04-01T19:05:00Z",
                "home_team": "New York Yankees",
                "away_team": "Boston Red Sox",
                "home_odds": -125,
                "away_odds": 115,
                "book_name": "archive-book",
                "fetched_at": "2024-04-01T19:10:00Z",
            },
        ]
    ).to_csv(source_path, index=False)

    db_path = tmp_path / "historical_odds.db"
    imported_row_count = import_historical_odds(
        source_path=source_path,
        db_path=db_path,
        default_market_type="f5_ml",
    )
    assert imported_row_count == 2

    with sqlite3.connect(db_path) as connection:
        game_pk = int(connection.execute("SELECT game_pk FROM games LIMIT 1").fetchone()[0])

    snapshots = load_historical_odds_for_games(
        db_path=db_path,
        game_pks=[game_pk],
        market_type="f5_ml",
        book_name="archive-book",
    )

    assert len(snapshots) == 1
    assert int(snapshots.iloc[0]["home_odds"]) == -125
    assert int(snapshots.iloc[0]["away_odds"]) == 115
