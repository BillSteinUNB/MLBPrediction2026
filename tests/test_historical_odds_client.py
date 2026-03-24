from __future__ import annotations

from pathlib import Path
import sqlite3

import pandas as pd
import pytest

from src.clients.historical_odds_client import (
    import_historical_odds,
    load_historical_odds_for_games,
)


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


def test_import_historical_odds_can_load_opening_snapshot(tmp_path: Path) -> None:
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
                "fetched_at": "2024-04-01T14:05:00Z",
            },
            {
                "commence_time": "2024-04-01T19:05:00Z",
                "home_team": "New York Yankees",
                "away_team": "Boston Red Sox",
                "home_odds": -125,
                "away_odds": 115,
                "book_name": "archive-book",
                "fetched_at": "2024-04-01T19:00:00Z",
            },
        ]
    ).to_csv(source_path, index=False)

    db_path = tmp_path / "historical_odds.db"
    import_historical_odds(
        source_path=source_path,
        db_path=db_path,
        default_market_type="f5_ml",
    )

    with sqlite3.connect(db_path) as connection:
        game_pk = int(connection.execute("SELECT game_pk FROM games LIMIT 1").fetchone()[0])

    snapshots = load_historical_odds_for_games(
        db_path=db_path,
        game_pks=[game_pk],
        market_type="f5_ml",
        book_name="archive-book",
        snapshot_selection="opening",
    )

    assert len(snapshots) == 1
    assert int(snapshots.iloc[0]["home_odds"]) == -120
    assert int(snapshots.iloc[0]["away_odds"]) == 110


def test_import_historical_odds_uses_game_pk_when_present(tmp_path: Path) -> None:
    source_path = tmp_path / "historical_odds.csv"
    pd.DataFrame(
        [
            {
                "game_pk": 777,
                "home_odds": -105,
                "away_odds": -115,
                "fetched_at": "2024-04-01T19:05:00Z",
                "book_name": "archive-book",
            }
        ]
    ).to_csv(source_path, index=False)

    db_path = tmp_path / "historical_odds.db"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
                version INTEGER NOT NULL,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS games (
                game_pk INTEGER PRIMARY KEY,
                date TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                home_starter_id INTEGER,
                away_starter_id INTEGER,
                venue TEXT NOT NULL,
                is_dome INTEGER NOT NULL DEFAULT 0,
                is_abs_active INTEGER NOT NULL DEFAULT 1,
                f5_home_score INTEGER,
                f5_away_score INTEGER,
                final_home_score INTEGER,
                final_away_score INTEGER,
                status TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS odds_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_pk INTEGER NOT NULL,
                book_name TEXT NOT NULL,
                market_type TEXT NOT NULL,
                home_odds INTEGER NOT NULL,
                away_odds INTEGER NOT NULL,
                home_point REAL,
                away_point REAL,
                fetched_at TEXT NOT NULL,
                is_frozen INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (game_pk) REFERENCES games (game_pk)
            );
            INSERT INTO games (game_pk, date, home_team, away_team, venue, is_dome, is_abs_active, status)
            VALUES (777, '2024-04-01T19:05:00+00:00', 'NYY', 'BOS', 'Yankee Stadium', 0, 1, 'scheduled');
            """
        )
        connection.commit()

    imported_row_count = import_historical_odds(
        source_path=source_path,
        db_path=db_path,
        default_market_type="f5_ml",
    )

    assert imported_row_count == 1
    snapshots = load_historical_odds_for_games(
        db_path=db_path,
        game_pks=[777],
        market_type="f5_ml",
        book_name="archive-book",
    )
    assert len(snapshots) == 1
    assert int(snapshots.iloc[0]["home_odds"]) == -105


def test_import_and_load_historical_odds_filters_absurd_prices(tmp_path: Path) -> None:
    source_path = tmp_path / "historical_odds.csv"
    pd.DataFrame(
        [
            {
                "game_pk": 778,
                "home_odds": -120,
                "away_odds": 110,
                "fetched_at": "2024-04-01T19:05:00Z",
                "book_name": "archive-book",
            },
            {
                "game_pk": 778,
                "home_odds": -10000,
                "away_odds": 2800,
                "fetched_at": "2024-04-01T19:10:00Z",
                "book_name": "archive-book",
            },
        ]
    ).to_csv(source_path, index=False)

    db_path = tmp_path / "historical_odds.db"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
                version INTEGER NOT NULL,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS games (
                game_pk INTEGER PRIMARY KEY,
                date TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                home_starter_id INTEGER,
                away_starter_id INTEGER,
                venue TEXT NOT NULL,
                is_dome INTEGER NOT NULL DEFAULT 0,
                is_abs_active INTEGER NOT NULL DEFAULT 1,
                f5_home_score INTEGER,
                f5_away_score INTEGER,
                final_home_score INTEGER,
                final_away_score INTEGER,
                status TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS odds_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_pk INTEGER NOT NULL,
                book_name TEXT NOT NULL,
                market_type TEXT NOT NULL,
                home_odds INTEGER NOT NULL,
                away_odds INTEGER NOT NULL,
                home_point REAL,
                away_point REAL,
                fetched_at TEXT NOT NULL,
                is_frozen INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (game_pk) REFERENCES games (game_pk)
            );
            INSERT INTO games (game_pk, date, home_team, away_team, venue, is_dome, is_abs_active, status)
            VALUES (778, '2024-04-01T19:05:00+00:00', 'NYY', 'BOS', 'Yankee Stadium', 0, 1, 'scheduled');
            """
        )
        connection.commit()

    imported_row_count = import_historical_odds(
        source_path=source_path,
        db_path=db_path,
        default_market_type="f5_ml",
    )

    assert imported_row_count == 1
    snapshots = load_historical_odds_for_games(
        db_path=db_path,
        game_pks=[778],
        market_type="f5_ml",
        book_name="archive-book",
    )
    assert len(snapshots) == 1
    assert int(snapshots.iloc[0]["home_odds"]) == -120


def test_import_historical_runline_odds_preserves_points(tmp_path: Path) -> None:
    source_path = tmp_path / "historical_rl.csv"
    pd.DataFrame(
        [
            {
                "game_pk": 779,
                "market_type": "f5_rl",
                "home_odds": 105,
                "away_odds": -125,
                "home_point": 0.5,
                "away_point": -0.5,
                "fetched_at": "2024-04-01T19:05:00Z",
                "book_name": "archive-book",
            }
        ]
    ).to_csv(source_path, index=False)

    db_path = tmp_path / "historical_rl.db"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
                version INTEGER NOT NULL,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS games (
                game_pk INTEGER PRIMARY KEY,
                date TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                home_starter_id INTEGER,
                away_starter_id INTEGER,
                venue TEXT NOT NULL,
                is_dome INTEGER NOT NULL DEFAULT 0,
                is_abs_active INTEGER NOT NULL DEFAULT 1,
                f5_home_score INTEGER,
                f5_away_score INTEGER,
                final_home_score INTEGER,
                final_away_score INTEGER,
                status TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS odds_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_pk INTEGER NOT NULL,
                book_name TEXT NOT NULL,
                market_type TEXT NOT NULL,
                home_odds INTEGER NOT NULL,
                away_odds INTEGER NOT NULL,
                home_point REAL,
                away_point REAL,
                fetched_at TEXT NOT NULL,
                is_frozen INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (game_pk) REFERENCES games (game_pk)
            );
            INSERT INTO games (game_pk, date, home_team, away_team, venue, is_dome, is_abs_active, status)
            VALUES (779, '2024-04-01T19:05:00+00:00', 'NYY', 'BOS', 'Yankee Stadium', 0, 1, 'scheduled');
            """
        )
        connection.commit()

    imported_row_count = import_historical_odds(
        source_path=source_path,
        db_path=db_path,
        default_market_type="f5_rl",
    )

    assert imported_row_count == 1
    snapshots = load_historical_odds_for_games(
        db_path=db_path,
        game_pks=[779],
        market_type="f5_rl",
        book_name="archive-book",
    )
    assert len(snapshots) == 1
    assert float(snapshots.iloc[0]["home_point"]) == pytest.approx(0.5)
    assert float(snapshots.iloc[0]["away_point"]) == pytest.approx(-0.5)
