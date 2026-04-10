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
                "fetched_at": "2024-04-01T18:00:00Z",
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


def test_load_historical_odds_excludes_post_start_canonical_latest_snapshot(tmp_path: Path) -> None:
    source_path = tmp_path / "historical_odds.csv"
    pd.DataFrame(
        [
            {
                "game_pk": 780,
                "home_odds": -118,
                "away_odds": 108,
                "fetched_at": "2024-04-01T18:30:00Z",
                "book_name": "archive-book",
            },
            {
                "game_pk": 780,
                "home_odds": -150,
                "away_odds": 130,
                "fetched_at": "2024-04-01T19:10:00Z",
                "book_name": "archive-book",
            },
        ]
    ).to_csv(source_path, index=False)

    db_path = tmp_path / "historical_cutoff.db"
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
            VALUES (780, '2024-04-01T19:05:00+00:00', 'NYY', 'BOS', 'Yankee Stadium', 0, 1, 'scheduled');
            """
        )
        connection.commit()

    import_historical_odds(
        source_path=source_path,
        db_path=db_path,
        default_market_type="f5_ml",
    )

    snapshots = load_historical_odds_for_games(
        db_path=db_path,
        games_frame=pd.DataFrame(
            [
                {
                    "game_pk": 780,
                    "scheduled_start": "2024-04-01T19:05:00Z",
                    "as_of_timestamp": "2024-04-01T19:00:00Z",
                }
            ]
        ),
        market_type="f5_ml",
        book_name="archive-book",
        snapshot_selection="latest",
    )

    assert len(snapshots) == 1
    assert int(snapshots.iloc[0]["home_odds"]) == -118
    assert int(snapshots.iloc[0]["away_odds"]) == 108


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


def test_import_historical_away_team_total_odds_preserves_total_columns(tmp_path: Path) -> None:
    source_path = tmp_path / "historical_team_total.csv"
    pd.DataFrame(
        [
            {
                "game_pk": 780,
                "market_type": "full_game_team_total_away",
                "total_point": 4.5,
                "over_odds": -118,
                "under_odds": -102,
                "fetched_at": "2024-04-01T19:05:00Z",
                "book_name": "archive-book",
            }
        ]
    ).to_csv(source_path, index=False)

    db_path = tmp_path / "historical_team_total.db"
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
            VALUES (780, '2024-04-01T19:05:00+00:00', 'NYY', 'BOS', 'Yankee Stadium', 0, 1, 'scheduled');
            """
        )
        connection.commit()

    imported_row_count = import_historical_odds(
        source_path=source_path,
        db_path=db_path,
        default_market_type="full_game_team_total_away",
    )

    assert imported_row_count == 1
    snapshots = load_historical_odds_for_games(
        db_path=db_path,
        game_pks=[780],
        market_type="full_game_team_total_away",
        book_name="archive-book",
    )
    assert len(snapshots) == 1
    assert float(snapshots.iloc[0]["total_point"]) == pytest.approx(4.5)
    assert int(snapshots.iloc[0]["over_odds"]) == -118
    assert int(snapshots.iloc[0]["under_odds"]) == -102
    assert pd.isna(snapshots.iloc[0]["home_odds"])
    assert pd.isna(snapshots.iloc[0]["away_odds"])
    assert str(snapshots.iloc[0]["source_schema"]) == "canonical"


def _seed_old_scraper_db(db_path: Path) -> Path:
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE games (
                game_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT,
                game_date TEXT NOT NULL,
                commence_time_utc TEXT,
                away_team TEXT NOT NULL,
                home_team TEXT NOT NULL,
                game_type TEXT,
                away_pitcher TEXT,
                home_pitcher TEXT
            );
            CREATE TABLE odds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT,
                game_date TEXT NOT NULL,
                commence_time TEXT,
                away_team TEXT NOT NULL,
                home_team TEXT NOT NULL,
                game_type TEXT,
                away_pitcher TEXT,
                home_pitcher TEXT,
                fetched_at TEXT,
                bookmaker TEXT NOT NULL,
                market_type TEXT NOT NULL,
                side TEXT NOT NULL,
                point TEXT,
                price TEXT NOT NULL,
                commence_time_utc TEXT,
                is_opening INTEGER DEFAULT 0,
                game_id INTEGER
            );
            """
        )
        connection.execute(
            """
            INSERT INTO games (event_id, game_date, commence_time_utc, away_team, home_team, game_type)
            VALUES ('evt-1001', '2025-04-01', '2025-04-01T23:05:00Z', 'BOS', 'NYY', 'R')
            """
        )
        connection.executemany(
            """
            INSERT INTO odds (
                event_id, game_date, commence_time, away_team, home_team, fetched_at,
                bookmaker, market_type, side, point, price, commence_time_utc, is_opening
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "evt-1001",
                    "2025-04-01",
                    "2025-04-01T23:05:00Z",
                    "BOS",
                    "NYY",
                    "2025-04-01T16:00:00Z",
                    "DraftKings",
                    "full_game_total",
                    "over",
                    "8.5",
                    "-110",
                    "2025-04-01T23:05:00Z",
                    1,
                ),
                (
                    "evt-1001",
                    "2025-04-01",
                    "2025-04-01T23:05:00Z",
                    "BOS",
                    "NYY",
                    "2025-04-01T16:00:00Z",
                    "DraftKings",
                    "full_game_team_total_away",
                    "over",
                    "4.5",
                    "-108",
                    "2025-04-01T23:05:00Z",
                    1,
                ),
                (
                    "evt-1001",
                    "2025-04-01",
                    "2025-04-01T23:05:00Z",
                    "BOS",
                    "NYY",
                    "2025-04-01T16:00:00Z",
                    "DraftKings",
                    "full_game_team_total_away",
                    "under",
                    "4.5",
                    "-112",
                    "2025-04-01T23:05:00Z",
                    1,
                ),
                (
                    "evt-1001",
                    "2025-04-01",
                    "2025-04-01T23:05:00Z",
                    "BOS",
                    "NYY",
                    "2025-04-01T16:00:00Z",
                    "FanDuel",
                    "full_game_team_total_away",
                    "over",
                    "4.5",
                    "-120",
                    "2025-04-01T23:05:00Z",
                    1,
                ),
                (
                    "evt-1001",
                    "2025-04-01",
                    "2025-04-01T23:05:00Z",
                    "BOS",
                    "NYY",
                    "2025-04-01T16:00:00Z",
                    "FanDuel",
                    "full_game_team_total_away",
                    "under",
                    "4.5",
                    "100",
                    "2025-04-01T23:05:00Z",
                    1,
                ),
                (
                    "evt-1001",
                    "2025-04-01",
                    "2025-04-01T23:05:00Z",
                    "BOS",
                    "NYY",
                    "2025-04-01T16:00:00Z",
                    "DraftKings",
                    "full_game_total",
                    "under",
                    "8.5",
                    "-110",
                    "2025-04-01T23:05:00Z",
                    1,
                ),
                (
                    "evt-1001",
                    "2025-04-01",
                    "2025-04-01T23:05:00Z",
                    "BOS",
                    "NYY",
                    "2025-04-01T16:00:00Z",
                    "FanDuel",
                    "full_game_total",
                    "over",
                    "8.5",
                    "-120",
                    "2025-04-01T23:05:00Z",
                    1,
                ),
                (
                    "evt-1001",
                    "2025-04-01",
                    "2025-04-01T23:05:00Z",
                    "BOS",
                    "NYY",
                    "2025-04-01T16:00:00Z",
                    "FanDuel",
                    "full_game_total",
                    "under",
                    "8.5",
                    "100",
                    "2025-04-01T23:05:00Z",
                    1,
                ),
                (
                    "evt-1001",
                    "2025-04-01",
                    "2025-04-01T23:05:00Z",
                    "BOS",
                    "NYY",
                    "2025-04-01T16:00:00Z",
                    "DraftKings",
                    "f5_ml",
                    "away",
                    "",
                    "120",
                    "2025-04-01T23:05:00Z",
                    1,
                ),
                (
                    "evt-1001",
                    "2025-04-01",
                    "2025-04-01T23:05:00Z",
                    "BOS",
                    "NYY",
                    "2025-04-01T16:00:00Z",
                    "DraftKings",
                    "f5_ml",
                    "home",
                    "",
                    "-130",
                    "2025-04-01T23:05:00Z",
                    1,
                ),
            ],
        )
        connection.commit()
    return db_path


def _seed_oddsportal_old_scraper_db(db_path: Path) -> Path:
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE games (
                game_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT,
                game_date TEXT NOT NULL,
                commence_time_utc TEXT,
                away_team TEXT NOT NULL,
                home_team TEXT NOT NULL,
                game_type TEXT,
                away_pitcher TEXT,
                home_pitcher TEXT
            );
            CREATE TABLE odds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT,
                game_date TEXT NOT NULL,
                commence_time TEXT,
                away_team TEXT NOT NULL,
                home_team TEXT NOT NULL,
                game_type TEXT,
                away_pitcher TEXT,
                home_pitcher TEXT,
                fetched_at TEXT,
                bookmaker TEXT NOT NULL,
                market_type TEXT NOT NULL,
                side TEXT NOT NULL,
                point TEXT,
                price TEXT NOT NULL,
                commence_time_utc TEXT,
                is_opening INTEGER DEFAULT 0,
                game_id INTEGER
            );
            INSERT INTO games (event_id, game_date, commence_time_utc, away_team, home_team, game_type)
            VALUES ('evt-op-1001', '2025-04-01', '2025-04-01T23:05:00Z', 'BOS', 'NYY', 'R');
            """
        )
        connection.executemany(
            """
            INSERT INTO odds (
                event_id, game_date, commence_time, away_team, home_team, fetched_at,
                bookmaker, market_type, side, point, price, commence_time_utc, is_opening
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "evt-op-1001",
                    "2025-04-01",
                    "2025-04-01T23:05:00Z",
                    "BOS",
                    "NYY",
                    "2025-04-01T16:00:00Z",
                    "OddsPortal:provider_899",
                    "full_game_total",
                    "over",
                    "8.5",
                    "-102",
                    "2025-04-01T23:05:00Z",
                    1,
                ),
                (
                    "evt-op-1001",
                    "2025-04-01",
                    "2025-04-01T23:05:00Z",
                    "BOS",
                    "NYY",
                    "2025-04-01T16:00:00Z",
                    "OddsPortal:provider_899",
                    "full_game_total",
                    "under",
                    "8.5",
                    "-118",
                    "2025-04-01T23:05:00Z",
                    1,
                ),
            ],
        )
        connection.commit()
    return db_path


def test_load_historical_odds_matches_old_scraper_totals_by_frame(tmp_path: Path) -> None:
    db_path = _seed_old_scraper_db(tmp_path / "old_scraper.db")
    frame = pd.DataFrame(
        [
            {
                "game_pk": 1001,
                "game_date": "2025-04-01",
                "scheduled_start": "2025-04-01T23:05:00Z",
                "home_team": "NYY",
                "away_team": "BOS",
            }
        ]
    )

    snapshots = load_historical_odds_for_games(
        db_path=db_path,
        games_frame=frame,
        market_type="full_game_total",
        snapshot_selection="opening",
    )

    assert len(snapshots) == 1
    assert int(snapshots.iloc[0]["game_pk"]) == 1001
    assert float(snapshots.iloc[0]["total_point"]) == pytest.approx(8.5)
    assert int(snapshots.iloc[0]["over_odds"]) == -115
    assert int(snapshots.iloc[0]["under_odds"]) == -105
    assert str(snapshots.iloc[0]["book_name"]) == "consensus"
    assert str(snapshots.iloc[0]["source_schema"]) == "old_scraper"


def test_load_historical_odds_matches_old_scraper_moneyline_by_frame(tmp_path: Path) -> None:
    db_path = _seed_old_scraper_db(tmp_path / "old_scraper.db")
    frame = pd.DataFrame(
        [
            {
                "game_pk": 1001,
                "game_date": "2025-04-01",
                "scheduled_start": "2025-04-01T23:05:00Z",
                "home_team": "NYY",
                "away_team": "BOS",
            }
        ]
    )

    snapshots = load_historical_odds_for_games(
        db_path=db_path,
        games_frame=frame,
        market_type="f5_ml",
        book_name="DraftKings",
        snapshot_selection="opening",
    )

    assert len(snapshots) == 1
    assert int(snapshots.iloc[0]["home_odds"]) == -130
    assert int(snapshots.iloc[0]["away_odds"]) == 120
    assert str(snapshots.iloc[0]["source_schema"]) == "old_scraper"


def test_load_historical_odds_matches_old_scraper_away_team_total_by_frame(tmp_path: Path) -> None:
    db_path = _seed_old_scraper_db(tmp_path / "old_scraper.db")
    frame = pd.DataFrame(
        [
            {
                "game_pk": 1001,
                "game_date": "2025-04-01",
                "scheduled_start": "2025-04-01T23:05:00Z",
                "home_team": "NYY",
                "away_team": "BOS",
            }
        ]
    )

    snapshots = load_historical_odds_for_games(
        db_path=db_path,
        games_frame=frame,
        market_type="full_game_team_total_away",
        snapshot_selection="opening",
    )

    assert len(snapshots) == 1
    assert int(snapshots.iloc[0]["game_pk"]) == 1001
    assert float(snapshots.iloc[0]["total_point"]) == pytest.approx(4.5)
    assert int(snapshots.iloc[0]["over_odds"]) == -114
    assert int(snapshots.iloc[0]["under_odds"]) == -106
    assert str(snapshots.iloc[0]["book_name"]) == "consensus"
    assert str(snapshots.iloc[0]["source_schema"]) == "old_scraper"


def test_load_historical_odds_merges_legacy_and_oddsportal_old_scraper_sources(tmp_path: Path) -> None:
    legacy_db = _seed_old_scraper_db(tmp_path / "legacy_old_scraper.db")
    oddsportal_db = _seed_oddsportal_old_scraper_db(tmp_path / "oddsportal_old_scraper.db")
    frame = pd.DataFrame(
        [
            {
                "game_pk": 1001,
                "game_date": "2025-04-01",
                "scheduled_start": "2025-04-01T23:05:00Z",
                "home_team": "NYY",
                "away_team": "BOS",
            }
        ]
    )

    legacy_only = load_historical_odds_for_games(
        db_path=legacy_db,
        games_frame=frame,
        market_type="full_game_total",
        snapshot_selection="opening",
    )
    merged = load_historical_odds_for_games(
        db_path=f"{legacy_db};{oddsportal_db}",
        games_frame=frame,
        market_type="full_game_total",
        snapshot_selection="opening",
    )

    assert len(legacy_only) == 1
    assert len(merged) == 1
    assert str(merged.iloc[0]["book_name"]) == "consensus"
    assert str(merged.iloc[0]["source_schema"]) == "old_scraper"
    assert str(merged.iloc[0]["source_origin"]) == "oddsportal"
    assert str(merged.iloc[0]["source_db_path"]) == str(oddsportal_db)
    assert int(legacy_only.iloc[0]["over_odds"]) != int(merged.iloc[0]["over_odds"])


def test_load_historical_odds_excludes_post_start_old_scraper_latest_snapshot(tmp_path: Path) -> None:
    db_path = tmp_path / "old_scraper_cutoff.db"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE games (
                game_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT,
                game_date TEXT NOT NULL,
                commence_time_utc TEXT,
                away_team TEXT NOT NULL,
                home_team TEXT NOT NULL,
                game_type TEXT,
                away_pitcher TEXT,
                home_pitcher TEXT
            );
            CREATE TABLE odds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT,
                game_date TEXT NOT NULL,
                commence_time TEXT,
                away_team TEXT NOT NULL,
                home_team TEXT NOT NULL,
                game_type TEXT,
                away_pitcher TEXT,
                home_pitcher TEXT,
                fetched_at TEXT,
                bookmaker TEXT NOT NULL,
                market_type TEXT NOT NULL,
                side TEXT NOT NULL,
                point TEXT,
                price TEXT NOT NULL,
                commence_time_utc TEXT,
                is_opening INTEGER DEFAULT 0,
                game_id INTEGER
            );
            INSERT INTO games (event_id, game_date, commence_time_utc, away_team, home_team, game_type)
            VALUES ('evt-2001', '2025-04-01', '2025-04-01T23:05:00Z', 'BOS', 'NYY', 'R');
            """
        )
        connection.executemany(
            """
            INSERT INTO odds (
                event_id, game_date, commence_time, away_team, home_team, fetched_at,
                bookmaker, market_type, side, point, price, commence_time_utc, is_opening
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "evt-2001",
                    "2025-04-01",
                    "2025-04-01T23:05:00Z",
                    "BOS",
                    "NYY",
                    "2025-04-01T18:00:00Z",
                    "DraftKings",
                    "f5_ml",
                    "away",
                    "",
                    "120",
                    "2025-04-01T23:05:00Z",
                    1,
                ),
                (
                    "evt-2001",
                    "2025-04-01",
                    "2025-04-01T23:05:00Z",
                    "BOS",
                    "NYY",
                    "2025-04-01T18:00:00Z",
                    "DraftKings",
                    "f5_ml",
                    "home",
                    "",
                    "-130",
                    "2025-04-01T23:05:00Z",
                    1,
                ),
                (
                    "evt-2001",
                    "2025-04-01",
                    "2025-04-01T23:05:00Z",
                    "BOS",
                    "NYY",
                    "2025-04-01T23:10:00Z",
                    "DraftKings",
                    "f5_ml",
                    "away",
                    "",
                    "155",
                    "2025-04-01T23:05:00Z",
                    0,
                ),
                (
                    "evt-2001",
                    "2025-04-01",
                    "2025-04-01T23:05:00Z",
                    "BOS",
                    "NYY",
                    "2025-04-01T23:10:00Z",
                    "DraftKings",
                    "f5_ml",
                    "home",
                    "",
                    "-175",
                    "2025-04-01T23:05:00Z",
                    0,
                ),
            ],
        )
        connection.commit()

    snapshots = load_historical_odds_for_games(
        db_path=db_path,
        games_frame=pd.DataFrame(
            [
                {
                    "game_pk": 2001,
                    "game_date": "2025-04-01",
                    "scheduled_start": "2025-04-01T23:05:00Z",
                    "as_of_timestamp": "2025-04-01T23:00:00Z",
                    "home_team": "NYY",
                    "away_team": "BOS",
                }
            ]
        ),
        market_type="f5_ml",
        book_name="DraftKings",
        snapshot_selection="latest",
    )

    assert len(snapshots) == 1
    assert int(snapshots.iloc[0]["home_odds"]) == -130
    assert int(snapshots.iloc[0]["away_odds"]) == 120
    assert str(snapshots.iloc[0]["source_schema"]) == "old_scraper"


def test_load_historical_odds_uses_old_scraper_opener_rows_even_when_backfill_scrape_time_is_postgame(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "old_scraper_opener_backfill.db"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE games (
                game_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT,
                game_date TEXT NOT NULL,
                commence_time_utc TEXT,
                away_team TEXT NOT NULL,
                home_team TEXT NOT NULL,
                game_type TEXT,
                away_pitcher TEXT,
                home_pitcher TEXT
            );
            CREATE TABLE odds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT,
                game_date TEXT NOT NULL,
                commence_time TEXT,
                away_team TEXT NOT NULL,
                home_team TEXT NOT NULL,
                game_type TEXT,
                away_pitcher TEXT,
                home_pitcher TEXT,
                fetched_at TEXT,
                bookmaker TEXT NOT NULL,
                market_type TEXT NOT NULL,
                side TEXT NOT NULL,
                point TEXT,
                price TEXT NOT NULL,
                commence_time_utc TEXT,
                is_opening INTEGER DEFAULT 0,
                game_id INTEGER
            );
            INSERT INTO games (event_id, game_date, commence_time_utc, away_team, home_team, game_type)
            VALUES ('evt-3001', '2025-04-01', '2025-04-01T23:05:00Z', 'BOS', 'NYY', 'R');
            """
        )
        connection.executemany(
            """
            INSERT INTO odds (
                event_id, game_date, commence_time, away_team, home_team, fetched_at,
                bookmaker, market_type, side, point, price, commence_time_utc, is_opening
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "evt-3001",
                    "2025-04-01",
                    "2025-04-01T23:05:00Z",
                    "BOS",
                    "NYY",
                    "2026-03-28T21:22:02.208406Z",
                    "Opener",
                    "full_game_ml",
                    "away",
                    "",
                    "104",
                    "2025-04-01T23:05:00Z",
                    1,
                ),
                (
                    "evt-3001",
                    "2025-04-01",
                    "2025-04-01T23:05:00Z",
                    "BOS",
                    "NYY",
                    "2026-03-28T21:22:02.208406Z",
                    "Opener",
                    "full_game_ml",
                    "home",
                    "",
                    "-125",
                    "2025-04-01T23:05:00Z",
                    1,
                ),
                (
                    "evt-3001",
                    "2025-04-01",
                    "2025-04-01T23:05:00Z",
                    "BOS",
                    "NYY",
                    "2026-03-28T21:22:02.208406Z",
                    "DraftKings",
                    "full_game_ml",
                    "away",
                    "",
                    "120",
                    "2025-04-01T23:05:00Z",
                    0,
                ),
                (
                    "evt-3001",
                    "2025-04-01",
                    "2025-04-01T23:05:00Z",
                    "BOS",
                    "NYY",
                    "2026-03-28T21:22:02.208406Z",
                    "DraftKings",
                    "full_game_ml",
                    "home",
                    "",
                    "-130",
                    "2025-04-01T23:05:00Z",
                    0,
                ),
            ],
        )
        connection.commit()

    snapshots = load_historical_odds_for_games(
        db_path=db_path,
        games_frame=pd.DataFrame(
            [
                {
                    "game_pk": 3001,
                    "game_date": "2025-04-01",
                    "scheduled_start": "2025-04-01T23:05:00Z",
                    "as_of_timestamp": "2025-03-31T00:00:00Z",
                    "home_team": "NYY",
                    "away_team": "BOS",
                }
            ]
        ),
        market_type="full_game_ml",
        snapshot_selection="opening",
    )

    assert len(snapshots) == 1
    assert int(snapshots.iloc[0]["home_odds"]) == -125
    assert int(snapshots.iloc[0]["away_odds"]) == 104
    assert str(snapshots.iloc[0]["book_name"]) == "consensus"
    assert str(snapshots.iloc[0]["source_schema"]) == "old_scraper"
