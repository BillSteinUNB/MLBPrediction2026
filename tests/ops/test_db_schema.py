from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.db import SCHEMA_VERSION, init_db


def _table_columns(db_path: Path, table_name: str) -> dict[str, tuple[str, int, str | None, int]]:
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()

    return {
        row[1]: (row[2], row[3], row[4], row[5])
        for row in rows
    }


def test_init_db_creates_all_required_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "schema.db"

    init_db(db_path)

    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

    assert {
        "schema_version",
        "games",
        "features",
        "pitcher_siera_cache",
        "predictions",
        "odds_snapshots",
        "bets",
        "bet_performance",
        "bankroll_ledger",
    }.issubset(tables)


def test_games_uses_game_pk_as_primary_key(tmp_path: Path) -> None:
    db_path = tmp_path / "schema.db"

    init_db(db_path)

    game_pk_type, _, _, game_pk_primary_key = _table_columns(db_path, "games")["game_pk"]

    assert game_pk_type.upper() == "INTEGER"
    assert game_pk_primary_key == 1


def test_features_table_requires_as_of_timestamp(tmp_path: Path) -> None:
    db_path = tmp_path / "schema.db"

    init_db(db_path)

    feature_columns = _table_columns(db_path, "features")

    as_of_type, as_of_not_null, _, _ = feature_columns["as_of_timestamp"]
    assert as_of_type.upper() == "TEXT"
    assert as_of_not_null == 1

    with sqlite3.connect(db_path) as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO features (game_pk, feature_name, feature_value, window_size)
                VALUES (?, ?, ?, ?)
                """,
                (1, "team_wrc_plus", 114.2, 7),
            )


def test_odds_snapshots_has_is_frozen_column(tmp_path: Path) -> None:
    db_path = tmp_path / "schema.db"

    init_db(db_path)

    is_frozen_type, is_frozen_not_null, is_frozen_default, _ = _table_columns(
        db_path,
        "odds_snapshots",
    )["is_frozen"]

    assert is_frozen_type.upper() == "INTEGER"
    assert is_frozen_not_null == 1
    assert is_frozen_default == "0"


def test_bet_performance_tracks_market_probability_and_clv_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "schema.db"

    init_db(db_path)

    performance_columns = _table_columns(db_path, "bet_performance")

    assert performance_columns["bet_id"][0].upper() == "INTEGER"
    assert performance_columns["bet_id"][1] == 1
    assert performance_columns["book_name"][0].upper() == "TEXT"
    assert performance_columns["book_name"][1] == 1
    assert performance_columns["model_probability"][0].upper() == "REAL"
    assert performance_columns["model_probability"][1] == 1
    assert performance_columns["market_probability"][0].upper() == "REAL"
    assert performance_columns["market_probability"][1] == 1
    assert performance_columns["clv"][0].upper() == "REAL"
    assert performance_columns["placed_at"][0].upper() == "TEXT"
    assert performance_columns["placed_at"][1] == 1


def test_bets_table_tracks_persisted_decision_metadata(tmp_path: Path) -> None:
    db_path = tmp_path / "schema.db"

    init_db(db_path)

    bet_columns = _table_columns(db_path, "bets")

    assert bet_columns["book_name"][0].upper() == "TEXT"
    assert bet_columns["source_model"][0].upper() == "TEXT"
    assert bet_columns["source_model_version"][0].upper() == "TEXT"
    assert bet_columns["model_probability"][0].upper() == "REAL"
    assert bet_columns["fair_probability"][0].upper() == "REAL"
    assert bet_columns["ev"][0].upper() == "REAL"
    assert bet_columns["line_at_bet"][0].upper() == "REAL"


def test_init_db_allows_total_market_bets_with_over_under_sides(tmp_path: Path) -> None:
    db_path = tmp_path / "schema.db"

    init_db(db_path)

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO games (game_pk, date, home_team, away_team, venue, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (9999, "2026-04-01T20:05:00+00:00", "NYY", "BOS", "Yankee Stadium", "scheduled"),
        )
        connection.execute(
            """
            INSERT INTO bets (
                game_pk,
                market_type,
                side,
                edge_pct,
                kelly_stake,
                odds_at_bet,
                result
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (9999, "full_game_total", "over", 0.04, 10.0, -110, "PENDING"),
        )
        connection.execute(
            """
            INSERT INTO bet_performance (
                bet_id,
                game_pk,
                market_type,
                side,
                book_name,
                model_probability,
                market_probability,
                edge_pct,
                odds_at_bet,
                stake,
                result,
                placed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                9999,
                "full_game_total",
                "over",
                "TestBook",
                0.56,
                0.52,
                0.04,
                -110,
                10.0,
                "PENDING",
                "2026-04-01T19:30:00+00:00",
            ),
        )


def test_init_db_tracks_schema_version(tmp_path: Path) -> None:
    db_path = tmp_path / "schema.db"

    init_db(db_path)

    with sqlite3.connect(db_path) as connection:
        version_row = connection.execute(
            "SELECT version FROM schema_version WHERE singleton_id = 1"
        ).fetchone()

    assert version_row == (SCHEMA_VERSION,)


def test_init_db_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "schema.db"

    init_db(db_path)
    init_db(db_path)

    with sqlite3.connect(db_path) as connection:
        version_rows = connection.execute("SELECT COUNT(*) FROM schema_version").fetchone()
        tables = connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchone()

    assert version_rows == (1,)
    assert tables == (11,)


def test_pitcher_siera_cache_table_includes_seasonal_siera_fields(tmp_path: Path) -> None:
    db_path = tmp_path / "schema.db"

    init_db(db_path)

    siera_columns = _table_columns(db_path, "pitcher_siera_cache")

    assert siera_columns["season"][0].upper() == "INTEGER"
    assert siera_columns["pitcher_name"][0].upper() == "TEXT"
    assert siera_columns["pitcher_name"][1] == 1
    assert siera_columns["siera"][0].upper() == "REAL"
    assert siera_columns["siera"][1] == 1
    assert siera_columns["fetched_at"][0].upper() == "TEXT"
    assert siera_columns["fetched_at"][1] == 1
