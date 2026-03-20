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
    assert performance_columns["model_probability"][0].upper() == "REAL"
    assert performance_columns["model_probability"][1] == 1
    assert performance_columns["market_probability"][0].upper() == "REAL"
    assert performance_columns["market_probability"][1] == 1
    assert performance_columns["clv"][0].upper() == "REAL"
    assert performance_columns["placed_at"][0].upper() == "TEXT"
    assert performance_columns["placed_at"][1] == 1


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
    assert tables == (8,)
