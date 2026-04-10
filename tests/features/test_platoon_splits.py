from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.db import init_db
from src.features.offense import load_team_platoon_splits_lookup


def _seed_rows(db_path: Path, rows: list[tuple[str, int, str, float]]) -> None:
    database_path = init_db(db_path)
    with sqlite3.connect(database_path) as connection:
        connection.executemany(
            """
            INSERT INTO team_platoon_splits
                (team_abbr, season, vs_hand, woba, xwoba, k_pct, bb_pct, pa)
            VALUES (?, ?, ?, ?, NULL, NULL, NULL, 500)
            ON CONFLICT(team_abbr, season, vs_hand)
            DO UPDATE SET
                woba = excluded.woba,
                pa = excluded.pa
            """,
            rows,
        )
        connection.commit()


def test_load_team_platoon_splits_prefers_prior_season_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "platoon.db"
    _seed_rows(
        db_path,
        [
            ("NYY", 2024, "L", 0.341),
            ("NYY", 2024, "R", 0.355),
            ("NYY", 2025, "L", 0.390),
            ("NYY", 2025, "R", 0.401),
        ],
    )

    splits = load_team_platoon_splits_lookup(2025, db_path=db_path)

    assert splits["NYY"]["vs_LHP"] == pytest.approx(0.341)
    assert splits["NYY"]["vs_RHP"] == pytest.approx(0.355)


def test_load_team_platoon_splits_does_not_fall_back_to_current_season_by_default(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "platoon.db"
    _seed_rows(
        db_path,
        [
            ("NYY", 2018, "L", 0.332),
            ("NYY", 2018, "R", 0.347),
        ],
    )

    splits = load_team_platoon_splits_lookup(2018, db_path=db_path)

    assert splits == {}


def test_load_team_platoon_splits_can_opt_into_current_season_fallback(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "platoon.db"
    _seed_rows(
        db_path,
        [
            ("NYY", 2018, "L", 0.332),
            ("NYY", 2018, "R", 0.347),
        ],
    )

    splits = load_team_platoon_splits_lookup(
        2018,
        db_path=db_path,
        allow_current_season_fallback=True,
    )

    assert splits["NYY"]["vs_LHP"] == pytest.approx(0.332)
    assert splits["NYY"]["vs_RHP"] == pytest.approx(0.347)


def test_load_team_platoon_splits_uses_default_db_when_working_db_is_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback_db_path = tmp_path / "default_platoon.db"
    working_db_path = tmp_path / "working.db"
    _seed_rows(
        fallback_db_path,
        [
            ("BOS", 2024, "L", 0.301),
            ("BOS", 2024, "R", 0.329),
        ],
    )
    init_db(working_db_path)
    monkeypatch.setattr("src.features.offense.DEFAULT_DB_PATH", fallback_db_path)

    splits = load_team_platoon_splits_lookup(2025, db_path=working_db_path)

    assert splits["BOS"]["vs_LHP"] == pytest.approx(0.301)
    assert splits["BOS"]["vs_RHP"] == pytest.approx(0.329)
