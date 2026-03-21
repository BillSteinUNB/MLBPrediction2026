from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from src.db import init_db
from src.features.umpires import compute_umpire_features


def _seed_game(
    db_path: Path,
    *,
    game_pk: int,
    game_date: str,
    home_team: str = "NYY",
    away_team: str = "BOS",
    final_home_score: int | None = None,
    final_away_score: int | None = None,
    f5_home_score: int | None = None,
    f5_away_score: int | None = None,
    status: str = "final",
) -> None:
    with sqlite3.connect(init_db(db_path)) as connection:
        connection.execute(
            """
            INSERT INTO games (
                game_pk, date, home_team, away_team, venue, is_dome, is_abs_active,
                f5_home_score, f5_away_score, final_home_score, final_away_score, status
            )
            VALUES (?, ?, ?, ?, ?, 0, 1, ?, ?, ?, ?, ?)
            """,
            (
                game_pk,
                f"{game_date}T19:05:00+00:00",
                home_team,
                away_team,
                "Yankee Stadium",
                f5_home_score,
                f5_away_score,
                final_home_score,
                final_away_score,
                status,
            ),
        )
        connection.commit()


def test_compute_umpire_features_uses_prior_plate_umpire_history(tmp_path: Path) -> None:
    db_path = tmp_path / "umpires.db"
    _seed_game(
        db_path,
        game_pk=1001,
        game_date="2024-04-01",
        final_home_score=5,
        final_away_score=3,
        f5_home_score=2,
        f5_away_score=1,
    )
    _seed_game(
        db_path,
        game_pk=1002,
        game_date="2024-04-02",
        final_home_score=4,
        final_away_score=2,
        f5_home_score=1,
        f5_away_score=0,
        status="scheduled",
    )

    def _fake_umpire_fetcher(*, season: int, refresh: bool = False) -> pd.DataFrame:
        _ = season
        _ = refresh
        return pd.DataFrame(
            [
                {"date": "2024-04-01", "hometeam": "NYA", "visteam": "BOS", "umphome": "westj902"},
                {"date": "2024-04-02", "hometeam": "NYA", "visteam": "BOS", "umphome": "westj902"},
            ]
        )

    features = compute_umpire_features(
        "2024-04-02",
        db_path=db_path,
        umpire_fetcher=_fake_umpire_fetcher,
    )

    feature_map = {feature.feature_name: feature.feature_value for feature in features if feature.game_pk == 1002}

    assert feature_map["plate_umpire_known"] == 1.0
    assert feature_map["plate_umpire_total_runs_avg_30g"] == 8.0
    assert feature_map["plate_umpire_f5_total_runs_avg_30g"] == 3.0
    assert feature_map["plate_umpire_home_win_pct_30g"] == 1.0
    assert feature_map["plate_umpire_sample_size_30g"] == 1.0
