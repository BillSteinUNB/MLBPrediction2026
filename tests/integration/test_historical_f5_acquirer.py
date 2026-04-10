from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.clients.historical_f5_acquirer import (
    normalize_sbr_f5_moneyline,
    normalize_sbr_f5_runline,
    seed_games_from_training_data,
)


def test_seed_games_from_training_data_populates_db(tmp_path: Path) -> None:
    training_path = tmp_path / "training.parquet"
    pd.DataFrame(
        [
            {
                "game_pk": 123,
                "season": 2024,
                "game_date": "2024-04-01",
                "scheduled_start": "2024-04-01T19:05:00Z",
                "home_team": "NYY",
                "away_team": "BOS",
                "venue": "Yankee Stadium",
                "status": "final",
                "f5_home_score": 3,
                "f5_away_score": 1,
                "final_home_score": 5,
                "final_away_score": 2,
                "abs_active": True,
            }
        ]
    ).to_parquet(training_path, index=False)

    db_path = tmp_path / "mlb.db"
    inserted = seed_games_from_training_data(db_path=db_path, training_path=training_path)
    assert inserted == 1

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT game_pk, home_team, away_team, status FROM games WHERE game_pk = 123"
        ).fetchone()

    assert row == (123, "NYY", "BOS", "final")


def test_normalize_sbr_f5_moneyline_matches_training_schedule(tmp_path: Path) -> None:
    training_path = tmp_path / "training.parquet"
    pd.DataFrame(
        [
            {
                "game_pk": 456,
                "season": 2024,
                "game_date": "2024-04-01",
                "scheduled_start": "2024-04-01T19:05:00Z",
                "home_team": "NYY",
                "away_team": "BOS",
                "venue": "Yankee Stadium",
                "status": "final",
                "f5_home_score": 0,
                "f5_away_score": 0,
                "final_home_score": 0,
                "final_away_score": 0,
                "abs_active": True,
            }
        ]
    ).to_parquet(training_path, index=False)

    raw_games = [
        {
            "gameView": {
                "gameId": 999,
                "startDate": "2024-04-01T19:05:00Z",
                "homeTeam": {"fullName": "New York Yankees"},
                "awayTeam": {"fullName": "Boston Red Sox"},
            },
            "oddsViews": [
                {
                    "sportsbook": "FanDuel",
                    "openingLine": {"homeOdds": -110, "awayOdds": 100},
                    "currentLine": {"homeOdds": -120, "awayOdds": 110},
                }
            ],
        }
    ]

    normalized, unmatched = normalize_sbr_f5_moneyline(raw_games, training_path=training_path)

    assert unmatched.empty
    assert len(normalized) == 2
    assert set(normalized["game_pk"].astype(int)) == {456}
    assert set(normalized["book_name"]) == {"sbr:fanduel"}
    assert set(normalized["market_type"]) == {"f5_ml"}
    assert normalized.loc[normalized["is_opening"], "home_odds"].iloc[0] == -110
    assert normalized.loc[~normalized["is_opening"], "home_odds"].iloc[0] == -120


def test_normalize_sbr_f5_moneyline_collapses_duplicate_team_tokens(tmp_path: Path) -> None:
    training_path = tmp_path / "training.parquet"
    pd.DataFrame(
        [
            {
                "game_pk": 789,
                "season": 2025,
                "game_date": "2025-04-02",
                "scheduled_start": "2025-04-02T19:35:00Z",
                "home_team": "OAK",
                "away_team": "CHC",
                "venue": "Sutter Health Park",
                "status": "final",
                "f5_home_score": 0,
                "f5_away_score": 0,
                "final_home_score": 0,
                "final_away_score": 0,
                "abs_active": True,
            }
        ]
    ).to_parquet(training_path, index=False)

    raw_games = [
        {
            "gameView": {
                "gameId": 111,
                "startDate": "2025-04-02T19:35:00Z",
                "homeTeam": {"fullName": "Athletics Athletics"},
                "awayTeam": {"fullName": "Chicago Cubs"},
            },
            "oddsViews": [
                {
                    "sportsbook": "Caesars",
                    "openingLine": {"homeOdds": 120, "awayOdds": -140},
                    "currentLine": {"homeOdds": 115, "awayOdds": -135},
                }
            ],
        }
    ]

    normalized, unmatched = normalize_sbr_f5_moneyline(raw_games, training_path=training_path)

    assert unmatched.empty
    assert len(normalized) == 2
    assert set(normalized["game_pk"].astype(int)) == {789}
    assert set(normalized["book_name"]) == {"sbr:caesars"}


def test_normalize_sbr_f5_moneyline_filters_absurd_odds(tmp_path: Path) -> None:
    training_path = tmp_path / "training.parquet"
    pd.DataFrame(
        [
            {
                "game_pk": 790,
                "season": 2025,
                "game_date": "2025-04-03",
                "scheduled_start": "2025-04-03T19:35:00Z",
                "home_team": "NYY",
                "away_team": "BOS",
                "venue": "Yankee Stadium",
                "status": "final",
                "f5_home_score": 0,
                "f5_away_score": 0,
                "final_home_score": 0,
                "final_away_score": 0,
                "abs_active": True,
            }
        ]
    ).to_parquet(training_path, index=False)

    raw_games = [
        {
            "gameView": {
                "gameId": 222,
                "startDate": "2025-04-03T19:35:00Z",
                "homeTeam": {"fullName": "New York Yankees"},
                "awayTeam": {"fullName": "Boston Red Sox"},
            },
            "oddsViews": [
                {
                    "sportsbook": "Caesars",
                    "openingLine": {"homeOdds": -110, "awayOdds": 100},
                    "currentLine": {"homeOdds": -10000, "awayOdds": 2800},
                }
            ],
        }
    ]

    normalized, unmatched = normalize_sbr_f5_moneyline(raw_games, training_path=training_path)

    assert unmatched.empty
    assert len(normalized) == 1
    assert normalized["is_opening"].tolist() == [True]


def test_normalize_sbr_f5_runline_carries_points(tmp_path: Path) -> None:
    training_path = tmp_path / "training.parquet"
    pd.DataFrame(
        [
            {
                "game_pk": 901,
                "season": 2024,
                "game_date": "2024-04-01",
                "scheduled_start": "2024-04-01T19:05:00Z",
                "home_team": "NYY",
                "away_team": "BOS",
                "venue": "Yankee Stadium",
                "status": "final",
                "f5_home_score": 0,
                "f5_away_score": 0,
                "final_home_score": 0,
                "final_away_score": 0,
                "abs_active": True,
            }
        ]
    ).to_parquet(training_path, index=False)

    raw_games = [
        {
            "gameView": {
                "gameId": 1234,
                "startDate": "2024-04-01T19:05:00Z",
                "homeTeam": {"fullName": "New York Yankees"},
                "awayTeam": {"fullName": "Boston Red Sox"},
            },
            "oddsViews": [
                {
                    "sportsbook": "Bovada",
                    "openingLine": {
                        "homeOdds": 105,
                        "awayOdds": -125,
                        "homeSpread": 0.5,
                        "awaySpread": -0.5,
                    },
                    "currentLine": {
                        "homeOdds": 115,
                        "awayOdds": -135,
                        "homeSpread": 1.5,
                        "awaySpread": -1.5,
                    },
                }
            ],
        }
    ]

    normalized, unmatched = normalize_sbr_f5_runline(raw_games, training_path=training_path)

    assert unmatched.empty
    assert len(normalized) == 2
    assert set(normalized["market_type"]) == {"f5_rl"}
    assert set(normalized["book_name"]) == {"sbr:bovada"}
    assert normalized.loc[normalized["is_opening"], "home_point"].iloc[0] == pytest.approx(0.5)
    assert normalized.loc[~normalized["is_opening"], "away_point"].iloc[0] == pytest.approx(-1.5)
