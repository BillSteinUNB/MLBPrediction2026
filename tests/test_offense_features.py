from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.db import init_db


def _seed_game(
    db_path: Path,
    *,
    game_pk: int,
    game_date: str,
    home_team: str,
    away_team: str,
) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO games (game_pk, date, home_team, away_team, venue, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (game_pk, game_date, home_team, away_team, "Test Park", "scheduled"),
        )
        connection.commit()


def _team_logs_with_current_game() -> dict[tuple[int, str], pd.DataFrame]:
    return {
        (2025, "NYY"): pd.DataFrame(
            {
                "Date": ["2025-04-01", "2025-04-02", "2025-04-03"],
                "AB": [30, 32, 35],
                "H": [10, 8, 20],
                "2B": [2, 1, 5],
                "3B": [0, 0, 1],
                "HR": [1, 1, 4],
                "BB": [3, 2, 8],
                "SO": [6, 7, 3],
                "HBP": [1, 1, 1],
                "SF": [1, 1, 1],
            }
        ),
        (2025, "BOS"): pd.DataFrame(
            {
                "Date": ["2025-04-01", "2025-04-02", "2025-04-03"],
                "AB": [31, 30, 29],
                "H": [7, 9, 3],
                "2B": [1, 2, 0],
                "3B": [0, 0, 0],
                "HR": [1, 1, 0],
                "BB": [2, 3, 1],
                "SO": [9, 8, 12],
                "HBP": [0, 1, 0],
                "SF": [1, 0, 0],
            }
        ),
        (2024, "NYY"): pd.DataFrame(
            {
                "Date": ["2024-08-01"],
                "AB": [30],
                "H": [12],
                "2B": [3],
                "3B": [0],
                "HR": [5],
                "BB": [5],
                "SO": [6],
                "HBP": [1],
                "SF": [1],
            }
        ),
        (2024, "BOS"): pd.DataFrame(
            {
                "Date": ["2024-08-01"],
                "AB": [31],
                "H": [11],
                "2B": [2],
                "3B": [0],
                "HR": [2],
                "BB": [4],
                "SO": [7],
                "HBP": [0],
                "SF": [0],
            }
        ),
    }


def _only_prior_game_logs() -> dict[tuple[int, str], pd.DataFrame]:
    return {
        (2025, "NYY"): pd.DataFrame(
            {
                "Date": ["2025-04-01"],
                "AB": [30],
                "H": [8],
                "2B": [1],
                "3B": [0],
                "HR": [1],
                "BB": [2],
                "SO": [10],
                "HBP": [0],
                "SF": [1],
            }
        ),
        (2025, "BOS"): pd.DataFrame(
            {
                "Date": ["2025-04-01"],
                "AB": [31],
                "H": [7],
                "2B": [1],
                "3B": [0],
                "HR": [1],
                "BB": [3],
                "SO": [9],
                "HBP": [1],
                "SF": [0],
            }
        ),
        (2024, "NYY"): pd.DataFrame(
            {
                "Date": ["2024-08-01"],
                "AB": [30],
                "H": [12],
                "2B": [3],
                "3B": [0],
                "HR": [5],
                "BB": [5],
                "SO": [6],
                "HBP": [1],
                "SF": [1],
            }
        ),
        (2024, "BOS"): pd.DataFrame(
            {
                "Date": ["2024-08-01"],
                "AB": [31],
                "H": [11],
                "2B": [2],
                "3B": [0],
                "HR": [2],
                "BB": [4],
                "SO": [7],
                "HBP": [0],
                "SF": [0],
            }
        ),
    }


def _season_snapshot_batting_stats() -> dict[int, pd.DataFrame]:
    return {
        2025: pd.DataFrame(
            {
                "player_id": [101, 102, 201, 202],
                "PA": [100, 300, 250, 250],
                "wRC+": [120.0, 80.0, 105.0, 95.0],
                "wOBA": [0.36, 0.30, 0.33, 0.31],
                "ISO": [0.24, 0.16, 0.18, 0.15],
                "BABIP": [0.33, 0.29, 0.31, 0.30],
                "K%": [20.0, 26.0, 23.0, 24.0],
                "BB%": [12.0, 8.0, 9.0, 8.0],
            }
        ),
        2024: pd.DataFrame(
            {
                "player_id": [101, 102, 201, 202],
                "PA": [500, 500, 500, 500],
                "wRC+": [110.0, 90.0, 102.0, 98.0],
                "wOBA": [0.34, 0.32, 0.32, 0.31],
                "ISO": [0.20, 0.18, 0.17, 0.16],
                "BABIP": [0.32, 0.30, 0.30, 0.29],
                "K%": [21.0, 25.0, 22.0, 23.0],
                "BB%": [11.0, 9.0, 9.0, 8.0],
            }
        ),
    }


def _dated_batting_logs() -> dict[int, pd.DataFrame]:
    return {
        2025: pd.DataFrame(
            {
                "player_id": [101, 101, 101, 102, 102, 102, 201, 202],
                "game_date": [
                    "2025-04-01",
                    "2025-04-02",
                    "2025-04-03",
                    "2025-04-01",
                    "2025-04-02",
                    "2025-04-03",
                    "2025-04-02",
                    "2025-04-02",
                ],
                "PA": [4, 4, 10, 5, 4, 10, 4, 4],
                "wRC+": [140.0, 110.0, 0.0, 90.0, 95.0, 0.0, 102.0, 98.0],
                "wOBA": [0.41, 0.36, 0.0, 0.28, 0.30, 0.0, 0.33, 0.31],
                "ISO": [0.25, 0.21, 0.0, 0.16, 0.17, 0.0, 0.18, 0.15],
                "BABIP": [0.34, 0.31, 0.0, 0.29, 0.30, 0.0, 0.31, 0.30],
                "K%": [18.0, 19.0, 45.0, 24.0, 23.0, 40.0, 23.0, 24.0],
                "BB%": [13.0, 11.0, 0.0, 8.0, 9.0, 0.0, 9.0, 8.0],
            }
        )
    }


def _woba(row: dict[str, float]) -> float:
    singles = row["H"] - row["2B"] - row["3B"] - row["HR"]
    numerator = (
        row["BB"] * 0.69
        + row["HBP"] * 0.72
        + singles * 0.89
        + row["2B"] * 1.27
        + row["3B"] * 1.62
        + row["HR"] * 2.10
    )
    denominator = row["AB"] + row["BB"] + row["HBP"] + row["SF"]
    return numerator / denominator


def test_compute_offensive_features_excludes_current_game_and_sets_as_of_timestamp(
    tmp_path: Path,
) -> None:
    from src.features.offense import compute_offensive_features

    db_path = tmp_path / "offense.db"
    init_db(db_path)
    _seed_game(
        db_path,
        game_pk=555,
        game_date="2025-04-03T20:05:00+00:00",
        home_team="NYY",
        away_team="BOS",
    )

    team_logs = _team_logs_with_current_game()
    current_home = pd.Series(
        [_woba(row.to_dict()) for _, row in team_logs[(2025, "NYY")].iloc[:2].iterrows()]
    ).mean()
    prior_home = _woba(team_logs[(2024, "NYY")].iloc[0].to_dict())
    expected_home = (current_home * 2 + prior_home * 30) / 32
    leaked_home = pd.Series(
        [_woba(row.to_dict()) for _, row in team_logs[(2025, "NYY")].iterrows()]
    ).mean()
    leaked_home = (leaked_home * 3 + prior_home * 30) / 33

    def fake_team_logs_fetcher(season: int, team: str, refresh: bool = False) -> pd.DataFrame:
        _ = refresh
        return team_logs[(season, team)].copy()

    def fake_batting_fetcher(season: int, min_pa: int = 50, refresh: bool = False) -> pd.DataFrame:
        _ = min_pa
        _ = refresh
        return _season_snapshot_batting_stats()[season].copy()

    rows = compute_offensive_features(
        "2025-04-03",
        db_path=db_path,
        windows=(2,),
        regression_weight=30,
        team_logs_fetcher=fake_team_logs_fetcher,
        batting_stats_fetcher=fake_batting_fetcher,
    )

    by_name = {(row.game_pk, row.feature_name): row for row in rows}
    home_woba = by_name[(555, "home_team_woba_2g")].feature_value
    assert home_woba == pytest.approx(expected_home)
    assert home_woba < leaked_home

    with sqlite3.connect(db_path) as connection:
        as_of = connection.execute(
            "SELECT as_of_timestamp FROM features WHERE game_pk = ? AND feature_name = ?",
            (555, "home_team_woba_2g"),
        ).fetchone()[0]
        lineup_feature = connection.execute(
            "SELECT feature_value FROM features WHERE game_pk = ? AND feature_name = ?",
            (555, "home_lineup_woba_2g"),
        ).fetchone()

    assert as_of == "2025-04-02T00:00:00+00:00"
    assert lineup_feature is not None
    assert lineup_feature[0] == pytest.approx(home_woba)


def test_compute_offensive_features_applies_marcel_blending_early_season(tmp_path: Path) -> None:
    from src.features.offense import compute_offensive_features

    db_path = tmp_path / "offense.db"
    init_db(db_path)
    _seed_game(
        db_path,
        game_pk=556,
        game_date="2025-04-02T20:05:00+00:00",
        home_team="NYY",
        away_team="BOS",
    )

    team_logs = _only_prior_game_logs()
    current_home = _woba(team_logs[(2025, "NYY")].iloc[0].to_dict())
    prior_home = _woba(team_logs[(2024, "NYY")].iloc[0].to_dict())
    expected_home = (current_home * 1 + prior_home * 30) / 31

    def fake_team_logs_fetcher(season: int, team: str, refresh: bool = False) -> pd.DataFrame:
        _ = refresh
        return team_logs[(season, team)].copy()

    def fake_batting_fetcher(season: int, min_pa: int = 50, refresh: bool = False) -> pd.DataFrame:
        _ = min_pa
        _ = refresh
        return _season_snapshot_batting_stats()[season].copy()

    rows = compute_offensive_features(
        "2025-04-02",
        db_path=db_path,
        windows=(7,),
        regression_weight=30,
        team_logs_fetcher=fake_team_logs_fetcher,
        batting_stats_fetcher=fake_batting_fetcher,
    )

    home_team_row = next(row for row in rows if row.feature_name == "home_team_woba_7g")
    home_lineup_row = next(row for row in rows if row.feature_name == "home_lineup_woba_7g")

    assert current_home < home_team_row.feature_value < prior_home
    assert home_team_row.feature_value == pytest.approx(expected_home)
    assert home_lineup_row.feature_value == pytest.approx(home_team_row.feature_value)


def test_compute_offensive_features_falls_back_to_team_metrics_for_undated_batting_snapshots(
    tmp_path: Path,
) -> None:
    from src.features.offense import compute_offensive_features

    db_path = tmp_path / "offense.db"
    init_db(db_path)
    _seed_game(
        db_path,
        game_pk=557,
        game_date="2025-04-03T20:05:00+00:00",
        home_team="NYY",
        away_team="BOS",
    )

    team_logs = _team_logs_with_current_game()
    batting_stats = _season_snapshot_batting_stats()

    def fake_team_logs_fetcher(season: int, team: str, refresh: bool = False) -> pd.DataFrame:
        _ = refresh
        return team_logs[(season, team)].copy()

    def fake_batting_fetcher(season: int, min_pa: int = 50, refresh: bool = False) -> pd.DataFrame:
        _ = min_pa
        _ = refresh
        return batting_stats[season].copy()

    rows = compute_offensive_features(
        "2025-04-03",
        db_path=db_path,
        windows=(7,),
        regression_weight=30,
        lineup_player_ids={(557, "NYY"): [101, 102], (557, "BOS"): [201, 202]},
        team_logs_fetcher=fake_team_logs_fetcher,
        batting_stats_fetcher=fake_batting_fetcher,
    )

    by_name = {row.feature_name: row.feature_value for row in rows}
    assert by_name["home_lineup_woba_7g"] == pytest.approx(by_name["home_team_woba_7g"])
    assert by_name["home_lineup_wrc_plus_7g"] == pytest.approx(by_name["home_team_wrc_plus_7g"])


def test_compute_offensive_features_uses_as_of_dated_lineup_metrics_when_available(
    tmp_path: Path,
) -> None:
    from src.features.offense import compute_offensive_features

    db_path = tmp_path / "offense.db"
    init_db(db_path)
    _seed_game(
        db_path,
        game_pk=558,
        game_date="2025-04-03T20:05:00+00:00",
        home_team="NYY",
        away_team="BOS",
    )

    team_logs = _team_logs_with_current_game()
    batting_logs = _dated_batting_logs()
    prior_home = _woba(team_logs[(2024, "NYY")].iloc[0].to_dict())
    lineup_current_woba = (0.36 * 4 + 0.30 * 4) / 8
    leaked_lineup_woba = (0.41 * 4 + 0.36 * 4 + 0.0 * 10 + 0.28 * 5 + 0.30 * 4 + 0.0 * 10) / 37
    expected_lineup_woba = (lineup_current_woba + prior_home * 2) / 3
    leaked_lineup_feature = (leaked_lineup_woba + prior_home * 2) / 3

    def fake_team_logs_fetcher(season: int, team: str, refresh: bool = False) -> pd.DataFrame:
        _ = refresh
        return team_logs[(season, team)].copy()

    def fake_batting_fetcher(season: int, min_pa: int = 50, refresh: bool = False) -> pd.DataFrame:
        _ = min_pa
        _ = refresh
        return batting_logs[season].copy()

    rows = compute_offensive_features(
        "2025-04-03",
        db_path=db_path,
        windows=(1,),
        regression_weight=2,
        lineup_player_ids={(558, "NYY"): [101, 102], (558, "BOS"): [201, 202]},
        team_logs_fetcher=fake_team_logs_fetcher,
        batting_stats_fetcher=fake_batting_fetcher,
    )

    by_name = {row.feature_name: row.feature_value for row in rows}
    assert by_name["home_lineup_woba_1g"] != pytest.approx(by_name["home_team_woba_1g"])
    assert by_name["home_lineup_woba_1g"] == pytest.approx(expected_lineup_woba)
    assert by_name["home_lineup_woba_1g"] > leaked_lineup_feature
