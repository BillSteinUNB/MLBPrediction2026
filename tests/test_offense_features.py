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


def _season_statcast_xwoba() -> dict[int, pd.DataFrame]:
    return {
        2025: pd.DataFrame(
            {
                "game_pk": [9001, 9001, 9002, 9002, 9001, 9001, 9002, 9002],
                "game_date": [
                    "2025-04-01",
                    "2025-04-01",
                    "2025-04-02",
                    "2025-04-02",
                    "2025-04-01",
                    "2025-04-01",
                    "2025-04-02",
                    "2025-04-02",
                ],
                "team": ["NYY", "NYY", "NYY", "NYY", "BOS", "BOS", "BOS", "BOS"],
                "player_id": [101, 102, 101, 102, 201, 202, 201, 202],
                "pa": [4, 5, 4, 4, 4, 4, 4, 4],
                "bbe": [2, 1, 2, 2, 1, 2, 2, 1],
                "xwoba": [0.39, 0.27, 0.34, 0.32, 0.31, 0.30, 0.28, 0.26],
                "barrel_pct": [50.0, 0.0, 50.0, 0.0, 0.0, 50.0, 0.0, 0.0],
            }
        ),
        2024: pd.DataFrame(
            {
                "game_pk": [8001, 8001, 8001, 8001],
                "game_date": ["2024-08-01", "2024-08-01", "2024-08-01", "2024-08-01"],
                "team": ["NYY", "NYY", "BOS", "BOS"],
                "player_id": [101, 102, 201, 202],
                "pa": [5, 4, 4, 4],
                "bbe": [3, 1, 2, 2],
                "xwoba": [0.33, 0.29, 0.31, 0.30],
                "barrel_pct": [33.3333333333, 0.0, 0.0, 50.0],
            }
        ),
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


def test_compute_offensive_features_blends_team_xwoba_and_gap_from_statcast(
    tmp_path: Path,
) -> None:
    from src.features.offense import compute_offensive_features

    db_path = tmp_path / "offense_xwoba.db"
    init_db(db_path)
    _seed_game(
        db_path,
        game_pk=561,
        game_date="2025-04-03T20:05:00+00:00",
        home_team="NYY",
        away_team="BOS",
    )

    team_logs = _team_logs_with_current_game()
    statcast_metrics = _season_statcast_xwoba()
    current_home_woba = pd.Series(
        [_woba(row.to_dict()) for _, row in team_logs[(2025, "NYY")].iloc[:2].iterrows()]
    ).mean()
    current_home_xwoba = (((0.39 * 4) + (0.27 * 5)) / 9 + ((0.34 * 4) + (0.32 * 4)) / 8) / 2
    current_home_barrel_pct = (((1 / 3) * 100.0) + ((1 / 4) * 100.0)) / 2
    prior_home_woba = _woba(team_logs[(2024, "NYY")].iloc[0].to_dict())
    prior_home_xwoba = (0.33 * 5 + 0.29 * 4) / 9
    prior_home_barrel_pct = (1 / 4) * 100.0
    expected_home_xwoba = (current_home_xwoba * 2 + prior_home_xwoba * 30) / 32
    expected_home_barrel_pct = (current_home_barrel_pct * 2 + prior_home_barrel_pct * 30) / 32
    expected_home_gap = (
        ((current_home_woba - current_home_xwoba) * 2)
        + ((prior_home_woba - prior_home_xwoba) * 30)
    ) / 32

    def fake_team_logs_fetcher(season: int, team: str, refresh: bool = False) -> pd.DataFrame:
        _ = refresh
        return team_logs[(season, team)].copy()

    def fake_batting_fetcher(season: int, min_pa: int = 50, refresh: bool = False) -> pd.DataFrame:
        _ = min_pa
        _ = refresh
        return _season_snapshot_batting_stats()[season].copy()

    def fake_offense_statcast_fetcher(
        season: int,
        *,
        db_path: Path,
        end_date,
        refresh: bool = False,
    ) -> pd.DataFrame:
        _ = db_path
        _ = end_date
        _ = refresh
        return statcast_metrics[season].copy()

    rows = compute_offensive_features(
        "2025-04-03",
        db_path=db_path,
        windows=(2,),
        regression_weight=30,
        team_logs_fetcher=fake_team_logs_fetcher,
        batting_stats_fetcher=fake_batting_fetcher,
        offense_statcast_fetcher=fake_offense_statcast_fetcher,
    )

    by_name = {row.feature_name: row.feature_value for row in rows}
    assert by_name["home_team_xwoba_2g"] == pytest.approx(expected_home_xwoba)
    assert by_name["home_team_woba_minus_xwoba_2g"] == pytest.approx(expected_home_gap)
    assert by_name["home_team_barrel_pct_2g"] == pytest.approx(expected_home_barrel_pct)
    assert by_name["home_lineup_xwoba_2g"] == pytest.approx(by_name["home_team_xwoba_2g"])
    assert by_name["home_lineup_barrel_pct_2g"] == pytest.approx(by_name["home_team_barrel_pct_2g"])


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
    statcast_metrics = _season_statcast_xwoba()
    prior_home = _woba(team_logs[(2024, "NYY")].iloc[0].to_dict())
    lineup_current_woba = (0.36 * 4 + 0.30 * 4) / 8
    lineup_current_xwoba = (0.34 * 4 + 0.32 * 4) / 8
    lineup_prior_xwoba = (0.33 + 0.29) / 2
    team_prior_xwoba = (0.33 * 5 + 0.29 * 4) / 9
    leaked_lineup_woba = (0.41 * 4 + 0.36 * 4 + 0.0 * 10 + 0.28 * 5 + 0.30 * 4 + 0.0 * 10) / 37
    expected_lineup_woba = (lineup_current_woba * 2 + prior_home * 2) / 4
    expected_lineup_xwoba = (lineup_current_xwoba * 2 + lineup_prior_xwoba * 2) / 4
    expected_lineup_gap = ((lineup_current_woba - lineup_current_xwoba) * 2 + (prior_home - team_prior_xwoba) * 2) / 4
    leaked_lineup_feature = (leaked_lineup_woba * 2 + prior_home * 2) / 4

    def fake_team_logs_fetcher(season: int, team: str, refresh: bool = False) -> pd.DataFrame:
        _ = refresh
        return team_logs[(season, team)].copy()

    def fake_batting_fetcher(season: int, min_pa: int = 50, refresh: bool = False) -> pd.DataFrame:
        _ = min_pa
        _ = refresh
        return batting_logs[season].copy()

    def fake_offense_statcast_fetcher(
        season: int,
        *,
        db_path: Path,
        end_date,
        refresh: bool = False,
    ) -> pd.DataFrame:
        _ = db_path
        _ = end_date
        _ = refresh
        return statcast_metrics[season].copy()

    rows = compute_offensive_features(
        "2025-04-03",
        db_path=db_path,
        windows=(1,),
        regression_weight=2,
        lineup_player_ids={(558, "NYY"): [101, 102], (558, "BOS"): [201, 202]},
        team_logs_fetcher=fake_team_logs_fetcher,
        batting_stats_fetcher=fake_batting_fetcher,
        offense_statcast_fetcher=fake_offense_statcast_fetcher,
    )

    by_name = {row.feature_name: row.feature_value for row in rows}
    assert by_name["home_lineup_woba_1g"] != pytest.approx(by_name["home_team_woba_1g"])
    assert by_name["home_lineup_woba_1g"] == pytest.approx(expected_lineup_woba)
    assert by_name["home_lineup_woba_1g"] > leaked_lineup_feature
    assert by_name["home_lineup_xwoba_1g"] == pytest.approx(expected_lineup_xwoba)
    assert by_name["home_lineup_woba_minus_xwoba_1g"] == pytest.approx(expected_lineup_gap)


def test_compute_offensive_features_falls_back_to_actual_woba_when_team_xwoba_is_missing(
    tmp_path: Path,
) -> None:
    from src.features.offense import compute_offensive_features

    db_path = tmp_path / "offense_xwoba_fallback.db"
    init_db(db_path)
    _seed_game(
        db_path,
        game_pk=562,
        game_date="2025-04-03T20:05:00+00:00",
        home_team="NYY",
        away_team="BOS",
    )

    team_logs = _team_logs_with_current_game()

    def fake_team_logs_fetcher(season: int, team: str, refresh: bool = False) -> pd.DataFrame:
        _ = refresh
        return team_logs[(season, team)].copy()

    def fake_batting_fetcher(season: int, min_pa: int = 50, refresh: bool = False) -> pd.DataFrame:
        _ = min_pa
        _ = refresh
        return _season_snapshot_batting_stats()[season].copy()

    def fake_offense_statcast_fetcher(
        season: int,
        *,
        db_path: Path,
        end_date,
        refresh: bool = False,
    ) -> pd.DataFrame:
        _ = season
        _ = db_path
        _ = end_date
        _ = refresh
        return pd.DataFrame(
            columns=["game_pk", "game_date", "team", "player_id", "pa", "xwoba"]
        )

    rows = compute_offensive_features(
        "2025-04-03",
        db_path=db_path,
        windows=(2,),
        regression_weight=30,
        team_logs_fetcher=fake_team_logs_fetcher,
        batting_stats_fetcher=fake_batting_fetcher,
        offense_statcast_fetcher=fake_offense_statcast_fetcher,
    )

    by_name = {row.feature_name: row.feature_value for row in rows}
    assert by_name["home_team_xwoba_2g"] == pytest.approx(by_name["home_team_woba_2g"])
    assert by_name["home_team_woba_minus_xwoba_2g"] == pytest.approx(0.0)
    assert by_name["home_team_barrel_pct_2g"] == pytest.approx(7.0)


def test_compute_game_level_metrics_prefers_game_pk_alignment_for_team_xwoba() -> None:
    from src.features.offense import _compute_game_level_metrics

    raw_logs = pd.DataFrame(
        {
            "game_pk": [9002, 9001],
            "Date": ["2025-04-02", "2025-04-01"],
            "AB": [32, 30],
            "H": [8, 10],
            "2B": [1, 2],
            "3B": [0, 0],
            "HR": [1, 1],
            "BB": [2, 3],
            "SO": [7, 6],
            "HBP": [1, 1],
            "SF": [1, 1],
        }
    )
    statcast_metrics = _season_statcast_xwoba()[2025].loc[
        lambda df: df["team"].eq("NYY")
    ].copy()

    result = _compute_game_level_metrics(raw_logs, statcast_metrics=statcast_metrics)

    assert result["game_pk"].tolist() == [9001, 9002]
    assert result["xwoba"].iloc[0] == pytest.approx(((0.39 * 4) + (0.27 * 5)) / 9)
    assert result["xwoba"].iloc[1] == pytest.approx(((0.34 * 4) + (0.32 * 4)) / 8)
    assert result["woba_minus_xwoba"].abs().gt(0).all()


def test_normalize_statcast_offense_metrics_strips_timezone_from_game_dates() -> None:
    from src.features.offense import _normalize_statcast_offense_metrics

    normalized = _normalize_statcast_offense_metrics(
        pd.DataFrame(
            [
                {
                    "game_pk": 9001,
                    "game_date": "2025-04-01T00:00:00+00:00",
                    "team": "NYY",
                    "player_id": 101,
                    "pa": 4,
                    "xwoba": 0.350,
                    "bbe": 2,
                    "barrel_pct": 12.5,
                }
            ]
        )
    )

    assert normalized["game_date"].dt.tz is None
    assert normalized["game_date"].dt.strftime("%Y-%m-%d").tolist() == ["2025-04-01"]


def test_build_statcast_offense_metrics_derives_batting_team_and_barrel_pct() -> None:
    from src.features.offense import _build_statcast_offense_metrics

    season_games = pd.DataFrame(
        {
            "game_pk": [7777],
            "game_date": ["2025-04-01"],
        }
    )
    statcast_frame = pd.DataFrame(
        {
            "game_pk": [7777, 7777, 7777, 7777],
            "at_bat_number": [1, 1, 2, 3],
            "pitch_number": [1, 2, 1, 1],
            "inning_topbot": ["Top", "Top", "Top", "Bot"],
            "away_team": ["BOS", "BOS", "BOS", "BOS"],
            "home_team": ["NYY", "NYY", "NYY", "NYY"],
            "batter": [11, 11, 12, 21],
            "events": [None, "single", "strikeout", "double"],
            "bb_type": [None, "line_drive", None, "fly_ball"],
            "launch_speed": [None, 100.0, None, 91.0],
            "launch_angle": [None, 27.0, None, 25.0],
            "estimated_woba_using_speedangle": [None, 0.700, None, 0.500],
        }
    )

    result = _build_statcast_offense_metrics(season_games, statcast_frame)
    assert result["team"].tolist() == ["BOS", "BOS", "NYY"]
    assert result["player_id"].tolist() == [11, 12, 21]
    assert result["pa"].tolist() == [1.0, 1.0, 1.0]
    assert result["bbe"].tolist() == [1.0, 0.0, 1.0]
    assert result["barrel_pct"].iloc[0] == pytest.approx(100.0)
    assert pd.isna(result["barrel_pct"].iloc[1])
    assert result["barrel_pct"].iloc[2] == pytest.approx(0.0)


def test_compute_offensive_features_uses_league_average_prior_for_first_year_lineup_players(
    tmp_path: Path,
) -> None:
    from src.features.offense import compute_offensive_features

    db_path = tmp_path / "offense.db"
    init_db(db_path)
    _seed_game(
        db_path,
        game_pk=559,
        game_date="2025-04-02T20:05:00+00:00",
        home_team="NYY",
        away_team="BOS",
    )

    team_logs = {
        (2025, "NYY"): pd.DataFrame(
            {
                "Date": ["2025-04-01"],
                "AB": [30],
                "H": [8],
                "2B": [1],
                "3B": [0],
                "HR": [1],
                "BB": [3],
                "SO": [8],
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
                "BB": [2],
                "SO": [9],
                "HBP": [0],
                "SF": [1],
            }
        ),
        (2024, "NYY"): pd.DataFrame(
            {
                "Date": ["2024-08-01"],
                "AB": [30],
                "H": [12],
                "2B": [3],
                "3B": [0],
                "HR": [4],
                "BB": [4],
                "SO": [6],
                "HBP": [0],
                "SF": [1],
            }
        ),
        (2024, "BOS"): pd.DataFrame(
            {
                "Date": ["2024-08-01"],
                "AB": [31],
                "H": [10],
                "2B": [2],
                "3B": [0],
                "HR": [2],
                "BB": [3],
                "SO": [7],
                "HBP": [0],
                "SF": [1],
            }
        ),
    }
    batting_by_season = {
        2025: pd.DataFrame(
            {
                "player_id": [101, 103],
                "game_date": ["2025-04-01", "2025-04-01"],
                "PA": [4, 4],
                "wRC+": [110.0, 130.0],
                "wOBA": [0.36, 0.42],
                "ISO": [0.18, 0.24],
                "BABIP": [0.31, 0.34],
                "K%": [19.0, 21.0],
                "BB%": [11.0, 10.0],
            }
        ),
        2024: pd.DataFrame(
            {
                "player_id": [101],
                "PA": [500],
                "wRC+": [108.0],
                "wOBA": [0.34],
                "ISO": [0.19],
                "BABIP": [0.30],
                "K%": [20.0],
                "BB%": [10.0],
            }
        ),
    }
    lineup_current_woba = (0.36 * 4 + 0.42 * 4) / 8
    rookie_prior_woba = (0.34 + 0.320) / 2
    expected_lineup_woba = (lineup_current_woba * 1 + rookie_prior_woba * 2) / 3
    team_prior_woba = _woba(team_logs[(2024, "NYY")].iloc[0].to_dict())
    team_prior_blend = (lineup_current_woba * 1 + team_prior_woba * 2) / 3

    def fake_team_logs_fetcher(season: int, team: str, refresh: bool = False) -> pd.DataFrame:
        _ = refresh
        return team_logs[(season, team)].copy()

    def fake_batting_fetcher(season: int, min_pa: int = 50, refresh: bool = False) -> pd.DataFrame:
        _ = min_pa
        _ = refresh
        return batting_by_season[season].copy()

    rows = compute_offensive_features(
        "2025-04-02",
        db_path=db_path,
        windows=(1,),
        regression_weight=2,
        lineup_player_ids={(559, "NYY"): [101, 103]},
        team_logs_fetcher=fake_team_logs_fetcher,
        batting_stats_fetcher=fake_batting_fetcher,
    )

    by_name = {row.feature_name: row.feature_value for row in rows}
    assert by_name["home_lineup_woba_1g"] == pytest.approx(expected_lineup_woba)
    assert by_name["home_lineup_woba_1g"] != pytest.approx(team_prior_blend)


def test_compute_offensive_features_supports_current_team_log_schema(
    tmp_path: Path,
) -> None:
    from src.features.offense import compute_offensive_features

    db_path = tmp_path / "offense_current_schema.db"
    init_db(db_path)
    _seed_game(
        db_path,
        game_pk=560,
        game_date="2025-04-03T20:05:00+00:00",
        home_team="NYY",
        away_team="BOS",
    )

    team_logs = {
        (2025, "NYY"): pd.DataFrame(
            {
                "Date": ["2025-04-01", "2025-04-02"],
                "Batting Stats_AB": [30, 32],
                "Batting Stats_H": [10, 8],
                "Batting Stats_2B": [2, 1],
                "Batting Stats_3B": [0, 0],
                "Batting Stats_HR": [1, 1],
                "Batting Stats_BB": [3, 2],
                "Batting Stats_SO": [6, 7],
                "Batting Stats_HBP": [1, 1],
                "Batting Stats_SF": [1, 1],
                "Batting Stats_SH": [0, 0],
            }
        ),
        (2025, "BOS"): pd.DataFrame(
            {
                "Date": ["2025-04-01", "2025-04-02"],
                "Batting Stats_AB": [31, 30],
                "Batting Stats_H": [7, 9],
                "Batting Stats_2B": [1, 2],
                "Batting Stats_3B": [0, 0],
                "Batting Stats_HR": [1, 1],
                "Batting Stats_BB": [2, 3],
                "Batting Stats_SO": [9, 8],
                "Batting Stats_HBP": [0, 1],
                "Batting Stats_SF": [1, 0],
                "Batting Stats_SH": [0, 0],
            }
        ),
        (2024, "NYY"): pd.DataFrame(
            {
                "Date": ["2024-08-01"],
                "Batting Stats_AB": [30],
                "Batting Stats_H": [12],
                "Batting Stats_2B": [3],
                "Batting Stats_3B": [0],
                "Batting Stats_HR": [5],
                "Batting Stats_BB": [5],
                "Batting Stats_SO": [6],
                "Batting Stats_HBP": [1],
                "Batting Stats_SF": [1],
                "Batting Stats_SH": [0],
            }
        ),
        (2024, "BOS"): pd.DataFrame(
            {
                "Date": ["2024-08-01"],
                "Batting Stats_AB": [31],
                "Batting Stats_H": [11],
                "Batting Stats_2B": [2],
                "Batting Stats_3B": [0],
                "Batting Stats_HR": [2],
                "Batting Stats_BB": [4],
                "Batting Stats_SO": [7],
                "Batting Stats_HBP": [0],
                "Batting Stats_SF": [0],
                "Batting Stats_SH": [0],
            }
        ),
    }

    def fake_team_logs_fetcher(season: int, team: str, refresh: bool = False) -> pd.DataFrame:
        _ = refresh
        return team_logs[(season, team)].copy()

    rows = compute_offensive_features(
        "2025-04-03",
        db_path=db_path,
        windows=(7,),
        regression_weight=0,
        team_logs_fetcher=fake_team_logs_fetcher,
        batting_stats_fetcher=lambda *_args, **_kwargs: pd.DataFrame(),
    )

    by_name = {row.feature_name: row.feature_value for row in rows}
    assert by_name["home_team_woba_7g"] > 0.32
    assert by_name["away_team_woba_7g"] > 0.30
