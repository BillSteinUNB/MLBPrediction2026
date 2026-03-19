from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.db import init_db


C_WEIGHT = 1.30
SS_WEIGHT = 1.25
LF_WEIGHT = 0.90
ABS_RETENTION = 0.75


def _seed_game(
    db_path: Path,
    *,
    game_pk: int,
    game_date: str,
    home_team: str,
    away_team: str,
    is_abs_active: int = 1,
) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO games (
                game_pk,
                date,
                home_team,
                away_team,
                venue,
                status,
                is_abs_active
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (game_pk, game_date, home_team, away_team, "Test Park", "scheduled", is_abs_active),
        )
        connection.commit()


def _fake_fielding_fetcher(fielding_by_season: dict[int, pd.DataFrame]):
    def _fetcher(season: int, **_kwargs) -> pd.DataFrame:
        return fielding_by_season.get(season, pd.DataFrame()).copy()

    return _fetcher


def _fake_framing_fetcher(framing_by_season: dict[int, pd.DataFrame]):
    def _fetcher(season: int, **_kwargs) -> pd.DataFrame:
        return framing_by_season.get(season, pd.DataFrame()).copy()

    return _fetcher


def _fake_team_logs_fetcher(team_logs: dict[tuple[int, str], pd.DataFrame]):
    def _fetcher(season: int, team: str, **_kwargs) -> pd.DataFrame:
        return team_logs.get((season, team), pd.DataFrame()).copy()

    return _fetcher


def test_compute_defense_features_excludes_current_game_applies_weights_and_adjusts_framing(
    tmp_path: Path,
) -> None:
    from src.features.defense import compute_defense_features

    db_path = tmp_path / "defense.db"
    init_db(db_path)
    _seed_game(
        db_path,
        game_pk=801,
        game_date="2025-04-04T20:05:00+00:00",
        home_team="NYY",
        away_team="BOS",
    )

    fielding_by_season = {
        2025: pd.DataFrame(
            [
                {"game_date": "2025-04-01", "team": "NYY", "position": "C", "DRS": 2.0, "OAA": 1.0},
                {"game_date": "2025-04-01", "team": "NYY", "position": "SS", "DRS": 1.0, "OAA": 2.0},
                {"game_date": "2025-04-03", "team": "NYY", "position": "C", "DRS": 4.0, "OAA": 3.0},
                {"game_date": "2025-04-03", "team": "NYY", "position": "LF", "DRS": 2.0, "OAA": 1.0},
                {"game_date": "2025-04-04", "team": "NYY", "position": "C", "DRS": 20.0, "OAA": 20.0},
                {"game_date": "2025-04-01", "team": "BOS", "position": "C", "DRS": 1.0, "OAA": 1.0},
                {"game_date": "2025-04-03", "team": "BOS", "position": "SS", "DRS": 2.0, "OAA": 1.0},
                {"game_date": "2025-04-04", "team": "BOS", "position": "SS", "DRS": 25.0, "OAA": 25.0},
            ]
        )
    }
    framing_by_season = {
        2025: pd.DataFrame(
            [
                {"game_date": "2025-04-01", "team": "NYY", "runs_extra_strikes": 2.0},
                {"game_date": "2025-04-03", "team": "NYY", "runs_extra_strikes": 4.0},
                {"game_date": "2025-04-04", "team": "NYY", "runs_extra_strikes": 10.0},
                {"game_date": "2025-04-01", "team": "BOS", "runs_extra_strikes": 1.0},
                {"game_date": "2025-04-03", "team": "BOS", "runs_extra_strikes": 2.0},
            ]
        )
    }
    team_logs = {
        (2025, "NYY"): pd.DataFrame(
            [
                {"Date": "2025-04-01", "Opp": "BOS", "AB": 31, "H": 9, "HR": 1, "SO": 7, "SF": 1},
                {"Date": "2025-04-03", "Opp": "BOS", "AB": 30, "H": 8, "HR": 1, "SO": 6, "SF": 1},
                {"Date": "2025-04-04", "Opp": "BOS", "AB": 30, "H": 20, "HR": 4, "SO": 2, "SF": 1},
            ]
        ),
        (2025, "BOS"): pd.DataFrame(
            [
                {"Date": "2025-04-01", "Opp": "NYY", "AB": 30, "H": 8, "HR": 1, "SO": 6, "SF": 1},
                {"Date": "2025-04-03", "Opp": "NYY", "AB": 31, "H": 7, "HR": 0, "SO": 8, "SF": 0},
                {"Date": "2025-04-04", "Opp": "NYY", "AB": 30, "H": 18, "HR": 3, "SO": 2, "SF": 1},
            ]
        ),
    }

    rows = compute_defense_features(
        "2025-04-04",
        db_path=db_path,
        windows=(30,),
        fielding_fetcher=_fake_fielding_fetcher(fielding_by_season),
        framing_fetcher=_fake_framing_fetcher(framing_by_season),
        team_logs_fetcher=_fake_team_logs_fetcher(team_logs),
    )

    by_name = {row.feature_name: row.feature_value for row in rows}
    expected_home_drs = ((2.0 * C_WEIGHT + 1.0 * SS_WEIGHT) + (4.0 * C_WEIGHT + 2.0 * LF_WEIGHT)) / 2
    expected_home_oaa = ((1.0 * C_WEIGHT + 2.0 * SS_WEIGHT) + (3.0 * C_WEIGHT + 1.0 * LF_WEIGHT)) / 2
    expected_home_framing = ((2.0 * ABS_RETENTION) + (4.0 * ABS_RETENTION)) / 2
    expected_home_def_efficiency = ((17.0 / 24.0) + (16.0 / 23.0)) / 2
    leaked_home_drs = (
        (2.0 * C_WEIGHT + 1.0 * SS_WEIGHT)
        + (4.0 * C_WEIGHT + 2.0 * LF_WEIGHT)
        + (20.0 * C_WEIGHT)
    ) / 3

    assert by_name["home_team_drs_season"] == pytest.approx(expected_home_drs)
    assert by_name["home_team_oaa_season"] == pytest.approx(expected_home_oaa)
    assert by_name["home_team_adjusted_framing_season"] == pytest.approx(expected_home_framing)
    assert by_name["home_team_defensive_efficiency_season"] == pytest.approx(
        expected_home_def_efficiency
    )
    assert by_name["home_team_drs_30g"] == pytest.approx(expected_home_drs)
    assert by_name["home_team_drs_season"] < leaked_home_drs

    with sqlite3.connect(db_path) as connection:
        stored = connection.execute(
            """
            SELECT as_of_timestamp, window_size
            FROM features
            WHERE game_pk = ? AND feature_name = ?
            """,
            (801, "home_team_adjusted_framing_30g"),
        ).fetchone()

    assert stored == ("2025-04-03T00:00:00+00:00", 30)


def test_compute_defense_features_builds_season_thirty_and_sixty_game_windows(
    tmp_path: Path,
) -> None:
    from src.features.defense import compute_defense_features

    db_path = tmp_path / "defense.db"
    init_db(db_path)
    _seed_game(
        db_path,
        game_pk=802,
        game_date="2025-06-05T20:05:00+00:00",
        home_team="NYY",
        away_team="BOS",
    )

    dates = pd.date_range("2025-04-01", periods=65, freq="D")
    nyy_fielding_rows = [
        {
            "game_date": game_day.strftime("%Y-%m-%d"),
            "team": "NYY",
            "position": "SS",
            "DRS": float(index),
            "OAA": float(index * 2),
        }
        for index, game_day in enumerate(dates, start=1)
    ]
    bos_fielding_rows = [
        {
            "game_date": game_day.strftime("%Y-%m-%d"),
            "team": "BOS",
            "position": "SS",
            "DRS": 1.0,
            "OAA": 1.0,
        }
        for game_day in dates
    ]

    framing_by_season = {
        2025: pd.DataFrame(
            [
                *[
                    {
                        "game_date": game_day.strftime("%Y-%m-%d"),
                        "team": "NYY",
                        "runs_extra_strikes": float(index),
                    }
                    for index, game_day in enumerate(dates, start=1)
                ],
                *[
                    {
                        "game_date": game_day.strftime("%Y-%m-%d"),
                        "team": "BOS",
                        "runs_extra_strikes": 1.0,
                    }
                    for game_day in dates
                ],
            ]
        )
    }
    team_logs = {
        (2025, "NYY"): pd.DataFrame(
            [
                {
                    "Date": game_day.strftime("%Y-%m-%d"),
                    "Opp": "BOS",
                    "AB": 30,
                    "H": 8,
                    "HR": 1,
                    "SO": 7,
                    "SF": 1,
                }
                for game_day in dates
            ]
        ),
        (2025, "BOS"): pd.DataFrame(
            [
                {
                    "Date": game_day.strftime("%Y-%m-%d"),
                    "Opp": "NYY",
                    "AB": 30,
                    "H": 6 + (index % 3),
                    "HR": 1,
                    "SO": 6,
                    "SF": 1,
                }
                for index, game_day in enumerate(dates, start=1)
            ]
        ),
    }

    rows = compute_defense_features(
        "2025-06-05",
        db_path=db_path,
        windows=(30, 60),
        fielding_fetcher=_fake_fielding_fetcher(
            {2025: pd.DataFrame([*nyy_fielding_rows, *bos_fielding_rows])}
        ),
        framing_fetcher=_fake_framing_fetcher(framing_by_season),
        team_logs_fetcher=_fake_team_logs_fetcher(team_logs),
    )

    by_name = {row.feature_name: row.feature_value for row in rows}
    expected_season_drs = pd.Series(range(1, 66), dtype=float).mean() * SS_WEIGHT
    expected_30g_drs = pd.Series(range(36, 66), dtype=float).mean() * SS_WEIGHT
    expected_60g_drs = pd.Series(range(6, 66), dtype=float).mean() * SS_WEIGHT
    expected_season_oaa = pd.Series([value * 2 for value in range(1, 66)], dtype=float).mean() * SS_WEIGHT
    expected_30g_framing = pd.Series(range(36, 66), dtype=float).mean() * ABS_RETENTION

    assert by_name["home_team_drs_season"] == pytest.approx(expected_season_drs)
    assert by_name["home_team_drs_30g"] == pytest.approx(expected_30g_drs)
    assert by_name["home_team_drs_60g"] == pytest.approx(expected_60g_drs)
    assert by_name["home_team_oaa_season"] == pytest.approx(expected_season_oaa)
    assert by_name["home_team_adjusted_framing_30g"] == pytest.approx(expected_30g_framing)
    assert by_name["home_team_drs_30g"] > by_name["home_team_drs_60g"] > by_name["home_team_drs_season"]


def test_compute_defense_features_uses_snapshot_totals_when_daily_history_is_unavailable(
    tmp_path: Path,
) -> None:
    from src.features.defense import compute_defense_features

    db_path = tmp_path / "defense.db"
    init_db(db_path)
    _seed_game(
        db_path,
        game_pk=803,
        game_date="2025-05-01T20:05:00+00:00",
        home_team="NYY",
        away_team="BOS",
    )

    prior_dates = pd.date_range("2025-04-01", periods=20, freq="D")
    team_logs = {
        (2025, "NYY"): pd.DataFrame(
            [
                {
                    "Date": game_day.strftime("%Y-%m-%d"),
                    "Opp": "BOS",
                    "AB": 30,
                    "H": 8,
                    "HR": 1,
                    "SO": 7,
                    "SF": 1,
                }
                for game_day in prior_dates
            ]
        ),
        (2025, "BOS"): pd.DataFrame(
            [
                {
                    "Date": game_day.strftime("%Y-%m-%d"),
                    "Opp": "NYY",
                    "AB": 30,
                    "H": 7,
                    "HR": 1,
                    "SO": 6,
                    "SF": 1,
                }
                for game_day in prior_dates
            ]
        ),
    }

    rows = compute_defense_features(
        "2025-05-01",
        db_path=db_path,
        windows=(30, 60),
        fielding_fetcher=_fake_fielding_fetcher(
            {
                2025: pd.DataFrame(
                    {
                        "Name": ["NYY Shortstop", "BOS Shortstop"],
                        "Team": ["NYY", "BOS"],
                        "Pos": ["SS", "SS"],
                        "DRS": [40.0, 20.0],
                        "OAA": [30.0, 15.0],
                    }
                )
            }
        ),
        framing_fetcher=_fake_framing_fetcher(
            {
                2025: pd.DataFrame(
                    {
                        "team": ["NYY", "BOS"],
                        "runs_extra_strikes": [6.0, 3.0],
                    }
                )
            }
        ),
        team_logs_fetcher=_fake_team_logs_fetcher(team_logs),
    )

    by_name = {row.feature_name: row.feature_value for row in rows}
    expected_home_drs_per_game = (40.0 * SS_WEIGHT) / 20
    expected_home_oaa_per_game = (30.0 * SS_WEIGHT) / 20
    expected_home_adjusted_framing_per_game = (6.0 * ABS_RETENTION) / 20

    assert by_name["home_team_drs_season"] == pytest.approx(expected_home_drs_per_game)
    assert by_name["home_team_drs_30g"] == pytest.approx(expected_home_drs_per_game)
    assert by_name["home_team_drs_60g"] == pytest.approx(expected_home_drs_per_game)
    assert by_name["home_team_oaa_season"] == pytest.approx(expected_home_oaa_per_game)
    assert by_name["home_team_adjusted_framing_season"] == pytest.approx(
        expected_home_adjusted_framing_per_game
    )
