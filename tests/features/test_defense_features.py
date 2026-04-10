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


def _defense_payload(rows: list[object]) -> list[tuple[int, str, int | None, str, float]]:
    return sorted(
        [
            (
                int(row.game_pk),
                str(row.feature_name),
                row.window_size,
                row.as_of_timestamp.isoformat(),
                round(float(row.feature_value), 8),
            )
            for row in rows
        ]
    )


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
    current_home_drs = ((2.0 * C_WEIGHT + 1.0 * SS_WEIGHT) + (4.0 * C_WEIGHT + 2.0 * LF_WEIGHT)) / 2
    current_home_oaa = ((1.0 * C_WEIGHT + 2.0 * SS_WEIGHT) + (3.0 * C_WEIGHT + 1.0 * LF_WEIGHT)) / 2
    current_home_framing = ((2.0 * ABS_RETENTION) + (4.0 * ABS_RETENTION)) / 2
    current_home_def_efficiency = ((17.0 / 24.0) + (16.0 / 23.0)) / 2
    expected_home_drs = (current_home_drs * 2) / 52
    expected_home_oaa = (current_home_oaa * 2) / 52
    expected_home_raw_framing = (3.0 * 2) / 52
    expected_home_framing = (current_home_framing * 2) / 52
    expected_home_def_efficiency = (current_home_def_efficiency * 2 + 0.700 * 50) / 52
    leaked_home_drs = (
        (2.0 * C_WEIGHT + 1.0 * SS_WEIGHT)
        + (4.0 * C_WEIGHT + 2.0 * LF_WEIGHT)
        + (20.0 * C_WEIGHT)
    ) / 3

    assert by_name["home_team_drs_season"] == pytest.approx(expected_home_drs)
    assert by_name["home_team_oaa_season"] == pytest.approx(expected_home_oaa)
    assert by_name["home_team_raw_framing_season"] == pytest.approx(expected_home_raw_framing)
    assert by_name["home_team_adjusted_framing_season"] == pytest.approx(expected_home_framing)
    assert by_name["home_team_framing_retention_proxy_season"] == pytest.approx(ABS_RETENTION)
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


def test_compute_defense_features_neutralizes_framing_retention_when_abs_is_inactive(
    tmp_path: Path,
) -> None:
    from src.features.defense import compute_defense_features

    db_path = tmp_path / "defense_abs_off.db"
    init_db(db_path)
    _seed_game(
        db_path,
        game_pk=807,
        game_date="2025-04-04T20:05:00+00:00",
        home_team="NYY",
        away_team="BOS",
        is_abs_active=0,
    )

    rows = compute_defense_features(
        "2025-04-04",
        db_path=db_path,
        windows=(30,),
        fielding_fetcher=_fake_fielding_fetcher({}),
        framing_fetcher=_fake_framing_fetcher(
            {
                2025: pd.DataFrame(
                    [
                        {"game_date": "2025-04-01", "team": "NYY", "runs_extra_strikes": 2.0},
                        {"game_date": "2025-04-01", "team": "BOS", "runs_extra_strikes": 1.0},
                    ]
                )
            }
        ),
        team_logs_fetcher=_fake_team_logs_fetcher(
            {
                (2025, "NYY"): pd.DataFrame(
                    [{"Date": "2025-04-01", "Opp": "BOS", "AB": 30, "H": 7, "HR": 1, "SO": 7, "SF": 1}]
                ),
                (2025, "BOS"): pd.DataFrame(
                    [{"Date": "2025-04-01", "Opp": "NYY", "AB": 30, "H": 7, "HR": 1, "SO": 7, "SF": 1}]
                ),
            }
        ),
    )

    by_name = {row.feature_name: row.feature_value for row in rows}

    assert by_name["home_team_adjusted_framing_season"] == pytest.approx(
        by_name["home_team_raw_framing_season"]
    )
    assert by_name["home_team_framing_retention_proxy_season"] == pytest.approx(1.0)


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
    blend_factor = 65 / (65 + 50)
    expected_season_drs = pd.Series(range(1, 66), dtype=float).mean() * SS_WEIGHT * blend_factor
    expected_30g_drs = pd.Series(range(36, 66), dtype=float).mean() * SS_WEIGHT * blend_factor
    expected_60g_drs = pd.Series(range(6, 66), dtype=float).mean() * SS_WEIGHT * blend_factor
    expected_season_oaa = (
        pd.Series([value * 2 for value in range(1, 66)], dtype=float).mean() * SS_WEIGHT * blend_factor
    )
    expected_30g_framing = pd.Series(range(36, 66), dtype=float).mean() * ABS_RETENTION * blend_factor

    assert by_name["home_team_drs_season"] == pytest.approx(expected_season_drs)
    assert by_name["home_team_drs_30g"] == pytest.approx(expected_30g_drs)
    assert by_name["home_team_drs_60g"] == pytest.approx(expected_60g_drs)
    assert by_name["home_team_oaa_season"] == pytest.approx(expected_season_oaa)
    assert by_name["home_team_adjusted_framing_30g"] == pytest.approx(expected_30g_framing)
    assert by_name["home_team_drs_30g"] > by_name["home_team_drs_60g"] > by_name["home_team_drs_season"]


def test_compute_defense_features_uses_prior_season_baselines_when_current_season_only_has_snapshots(
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
        (2024, "NYY"): pd.DataFrame(
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
                for game_day in pd.date_range("2024-04-01", periods=20, freq="D")
            ]
        ),
        (2024, "BOS"): pd.DataFrame(
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
                for game_day in pd.date_range("2024-04-01", periods=20, freq="D")
            ]
        ),
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
                2024: pd.DataFrame(
                    {
                        "Name": ["NYY Shortstop", "BOS Shortstop"],
                        "Team": ["NYY", "BOS"],
                        "Pos": ["SS", "SS"],
                        "DRS": [12.0, 8.0],
                        "OAA": [18.0, 6.0],
                    }
                ),
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
                2024: pd.DataFrame(
                    {
                        "team": ["NYY", "BOS"],
                        "runs_extra_strikes": [4.0, 2.0],
                    }
                ),
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
    expected_home_drs_per_game = (12.0 * SS_WEIGHT) / 20
    expected_home_oaa_per_game = (18.0 * SS_WEIGHT) / 20
    expected_home_adjusted_framing_per_game = (4.0 * ABS_RETENTION) / 20
    leaked_current_snapshot_drs = (40.0 * SS_WEIGHT) / 20

    assert by_name["home_team_drs_season"] == pytest.approx(expected_home_drs_per_game)
    assert by_name["home_team_drs_30g"] == pytest.approx(expected_home_drs_per_game)
    assert by_name["home_team_drs_60g"] == pytest.approx(expected_home_drs_per_game)
    assert by_name["home_team_drs_season"] != pytest.approx(leaked_current_snapshot_drs)
    assert by_name["home_team_oaa_season"] == pytest.approx(expected_home_oaa_per_game)
    assert by_name["home_team_adjusted_framing_season"] == pytest.approx(
        expected_home_adjusted_framing_per_game
    )


def test_compute_defense_features_falls_back_to_league_averages_without_team_history(
    tmp_path: Path,
) -> None:
    from src.features.defense import compute_defense_features

    db_path = tmp_path / "defense.db"
    init_db(db_path)
    _seed_game(
        db_path,
        game_pk=804,
        game_date="2025-04-02T20:05:00+00:00",
        home_team="NYY",
        away_team="BOS",
    )

    rows = compute_defense_features(
        "2025-04-02",
        db_path=db_path,
        windows=(30, 60),
        fielding_fetcher=_fake_fielding_fetcher({}),
        framing_fetcher=_fake_framing_fetcher({}),
        team_logs_fetcher=_fake_team_logs_fetcher({}),
    )

    by_name = {row.feature_name: row.feature_value for row in rows}

    assert by_name["home_team_drs_season"] == pytest.approx(0.0)
    assert by_name["home_team_oaa_season"] == pytest.approx(0.0)
    assert by_name["home_team_adjusted_framing_season"] == pytest.approx(0.0)
    assert by_name["home_team_defensive_efficiency_season"] == pytest.approx(0.700)
    assert by_name["home_team_drs_30g"] == pytest.approx(0.0)
    assert by_name["home_team_drs_60g"] == pytest.approx(0.0)


def test_compute_defense_features_supports_current_team_log_schema(tmp_path: Path) -> None:
    from src.features.defense import compute_defense_features

    db_path = tmp_path / "defense.db"
    init_db(db_path)
    _seed_game(
        db_path,
        game_pk=805,
        game_date="2025-04-03T20:05:00+00:00",
        home_team="NYY",
        away_team="BOS",
    )

    team_logs = {
        (2025, "NYY"): pd.DataFrame(
            {
                "Date": ["2025-04-01", "2025-04-02"],
                "Opp": ["BOS", "BOS"],
                "Batting Stats_AB": [31, 30],
                "Batting Stats_H": [9, 7],
                "Batting Stats_HR": [1, 0],
                "Batting Stats_SO": [8, 6],
                "Batting Stats_SF": [1, 1],
            }
        ),
        (2025, "BOS"): pd.DataFrame(
            {
                "Date": ["2025-04-01", "2025-04-02"],
                "Opp": ["NYY", "NYY"],
                "Batting Stats_AB": [30, 31],
                "Batting Stats_H": [6, 8],
                "Batting Stats_HR": [1, 1],
                "Batting Stats_SO": [9, 7],
                "Batting Stats_SF": [1, 0],
            }
        ),
    }

    rows = compute_defense_features(
        "2025-04-03",
        db_path=db_path,
        windows=(30,),
        fielding_fetcher=_fake_fielding_fetcher({}),
        framing_fetcher=_fake_framing_fetcher({}),
        team_logs_fetcher=_fake_team_logs_fetcher(team_logs),
    )

    by_name = {row.feature_name: row.feature_value for row in rows}

    assert by_name["home_team_defensive_efficiency_season"] != pytest.approx(0.700)
    assert by_name["away_team_defensive_efficiency_season"] != pytest.approx(0.700)


def test_compute_defense_features_accepts_tz_aware_input_dates(tmp_path: Path) -> None:
    from src.features.defense import compute_defense_features

    db_path = tmp_path / "defense_tz.db"
    init_db(db_path)
    _seed_game(
        db_path,
        game_pk=806,
        game_date="2025-04-04T20:05:00+00:00",
        home_team="NYY",
        away_team="BOS",
    )

    fielding_by_season = {
        2025: pd.DataFrame(
            [
                {"game_date": pd.Timestamp("2025-04-01T00:00:00Z"), "team": "NYY", "position": "SS", "DRS": 2.0, "OAA": 1.0},
                {"game_date": pd.Timestamp("2025-04-03T00:00:00Z"), "team": "BOS", "position": "SS", "DRS": 1.0, "OAA": 1.0},
            ]
        )
    }
    framing_by_season = {
        2025: pd.DataFrame(
            [
                {"game_date": pd.Timestamp("2025-04-01T00:00:00Z"), "team": "NYY", "runs_extra_strikes": 1.0},
                {"game_date": pd.Timestamp("2025-04-03T00:00:00Z"), "team": "BOS", "runs_extra_strikes": 1.0},
            ]
        )
    }
    team_logs = {
        (2025, "NYY"): pd.DataFrame([{"Date": pd.Timestamp("2025-04-01T00:00:00Z"), "Opp": "BOS", "AB": 30, "H": 8, "HR": 1, "SO": 7, "SF": 1}]),
        (2025, "BOS"): pd.DataFrame([{"Date": pd.Timestamp("2025-04-03T00:00:00Z"), "Opp": "NYY", "AB": 30, "H": 7, "HR": 0, "SO": 8, "SF": 1}]),
    }

    rows = compute_defense_features(
        "2025-04-04",
        db_path=db_path,
        windows=(30,),
        fielding_fetcher=_fake_fielding_fetcher(fielding_by_season),
        framing_fetcher=_fake_framing_fetcher(framing_by_season),
        team_logs_fetcher=_fake_team_logs_fetcher(team_logs),
    )

    assert rows


def test_compute_defense_features_for_schedule_matches_day_by_day_results(
    tmp_path: Path,
) -> None:
    from src.features.defense import (
        compute_defense_features,
        compute_defense_features_for_schedule,
    )

    schedule = pd.DataFrame(
        [
            {
                "game_pk": 8801,
                "game_date": "2025-04-03",
                "home_team": "NYY",
                "away_team": "BOS",
                "is_abs_active": 1,
            },
            {
                "game_pk": 8802,
                "game_date": "2025-04-04",
                "home_team": "BOS",
                "away_team": "NYY",
                "is_abs_active": 1,
            },
        ]
    )

    fielding_by_season = {
        2025: pd.DataFrame(
            [
                {"game_date": "2025-04-01", "team": "NYY", "position": "SS", "DRS": 2.0, "OAA": 3.0},
                {"game_date": "2025-04-02", "team": "NYY", "position": "C", "DRS": 1.0, "OAA": 2.0},
                {"game_date": "2025-04-03", "team": "NYY", "position": "LF", "DRS": 4.0, "OAA": 1.0},
                {"game_date": "2025-04-01", "team": "BOS", "position": "SS", "DRS": 1.0, "OAA": 1.0},
                {"game_date": "2025-04-02", "team": "BOS", "position": "C", "DRS": 2.0, "OAA": 2.0},
                {"game_date": "2025-04-03", "team": "BOS", "position": "LF", "DRS": 3.0, "OAA": 2.0},
            ]
        ),
        2024: pd.DataFrame(
            [
                {"game_date": "2024-09-01", "team": "NYY", "position": "SS", "DRS": 6.0, "OAA": 5.0},
                {"game_date": "2024-09-01", "team": "BOS", "position": "SS", "DRS": 4.0, "OAA": 3.0},
            ]
        ),
    }
    framing_by_season = {
        2025: pd.DataFrame(
            [
                {"game_date": "2025-04-01", "team": "NYY", "runs_extra_strikes": 1.0},
                {"game_date": "2025-04-02", "team": "NYY", "runs_extra_strikes": 2.0},
                {"game_date": "2025-04-03", "team": "NYY", "runs_extra_strikes": 3.0},
                {"game_date": "2025-04-01", "team": "BOS", "runs_extra_strikes": 1.0},
                {"game_date": "2025-04-02", "team": "BOS", "runs_extra_strikes": 1.5},
                {"game_date": "2025-04-03", "team": "BOS", "runs_extra_strikes": 2.0},
            ]
        ),
        2024: pd.DataFrame(
            [
                {"game_date": "2024-09-01", "team": "NYY", "runs_extra_strikes": 2.0},
                {"game_date": "2024-09-01", "team": "BOS", "runs_extra_strikes": 1.0},
            ]
        ),
    }
    team_logs = {
        (2025, "NYY"): pd.DataFrame(
            [
                {"Date": "2025-04-01", "Opp": "BOS", "AB": 30, "H": 7, "HR": 1, "SO": 8, "SF": 1},
                {"Date": "2025-04-02", "Opp": "BOS", "AB": 31, "H": 8, "HR": 1, "SO": 7, "SF": 1},
                {"Date": "2025-04-03", "Opp": "BOS", "AB": 29, "H": 5, "HR": 0, "SO": 9, "SF": 1},
            ]
        ),
        (2025, "BOS"): pd.DataFrame(
            [
                {"Date": "2025-04-01", "Opp": "NYY", "AB": 30, "H": 6, "HR": 1, "SO": 9, "SF": 1},
                {"Date": "2025-04-02", "Opp": "NYY", "AB": 29, "H": 7, "HR": 0, "SO": 8, "SF": 1},
                {"Date": "2025-04-03", "Opp": "NYY", "AB": 30, "H": 9, "HR": 1, "SO": 6, "SF": 1},
            ]
        ),
        (2024, "NYY"): pd.DataFrame(
            [{"Date": "2024-09-01", "Opp": "BOS", "AB": 30, "H": 7, "HR": 1, "SO": 8, "SF": 1}]
        ),
        (2024, "BOS"): pd.DataFrame(
            [{"Date": "2024-09-01", "Opp": "NYY", "AB": 30, "H": 8, "HR": 1, "SO": 7, "SF": 1}]
        ),
    }

    day_db_path = tmp_path / "defense_day.db"
    bulk_db_path = tmp_path / "defense_bulk.db"
    init_db(day_db_path)
    init_db(bulk_db_path)
    for row in schedule.to_dict(orient="records"):
        _seed_game(
            day_db_path,
            game_pk=int(row["game_pk"]),
            game_date=f"{row['game_date']}T20:05:00+00:00",
            home_team=str(row["home_team"]),
            away_team=str(row["away_team"]),
            is_abs_active=int(row["is_abs_active"]),
        )
        _seed_game(
            bulk_db_path,
            game_pk=int(row["game_pk"]),
            game_date=f"{row['game_date']}T20:05:00+00:00",
            home_team=str(row["home_team"]),
            away_team=str(row["away_team"]),
            is_abs_active=int(row["is_abs_active"]),
        )

    expected_rows = []
    for game_date in schedule["game_date"].tolist():
        expected_rows.extend(
            compute_defense_features(
                game_date,
                db_path=day_db_path,
                fielding_fetcher=_fake_fielding_fetcher(fielding_by_season),
                framing_fetcher=_fake_framing_fetcher(framing_by_season),
                team_logs_fetcher=_fake_team_logs_fetcher(team_logs),
            )
        )

    actual_rows = compute_defense_features_for_schedule(
        schedule,
        db_path=bulk_db_path,
        fielding_fetcher=_fake_fielding_fetcher(fielding_by_season),
        framing_fetcher=_fake_framing_fetcher(framing_by_season),
        team_logs_fetcher=_fake_team_logs_fetcher(team_logs),
    )

    assert _defense_payload(actual_rows) == _defense_payload(expected_rows)


def test_compute_defense_features_for_schedule_deduplicates_duplicate_game_rows(
    tmp_path: Path,
) -> None:
    from src.features.defense import compute_defense_features_for_schedule

    base_schedule = pd.DataFrame(
        [
            {
                "game_pk": 9901,
                "game_date": "2025-04-03",
                "home_team": "NYY",
                "away_team": "BOS",
                "is_abs_active": 1,
            }
        ]
    )
    duplicate_schedule = pd.concat([base_schedule, base_schedule], ignore_index=True)

    fielding_by_season = {
        2025: pd.DataFrame(
            [
                {"game_date": "2025-04-01", "team": "NYY", "position": "SS", "DRS": 2.0, "OAA": 3.0},
                {"game_date": "2025-04-01", "team": "BOS", "position": "SS", "DRS": 1.0, "OAA": 1.0},
            ]
        ),
        2024: pd.DataFrame(
            [
                {"game_date": "2024-09-01", "team": "NYY", "position": "SS", "DRS": 6.0, "OAA": 5.0},
                {"game_date": "2024-09-01", "team": "BOS", "position": "SS", "DRS": 4.0, "OAA": 3.0},
            ]
        ),
    }
    framing_by_season = {
        2025: pd.DataFrame(
            [
                {"game_date": "2025-04-01", "team": "NYY", "runs_extra_strikes": 1.0},
                {"game_date": "2025-04-01", "team": "BOS", "runs_extra_strikes": 1.0},
            ]
        ),
        2024: pd.DataFrame(
            [
                {"game_date": "2024-09-01", "team": "NYY", "runs_extra_strikes": 2.0},
                {"game_date": "2024-09-01", "team": "BOS", "runs_extra_strikes": 1.0},
            ]
        ),
    }
    team_logs = {
        (2025, "NYY"): pd.DataFrame(
            [{"Date": "2025-04-01", "Opp": "BOS", "AB": 30, "H": 7, "HR": 1, "SO": 8, "SF": 1}]
        ),
        (2025, "BOS"): pd.DataFrame(
            [{"Date": "2025-04-01", "Opp": "NYY", "AB": 30, "H": 6, "HR": 1, "SO": 9, "SF": 1}]
        ),
        (2024, "NYY"): pd.DataFrame(
            [{"Date": "2024-09-01", "Opp": "BOS", "AB": 30, "H": 7, "HR": 1, "SO": 8, "SF": 1}]
        ),
        (2024, "BOS"): pd.DataFrame(
            [{"Date": "2024-09-01", "Opp": "NYY", "AB": 30, "H": 8, "HR": 1, "SO": 7, "SF": 1}]
        ),
    }

    single_db_path = tmp_path / "defense_single.db"
    duplicate_db_path = tmp_path / "defense_duplicate.db"
    init_db(single_db_path)
    init_db(duplicate_db_path)
    for db_path in (single_db_path, duplicate_db_path):
        _seed_game(
            db_path,
            game_pk=9901,
            game_date="2025-04-03T20:05:00+00:00",
            home_team="NYY",
            away_team="BOS",
            is_abs_active=1,
        )

    expected_rows = compute_defense_features_for_schedule(
        base_schedule,
        db_path=single_db_path,
        windows=(30,),
        fielding_fetcher=_fake_fielding_fetcher(fielding_by_season),
        framing_fetcher=_fake_framing_fetcher(framing_by_season),
        team_logs_fetcher=_fake_team_logs_fetcher(team_logs),
    )
    actual_rows = compute_defense_features_for_schedule(
        duplicate_schedule,
        db_path=duplicate_db_path,
        windows=(30,),
        fielding_fetcher=_fake_fielding_fetcher(fielding_by_season),
        framing_fetcher=_fake_framing_fetcher(framing_by_season),
        team_logs_fetcher=_fake_team_logs_fetcher(team_logs),
    )

    assert _defense_payload(actual_rows) == _defense_payload(expected_rows)
