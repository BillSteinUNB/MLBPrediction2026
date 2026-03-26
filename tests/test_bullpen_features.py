from __future__ import annotations

import sqlite3
from datetime import date
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
    home_starter_id: int | None = None,
    away_starter_id: int | None = None,
) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO games (
                game_pk,
                date,
                home_team,
                away_team,
                home_starter_id,
                away_starter_id,
                venue,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                game_pk,
                game_date,
                home_team,
                away_team,
                home_starter_id,
                away_starter_id,
                "Test Park",
                "scheduled",
            ),
        )
        connection.commit()


def _bullpen_metrics_by_season() -> dict[int, pd.DataFrame]:
    return {
        2025: pd.DataFrame(
            [
                {
                    "game_pk": 5001,
                    "game_date": "2025-04-03",
                    "team": "NYY",
                    "pitcher_id": 15,
                    "pitch_count": 30,
                    "innings_pitched": 2.0,
                    "xfip": 3.8,
                    "inherited_runners": 2,
                    "inherited_runners_scored": 1,
                },
                {
                    "game_pk": 5002,
                    "game_date": "2025-04-05",
                    "team": "NYY",
                    "pitcher_id": 11,
                    "pitch_count": 20,
                    "innings_pitched": 1.0,
                    "xfip": 3.0,
                    "inherited_runners": 1,
                    "inherited_runners_scored": 0,
                },
                {
                    "game_pk": 5003,
                    "game_date": "2025-04-06",
                    "team": "NYY",
                    "pitcher_id": 12,
                    "pitch_count": 15,
                    "innings_pitched": 1.0,
                    "xfip": 4.0,
                    "inherited_runners": 2,
                    "inherited_runners_scored": 1,
                },
                {
                    "game_pk": 5004,
                    "game_date": "2025-04-07",
                    "team": "NYY",
                    "pitcher_id": 13,
                    "pitch_count": 18,
                    "innings_pitched": 1.0,
                    "xfip": 3.5,
                    "inherited_runners": 0,
                    "inherited_runners_scored": 0,
                },
                {
                    "game_pk": 5005,
                    "game_date": "2025-04-08",
                    "team": "NYY",
                    "pitcher_id": 11,
                    "pitch_count": 22,
                    "innings_pitched": 1.0,
                    "xfip": 3.2,
                    "inherited_runners": 1,
                    "inherited_runners_scored": 1,
                },
                {
                    "game_pk": 5006,
                    "game_date": "2025-04-09",
                    "team": "NYY",
                    "pitcher_id": 14,
                    "pitch_count": 16,
                    "innings_pitched": 1.0,
                    "xfip": 4.5,
                    "inherited_runners": 3,
                    "inherited_runners_scored": 1,
                },
                {
                    "game_pk": 6001,
                    "game_date": "2025-04-06",
                    "team": "BOS",
                    "pitcher_id": 21,
                    "pitch_count": 14,
                    "innings_pitched": 1.0,
                    "xfip": 4.2,
                    "inherited_runners": 1,
                    "inherited_runners_scored": 1,
                },
                {
                    "game_pk": 6002,
                    "game_date": "2025-04-08",
                    "team": "BOS",
                    "pitcher_id": 22,
                    "pitch_count": 18,
                    "innings_pitched": 1.0,
                    "xfip": 3.9,
                    "inherited_runners": 0,
                    "inherited_runners_scored": 0,
                },
                {
                    "game_pk": 6003,
                    "game_date": "2025-04-09",
                    "team": "BOS",
                    "pitcher_id": 23,
                    "pitch_count": 12,
                    "innings_pitched": 1.0,
                    "xfip": 3.7,
                    "inherited_runners": 2,
                    "inherited_runners_scored": 0,
                },
                {
                    "game_pk": 7001,
                    "game_date": "2025-04-10",
                    "team": "NYY",
                    "pitcher_id": 99,
                    "pitch_count": 200,
                    "innings_pitched": 3.0,
                    "xfip": 9.5,
                    "inherited_runners": 10,
                    "inherited_runners_scored": 10,
                },
                {
                    "game_pk": 7001,
                    "game_date": "2025-04-10",
                    "team": "BOS",
                    "pitcher_id": 88,
                    "pitch_count": 120,
                    "innings_pitched": 2.0,
                    "xfip": 8.0,
                    "inherited_runners": 8,
                    "inherited_runners_scored": 8,
                },
            ]
        ),
        2024: pd.DataFrame(),
    }


def _early_season_metrics() -> dict[int, pd.DataFrame]:
    return {
        2025: pd.DataFrame(
            [
                {
                    "game_pk": 4001,
                    "game_date": "2025-04-01",
                    "team": "NYY",
                    "pitcher_id": 11,
                    "pitch_count": 24,
                    "innings_pitched": 1.0,
                    "xfip": 2.8,
                    "inherited_runners": 2,
                    "inherited_runners_scored": 1,
                },
                {
                    "game_pk": 4002,
                    "game_date": "2025-04-01",
                    "team": "BOS",
                    "pitcher_id": 21,
                    "pitch_count": 18,
                    "innings_pitched": 1.0,
                    "xfip": 3.6,
                    "inherited_runners": 1,
                    "inherited_runners_scored": 0,
                },
            ]
        ),
        2024: pd.DataFrame(),
    }


def _fake_bullpen_metrics_fetcher(metrics_by_season: dict[int, pd.DataFrame]):
    def _fetcher(
        season: int,
        *,
        db_path: str | Path,
        end_date=None,
        refresh: bool = False,
    ) -> pd.DataFrame:
        _ = db_path
        _ = refresh
        dataframe = metrics_by_season.get(season, pd.DataFrame()).copy()
        if dataframe.empty or end_date is None:
            return dataframe
        return dataframe.loc[pd.to_datetime(dataframe["game_date"]).dt.date <= end_date].reset_index(
            drop=True
        )

    return _fetcher


def test_compute_bullpen_features_calculates_fatigue_and_rolling_rates(
    tmp_path: Path,
) -> None:
    from src.features.bullpen import compute_bullpen_features

    db_path = tmp_path / "bullpen.db"
    init_db(db_path)
    _seed_game(
        db_path,
        game_pk=7001,
        game_date="2025-04-10T20:05:00+00:00",
        home_team="NYY",
        away_team="BOS",
    )

    rows = compute_bullpen_features(
        "2025-04-10",
        db_path=db_path,
        bullpen_metrics_fetcher=_fake_bullpen_metrics_fetcher(_bullpen_metrics_by_season()),
    )

    by_name = {row.feature_name: row.feature_value for row in rows}
    expected_home_xfip = (3.8 * 2 + 3.0 + 4.0 + 3.5 + 3.2 + 4.5) / 7
    leaked_home_xfip = (expected_home_xfip * 7 + 9.5 * 3) / 10

    assert by_name["home_team_bullpen_pitch_count_3d"] == pytest.approx(56.0)
    assert by_name["home_team_bullpen_pitch_count_5d"] == pytest.approx(91.0)
    assert by_name["home_team_bullpen_avg_rest_days_top5"] == pytest.approx(2.4)
    assert by_name["home_team_bullpen_ir_pct_30g"] == pytest.approx(4 / 9)
    assert by_name["home_team_bullpen_xfip"] == pytest.approx(expected_home_xfip)
    assert by_name["home_team_bullpen_high_leverage_available_count"] == pytest.approx(3.0)
    assert by_name["away_team_bullpen_pitch_count_3d"] == pytest.approx(30.0)
    assert by_name["home_team_bullpen_xfip"] < leaked_home_xfip

    with sqlite3.connect(db_path) as connection:
        stored = connection.execute(
            """
            SELECT as_of_timestamp, window_size
            FROM features
            WHERE game_pk = ? AND feature_name = ?
            """,
            (7001, "home_team_bullpen_ir_pct_30g"),
        ).fetchone()

    assert stored == ("2025-04-09T00:00:00+00:00", 30)

    compute_bullpen_features(
        "2025-04-10",
        db_path=db_path,
        bullpen_metrics_fetcher=_fake_bullpen_metrics_fetcher(_bullpen_metrics_by_season()),
    )

    with sqlite3.connect(db_path) as connection:
        feature_count = connection.execute(
            "SELECT COUNT(*) FROM features WHERE game_pk = ?",
            (7001,),
        ).fetchone()[0]

    assert feature_count == len(rows)


def test_compute_bullpen_features_returns_defaults_for_first_three_days(
    tmp_path: Path,
) -> None:
    from src.features.bullpen import compute_bullpen_features

    db_path = tmp_path / "bullpen.db"
    init_db(db_path)
    _seed_game(
        db_path,
        game_pk=7101,
        game_date="2025-04-02T20:05:00+00:00",
        home_team="NYY",
        away_team="BOS",
    )

    rows = compute_bullpen_features(
        "2025-04-02",
        db_path=db_path,
        bullpen_metrics_fetcher=_fake_bullpen_metrics_fetcher(_early_season_metrics()),
    )

    by_name = {row.feature_name: row.feature_value for row in rows}
    assert by_name["home_team_bullpen_pitch_count_3d"] == 0.0
    assert by_name["home_team_bullpen_pitch_count_5d"] == 0.0
    assert by_name["home_team_bullpen_avg_rest_days_top5"] == 3.0
    assert by_name["home_team_bullpen_ir_pct_30g"] == 0.0
    assert by_name["home_team_bullpen_xfip"] == pytest.approx(4.2)
    assert by_name["home_team_bullpen_high_leverage_available_count"] == 5.0


def test_compute_bullpen_features_for_schedule_matches_day_by_day_results(
    tmp_path: Path,
) -> None:
    from src.features.bullpen import (
        compute_bullpen_features,
        compute_bullpen_features_for_schedule,
    )

    metrics_fetcher = _fake_bullpen_metrics_fetcher(_bullpen_metrics_by_season())
    schedule = pd.DataFrame(
        [
            {
                "game_pk": 7001,
                "game_date": "2025-04-10",
                "scheduled_start": "2025-04-10T20:05:00+00:00",
                "home_team": "NYY",
                "away_team": "BOS",
            },
            {
                "game_pk": 7002,
                "game_date": "2025-04-11",
                "scheduled_start": "2025-04-11T20:05:00+00:00",
                "home_team": "BOS",
                "away_team": "NYY",
            },
        ]
    )

    day_db_path = tmp_path / "bullpen_day.db"
    bulk_db_path = tmp_path / "bullpen_bulk.db"
    init_db(day_db_path)
    init_db(bulk_db_path)
    for row in schedule.to_dict(orient="records"):
        _seed_game(
            day_db_path,
            game_pk=int(row["game_pk"]),
            game_date=str(row["scheduled_start"]),
            home_team=str(row["home_team"]),
            away_team=str(row["away_team"]),
        )
        _seed_game(
            bulk_db_path,
            game_pk=int(row["game_pk"]),
            game_date=str(row["scheduled_start"]),
            home_team=str(row["home_team"]),
            away_team=str(row["away_team"]),
        )

    expected_rows = []
    for game_date in schedule["game_date"].tolist():
        expected_rows.extend(
            compute_bullpen_features(
                game_date,
                db_path=day_db_path,
                bullpen_metrics_fetcher=metrics_fetcher,
            )
        )

    actual_rows = compute_bullpen_features_for_schedule(
        schedule,
        db_path=bulk_db_path,
        bullpen_metrics_fetcher=metrics_fetcher,
    )

    expected_payload = sorted(
        [
            (
                row.game_pk,
                row.feature_name,
                row.window_size,
                row.as_of_timestamp.isoformat(),
                round(float(row.feature_value), 8),
            )
            for row in expected_rows
        ]
    )
    actual_payload = sorted(
        [
            (
                row.game_pk,
                row.feature_name,
                row.window_size,
                row.as_of_timestamp.isoformat(),
                round(float(row.feature_value), 8),
            )
            for row in actual_rows
        ]
    )

    assert actual_payload == expected_payload


def test_fetch_season_bullpen_metrics_reuses_persisted_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.features import bullpen as bullpen_module

    db_path = tmp_path / "bullpen_cache.db"
    init_db(db_path)
    _seed_game(
        db_path,
        game_pk=7201,
        game_date="2025-04-10T20:05:00+00:00",
        home_team="NYY",
        away_team="BOS",
        home_starter_id=100,
        away_starter_id=400,
    )

    cached_metrics = pd.DataFrame(
        [
            {
                "game_pk": 7201,
                "game_date": "2025-04-10",
                "team": "NYY",
                "pitcher_id": 15,
                "pitch_count": 18,
                "innings_pitched": 1.0,
                "xfip": 3.6,
                "inherited_runners": 1.0,
                "inherited_runners_scored": 0.0,
            },
            {
                "game_pk": 7201,
                "game_date": "2025-04-10",
                "team": "BOS",
                "pitcher_id": 25,
                "pitch_count": 16,
                "innings_pitched": 1.0,
                "xfip": 4.1,
                "inherited_runners": 2.0,
                "inherited_runners_scored": 1.0,
            },
        ]
    )
    fetch_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(bullpen_module, "DERIVED_CACHE_ROOT", tmp_path / "derived_bullpen")

    def _fake_fetch_statcast_range(start_date, end_date, refresh=False):
        _ = refresh
        fetch_calls.append((str(start_date), str(end_date)))
        return pd.DataFrame({"placeholder": []})

    monkeypatch.setattr(bullpen_module, "fetch_statcast_range", _fake_fetch_statcast_range)
    monkeypatch.setattr(
        bullpen_module,
        "_build_relief_metrics_from_statcast",
        lambda games, statcast_frame: cached_metrics.loc[
            :, ["game_pk", "game_date", "team", "pitcher_id", "pitch_count", "innings_pitched", "xfip"]
        ].copy(),
    )
    monkeypatch.setattr(
        bullpen_module,
        "_build_inherited_runner_lookup",
        lambda games, season, refresh, team_logs_fetcher: cached_metrics.loc[
            :, ["game_pk", "game_date", "team", "inherited_runners", "inherited_runners_scored"]
        ].copy(),
    )

    first = bullpen_module._fetch_season_bullpen_metrics(
        2025,
        db_path=db_path,
        end_date=date(2025, 4, 10),
        refresh=False,
        team_logs_fetcher=lambda *_args, **_kwargs: pd.DataFrame(),
    )

    assert fetch_calls == [("2025-04-10", "2025-04-10")]
    assert first.equals(cached_metrics)

    def _unexpected_fetch(*_args, **_kwargs):
        raise AssertionError("statcast fetch should not run when cache is present")

    monkeypatch.setattr(bullpen_module, "fetch_statcast_range", _unexpected_fetch)

    second = bullpen_module._fetch_season_bullpen_metrics(
        2025,
        db_path=db_path,
        end_date=date(2025, 4, 10),
        refresh=False,
        team_logs_fetcher=lambda *_args, **_kwargs: pd.DataFrame(),
    )

    assert second.equals(cached_metrics)


def test_compute_bullpen_features_handles_empty_metrics_without_dt_accessor_failure(
    tmp_path: Path,
) -> None:
    from src.features.bullpen import compute_bullpen_features

    db_path = tmp_path / "bullpen.db"
    init_db(db_path)
    _seed_game(
        db_path,
        game_pk=7201,
        game_date="2025-04-10T20:05:00+00:00",
        home_team="NYY",
        away_team="BOS",
    )

    rows = compute_bullpen_features(
        "2025-04-10",
        db_path=db_path,
        bullpen_metrics_fetcher=_fake_bullpen_metrics_fetcher({2025: pd.DataFrame(), 2024: pd.DataFrame()}),
    )

    by_name = {row.feature_name: row.feature_value for row in rows}
    assert by_name["home_team_bullpen_xfip"] == pytest.approx(4.2)
    assert by_name["away_team_bullpen_high_leverage_available_count"] == pytest.approx(5.0)


def test_compute_bullpen_features_parses_flattened_pitching_ir_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.features import bullpen

    db_path = tmp_path / "bullpen.db"
    init_db(db_path)
    _seed_game(
        db_path,
        game_pk=8001,
        game_date="2025-04-05T19:05:00+00:00",
        home_team="NYY",
        away_team="BOS",
        home_starter_id=501,
        away_starter_id=701,
    )
    _seed_game(
        db_path,
        game_pk=8002,
        game_date="2025-04-07T19:05:00+00:00",
        home_team="TOR",
        away_team="NYY",
        home_starter_id=702,
        away_starter_id=502,
    )
    _seed_game(
        db_path,
        game_pk=8003,
        game_date="2025-04-08T19:05:00+00:00",
        home_team="NYY",
        away_team="TB",
        home_starter_id=503,
        away_starter_id=703,
    )
    _seed_game(
        db_path,
        game_pk=9001,
        game_date="2025-04-10T20:05:00+00:00",
        home_team="NYY",
        away_team="BOS",
        home_starter_id=504,
        away_starter_id=704,
    )

    statcast_frame = pd.DataFrame(
        [
            {
                "game_pk": 8001,
                "game_date": "2025-04-05",
                "inning_topbot": "top",
                "pitcher": 501,
                "at_bat_number": 1,
                "pitch_number": 1,
                "events": "field_out",
            },
            {
                "game_pk": 8001,
                "game_date": "2025-04-05",
                "inning_topbot": "top",
                "pitcher": 601,
                "at_bat_number": 2,
                "pitch_number": 1,
                "events": "strikeout",
            },
            {
                "game_pk": 8002,
                "game_date": "2025-04-07",
                "inning_topbot": "bot",
                "pitcher": 502,
                "at_bat_number": 1,
                "pitch_number": 1,
                "events": "field_out",
            },
            {
                "game_pk": 8002,
                "game_date": "2025-04-07",
                "inning_topbot": "bot",
                "pitcher": 602,
                "at_bat_number": 2,
                "pitch_number": 1,
                "events": "strikeout",
            },
            {
                "game_pk": 8003,
                "game_date": "2025-04-08",
                "inning_topbot": "top",
                "pitcher": 503,
                "at_bat_number": 1,
                "pitch_number": 1,
                "events": "field_out",
            },
            {
                "game_pk": 8003,
                "game_date": "2025-04-08",
                "inning_topbot": "top",
                "pitcher": 603,
                "at_bat_number": 2,
                "pitch_number": 1,
                "events": "strikeout",
            },
        ]
    )

    def fake_team_logs_fetcher(
        season: int,
        team: str,
        *,
        log_type: str = "batting",
        refresh: bool = False,
    ) -> pd.DataFrame:
        assert season == 2025
        assert log_type == "pitching"
        assert refresh is False
        if team != "NYY":
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "Pitching_Date": ["2025-04-05", "2025-04-07", "2025-04-08"],
                "Pitching_IR": [2, 1, 3],
                "Pitching_IS": [1, 0, 1],
            }
        )

    monkeypatch.setattr(bullpen, "DERIVED_CACHE_ROOT", tmp_path / "derived_bullpen")
    monkeypatch.setattr(bullpen, "fetch_statcast_range", lambda *_args, **_kwargs: statcast_frame.copy())

    rows = bullpen.compute_bullpen_features(
        "2025-04-10",
        db_path=db_path,
        team_logs_fetcher=fake_team_logs_fetcher,
    )

    by_name = {row.feature_name: row.feature_value for row in rows}
    assert by_name["home_team_bullpen_ir_pct_30g"] == pytest.approx(2 / 6)
    assert by_name["home_team_bullpen_ir_pct_30g"] > 0.0


def test_collapse_plate_appearances_keeps_same_at_bat_number_separate_across_games() -> None:
    from src.features.bullpen import _collapse_plate_appearances

    pitches = pd.DataFrame(
        [
            {
                "game_pk": 1001,
                "pitcher_id": 501,
                "at_bat_number": 1,
                "pitch_number": 1,
                "events": None,
            },
            {
                "game_pk": 1001,
                "pitcher_id": 501,
                "at_bat_number": 1,
                "pitch_number": 2,
                "events": "walk",
            },
            {
                "game_pk": 1002,
                "pitcher_id": 501,
                "at_bat_number": 1,
                "pitch_number": 1,
                "events": None,
            },
            {
                "game_pk": 1002,
                "pitcher_id": 501,
                "at_bat_number": 1,
                "pitch_number": 2,
                "events": "strikeout",
            },
        ]
    )

    collapsed = _collapse_plate_appearances(pitches)

    assert len(collapsed) == 2
    assert set(collapsed["game_pk"].tolist()) == {1001, 1002}
    assert set(collapsed["events"].tolist()) == {"walk", "strikeout"}
