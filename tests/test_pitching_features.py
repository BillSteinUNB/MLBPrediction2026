from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.db import init_db
from src.models.lineup import Lineup


def _seed_game(
    db_path: Path,
    *,
    game_pk: int,
    game_date: str,
    home_team: str,
    away_team: str,
    home_starter_id: int,
    away_starter_id: int,
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


def _start_metrics_by_season() -> dict[int, pd.DataFrame]:
    return {
        2025: pd.DataFrame(
            [
                {
                    "game_pk": 9001,
                    "game_date": "2025-04-01",
                    "team": "NYY",
                    "pitcher_id": 100,
                    "xFIP": 3.0,
                    "xERA": 3.2,
                    "K%": 24.0,
                    "BB%": 6.0,
                    "GB%": 45.0,
                    "HR/FB": 9.0,
                    "avg_fastball_velocity": 96.1,
                    "pitch_mix_entropy": 1.80,
                    "innings_pitched": 6.0,
                },
                {
                    "game_pk": 9002,
                    "game_date": "2025-04-03",
                    "team": "NYY",
                    "pitcher_id": 200,
                    "xFIP": 8.5,
                    "xERA": 6.9,
                    "K%": 12.0,
                    "BB%": 12.0,
                    "GB%": 33.0,
                    "HR/FB": 21.0,
                    "avg_fastball_velocity": 92.0,
                    "pitch_mix_entropy": 1.20,
                    "innings_pitched": 4.0,
                },
                {
                    "game_pk": 9003,
                    "game_date": "2025-04-06",
                    "team": "NYY",
                    "pitcher_id": 100,
                    "xFIP": 4.0,
                    "xERA": 4.1,
                    "K%": 22.0,
                    "BB%": 7.0,
                    "GB%": 43.0,
                    "HR/FB": 10.0,
                    "avg_fastball_velocity": 95.8,
                    "pitch_mix_entropy": 1.75,
                    "innings_pitched": 5.2,
                },
                {
                    "game_pk": 9004,
                    "game_date": "2025-04-08",
                    "team": "NYY",
                    "pitcher_id": 300,
                    "xFIP": 7.5,
                    "xERA": 6.5,
                    "K%": 14.0,
                    "BB%": 11.0,
                    "GB%": 35.0,
                    "HR/FB": 18.0,
                    "avg_fastball_velocity": 91.8,
                    "pitch_mix_entropy": 1.10,
                    "innings_pitched": 4.1,
                },
                {
                    "game_pk": 9010,
                    "game_date": "2025-04-10",
                    "team": "NYY",
                    "pitcher_id": 100,
                    "xFIP": 1.0,
                    "xERA": 1.5,
                    "K%": 35.0,
                    "BB%": 2.0,
                    "GB%": 55.0,
                    "HR/FB": 4.0,
                    "avg_fastball_velocity": 97.4,
                    "pitch_mix_entropy": 2.10,
                    "innings_pitched": 7.0,
                },
                {
                    "game_pk": 9101,
                    "game_date": "2025-04-02",
                    "team": "BOS",
                    "pitcher_id": 400,
                    "xFIP": 3.5,
                    "xERA": 3.4,
                    "K%": 25.0,
                    "BB%": 7.0,
                    "GB%": 44.0,
                    "HR/FB": 8.0,
                    "avg_fastball_velocity": 95.2,
                    "pitch_mix_entropy": 1.60,
                    "innings_pitched": 5.0,
                },
                {
                    "game_pk": 9102,
                    "game_date": "2025-04-07",
                    "team": "BOS",
                    "pitcher_id": 400,
                    "xFIP": 3.7,
                    "xERA": 3.6,
                    "K%": 24.0,
                    "BB%": 6.0,
                    "GB%": 46.0,
                    "HR/FB": 9.0,
                    "avg_fastball_velocity": 95.0,
                    "pitch_mix_entropy": 1.58,
                    "innings_pitched": 6.0,
                },
                {
                    "game_pk": 9201,
                    "game_date": "2025-04-01",
                    "team": "SEA",
                    "pitcher_id": 700,
                    "xFIP": 4.0,
                    "xERA": 3.8,
                    "K%": 21.0,
                    "BB%": 8.0,
                    "GB%": 42.0,
                    "HR/FB": 11.0,
                    "avg_fastball_velocity": 94.0,
                    "pitch_mix_entropy": 1.48,
                    "innings_pitched": 5.0,
                },
                {
                    "game_pk": 9301,
                    "game_date": "2025-03-28",
                    "team": "TB",
                    "pitcher_id": 900,
                    "xFIP": 8.0,
                    "xERA": 6.8,
                    "K%": 18.0,
                    "BB%": 10.0,
                    "GB%": 40.0,
                    "HR/FB": 20.0,
                    "avg_fastball_velocity": 93.5,
                    "pitch_mix_entropy": 1.30,
                    "innings_pitched": 1.0,
                },
                {
                    "game_pk": 9302,
                    "game_date": "2025-04-02",
                    "team": "TB",
                    "pitcher_id": 900,
                    "xFIP": 9.0,
                    "xERA": 7.0,
                    "K%": 17.0,
                    "BB%": 11.0,
                    "GB%": 39.0,
                    "HR/FB": 22.0,
                    "avg_fastball_velocity": 93.1,
                    "pitch_mix_entropy": 1.25,
                    "innings_pitched": 1.1,
                },
                {
                    "game_pk": 9303,
                    "game_date": "2025-04-12",
                    "team": "TB",
                    "pitcher_id": 901,
                    "xFIP": 3.0,
                    "xERA": 3.1,
                    "K%": 27.0,
                    "BB%": 6.0,
                    "GB%": 47.0,
                    "HR/FB": 7.0,
                    "avg_fastball_velocity": 96.0,
                    "pitch_mix_entropy": 1.72,
                    "innings_pitched": 5.0,
                },
                {
                    "game_pk": 9304,
                    "game_date": "2025-04-17",
                    "team": "TB",
                    "pitcher_id": 902,
                    "xFIP": 3.5,
                    "xERA": 3.4,
                    "K%": 26.0,
                    "BB%": 7.0,
                    "GB%": 45.0,
                    "HR/FB": 8.0,
                    "avg_fastball_velocity": 95.6,
                    "pitch_mix_entropy": 1.68,
                    "innings_pitched": 6.0,
                },
                {
                    "game_pk": 9401,
                    "game_date": "2025-04-10",
                    "team": "TOR",
                    "pitcher_id": 950,
                    "xFIP": 4.4,
                    "xERA": 4.1,
                    "K%": 22.0,
                    "BB%": 7.0,
                    "GB%": 44.0,
                    "HR/FB": 11.0,
                    "avg_fastball_velocity": 94.4,
                    "pitch_mix_entropy": 1.57,
                    "innings_pitched": 5.2,
                },
            ]
        ),
        2024: pd.DataFrame(
            [
                {
                    "game_pk": 8001,
                    "game_date": "2024-08-01",
                    "team": "NYY",
                    "pitcher_id": 100,
                    "xFIP": 2.5,
                    "xERA": 2.8,
                    "K%": 28.0,
                    "BB%": 5.0,
                    "GB%": 46.0,
                    "HR/FB": 7.0,
                    "avg_fastball_velocity": 96.4,
                    "pitch_mix_entropy": 1.84,
                    "innings_pitched": 6.1,
                },
                {
                    "game_pk": 8002,
                    "game_date": "2024-08-02",
                    "team": "BOS",
                    "pitcher_id": 400,
                    "xFIP": 3.1,
                    "xERA": 3.0,
                    "K%": 26.0,
                    "BB%": 6.0,
                    "GB%": 45.0,
                    "HR/FB": 8.0,
                    "avg_fastball_velocity": 95.4,
                    "pitch_mix_entropy": 1.62,
                    "innings_pitched": 6.0,
                },
                {
                    "game_pk": 8003,
                    "game_date": "2024-08-03",
                    "team": "SEA",
                    "pitcher_id": 700,
                    "xFIP": 3.2,
                    "xERA": 3.1,
                    "K%": 24.0,
                    "BB%": 7.0,
                    "GB%": 44.0,
                    "HR/FB": 9.0,
                    "avg_fastball_velocity": 94.3,
                    "pitch_mix_entropy": 1.50,
                    "innings_pitched": 5.2,
                },
                {
                    "game_pk": 8004,
                    "game_date": "2024-08-04",
                    "team": "TB",
                    "pitcher_id": 901,
                    "xFIP": 3.4,
                    "xERA": 3.3,
                    "K%": 25.0,
                    "BB%": 6.0,
                    "GB%": 45.0,
                    "HR/FB": 8.0,
                    "avg_fastball_velocity": 95.8,
                    "pitch_mix_entropy": 1.71,
                    "innings_pitched": 5.0,
                },
                {
                    "game_pk": 8005,
                    "game_date": "2024-08-05",
                    "team": "TOR",
                    "pitcher_id": 950,
                    "xFIP": 4.0,
                    "xERA": 3.9,
                    "K%": 23.0,
                    "BB%": 7.0,
                    "GB%": 43.0,
                    "HR/FB": 10.0,
                    "avg_fastball_velocity": 94.1,
                    "pitch_mix_entropy": 1.55,
                    "innings_pitched": 5.1,
                },
            ]
        ),
    }


def _fake_start_metrics_fetcher(metrics_by_season: dict[int, pd.DataFrame]):
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


def _lineup_for(
    *,
    game_pk: int,
    team: str,
    starter_id: int,
    is_opener: bool,
) -> Lineup:
    return Lineup(
        game_pk=game_pk,
        team=team,
        source="test",
        confirmed=True,
        as_of_timestamp=datetime(2025, 4, 20, 12, 0, tzinfo=timezone.utc),
        starting_pitcher_id=starter_id,
        projected_starting_pitcher_id=starter_id,
        is_opener=is_opener,
        is_bullpen_game=is_opener,
    )


def test_compute_pitching_features_uses_starter_starts_only_and_excludes_current_game(
    tmp_path: Path,
) -> None:
    from src.features.pitching import compute_pitching_features

    db_path = tmp_path / "pitching.db"
    init_db(db_path)
    _seed_game(
        db_path,
        game_pk=9010,
        game_date="2025-04-10T20:05:00+00:00",
        home_team="NYY",
        away_team="BOS",
        home_starter_id=100,
        away_starter_id=400,
    )

    rows = compute_pitching_features(
        "2025-04-10",
        db_path=db_path,
        windows=(2,),
        regression_weight=0,
        start_metrics_fetcher=_fake_start_metrics_fetcher(_start_metrics_by_season()),
        lineup_fetcher=lambda *_args, **_kwargs: [],
    )

    by_name = {row.feature_name: row.feature_value for row in rows}
    assert by_name["home_starter_xfip_2s"] == pytest.approx(3.5)
    assert by_name["home_starter_xera_2s"] == pytest.approx(3.65)
    assert by_name["home_starter_avg_fastball_velocity_2s"] == pytest.approx(95.95)
    assert by_name["home_starter_pitch_mix_entropy_2s"] == pytest.approx(1.775)
    assert by_name["away_starter_xfip_2s"] == pytest.approx(3.6)
    assert by_name["home_starter_is_opener"] == 0.0
    assert by_name["home_starter_uses_team_composite"] == 0.0

    with sqlite3.connect(db_path) as connection:
        as_of = connection.execute(
            "SELECT as_of_timestamp FROM features WHERE game_pk = ? AND feature_name = ?",
            (9010, "home_starter_xfip_2s"),
        ).fetchone()[0]

    assert as_of == "2025-04-09T00:00:00+00:00"


def test_compute_pitching_features_applies_pitcher_marcel_blend(tmp_path: Path) -> None:
    from src.features.pitching import compute_pitching_features

    db_path = tmp_path / "pitching.db"
    init_db(db_path)
    _seed_game(
        db_path,
        game_pk=9205,
        game_date="2025-04-02T20:05:00+00:00",
        home_team="SEA",
        away_team="TOR",
        home_starter_id=700,
        away_starter_id=950,
    )

    rows = compute_pitching_features(
        "2025-04-02",
        db_path=db_path,
        windows=(7,),
        regression_weight=15,
        start_metrics_fetcher=_fake_start_metrics_fetcher(_start_metrics_by_season()),
        lineup_fetcher=lambda *_args, **_kwargs: [],
    )

    by_name = {row.feature_name: row.feature_value for row in rows}
    expected_home_xfip = (4.0 * 1 + 3.2 * 15) / 16
    expected_home_xera = (3.8 * 1 + 3.1 * 15) / 16

    assert by_name["home_starter_xfip_7s"] == pytest.approx(expected_home_xfip)
    assert by_name["home_starter_xera_7s"] == pytest.approx(expected_home_xera)
    assert by_name["home_starter_xfip_7s"] < 4.0
    assert by_name["home_starter_xfip_7s"] > 3.2


def test_compute_pitching_features_uses_team_composite_when_lineup_flags_opener(
    tmp_path: Path,
) -> None:
    from src.features.pitching import compute_pitching_features

    db_path = tmp_path / "pitching.db"
    init_db(db_path)
    _seed_game(
        db_path,
        game_pk=9310,
        game_date="2025-04-10T20:05:00+00:00",
        home_team="NYY",
        away_team="BOS",
        home_starter_id=100,
        away_starter_id=400,
    )

    rows = compute_pitching_features(
        "2025-04-10",
        db_path=db_path,
        windows=(2,),
        regression_weight=0,
        start_metrics_fetcher=_fake_start_metrics_fetcher(_start_metrics_by_season()),
        lineup_fetcher=lambda *_args, **_kwargs: [
            _lineup_for(game_pk=9310, team="NYY", starter_id=100, is_opener=True)
        ],
    )

    by_name = {row.feature_name: row.feature_value for row in rows}
    assert by_name["home_starter_is_opener"] == 1.0
    assert by_name["home_starter_uses_team_composite"] == 1.0
    assert by_name["home_starter_xfip_2s"] == pytest.approx((4.0 + 7.5) / 2)
    assert by_name["home_starter_xfip_2s"] != pytest.approx((3.0 + 4.0) / 2)


def test_compute_pitching_features_detects_opener_and_uses_team_composite(
    tmp_path: Path,
) -> None:
    from src.features.pitching import compute_pitching_features

    db_path = tmp_path / "pitching.db"
    init_db(db_path)
    _seed_game(
        db_path,
        game_pk=9305,
        game_date="2025-04-20T20:05:00+00:00",
        home_team="TB",
        away_team="TOR",
        home_starter_id=900,
        away_starter_id=950,
    )

    rows = compute_pitching_features(
        "2025-04-20",
        db_path=db_path,
        windows=(2,),
        regression_weight=0,
        start_metrics_fetcher=_fake_start_metrics_fetcher(_start_metrics_by_season()),
        lineup_fetcher=lambda *_args, **_kwargs: [],
    )

    by_name = {row.feature_name: row.feature_value for row in rows}
    assert by_name["home_starter_is_opener"] == 1.0
    assert by_name["home_starter_uses_team_composite"] == 1.0
    assert by_name["home_starter_xfip_2s"] == pytest.approx((3.0 + 3.5) / 2)
    assert by_name["home_starter_xfip_2s"] != pytest.approx((8.0 + 9.0) / 2)


def test_compute_pitching_features_still_uses_history_based_opener_when_lineup_flag_is_false(
    tmp_path: Path,
) -> None:
    from src.features.pitching import compute_pitching_features

    db_path = tmp_path / "pitching.db"
    init_db(db_path)
    _seed_game(
        db_path,
        game_pk=9311,
        game_date="2025-04-20T20:05:00+00:00",
        home_team="TB",
        away_team="TOR",
        home_starter_id=900,
        away_starter_id=950,
    )

    rows = compute_pitching_features(
        "2025-04-20",
        db_path=db_path,
        windows=(2,),
        regression_weight=0,
        start_metrics_fetcher=_fake_start_metrics_fetcher(_start_metrics_by_season()),
        lineup_fetcher=lambda *_args, **_kwargs: [
            _lineup_for(game_pk=9311, team="TB", starter_id=900, is_opener=False)
        ],
    )

    by_name = {row.feature_name: row.feature_value for row in rows}
    assert by_name["home_starter_is_opener"] == 1.0
    assert by_name["home_starter_uses_team_composite"] == 1.0
    assert by_name["home_starter_xfip_2s"] == pytest.approx((3.0 + 3.5) / 2)
    assert by_name["home_starter_xfip_2s"] != pytest.approx((8.0 + 9.0) / 2)
