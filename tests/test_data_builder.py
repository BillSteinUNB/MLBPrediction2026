from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.db import init_db
from src.features.adjustments.weather import compute_weather_adjustment
from src.features.baselines import compute_baseline_features
from src.features.bullpen import compute_bullpen_features
from src.features.defense import compute_defense_features
from src.features.offense import compute_offensive_features
from src.features.pitching import compute_pitching_features
from src.model.data_builder import (
    assert_training_data_is_leakage_free,
    build_training_dataset,
    resolve_training_years,
)
from src.models.weather import WeatherData


def _schedule_row(
    game_pk: int,
    scheduled_start: str,
    home_team: str,
    away_team: str,
    venue: str,
    *,
    home_starter_id: int = 100,
    away_starter_id: int = 200,
    game_type: str = "R",
    status: str = "final",
    f5_home_score: int = 3,
    f5_away_score: int = 1,
    final_home_score: int = 5,
    final_away_score: int = 2,
) -> dict[str, object]:
    return {
        "game_pk": game_pk,
        "scheduled_start": scheduled_start,
        "home_team": home_team,
        "away_team": away_team,
        "home_starter_id": home_starter_id,
        "away_starter_id": away_starter_id,
        "venue": venue,
        "game_type": game_type,
        "status": status,
        "f5_home_score": f5_home_score,
        "f5_away_score": f5_away_score,
        "final_home_score": final_home_score,
        "final_away_score": final_away_score,
    }


def _seed_schedule(db_path: Path, schedule: pd.DataFrame) -> None:
    init_db(db_path)
    with sqlite3.connect(db_path) as connection:
        for row in schedule.to_dict(orient="records"):
            connection.execute(
                """
                INSERT OR REPLACE INTO games (
                    game_pk,
                    date,
                    home_team,
                    away_team,
                    home_starter_id,
                    away_starter_id,
                    venue,
                    is_dome,
                    is_abs_active,
                    f5_home_score,
                    f5_away_score,
                    final_home_score,
                    final_away_score,
                    status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(row["game_pk"]),
                    str(row["scheduled_start"]),
                    str(row["home_team"]),
                    str(row["away_team"]),
                    row.get("home_starter_id"),
                    row.get("away_starter_id"),
                    str(row["venue"]),
                    0,
                    1,
                    int(row["f5_home_score"]),
                    int(row["f5_away_score"]),
                    int(row["final_home_score"]),
                    int(row["final_away_score"]),
                    str(row["status"]),
                ),
            )
        connection.commit()


def _team_logs_by_season() -> dict[tuple[int, str], pd.DataFrame]:
    return {
        (2024, "NYY"): pd.DataFrame(
            [
                {"Date": "2024-09-25", "Opp": "BOS", "AB": 32, "H": 10, "2B": 2, "3B": 0, "HR": 1, "BB": 4, "SO": 7, "HBP": 1, "SF": 1, "SH": 0},
                {"Date": "2024-09-27", "Opp": "BOS", "AB": 31, "H": 8, "2B": 1, "3B": 0, "HR": 1, "BB": 3, "SO": 8, "HBP": 0, "SF": 1, "SH": 0},
            ]
        ),
        (2024, "BOS"): pd.DataFrame(
            [
                {"Date": "2024-09-25", "Opp": "NYY", "AB": 30, "H": 7, "2B": 1, "3B": 0, "HR": 1, "BB": 3, "SO": 9, "HBP": 0, "SF": 1, "SH": 0},
                {"Date": "2024-09-27", "Opp": "NYY", "AB": 31, "H": 9, "2B": 2, "3B": 0, "HR": 1, "BB": 2, "SO": 8, "HBP": 0, "SF": 1, "SH": 0},
            ]
        ),
        (2025, "NYY"): pd.DataFrame(
            [
                {"Date": "2025-03-31", "Opp": "BOS", "AB": 28, "H": 6, "2B": 1, "3B": 0, "HR": 0, "BB": 2, "SO": 8, "HBP": 0, "SF": 1, "SH": 0},
                {"Date": "2025-04-01", "Opp": "BOS", "AB": 29, "H": 7, "2B": 1, "3B": 0, "HR": 1, "BB": 3, "SO": 8, "HBP": 0, "SF": 1, "SH": 0},
                {"Date": "2025-04-03", "Opp": "BOS", "AB": 30, "H": 8, "2B": 1, "3B": 0, "HR": 1, "BB": 4, "SO": 7, "HBP": 0, "SF": 1, "SH": 0},
                {"Date": "2025-04-08", "Opp": "BOS", "AB": 31, "H": 9, "2B": 2, "3B": 0, "HR": 1, "BB": 3, "SO": 6, "HBP": 0, "SF": 1, "SH": 0},
                {"Date": "2025-04-10", "Opp": "BOS", "AB": 34, "H": 18, "2B": 5, "3B": 1, "HR": 3, "BB": 6, "SO": 2, "HBP": 0, "SF": 1, "SH": 0},
            ]
        ),
        (2025, "BOS"): pd.DataFrame(
            [
                {"Date": "2025-03-31", "Opp": "NYY", "AB": 29, "H": 5, "2B": 1, "3B": 0, "HR": 0, "BB": 2, "SO": 10, "HBP": 0, "SF": 1, "SH": 0},
                {"Date": "2025-04-01", "Opp": "NYY", "AB": 30, "H": 6, "2B": 1, "3B": 0, "HR": 0, "BB": 2, "SO": 9, "HBP": 0, "SF": 1, "SH": 0},
                {"Date": "2025-04-03", "Opp": "NYY", "AB": 31, "H": 7, "2B": 1, "3B": 0, "HR": 1, "BB": 3, "SO": 8, "HBP": 0, "SF": 1, "SH": 0},
                {"Date": "2025-04-08", "Opp": "NYY", "AB": 30, "H": 6, "2B": 1, "3B": 0, "HR": 0, "BB": 2, "SO": 9, "HBP": 0, "SF": 1, "SH": 0},
                {"Date": "2025-04-10", "Opp": "NYY", "AB": 33, "H": 1, "2B": 0, "3B": 0, "HR": 0, "BB": 1, "SO": 15, "HBP": 0, "SF": 0, "SH": 0},
            ]
        ),
    }


def _start_metrics_by_season() -> dict[int, pd.DataFrame]:
    return {
        2024: pd.DataFrame(
            [
                {"game_pk": 9001, "game_date": "2024-09-15", "team": "NYY", "pitcher_id": 100, "xfip": 3.55, "xera": 3.40, "k_pct": 26.0, "bb_pct": 7.0, "gb_pct": 44.0, "hr_fb_pct": 9.0, "avg_fastball_velocity": 95.2, "pitch_mix_entropy": 1.70, "innings_pitched": 6.0},
                {"game_pk": 9002, "game_date": "2024-09-15", "team": "BOS", "pitcher_id": 200, "xfip": 3.85, "xera": 3.75, "k_pct": 24.0, "bb_pct": 8.0, "gb_pct": 43.0, "hr_fb_pct": 10.0, "avg_fastball_velocity": 94.6, "pitch_mix_entropy": 1.58, "innings_pitched": 5.2},
            ]
        ),
        2025: pd.DataFrame(
            [
                {"game_pk": 4099, "game_date": "2025-03-31", "team": "NYY", "pitcher_id": 100, "xfip": 3.70, "xera": 3.60, "k_pct": 24.0, "bb_pct": 7.0, "gb_pct": 43.0, "hr_fb_pct": 10.0, "avg_fastball_velocity": 95.0, "pitch_mix_entropy": 1.68, "innings_pitched": 5.0},
                {"game_pk": 4100, "game_date": "2025-04-01", "team": "NYY", "pitcher_id": 100, "xfip": 3.60, "xera": 3.50, "k_pct": 25.0, "bb_pct": 6.5, "gb_pct": 44.0, "hr_fb_pct": 9.5, "avg_fastball_velocity": 95.1, "pitch_mix_entropy": 1.72, "innings_pitched": 5.2},
                {"game_pk": 4101, "game_date": "2025-04-03", "team": "NYY", "pitcher_id": 100, "xfip": 3.40, "xera": 3.35, "k_pct": 27.0, "bb_pct": 6.0, "gb_pct": 45.0, "hr_fb_pct": 9.0, "avg_fastball_velocity": 95.5, "pitch_mix_entropy": 1.80, "innings_pitched": 6.0},
                {"game_pk": 4102, "game_date": "2025-04-08", "team": "NYY", "pitcher_id": 100, "xfip": 3.20, "xera": 3.10, "k_pct": 28.0, "bb_pct": 5.5, "gb_pct": 46.0, "hr_fb_pct": 8.5, "avg_fastball_velocity": 95.8, "pitch_mix_entropy": 1.85, "innings_pitched": 6.1},
                {"game_pk": 4199, "game_date": "2025-03-31", "team": "BOS", "pitcher_id": 200, "xfip": 4.30, "xera": 4.20, "k_pct": 20.0, "bb_pct": 9.5, "gb_pct": 40.0, "hr_fb_pct": 11.5, "avg_fastball_velocity": 93.8, "pitch_mix_entropy": 1.42, "innings_pitched": 4.2},
                {"game_pk": 4200, "game_date": "2025-04-01", "team": "BOS", "pitcher_id": 200, "xfip": 4.20, "xera": 4.10, "k_pct": 21.0, "bb_pct": 9.0, "gb_pct": 41.0, "hr_fb_pct": 11.0, "avg_fastball_velocity": 94.0, "pitch_mix_entropy": 1.45, "innings_pitched": 5.0},
                {"game_pk": 4201, "game_date": "2025-04-03", "team": "BOS", "pitcher_id": 200, "xfip": 4.10, "xera": 4.00, "k_pct": 22.0, "bb_pct": 8.5, "gb_pct": 41.0, "hr_fb_pct": 10.5, "avg_fastball_velocity": 94.2, "pitch_mix_entropy": 1.48, "innings_pitched": 5.1},
                {"game_pk": 4202, "game_date": "2025-04-08", "team": "BOS", "pitcher_id": 200, "xfip": 3.95, "xera": 3.90, "k_pct": 23.0, "bb_pct": 8.0, "gb_pct": 42.0, "hr_fb_pct": 10.0, "avg_fastball_velocity": 94.4, "pitch_mix_entropy": 1.52, "innings_pitched": 5.2},
                {"game_pk": 4999, "game_date": "2025-04-10", "team": "NYY", "pitcher_id": 100, "xfip": 0.50, "xera": 0.50, "k_pct": 45.0, "bb_pct": 1.0, "gb_pct": 60.0, "hr_fb_pct": 1.0, "avg_fastball_velocity": 98.0, "pitch_mix_entropy": 2.50, "innings_pitched": 7.0},
                {"game_pk": 5999, "game_date": "2025-04-10", "team": "BOS", "pitcher_id": 200, "xfip": 9.50, "xera": 8.80, "k_pct": 8.0, "bb_pct": 15.0, "gb_pct": 30.0, "hr_fb_pct": 18.0, "avg_fastball_velocity": 90.0, "pitch_mix_entropy": 1.00, "innings_pitched": 3.0},
            ]
        ),
    }


def _fielding_by_season() -> dict[int, pd.DataFrame]:
    return {
        2024: pd.DataFrame(
            [
                {"game_date": "2024-09-15", "team": "NYY", "position": "SS", "DRS": 3.0, "OAA": 2.0},
                {"game_date": "2024-09-15", "team": "BOS", "position": "SS", "DRS": 1.0, "OAA": 1.0},
            ]
        ),
        2025: pd.DataFrame(
            [
                {"game_date": "2025-04-03", "team": "NYY", "position": "C", "DRS": 2.0, "OAA": 1.0},
                {"game_date": "2025-04-08", "team": "NYY", "position": "SS", "DRS": 3.0, "OAA": 2.0},
                {"game_date": "2025-04-10", "team": "NYY", "position": "C", "DRS": 20.0, "OAA": 20.0},
                {"game_date": "2025-04-03", "team": "BOS", "position": "C", "DRS": 1.0, "OAA": 1.0},
                {"game_date": "2025-04-08", "team": "BOS", "position": "SS", "DRS": 1.5, "OAA": 1.0},
                {"game_date": "2025-04-10", "team": "BOS", "position": "SS", "DRS": 15.0, "OAA": 15.0},
            ]
        ),
    }


def _framing_by_season() -> dict[int, pd.DataFrame]:
    return {
        2024: pd.DataFrame(
            [
                {"game_date": "2024-09-15", "team": "NYY", "runs_extra_strikes": 3.0},
                {"game_date": "2024-09-15", "team": "BOS", "runs_extra_strikes": 1.0},
            ]
        ),
        2025: pd.DataFrame(
            [
                {"game_date": "2025-04-03", "team": "NYY", "runs_extra_strikes": 2.0},
                {"game_date": "2025-04-08", "team": "NYY", "runs_extra_strikes": 4.0},
                {"game_date": "2025-04-10", "team": "NYY", "runs_extra_strikes": 9.0},
                {"game_date": "2025-04-03", "team": "BOS", "runs_extra_strikes": 1.0},
                {"game_date": "2025-04-08", "team": "BOS", "runs_extra_strikes": 2.0},
                {"game_date": "2025-04-10", "team": "BOS", "runs_extra_strikes": 6.0},
            ]
        ),
    }


def _bullpen_metrics_by_season() -> dict[int, pd.DataFrame]:
    return {
        2025: pd.DataFrame(
            [
                {"game_pk": 4099, "game_date": "2025-03-31", "team": "NYY", "pitcher_id": 301, "pitch_count": 16, "innings_pitched": 1.0, "xfip": 3.9, "inherited_runners": 1, "inherited_runners_scored": 0},
                {"game_pk": 4199, "game_date": "2025-03-31", "team": "BOS", "pitcher_id": 401, "pitch_count": 15, "innings_pitched": 1.0, "xfip": 4.3, "inherited_runners": 1, "inherited_runners_scored": 1},
                {"game_pk": 4101, "game_date": "2025-04-03", "team": "NYY", "pitcher_id": 301, "pitch_count": 18, "innings_pitched": 1.0, "xfip": 3.8, "inherited_runners": 2, "inherited_runners_scored": 1},
                {"game_pk": 4101, "game_date": "2025-04-03", "team": "NYY", "pitcher_id": 302, "pitch_count": 15, "innings_pitched": 1.0, "xfip": 3.4, "inherited_runners": 1, "inherited_runners_scored": 0},
                {"game_pk": 4102, "game_date": "2025-04-08", "team": "NYY", "pitcher_id": 301, "pitch_count": 21, "innings_pitched": 1.0, "xfip": 3.6, "inherited_runners": 1, "inherited_runners_scored": 1},
                {"game_pk": 4102, "game_date": "2025-04-08", "team": "NYY", "pitcher_id": 303, "pitch_count": 12, "innings_pitched": 1.0, "xfip": 3.2, "inherited_runners": 0, "inherited_runners_scored": 0},
                {"game_pk": 4201, "game_date": "2025-04-03", "team": "BOS", "pitcher_id": 401, "pitch_count": 17, "innings_pitched": 1.0, "xfip": 4.2, "inherited_runners": 2, "inherited_runners_scored": 1},
                {"game_pk": 4201, "game_date": "2025-04-03", "team": "BOS", "pitcher_id": 402, "pitch_count": 16, "innings_pitched": 1.0, "xfip": 4.0, "inherited_runners": 0, "inherited_runners_scored": 0},
                {"game_pk": 4202, "game_date": "2025-04-08", "team": "BOS", "pitcher_id": 401, "pitch_count": 19, "innings_pitched": 1.0, "xfip": 4.1, "inherited_runners": 1, "inherited_runners_scored": 0},
                {"game_pk": 4202, "game_date": "2025-04-08", "team": "BOS", "pitcher_id": 403, "pitch_count": 10, "innings_pitched": 1.0, "xfip": 3.9, "inherited_runners": 1, "inherited_runners_scored": 1},
                {"game_pk": 4999, "game_date": "2025-04-10", "team": "NYY", "pitcher_id": 301, "pitch_count": 40, "innings_pitched": 2.0, "xfip": 0.5, "inherited_runners": 5, "inherited_runners_scored": 0},
                {"game_pk": 5999, "game_date": "2025-04-10", "team": "BOS", "pitcher_id": 401, "pitch_count": 42, "innings_pitched": 2.0, "xfip": 9.5, "inherited_runners": 5, "inherited_runners_scored": 5},
            ]
        )
    }


def _fake_team_logs_fetcher(team_logs: dict[tuple[int, str], pd.DataFrame]):
    def _fetcher(season: int, team: str, **_kwargs) -> pd.DataFrame:
        return team_logs.get((season, team), pd.DataFrame()).copy()

    return _fetcher


def _fake_start_metrics_fetcher(metrics_by_season: dict[int, pd.DataFrame]):
    def _fetcher(season: int, *, db_path: Path, end_date, refresh: bool = False) -> pd.DataFrame:
        _ = db_path
        _ = refresh
        dataframe = metrics_by_season.get(season, pd.DataFrame()).copy()
        if dataframe.empty or end_date is None:
            return dataframe
        return dataframe.loc[pd.to_datetime(dataframe["game_date"]).dt.date <= end_date].reset_index(drop=True)

    return _fetcher


def _fake_fielding_fetcher(frames_by_season: dict[int, pd.DataFrame]):
    def _fetcher(season: int, **_kwargs) -> pd.DataFrame:
        return frames_by_season.get(season, pd.DataFrame()).copy()

    return _fetcher


def _fake_framing_fetcher(frames_by_season: dict[int, pd.DataFrame]):
    def _fetcher(season: int, **_kwargs) -> pd.DataFrame:
        return frames_by_season.get(season, pd.DataFrame()).copy()

    return _fetcher


def _fake_bullpen_metrics_fetcher(metrics_by_season: dict[int, pd.DataFrame]):
    def _fetcher(season: int, *, db_path: Path, end_date, refresh: bool = False) -> pd.DataFrame:
        _ = db_path
        _ = refresh
        dataframe = metrics_by_season.get(season, pd.DataFrame()).copy()
        if dataframe.empty or end_date is None:
            return dataframe
        return dataframe.loc[pd.to_datetime(dataframe["game_date"]).dt.date <= end_date].reset_index(drop=True)

    return _fetcher


def _fake_weather_fetcher(team_abbr: str, game_datetime, **_kwargs) -> WeatherData:
    _ = team_abbr
    _ = game_datetime
    return WeatherData(
        temperature_f=81.0,
        humidity_pct=62.0,
        wind_speed_mph=12.0,
        wind_direction_deg=198.0,
        pressure_hpa=1009.0,
        air_density=1.17,
        wind_factor=-11.4,
        is_dome_default=False,
        forecast_time=datetime(2025, 4, 10, 18, 0, tzinfo=timezone.utc),
        fetched_at=datetime(2025, 4, 10, 16, 0, tzinfo=timezone.utc),
    )


def _direct_module_feature_values(
    *,
    db_path: Path,
    target_date: str,
    team_logs: dict[tuple[int, str], pd.DataFrame],
    start_metrics: dict[int, pd.DataFrame],
    fielding: dict[int, pd.DataFrame],
    framing: dict[int, pd.DataFrame],
    bullpen_metrics: dict[int, pd.DataFrame],
) -> dict[str, float]:
    rows = [
        *compute_offensive_features(
            target_date,
            db_path=db_path,
            team_logs_fetcher=_fake_team_logs_fetcher(team_logs),
            batting_stats_fetcher=lambda *_args, **_kwargs: pd.DataFrame(),
        ),
        *compute_pitching_features(
            target_date,
            db_path=db_path,
            start_metrics_fetcher=_fake_start_metrics_fetcher(start_metrics),
            lineup_fetcher=lambda *_args, **_kwargs: [],
        ),
        *compute_defense_features(
            target_date,
            db_path=db_path,
            fielding_fetcher=_fake_fielding_fetcher(fielding),
            framing_fetcher=_fake_framing_fetcher(framing),
            team_logs_fetcher=_fake_team_logs_fetcher(team_logs),
        ),
        *compute_bullpen_features(
            target_date,
            db_path=db_path,
            bullpen_metrics_fetcher=_fake_bullpen_metrics_fetcher(bullpen_metrics),
        ),
        *compute_baseline_features(target_date, db_path=db_path),
    ]
    return {row.feature_name: row.feature_value for row in rows if row.game_pk == 4002}


def test_build_training_dataset_integrates_real_feature_modules_and_matches_inference_columns(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "training_sample.parquet"
    schedule = pd.DataFrame(
        [
            _schedule_row(
                4001,
                "2025-04-08T23:05:00Z",
                "NYY",
                "BOS",
                "Yankee Stadium",
                f5_home_score=2,
                f5_away_score=1,
                final_home_score=5,
                final_away_score=3,
            ),
            _schedule_row(
                4002,
                "2025-04-10T23:05:00Z",
                "NYY",
                "BOS",
                "Yankee Stadium",
                f5_home_score=3,
                f5_away_score=1,
                final_home_score=4,
                final_away_score=2,
            ),
            _schedule_row(
                4998,
                "2025-03-10T17:05:00Z",
                "NYY",
                "BOS",
                "Spring Park",
                game_type="S",
            ),
            _schedule_row(
                4999,
                "2025-10-10T23:05:00Z",
                "NYY",
                "BOS",
                "Yankee Stadium",
                game_type="F",
            ),
        ]
    )
    team_logs = _team_logs_by_season()
    start_metrics = _start_metrics_by_season()
    fielding = _fielding_by_season()
    framing = _framing_by_season()
    bullpen_metrics = _bullpen_metrics_by_season()

    direct_db_path = tmp_path / "direct_modules.db"
    _seed_schedule(direct_db_path, schedule.loc[schedule["game_type"] == "R"].copy())
    expected_feature_values = _direct_module_feature_values(
        db_path=direct_db_path,
        target_date="2025-04-10",
        team_logs=team_logs,
        start_metrics=start_metrics,
        fielding=fielding,
        framing=framing,
        bullpen_metrics=bullpen_metrics,
    )

    result = build_training_dataset(
        start_year=2025,
        end_year=2025,
        output_path=output_path,
        full_regular_seasons_target=1,
        shortened_season_game_threshold=0,
        schedule_fetcher=lambda year: schedule.copy(),
        batting_stats_fetcher=lambda *_args, **_kwargs: pd.DataFrame(),
        team_logs_fetcher=_fake_team_logs_fetcher(team_logs),
        fielding_stats_fetcher=_fake_fielding_fetcher(fielding),
        framing_stats_fetcher=_fake_framing_fetcher(framing),
        start_metrics_fetcher=_fake_start_metrics_fetcher(start_metrics),
        bullpen_metrics_fetcher=_fake_bullpen_metrics_fetcher(bullpen_metrics),
        lineup_fetcher=lambda *_args, **_kwargs: [],
        weather_fetcher=_fake_weather_fetcher,
    )

    dataset = result.dataframe
    assert dataset["game_pk"].tolist() == [4001, 4002]
    assert set(dataset["game_type"]) == {"R"}
    assert dataset["f5_ml_result"].tolist() == [1, 1]
    assert dataset["f5_rl_result"].tolist() == [0, 1]
    assert_training_data_is_leakage_free(dataset)

    target_row = dataset.loc[dataset["game_pk"] == 4002].iloc[0]
    for feature_name, expected_value in expected_feature_values.items():
        assert feature_name in dataset.columns
        assert target_row[feature_name] == pytest.approx(expected_value)

    expected_weather = compute_weather_adjustment(
        _fake_weather_fetcher("NYY", "2025-04-10T23:05:00Z"),
        team_code="NYY",
        venue="Yankee Stadium",
        is_dome=False,
    )
    assert target_row["weather_temp_factor"] == pytest.approx(expected_weather.temp_factor)
    assert target_row["weather_air_density_factor"] == pytest.approx(
        expected_weather.air_density_factor
    )
    assert target_row["weather_humidity_factor"] == pytest.approx(expected_weather.humidity_factor)
    assert target_row["weather_wind_factor"] == pytest.approx(expected_weather.wind_factor)
    assert target_row["weather_rain_risk"] == pytest.approx(expected_weather.rain_risk)
    assert target_row["weather_composite"] == pytest.approx(expected_weather.weather_composite)
    assert target_row["weather_data_missing"] == 0.0

    assert "home_team_woba_7g" in dataset.columns
    assert "home_starter_xfip_7s" in dataset.columns
    assert "home_team_drs_season" in dataset.columns
    assert "home_team_bullpen_xfip" in dataset.columns
    assert "home_team_log5_30g" in dataset.columns
    assert "home_offense_runs_scored_7g" not in dataset.columns
    assert "home_starter_xfip_prior" not in dataset.columns
    assert "home_team_wrc_plus_prior" not in dataset.columns
    assert output_path.exists()
    assert result.metadata_path.exists()

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["row_count"] == 2
    assert metadata["feature_column_count"] >= len(expected_feature_values) + 9
    assert metadata["data_version_hash"] == dataset["data_version_hash"].iat[0]
    assert dataset["data_version_hash"].nunique() == 1


def test_build_training_dataset_keeps_same_day_doubleheaders_on_prior_day_feature_snapshot(
    tmp_path: Path,
) -> None:
    schedule = pd.DataFrame(
        [
            _schedule_row(
                3001,
                "2025-04-01T23:05:00Z",
                "NYY",
                "BOS",
                "Yankee Stadium",
                f5_home_score=2,
                f5_away_score=0,
                final_home_score=6,
                final_away_score=2,
            ),
            _schedule_row(
                3002,
                "2025-04-02T17:05:00Z",
                "NYY",
                "BOS",
                "Yankee Stadium",
                f5_home_score=4,
                f5_away_score=1,
                final_home_score=8,
                final_away_score=3,
            ),
            _schedule_row(
                3003,
                "2025-04-02T23:05:00Z",
                "NYY",
                "BOS",
                "Yankee Stadium",
                f5_home_score=1,
                f5_away_score=5,
                final_home_score=2,
                final_away_score=9,
            ),
        ]
    )
    team_logs = _team_logs_by_season()
    start_metrics = _start_metrics_by_season()
    fielding = _fielding_by_season()
    framing = _framing_by_season()
    bullpen_metrics = _bullpen_metrics_by_season()

    dataset = build_training_dataset(
        start_year=2025,
        end_year=2025,
        output_path=tmp_path / "doubleheader_sample.parquet",
        full_regular_seasons_target=1,
        shortened_season_game_threshold=0,
        schedule_fetcher=lambda year: schedule.copy(),
        batting_stats_fetcher=lambda *_args, **_kwargs: pd.DataFrame(),
        team_logs_fetcher=_fake_team_logs_fetcher(team_logs),
        fielding_stats_fetcher=_fake_fielding_fetcher(fielding),
        framing_stats_fetcher=_fake_framing_fetcher(framing),
        start_metrics_fetcher=_fake_start_metrics_fetcher(start_metrics),
        bullpen_metrics_fetcher=_fake_bullpen_metrics_fetcher(bullpen_metrics),
        lineup_fetcher=lambda *_args, **_kwargs: [],
        weather_fetcher=_fake_weather_fetcher,
    ).dataframe

    earlier_game = dataset.loc[dataset["game_pk"] == 3002].iloc[0]
    later_game = dataset.loc[dataset["game_pk"] == 3003].iloc[0]

    assert later_game["as_of_timestamp"] == "2025-04-01T00:00:00+00:00"
    assert earlier_game["as_of_timestamp"] == later_game["as_of_timestamp"]
    assert later_game["home_team_woba_7g"] == pytest.approx(earlier_game["home_team_woba_7g"])
    assert later_game["home_starter_xfip_7s"] == pytest.approx(
        earlier_game["home_starter_xfip_7s"]
    )
    assert later_game["home_team_bullpen_xfip"] == pytest.approx(
        earlier_game["home_team_bullpen_xfip"]
    )
    assert later_game["home_team_log5_30g"] == pytest.approx(earlier_game["home_team_log5_30g"])


def test_resolve_training_years_backfills_shortened_season_with_previous_full_year() -> None:
    resolved_years = resolve_training_years(
        start_year=2019,
        end_year=2025,
        full_regular_seasons_target=7,
        season_row_counts={2019: 2430, 2020: 898, 2021: 2430, 2022: 2430, 2023: 2430, 2024: 2430, 2025: 2430},
        shortened_season_game_threshold=2000,
    )

    assert resolved_years == [2018, 2019, 2021, 2022, 2023, 2024, 2025]
