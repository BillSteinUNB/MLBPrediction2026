from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.features.adjustments.abs_adjustment import (
    DEFAULT_STRIKEOUT_RATE_DELTA,
    DEFAULT_WALK_RATE_DELTA,
)
from src.features.adjustments.weather import (
    NEUTRAL_WEATHER_FACTOR,
    compute_weather_adjustment,
)
from src.features.bullpen import (
    DEFAULT_AVG_REST_DAYS,
    DEFAULT_TOP_RELIEVER_COUNT,
    DEFAULT_XFIP,
)
from src.features.defense import DEFAULT_DEFENSIVE_EFFICIENCY
from src.features.offense import LEAGUE_WOBA_BASELINE, LEAGUE_WRC_PLUS_BASELINE
from src.features.pitching import DEFAULT_METRIC_BASELINES
from src.db import init_db
from src.features.baselines import compute_baseline_features
from src.features.bullpen import compute_bullpen_features
from src.features.defense import compute_defense_features
from src.features.offense import compute_offensive_features
from src.features.pitching import compute_pitching_features
from src.model.data_builder import (
    _compute_team_batting_splits,
    _derive_matchup_interaction_features,
    _feature_rows_to_frame,
    _fill_missing_feature_values,
    assert_training_data_is_complete,
    assert_training_data_is_leakage_free,
    build_live_feature_frame,
    build_training_dataset,
    resolve_training_years,
)
from src.models.lineup import Lineup, LineupPlayer
from src.models.weather import WeatherData


_VALIDATION_FIXTURE_SEASONS = (2018, 2019, 2021, 2022, 2023, 2024, 2025)
_VALIDATION_FIXTURE_CACHE: dict[int, pd.DataFrame] = {}


def _fake_team_batting_split_statcast_frame(game_date: str = "2025-04-09") -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for at_bat_number in range(1, 31):
        rows.append(
            {
                "game_pk": 4499,
                "game_date": game_date,
                "at_bat_number": at_bat_number,
                "pitch_number": 1,
                "inning_topbot": "Top",
                "away_team": "NYY",
                "home_team": "BOS",
                "p_throws": "L",
                "events": "single",
                "estimated_woba_using_speedangle": 0.410,
            }
        )
    for at_bat_number in range(31, 61):
        rows.append(
            {
                "game_pk": 4499,
                "game_date": game_date,
                "at_bat_number": at_bat_number,
                "pitch_number": 1,
                "inning_topbot": "Top",
                "away_team": "NYY",
                "home_team": "BOS",
                "p_throws": "R",
                "events": "single",
                "estimated_woba_using_speedangle": 0.330,
            }
        )
    for at_bat_number in range(61, 91):
        rows.append(
            {
                "game_pk": 4499,
                "game_date": game_date,
                "at_bat_number": at_bat_number,
                "pitch_number": 1,
                "inning_topbot": "Bot",
                "away_team": "NYY",
                "home_team": "BOS",
                "p_throws": "L",
                "events": "single",
                "estimated_woba_using_speedangle": 0.290,
            }
        )
    for at_bat_number in range(91, 121):
        rows.append(
            {
                "game_pk": 4499,
                "game_date": game_date,
                "at_bat_number": at_bat_number,
                "pitch_number": 1,
                "inning_topbot": "Bot",
                "away_team": "NYY",
                "home_team": "BOS",
                "p_throws": "R",
                "events": "single",
                "estimated_woba_using_speedangle": 0.350,
            }
        )

    return pd.DataFrame(rows)


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
        precipitation_probability=0.4,
        precipitation_mm=1.25,
        cloud_cover_pct=76.0,
        is_dome_default=False,
        forecast_time=datetime(2025, 4, 10, 18, 0, tzinfo=timezone.utc),
        fetched_at=datetime(2025, 4, 10, 16, 0, tzinfo=timezone.utc),
    )


def _shift_fixture_year(value: str, year: int) -> str:
    timestamp = pd.Timestamp(value)
    shifted = timestamp.replace(year=year)
    if "T" not in str(value):
        return shifted.date().isoformat()
    if shifted.tzinfo is None:
        shifted = shifted.tz_localize("UTC")
    else:
        shifted = shifted.tz_convert("UTC")
    return shifted.isoformat()


def _build_validation_schedule_by_year(rows_per_season: int) -> dict[int, pd.DataFrame]:
    games_per_day = 15
    schedule_by_year: dict[int, pd.DataFrame] = {}

    for season in range(2018, 2026):
        if season == 2020:
            schedule_by_year[season] = pd.DataFrame()
            continue

        season_base = pd.Timestamp(f"{season}-04-01T18:05:00Z")
        rows: list[dict[str, object]] = []
        for game_offset in range(rows_per_season):
            scheduled_start = season_base + pd.Timedelta(days=game_offset // games_per_day)
            scheduled_start += pd.Timedelta(minutes=(game_offset % games_per_day) * 20)

            if game_offset % 3 == 0:
                f5_home_score, f5_away_score = 4, 1
                final_home_score, final_away_score = 6, 2
            elif game_offset % 2 == 0:
                f5_home_score, f5_away_score = 2, 1
                final_home_score, final_away_score = 4, 3
            else:
                f5_home_score, f5_away_score = 1, 3
                final_home_score, final_away_score = 3, 5

            rows.append(
                _schedule_row(
                    game_pk=(season * 10_000) + game_offset,
                    scheduled_start=scheduled_start.isoformat(),
                    home_team="NYY",
                    away_team="BOS",
                    venue="Yankee Stadium",
                    f5_home_score=f5_home_score,
                    f5_away_score=f5_away_score,
                    final_home_score=final_home_score,
                    final_away_score=final_away_score,
                )
            )

        schedule_by_year[season] = pd.DataFrame(rows)

    return schedule_by_year


def _build_validation_team_logs_by_season() -> dict[tuple[int, str], pd.DataFrame]:
    templates = _team_logs_by_season()
    frames_by_season: dict[tuple[int, str], pd.DataFrame] = {}

    for season in range(2017, 2026):
        for team in ("NYY", "BOS"):
            frame = templates[(2025, team)].copy()
            frame["Date"] = frame["Date"].map(lambda value: _shift_fixture_year(value, season))
            frames_by_season[(season, team)] = frame

    return frames_by_season


def _build_validation_start_metrics_by_season() -> dict[int, pd.DataFrame]:
    template = _start_metrics_by_season()[2025]
    frames_by_season: dict[int, pd.DataFrame] = {}

    for season in range(2017, 2026):
        frame = template.copy()
        frame["game_date"] = frame["game_date"].map(
            lambda value: _shift_fixture_year(value, season)
        )
        frame["game_pk"] = frame["game_pk"].map(
            lambda value: (season * 10_000) + (int(value) % 10_000)
        )
        frames_by_season[season] = frame

    return frames_by_season


def _build_validation_fielding_by_season() -> dict[int, pd.DataFrame]:
    template = _fielding_by_season()[2025]
    frames_by_season: dict[int, pd.DataFrame] = {}

    for season in range(2017, 2026):
        frame = template.copy()
        frame["game_date"] = frame["game_date"].map(
            lambda value: _shift_fixture_year(value, season)
        )
        frames_by_season[season] = frame

    return frames_by_season


def _build_validation_framing_by_season() -> dict[int, pd.DataFrame]:
    template = _framing_by_season()[2025]
    frames_by_season: dict[int, pd.DataFrame] = {}

    for season in range(2017, 2026):
        frame = template.copy()
        frame["game_date"] = frame["game_date"].map(
            lambda value: _shift_fixture_year(value, season)
        )
        frames_by_season[season] = frame

    return frames_by_season


def _build_validation_bullpen_metrics_by_season() -> dict[int, pd.DataFrame]:
    template = _bullpen_metrics_by_season()[2025]
    frames_by_season: dict[int, pd.DataFrame] = {}

    for season in range(2017, 2026):
        frame = template.copy()
        frame["game_date"] = frame["game_date"].map(
            lambda value: _shift_fixture_year(value, season)
        )
        frame["game_pk"] = frame["game_pk"].map(
            lambda value: (season * 10_000) + (int(value) % 10_000)
        )
        frames_by_season[season] = frame

    return frames_by_season


def _write_cached_training_validation_fixture(
    output_path: Path,
    *,
    rows_per_season: int = 2430,
) -> pd.DataFrame:
    dataframe = _VALIDATION_FIXTURE_CACHE.get(rows_per_season)
    if dataframe is None:
        schedule_by_year = _build_validation_schedule_by_year(rows_per_season)
        dataframe = build_training_dataset(
            start_year=2018,
            end_year=2025,
            output_path=output_path,
            full_regular_seasons_target=len(_VALIDATION_FIXTURE_SEASONS),
            shortened_season_game_threshold=1,
            schedule_fetcher=lambda year: schedule_by_year.get(year, pd.DataFrame()).copy(),
            batting_stats_fetcher=lambda *_args, **_kwargs: pd.DataFrame(),
            team_logs_fetcher=_fake_team_logs_fetcher(_build_validation_team_logs_by_season()),
            fielding_stats_fetcher=_fake_fielding_fetcher(_build_validation_fielding_by_season()),
            framing_stats_fetcher=_fake_framing_fetcher(_build_validation_framing_by_season()),
            start_metrics_fetcher=_fake_start_metrics_fetcher(
                _build_validation_start_metrics_by_season()
            ),
            bullpen_metrics_fetcher=_fake_bullpen_metrics_fetcher(
                _build_validation_bullpen_metrics_by_season()
            ),
            lineup_fetcher=lambda *_args, **_kwargs: [],
            weather_fetcher=_fake_weather_fetcher,
        ).dataframe
        _VALIDATION_FIXTURE_CACHE[rows_per_season] = dataframe.copy()

    dataframe = dataframe.copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(output_path, index=False)
    return dataframe


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
        precipitation_probability=0.4,
    )
    assert target_row["weather_temp_factor"] == pytest.approx(expected_weather.temp_factor)
    assert target_row["weather_air_density_factor"] == pytest.approx(
        expected_weather.air_density_factor
    )
    assert target_row["weather_humidity_factor"] == pytest.approx(expected_weather.humidity_factor)
    assert target_row["weather_wind_factor"] == pytest.approx(expected_weather.wind_factor)
    assert target_row["weather_rain_risk"] == pytest.approx(expected_weather.rain_risk)
    assert target_row["weather_composite"] == pytest.approx(expected_weather.weather_composite)
    assert target_row["weather_precip_probability"] == pytest.approx(0.4)
    assert target_row["weather_precipitation_mm"] == pytest.approx(1.25)
    assert target_row["weather_cloud_cover_pct"] == pytest.approx(76.0)
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


def test_build_training_dataset_attaches_posted_f5_runline_targets(tmp_path: Path) -> None:
    schedule = pd.DataFrame(
        [
            _schedule_row(
                4001,
                "2025-04-03T23:05:00Z",
                "NYY",
                "BOS",
                "Yankee Stadium",
                f5_home_score=3,
                f5_away_score=2,
                final_home_score=5,
                final_away_score=4,
            ),
            _schedule_row(
                4002,
                "2025-04-10T23:05:00Z",
                "NYY",
                "BOS",
                "Yankee Stadium",
                f5_home_score=2,
                f5_away_score=1,
                final_home_score=4,
                final_away_score=2,
            ),
        ]
    )
    odds_db_path = tmp_path / "historical_odds.db"
    _seed_schedule(odds_db_path, schedule.copy())
    with sqlite3.connect(odds_db_path) as connection:
        connection.executemany(
            """
            INSERT INTO odds_snapshots (
                game_pk,
                book_name,
                market_type,
                home_odds,
                away_odds,
                home_point,
                away_point,
                fetched_at,
                is_frozen
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (4001, "sbr:caesars", "f5_rl", 105, -125, -0.5, 0.5, "2025-04-03T23:05:00+00:00", 1),
                (4002, "sbr:caesars", "f5_rl", 120, -140, -1.0, 1.0, "2025-04-10T23:05:00+00:00", 1),
            ],
        )
        connection.commit()

    result = build_training_dataset(
        start_year=2025,
        end_year=2025,
        output_path=tmp_path / "training.parquet",
        full_regular_seasons_target=1,
        shortened_season_game_threshold=0,
        schedule_fetcher=lambda year: schedule.copy(),
        batting_stats_fetcher=lambda *_args, **_kwargs: pd.DataFrame(),
        team_logs_fetcher=lambda *_args, **_kwargs: pd.DataFrame(),
        fielding_stats_fetcher=lambda *_args, **_kwargs: pd.DataFrame(),
        framing_stats_fetcher=lambda *_args, **_kwargs: pd.DataFrame(),
        lineup_fetcher=lambda *_args, **_kwargs: [],
        historical_odds_db_path=odds_db_path,
        historical_rl_book_name="sbr:caesars",
    )

    dataset = result.dataframe.set_index("game_pk")
    assert dataset.loc[4001, "posted_f5_rl_home_point"] == pytest.approx(-0.5)
    assert dataset.loc[4001, "posted_f5_rl_away_odds"] == -125
    assert dataset.loc[4001, "home_cover_at_posted_line"] == 1
    assert dataset.loc[4001, "away_cover_at_posted_line"] == 0
    assert dataset.loc[4001, "push_at_posted_line"] == 0
    assert dataset.loc[4002, "posted_f5_rl_home_point"] == pytest.approx(-1.0)
    assert dataset.loc[4002, "home_cover_at_posted_line"] == 0
    assert dataset.loc[4002, "away_cover_at_posted_line"] == 0
    assert dataset.loc[4002, "push_at_posted_line"] == 1


def test_feature_rows_to_frame_uses_latest_as_of_timestamp_per_feature() -> None:
    feature_rows = pd.DataFrame(
        [
            {
                "id": 2,
                "game_pk": 111,
                "feature_name": "home_team_log5_30g",
                "feature_value": 0.44,
                "as_of_timestamp": "2024-04-01T00:00:00+00:00",
            },
            {
                "id": 1,
                "game_pk": 111,
                "feature_name": "home_team_log5_30g",
                "feature_value": 0.41,
                "as_of_timestamp": "2024-03-31T00:00:00+00:00",
            },
            {
                "id": 3,
                "game_pk": 111,
                "feature_name": "park_runs_factor",
                "feature_value": 1.02,
                "as_of_timestamp": "2024-03-31T00:00:00+00:00",
            },
        ]
    )

    feature_frame = _feature_rows_to_frame(feature_rows)

    assert len(feature_frame) == 1
    assert float(feature_frame.loc[0, "home_team_log5_30g"]) == pytest.approx(0.44)
    assert float(feature_frame.loc[0, "park_runs_factor"]) == pytest.approx(1.02)


@pytest.mark.parametrize(
    ("refresh_raw_data", "expected_refresh"),
    [(False, False), (True, True)],
)
def test_build_training_dataset_separates_training_refresh_from_raw_data_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    refresh_raw_data: bool,
    expected_refresh: bool,
) -> None:
    monkeypatch.setattr(
        "src.model.data_builder.fetch_statcast_range",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )

    schedule = pd.DataFrame(
        [
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
            )
        ]
    )
    team_logs = _team_logs_by_season()
    start_metrics = _start_metrics_by_season()
    fielding = _fielding_by_season()
    framing = _framing_by_season()
    bullpen_metrics = _bullpen_metrics_by_season()
    refresh_calls: dict[str, list[bool]] = {
        "batting": [],
        "team_logs": [],
        "fielding": [],
        "framing": [],
        "start_metrics": [],
        "bullpen": [],
    }

    def _batting_fetcher(season: int, **kwargs) -> pd.DataFrame:
        refresh_calls["batting"].append(bool(kwargs.get("refresh", False)))
        if season == 2024:
            return pd.DataFrame(
                {
                    "player_id": [1, 2, 3, 4, 5, 201, 202, 203, 204, 205],
                    "Team": ["NYY"] * 5 + ["BOS"] * 5,
                    "PA": [200] * 10,
                    "wRC+": [100.0] * 10,
                    "wOBA": [0.320] * 10,
                    "ISO": [0.170] * 10,
                    "BABIP": [0.300] * 10,
                    "K%": [22.0] * 10,
                    "BB%": [8.0] * 10,
                }
            )
        return pd.DataFrame()

    def _team_logs_fetcher(season: int, team: str, **kwargs) -> pd.DataFrame:
        refresh_calls["team_logs"].append(bool(kwargs.get("refresh", False)))
        return team_logs.get((season, team), pd.DataFrame()).copy()

    def _fielding_fetcher(season: int, **kwargs) -> pd.DataFrame:
        refresh_calls["fielding"].append(bool(kwargs.get("refresh", False)))
        return fielding.get(season, pd.DataFrame()).copy()

    def _framing_fetcher(season: int, **kwargs) -> pd.DataFrame:
        refresh_calls["framing"].append(bool(kwargs.get("refresh", False)))
        return framing.get(season, pd.DataFrame()).copy()

    def _start_metrics_fetcher(
        season: int,
        *,
        db_path: Path,
        end_date,
        refresh: bool = False,
    ) -> pd.DataFrame:
        _ = db_path
        refresh_calls["start_metrics"].append(bool(refresh))
        dataframe = start_metrics.get(season, pd.DataFrame()).copy()
        if dataframe.empty or end_date is None:
            return dataframe
        return dataframe.loc[
            pd.to_datetime(dataframe["game_date"]).dt.date <= end_date
        ].reset_index(drop=True)

    def _bullpen_metrics_fetcher(
        season: int,
        *,
        db_path: Path,
        end_date,
        refresh: bool = False,
    ) -> pd.DataFrame:
        _ = db_path
        refresh_calls["bullpen"].append(bool(refresh))
        dataframe = bullpen_metrics.get(season, pd.DataFrame()).copy()
        if dataframe.empty or end_date is None:
            return dataframe
        return dataframe.loc[
            pd.to_datetime(dataframe["game_date"]).dt.date <= end_date
        ].reset_index(drop=True)

    result = build_training_dataset(
        start_year=2025,
        end_year=2025,
        output_path=tmp_path / f"refresh_behavior_{refresh_raw_data}.parquet",
        full_regular_seasons_target=1,
        shortened_season_game_threshold=0,
        refresh=True,
        refresh_raw_data=refresh_raw_data,
        schedule_fetcher=lambda _year: schedule.copy(),
        batting_stats_fetcher=_batting_fetcher,
        team_logs_fetcher=_team_logs_fetcher,
        fielding_stats_fetcher=_fielding_fetcher,
        framing_stats_fetcher=_framing_fetcher,
        start_metrics_fetcher=_start_metrics_fetcher,
        bullpen_metrics_fetcher=_bullpen_metrics_fetcher,
        lineup_fetcher=lambda *_args, **_kwargs: [
            Lineup(
                game_pk=4002,
                team="NYY",
                source="test",
                confirmed=True,
                as_of_timestamp=datetime(2025, 4, 10, 18, 0, tzinfo=timezone.utc),
                players=[
                    LineupPlayer(
                        batting_order=index + 1,
                        player_id=value,
                        player_name=f"NYY {value}",
                    )
                    for index, value in enumerate((1, 2, 3, 4, 5))
                ],
                starting_pitcher_id=100,
            ),
            Lineup(
                game_pk=4002,
                team="BOS",
                source="test",
                confirmed=True,
                as_of_timestamp=datetime(2025, 4, 10, 18, 0, tzinfo=timezone.utc),
                players=[
                    LineupPlayer(
                        batting_order=index + 1,
                        player_id=value,
                        player_name=f"BOS {value}",
                    )
                    for index, value in enumerate((201, 202, 203, 204, 205))
                ],
                starting_pitcher_id=200,
            ),
        ],
        weather_fetcher=_fake_weather_fetcher,
    )

    assert result.dataframe["game_pk"].tolist() == [4002]
    for fetcher_name, calls in refresh_calls.items():
        assert calls, f"{fetcher_name} fetcher was not exercised"
        assert set(calls) == {expected_refresh}


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


def test_build_training_dataset_uses_live_weather_fetcher_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule = pd.DataFrame(
        [
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
            )
        ]
    )
    team_logs = _team_logs_by_season()
    start_metrics = _start_metrics_by_season()
    fielding = _fielding_by_season()
    framing = _framing_by_season()
    bullpen_metrics = _bullpen_metrics_by_season()

    monkeypatch.setattr("src.model.data_builder.fetch_game_weather", _fake_weather_fetcher)

    dataset = build_training_dataset(
        start_year=2025,
        end_year=2025,
        output_path=tmp_path / "default_weather_sample.parquet",
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
    ).dataframe

    target_row = dataset.iloc[0]
    expected_weather = compute_weather_adjustment(
        _fake_weather_fetcher("NYY", "2025-04-10T23:05:00Z"),
        team_code="NYY",
        venue="Yankee Stadium",
        is_dome=False,
        precipitation_probability=0.4,
    )

    assert target_row["weather_temp_factor"] == pytest.approx(expected_weather.temp_factor)
    assert target_row["weather_air_density_factor"] == pytest.approx(
        expected_weather.air_density_factor
    )
    assert target_row["weather_humidity_factor"] == pytest.approx(expected_weather.humidity_factor)
    assert target_row["weather_wind_factor"] == pytest.approx(expected_weather.wind_factor)
    assert target_row["weather_rain_risk"] == pytest.approx(expected_weather.rain_risk)
    assert target_row["weather_composite"] == pytest.approx(expected_weather.weather_composite)
    assert target_row["weather_precip_probability"] == pytest.approx(0.4)
    assert target_row["weather_precipitation_mm"] == pytest.approx(1.25)
    assert target_row["weather_cloud_cover_pct"] == pytest.approx(76.0)
    assert target_row["weather_data_missing"] == 0.0


def test_build_training_dataset_skips_roster_turnover_without_lineup_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, dict[str, float] | None] = {}

    def capture_turnover(name: str):
        def _capture(*_args, **kwargs):
            captured[name] = kwargs.get("roster_turnover_by_team")
            return []

        return _capture

    monkeypatch.setattr("src.model.data_builder.compute_offensive_features", capture_turnover("offense"))
    monkeypatch.setattr("src.model.data_builder.compute_pitching_features", capture_turnover("pitching"))
    monkeypatch.setattr("src.model.data_builder.compute_defense_features", capture_turnover("defense"))
    monkeypatch.setattr("src.model.data_builder.compute_bullpen_features", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("src.model.data_builder.compute_baseline_features", lambda *_args, **_kwargs: [])

    schedule = pd.DataFrame(
        [
            _schedule_row(
                4502,
                "2025-04-10T23:05:00Z",
                "NYY",
                "BOS",
                "Yankee Stadium",
                home_starter_id=120,
                away_starter_id=200,
                f5_home_score=3,
                f5_away_score=1,
                final_home_score=4,
                final_away_score=2,
            )
        ]
    )
    prior_batting = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4, 5, 201, 202, 203, 204, 205, 206, 207, 208, 209],
            "Team": ["NYY"] * 5 + ["BOS"] * 9,
            "PA": [200] * 14,
            "wRC+": [100.0] * 14,
            "wOBA": [0.320] * 14,
            "ISO": [0.170] * 14,
            "BABIP": [0.300] * 14,
            "K%": [22.0] * 14,
            "BB%": [8.0] * 14,
        }
    )
    prior_start_metrics = pd.DataFrame(
        {
            "team": ["NYY", "BOS"],
            "pitcher_id": [100, 200],
            "game_date": ["2024-08-01", "2024-08-01"],
            "xfip": [4.0, 4.0],
            "xera": [4.0, 4.0],
            "k_pct": [22.0, 22.0],
            "bb_pct": [8.0, 8.0],
            "gb_pct": [43.0, 43.0],
            "hr_fb_pct": [11.0, 11.0],
            "avg_fastball_velocity": [94.0, 94.0],
            "pitch_mix_entropy": [1.5, 1.5],
            "innings_pitched": [5.0, 5.0],
        }
    )

    result = build_training_dataset(
        start_year=2025,
        end_year=2025,
        output_path=tmp_path / "turnover_gate.parquet",
        full_regular_seasons_target=1,
        shortened_season_game_threshold=0,
        schedule_fetcher=lambda _year: schedule.copy(),
        batting_stats_fetcher=lambda season, **_kwargs: prior_batting.copy() if season == 2024 else pd.DataFrame(),
        team_logs_fetcher=_fake_team_logs_fetcher({}),
        fielding_stats_fetcher=_fake_fielding_fetcher({}),
        framing_stats_fetcher=_fake_framing_fetcher({}),
        start_metrics_fetcher=_fake_start_metrics_fetcher({2024: prior_start_metrics}),
        bullpen_metrics_fetcher=_fake_bullpen_metrics_fetcher({}),
        lineup_fetcher=lambda *_args, **_kwargs: [],
        weather_fetcher=_fake_weather_fetcher,
    )

    assert result.dataframe["game_pk"].tolist() == [4502]
    assert captured["offense"] is None
    assert captured["pitching"] is None
    assert captured["defense"] is None


def test_build_training_dataset_uses_retrosheet_lineup_ids_when_no_history_is_provided(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_lineup_ids: dict[tuple[int, str], list[int]] | None = None

    def _capture_offense(*_args, **kwargs):
        nonlocal captured_lineup_ids
        lineup_ids = kwargs.get("lineup_player_ids")
        if lineup_ids:
            captured_lineup_ids = {
                key: [int(player_id) for player_id in player_ids]
                for key, player_ids in lineup_ids.items()
            }
        return []

    monkeypatch.setattr("src.model.data_builder.compute_offensive_features", _capture_offense)
    monkeypatch.setattr("src.model.data_builder.compute_pitching_features", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("src.model.data_builder.compute_defense_features", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("src.model.data_builder.compute_bullpen_features", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("src.model.data_builder.compute_baseline_features", lambda *_args, **_kwargs: [])

    retrosheet_lineups = pd.DataFrame(
        [
            {
                "gid": "NYA202504100",
                "team": "NYA",
                "opp": "BOS",
                "date": 20250410,
                "number": 0,
                "vishome": "h",
                "season": 2025,
                "start_l1": "judga001",
                "start_l2": "sotoj001",
                "start_l3": "stant001",
                "start_l4": "rizzo001",
                "start_l5": "volpa001",
                "start_l6": "torrg001",
                "start_l7": "wells001",
                "start_l8": "verdu001",
                "start_l9": "chisb001",
            },
            {
                "gid": "NYA202504100",
                "team": "BOS",
                "opp": "NYA",
                "date": 20250410,
                "number": 0,
                "vishome": "v",
                "season": 2025,
                "start_l1": "duran001",
                "start_l2": "dever001",
                "start_l3": "story001",
                "start_l4": "casas001",
                "start_l5": "abreu001",
                "start_l6": "wonge001",
                "start_l7": "refoe001",
                "start_l8": "hamil001",
                "start_l9": "wongc001",
            },
        ]
    )
    register = pd.DataFrame(
        {
            "key_retro": [
                "judga001",
                "sotoj001",
                "stant001",
                "rizzo001",
                "volpa001",
                "torrg001",
                "wells001",
                "verdu001",
                "chisb001",
                "duran001",
                "dever001",
                "story001",
                "casas001",
                "abreu001",
                "wonge001",
                "refoe001",
                "hamil001",
                "wongc001",
                "starterh",
                "startera",
            ],
            "key_mlbam": [
                "592450",
                "665742",
                "519317",
                "519203",
                "683011",
                "686223",
                "669224",
                "657557",
                "664702",
                "677649",
                "646240",
                "596115",
                "671213",
                "677800",
                "657136",
                "676789",
                "671218",
                "543939",
                "100",
                "200",
            ],
            "name_first": ["Aaron", "Juan", "Giancarlo", "Anthony", "Anthony", "Gleyber", "Austin", "Alex", "Jazz", "Jarren", "Rafael", "Trevor", "Triston", "Wilyer", "Connor", "Rob", "David", "Connor", "Home", "Away"],
            "name_last": ["Judge", "Soto", "Stanton", "Rizzo", "Volpe", "Torres", "Wells", "Verdugo", "Chisholm", "Duran", "Devers", "Story", "Casas", "Abreu", "Wong", "Refsnyder", "Hamilton", "Wong", "Starter", "Starter"],
            "bats": ["R", "L", "R", "L", "R", "R", "L", "L", "S", "L", "L", "R", "L", "R", "R", "R", "L", "R", None, None],
            "throws": ["R", "R", "R", "L", "R", "R", "R", "L", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "L"],
        }
    )

    monkeypatch.setattr(
        "src.model.data_builder.fetch_retrosheet_starting_lineups",
        lambda **_kwargs: retrosheet_lineups.copy(),
    )
    monkeypatch.setattr(
        "src.model.data_builder.fetch_chadwick_register",
        lambda **_kwargs: register.copy(),
    )
    monkeypatch.setattr(
        "src.model.data_builder.fetch_statcast_range",
        lambda *_args, **_kwargs: _fake_team_batting_split_statcast_frame(),
    )

    schedule = pd.DataFrame(
        [
            _schedule_row(
                5001,
                "2025-04-10T23:05:00Z",
                "NYY",
                "BOS",
                "Yankee Stadium",
                f5_home_score=3,
                f5_away_score=1,
                final_home_score=4,
                final_away_score=2,
            )
        ]
    )

    result = build_training_dataset(
        start_year=2025,
        end_year=2025,
        output_path=tmp_path / "retrosheet_lineups.parquet",
        full_regular_seasons_target=1,
        shortened_season_game_threshold=0,
        schedule_fetcher=lambda _year: schedule.copy(),
        team_logs_fetcher=_fake_team_logs_fetcher({}),
        fielding_stats_fetcher=_fake_fielding_fetcher({}),
        framing_stats_fetcher=_fake_framing_fetcher({}),
        start_metrics_fetcher=_fake_start_metrics_fetcher({}),
        bullpen_metrics_fetcher=_fake_bullpen_metrics_fetcher({}),
        weather_fetcher=_fake_weather_fetcher,
    )

    assert result.dataframe["game_pk"].tolist() == [5001]
    assert captured_lineup_ids is not None
    assert captured_lineup_ids[(5001, "NYY")][:3] == [592450, 665742, 519317]
    assert len(captured_lineup_ids[(5001, "NYY")]) == 9
    assert len(captured_lineup_ids[(5001, "BOS")]) == 9
    row = result.dataframe.iloc[0]
    assert row["home_lineup_known_bats_pct"] == pytest.approx(1.0)
    assert row["home_opposing_starter_throws_left"] == pytest.approx(1.0)
    assert row["home_lineup_platoon_advantage_pct"] > 0.0
    assert row["home_team_woba_vs_LHP"] == pytest.approx(0.410)
    assert row["home_team_woba_vs_opposing_hand"] == pytest.approx(0.410)
    assert row["away_team_woba_vs_RHP"] == pytest.approx(0.350)
    assert row["away_team_woba_vs_opposing_hand"] == pytest.approx(0.350)


def test_build_live_feature_frame_adds_schedule_travel_context(
    tmp_path: Path,
) -> None:
    current_schedule = pd.DataFrame(
        [
            _schedule_row(
                7002,
                "2025-04-10T23:05:00Z",
                "NYY",
                "SEA",
                "Yankee Stadium",
                status="scheduled",
                f5_home_score=0,
                f5_away_score=0,
                final_home_score=0,
                final_away_score=0,
            )
        ]
    )
    historical_games = pd.DataFrame(
        [
            _schedule_row(
                7001,
                "2025-04-09T03:05:00Z",
                "SEA",
                "TB",
                "T-Mobile Park",
                status="final",
            )
        ]
    )

    frame = build_live_feature_frame(
        target_date="2025-04-10",
        schedule=current_schedule,
        historical_games=historical_games,
        db_path=tmp_path / "live_schedule_context.db",
        lineups=[],
        weather_fetcher=None,
        batting_stats_fetcher=lambda *_args, **_kwargs: pd.DataFrame(),
        team_logs_fetcher=_fake_team_logs_fetcher({}),
        fielding_stats_fetcher=_fake_fielding_fetcher({}),
        framing_stats_fetcher=_fake_framing_fetcher({}),
        start_metrics_fetcher=_fake_start_metrics_fetcher({}),
        bullpen_metrics_fetcher=_fake_bullpen_metrics_fetcher({}),
    )

    row = frame.iloc[0]
    assert row["away_timezone_crossings_east"] == pytest.approx(3.0)
    assert row["away_is_day_after_night_game"] == pytest.approx(0.0)


def test_build_live_feature_frame_recomputes_same_day_features_and_ignores_stale_rows(
    tmp_path: Path,
) -> None:
    current_schedule = pd.DataFrame(
        [
            _schedule_row(
                4002,
                "2025-04-10T23:05:00Z",
                "NYY",
                "BOS",
                "Yankee Stadium",
                status="scheduled",
                f5_home_score=0,
                f5_away_score=0,
                final_home_score=0,
                final_away_score=0,
            )
        ]
    )
    historical_games = pd.DataFrame(
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
            )
        ]
    )
    team_logs = _team_logs_by_season()
    start_metrics = _start_metrics_by_season()
    fielding = _fielding_by_season()
    framing = _framing_by_season()
    bullpen_metrics = _bullpen_metrics_by_season()

    direct_db_path = tmp_path / "direct_live_modules.db"
    _seed_schedule(
        direct_db_path,
        pd.concat([historical_games, current_schedule], ignore_index=True),
    )
    expected_feature_values = _direct_module_feature_values(
        db_path=direct_db_path,
        target_date="2025-04-10",
        team_logs=team_logs,
        start_metrics=start_metrics,
        fielding=fielding,
        framing=framing,
        bullpen_metrics=bullpen_metrics,
    )

    live_db_path = init_db(tmp_path / "live_pipeline.db")
    with sqlite3.connect(live_db_path) as connection:
        connection.execute(
            """
            INSERT INTO features (game_pk, feature_name, feature_value, window_size, as_of_timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (4002, "home_team_woba_7g", -999.0, 7, "2025-04-01T00:00:00+00:00"),
        )
        connection.commit()

    frame = build_live_feature_frame(
        target_date="2025-04-10",
        schedule=current_schedule,
        historical_games=historical_games,
        db_path=live_db_path,
        lineups=[],
        weather_fetcher=_fake_weather_fetcher,
        batting_stats_fetcher=lambda *_args, **_kwargs: pd.DataFrame(),
        team_logs_fetcher=_fake_team_logs_fetcher(team_logs),
        fielding_stats_fetcher=_fake_fielding_fetcher(fielding),
        framing_stats_fetcher=_fake_framing_fetcher(framing),
        start_metrics_fetcher=_fake_start_metrics_fetcher(start_metrics),
        bullpen_metrics_fetcher=_fake_bullpen_metrics_fetcher(bullpen_metrics),
    )

    assert frame["game_pk"].tolist() == [4002]
    target_row = frame.iloc[0]
    assert target_row["as_of_timestamp"] == "2025-04-09T00:00:00+00:00"
    assert target_row["home_team_woba_7g"] == pytest.approx(expected_feature_values["home_team_woba_7g"])
    assert target_row["home_team_woba_7g"] != -999.0
    assert target_row["home_starter_xfip_7s"] == pytest.approx(
        expected_feature_values["home_starter_xfip_7s"]
    )
    assert target_row["home_team_drs_season"] == pytest.approx(
        expected_feature_values["home_team_drs_season"]
    )
    assert target_row["home_team_bullpen_xfip"] == pytest.approx(
        expected_feature_values["home_team_bullpen_xfip"]
    )
    assert target_row["home_team_log5_30g"] == pytest.approx(
        expected_feature_values["home_team_log5_30g"]
    )


def test_compute_team_batting_splits_uses_strict_lag_and_min_pa() -> None:
    rows: list[dict[str, object]] = []
    for at_bat_number in range(1, 31):
        rows.append(
            {
                "game_pk": 1,
                "game_date": "2025-04-09",
                "at_bat_number": at_bat_number,
                "pitch_number": 1,
                "inning_topbot": "Top",
                "away_team": "NYY",
                "home_team": "BOS",
                "p_throws": "L",
                "events": "single",
                "estimated_woba_using_speedangle": 0.400,
            }
        )
    for at_bat_number in range(31, 60):
        rows.append(
            {
                "game_pk": 1,
                "game_date": "2025-04-09",
                "at_bat_number": at_bat_number,
                "pitch_number": 1,
                "inning_topbot": "Top",
                "away_team": "NYY",
                "home_team": "BOS",
                "p_throws": "R",
                "events": "single",
                "estimated_woba_using_speedangle": 0.360,
            }
        )
    for at_bat_number in range(60, 96):
        rows.append(
            {
                "game_pk": 2,
                "game_date": "2025-04-10",
                "at_bat_number": at_bat_number,
                "pitch_number": 1,
                "inning_topbot": "Top",
                "away_team": "NYY",
                "home_team": "BOS",
                "p_throws": "R",
                "events": "single",
                "estimated_woba_using_speedangle": 0.500,
            }
        )

    splits = _compute_team_batting_splits(pd.DataFrame(rows), datetime(2025, 4, 10).date())

    assert splits["NYY"]["vs_LHP"] == pytest.approx(0.400)
    assert splits["NYY"]["vs_RHP"] == pytest.approx(LEAGUE_WOBA_BASELINE)


def test_compute_team_batting_splits_falls_back_to_event_woba_when_xwoba_missing() -> None:
    rows: list[dict[str, object]] = []
    for at_bat_number in range(1, 16):
        rows.append(
            {
                "game_pk": 3,
                "game_date": "2025-04-09",
                "at_bat_number": at_bat_number,
                "pitch_number": 1,
                "inning_topbot": "Bot",
                "away_team": "SEA",
                "home_team": "TB",
                "p_throws": "L",
                "events": "single",
            }
        )
    for at_bat_number in range(16, 31):
        rows.append(
            {
                "game_pk": 3,
                "game_date": "2025-04-09",
                "at_bat_number": at_bat_number,
                "pitch_number": 1,
                "inning_topbot": "Bot",
                "away_team": "SEA",
                "home_team": "TB",
                "p_throws": "L",
                "events": "field_out",
            }
        )

    splits = _compute_team_batting_splits(pd.DataFrame(rows), datetime(2025, 4, 10).date())

    assert splits["TB"]["vs_LHP"] == pytest.approx((15 * 0.89) / 30)
    assert splits["TB"]["vs_RHP"] == pytest.approx(LEAGUE_WOBA_BASELINE)


def test_build_live_feature_frame_uses_official_game_date_not_utc_rollover(
    tmp_path: Path,
) -> None:
    current_schedule = pd.DataFrame(
        [
            {
                **_schedule_row(
                    823244,
                    "2026-03-26T00:05:00Z",
                    "SF",
                    "NYY",
                    "Oracle Park",
                    home_starter_id=657277,
                    away_starter_id=608331,
                    status="scheduled",
                    f5_home_score=0,
                    f5_away_score=0,
                    final_home_score=0,
                    final_away_score=0,
                ),
                "game_date": "2026-03-25",
            }
        ]
    )

    frame = build_live_feature_frame(
        target_date="2026-03-25",
        schedule=current_schedule,
        historical_games=pd.DataFrame(),
        db_path=tmp_path / "live_rollover.db",
        lineups=[],
        weather_fetcher=_fake_weather_fetcher,
        batting_stats_fetcher=lambda *_args, **_kwargs: pd.DataFrame(),
        team_logs_fetcher=_fake_team_logs_fetcher({}),
        fielding_stats_fetcher=_fake_fielding_fetcher({}),
        framing_stats_fetcher=_fake_framing_fetcher({}),
        start_metrics_fetcher=_fake_start_metrics_fetcher({}),
        bullpen_metrics_fetcher=_fake_bullpen_metrics_fetcher({}),
    )

    assert frame["game_pk"].tolist() == [823244]
    assert frame.iloc[0]["game_date"] == "2026-03-25"


def test_build_live_feature_frame_threads_roster_turnover_into_live_feature_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, dict[str, float] | None] = {}

    def capture_turnover(name: str):
        def _capture(*_args, **kwargs):
            captured[name] = kwargs.get("roster_turnover_by_team")
            return []

        return _capture

    monkeypatch.setattr("src.model.data_builder.compute_offensive_features", capture_turnover("offense"))
    monkeypatch.setattr("src.model.data_builder.compute_pitching_features", capture_turnover("pitching"))
    monkeypatch.setattr("src.model.data_builder.compute_defense_features", capture_turnover("defense"))
    monkeypatch.setattr("src.model.data_builder.compute_bullpen_features", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("src.model.data_builder.compute_baseline_features", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        "src.model.data_builder.fetch_statcast_range",
        lambda *_args, **_kwargs: _fake_team_batting_split_statcast_frame(),
    )

    current_schedule = pd.DataFrame(
        [
            _schedule_row(
                4502,
                "2025-04-10T23:05:00Z",
                "NYY",
                "BOS",
                "Yankee Stadium",
                home_starter_id=120,
                away_starter_id=200,
                status="scheduled",
                f5_home_score=0,
                f5_away_score=0,
                final_home_score=0,
                final_away_score=0,
            )
        ]
    )
    lineups = [
        Lineup(
            game_pk=4502,
            team="NYY",
            source="test",
            confirmed=True,
            as_of_timestamp=datetime(2025, 4, 10, 18, 0, tzinfo=timezone.utc),
            starting_pitcher_id=120,
            starting_pitcher_throws="R",
            players=[
                LineupPlayer(
                    batting_order=index + 1,
                    player_id=player_id,
                    player_name=f"NYY {player_id}",
                    bats="L" if index < 5 else "R",
                )
                for index, player_id in enumerate(range(11, 20))
            ],
        ),
        Lineup(
            game_pk=4502,
            team="BOS",
            source="test",
            confirmed=True,
            as_of_timestamp=datetime(2025, 4, 10, 18, 0, tzinfo=timezone.utc),
            starting_pitcher_id=200,
            starting_pitcher_throws="L",
            players=[
                LineupPlayer(
                    batting_order=index + 1,
                    player_id=player_id,
                    player_name=f"BOS {player_id}",
                    bats="R" if index < 6 else "L",
                )
                for index, player_id in enumerate(range(201, 210))
            ],
        ),
    ]
    prior_batting = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4, 5, 201, 202, 203, 204, 205, 206, 207, 208, 209],
            "Team": ["NYY"] * 5 + ["BOS"] * 9,
            "PA": [200] * 14,
            "wRC+": [100.0] * 14,
            "wOBA": [0.320] * 14,
            "ISO": [0.170] * 14,
            "BABIP": [0.300] * 14,
            "K%": [22.0] * 14,
            "BB%": [8.0] * 14,
        }
    )
    prior_start_metrics = pd.DataFrame(
        {
            "team": ["NYY", "BOS"],
            "pitcher_id": [100, 200],
            "game_date": ["2024-08-01", "2024-08-01"],
            "xfip": [4.0, 4.0],
            "xera": [4.0, 4.0],
            "k_pct": [22.0, 22.0],
            "bb_pct": [8.0, 8.0],
            "gb_pct": [43.0, 43.0],
            "hr_fb_pct": [11.0, 11.0],
            "avg_fastball_velocity": [94.0, 94.0],
            "pitch_mix_entropy": [1.5, 1.5],
            "innings_pitched": [5.0, 5.0],
        }
    )

    frame = build_live_feature_frame(
        target_date="2025-04-10",
        schedule=current_schedule,
        historical_games=pd.DataFrame(),
        db_path=tmp_path / "live_pipeline.db",
        lineups=lineups,
        weather_fetcher=None,
        batting_stats_fetcher=lambda season, **_kwargs: prior_batting.copy() if season == 2024 else pd.DataFrame(),
        team_logs_fetcher=_fake_team_logs_fetcher({}),
        fielding_stats_fetcher=_fake_fielding_fetcher({}),
        framing_stats_fetcher=_fake_framing_fetcher({}),
        start_metrics_fetcher=_fake_start_metrics_fetcher({2024: prior_start_metrics}),
        bullpen_metrics_fetcher=_fake_bullpen_metrics_fetcher({}),
    )

    assert frame["game_pk"].tolist() == [4502]
    assert captured["offense"] is not None
    assert captured["pitching"] is not None
    assert captured["defense"] is not None
    assert captured["offense"]["NYY"] == pytest.approx(1.0)
    assert captured["offense"]["BOS"] == pytest.approx(0.0)
    assert captured["pitching"] == captured["offense"]
    assert captured["defense"] == captured["offense"]
    assert frame.iloc[0]["home_lineup_lhb_pct"] == pytest.approx(5 / 9)
    assert frame.iloc[0]["home_lineup_known_bats_pct"] == pytest.approx(1.0)
    assert frame.iloc[0]["home_lineup_platoon_advantage_pct"] == pytest.approx(4 / 9)
    assert frame.iloc[0]["home_opposing_starter_throws_left"] == pytest.approx(1.0)
    assert frame.iloc[0]["away_opposing_starter_throws_right"] == pytest.approx(1.0)
    assert frame.iloc[0]["home_team_woba_vs_LHP"] == pytest.approx(0.410)
    assert frame.iloc[0]["home_team_woba_vs_RHP"] == pytest.approx(0.330)
    assert frame.iloc[0]["home_team_woba_vs_opposing_hand"] == pytest.approx(0.410)
    assert frame.iloc[0]["away_team_woba_vs_LHP"] == pytest.approx(0.290)
    assert frame.iloc[0]["away_team_woba_vs_RHP"] == pytest.approx(0.350)
    assert frame.iloc[0]["away_team_woba_vs_opposing_hand"] == pytest.approx(0.350)


def test_derive_matchup_interaction_features_uses_lineup_and_starter_quality() -> None:
    features = _derive_matchup_interaction_features(
        {
            "home_lineup_woba_30g": 0.352,
            "away_lineup_woba_30g": 0.331,
            "home_starter_xera_30s": 3.8,
            "away_starter_xera_30s": 4.1,
        }
    )

    expected_away_starter_xwoba = 0.320 + ((4.1 - 3.2) / 15.0)
    expected_home_starter_xwoba = 0.320 + ((3.8 - 3.2) / 15.0)
    assert features["home_offense_vs_away_starter_woba_gap"] == pytest.approx(
        0.352 - expected_away_starter_xwoba
    )
    assert features["away_offense_vs_home_starter_woba_gap"] == pytest.approx(
        0.331 - expected_home_starter_xwoba
    )


def test_fill_missing_feature_values_uses_module_defaults_instead_of_dataset_means() -> None:
    raw = pd.DataFrame(
        [
            {
                "game_pk": 1,
                "season": 2025,
                "game_date": "2025-04-10",
                "scheduled_start": "2025-04-10T23:05:00+00:00",
                "as_of_timestamp": "2025-04-09T00:00:00+00:00",
                "home_team": "NYY",
                "away_team": "BOS",
                "venue": "Yankee Stadium",
                "game_type": "R",
                "status": "final",
                "f5_home_score": 3,
                "f5_away_score": 1,
                "final_home_score": 4,
                "final_away_score": 2,
                "f5_margin": 2.0,
                "final_margin": 2.0,
                "f5_tied_after_5": 0,
                "f5_ml_result": 1,
                "f5_rl_result": 1,
                "home_lineup_wrc_plus_7g": pd.NA,
                "home_team_woba_7g": pd.NA,
                "away_starter_is_opener": pd.NA,
                "away_starter_xfip_7s": pd.NA,
                "home_starter_days_rest": pd.NA,
                "away_starter_last_start_pitch_count": pd.NA,
                "home_starter_cumulative_pitch_load_5s": pd.NA,
                "home_team_drs_season": pd.NA,
                "away_team_defensive_efficiency_season": pd.NA,
                "home_team_bullpen_avg_rest_days_top5": pd.NA,
                "away_team_bullpen_high_leverage_available_count": pd.NA,
                "home_team_bullpen_xfip": pd.NA,
                "home_team_pythagorean_wp_30g": pd.NA,
                "away_team_log5_30g": pd.NA,
                "park_runs_factor": pd.NA,
                "park_hr_factor": pd.NA,
                "abs_active": pd.NA,
                "abs_walk_rate_delta": pd.NA,
                "abs_strikeout_rate_delta": pd.NA,
                "weather_composite": pd.NA,
                "weather_wind_factor": pd.NA,
                "weather_data_missing": pd.NA,
            },
            {
                "game_pk": 2,
                "season": 2025,
                "game_date": "2025-09-10",
                "scheduled_start": "2025-09-10T23:05:00+00:00",
                "as_of_timestamp": "2025-09-09T00:00:00+00:00",
                "home_team": "NYY",
                "away_team": "BOS",
                "venue": "Yankee Stadium",
                "game_type": "R",
                "status": "final",
                "f5_home_score": 5,
                "f5_away_score": 0,
                "final_home_score": 6,
                "final_away_score": 1,
                "f5_margin": 5.0,
                "final_margin": 5.0,
                "f5_tied_after_5": 0,
                "f5_ml_result": 1,
                "f5_rl_result": 1,
                "home_lineup_wrc_plus_7g": 152.0,
                "home_team_woba_7g": 0.401,
                "away_starter_is_opener": 1.0,
                "away_starter_xfip_7s": 2.35,
                "home_starter_days_rest": 4.0,
                "away_starter_last_start_pitch_count": 104.0,
                "home_starter_cumulative_pitch_load_5s": 96.0,
                "home_team_drs_season": 9.0,
                "away_team_defensive_efficiency_season": 0.742,
                "home_team_bullpen_avg_rest_days_top5": 6.0,
                "away_team_bullpen_high_leverage_available_count": 1.0,
                "home_team_bullpen_xfip": 3.1,
                "home_team_pythagorean_wp_30g": 0.74,
                "away_team_log5_30g": 0.22,
                "park_runs_factor": 1.17,
                "park_hr_factor": 1.11,
                "abs_active": 0.0,
                "abs_walk_rate_delta": 0.0,
                "abs_strikeout_rate_delta": 0.0,
                "weather_composite": 1.08,
                "weather_wind_factor": 7.5,
                "weather_data_missing": 0.0,
            },
        ]
    )

    filled = _fill_missing_feature_values(raw)
    first_row = filled.iloc[0]
    second_row = filled.iloc[1]

    assert first_row["home_lineup_wrc_plus_7g"] == pytest.approx(LEAGUE_WRC_PLUS_BASELINE)
    assert first_row["home_team_woba_7g"] == pytest.approx(LEAGUE_WOBA_BASELINE)
    assert first_row["away_starter_is_opener"] == 0.0
    assert first_row["away_starter_xfip_7s"] == pytest.approx(DEFAULT_METRIC_BASELINES["xfip"])
    assert first_row["home_starter_days_rest"] == pytest.approx(5.0)
    assert first_row["away_starter_last_start_pitch_count"] == pytest.approx(90.0)
    assert first_row["home_starter_cumulative_pitch_load_5s"] == pytest.approx(90.0)
    assert first_row["home_team_drs_season"] == 0.0
    assert first_row["away_team_defensive_efficiency_season"] == pytest.approx(
        DEFAULT_DEFENSIVE_EFFICIENCY
    )
    assert first_row["home_team_bullpen_avg_rest_days_top5"] == pytest.approx(
        DEFAULT_AVG_REST_DAYS
    )
    assert first_row["away_team_bullpen_high_leverage_available_count"] == pytest.approx(
        float(DEFAULT_TOP_RELIEVER_COUNT)
    )
    assert first_row["home_team_bullpen_xfip"] == pytest.approx(DEFAULT_XFIP)
    assert first_row["home_team_pythagorean_wp_30g"] == pytest.approx(0.5)
    assert first_row["away_team_log5_30g"] == pytest.approx(0.5)
    assert first_row["park_runs_factor"] == pytest.approx(1.0)
    assert first_row["park_hr_factor"] == pytest.approx(1.0)
    assert first_row["abs_active"] == pytest.approx(1.0)
    assert first_row["abs_walk_rate_delta"] == pytest.approx(DEFAULT_WALK_RATE_DELTA)
    assert first_row["abs_strikeout_rate_delta"] == pytest.approx(DEFAULT_STRIKEOUT_RATE_DELTA)
    assert first_row["weather_composite"] == pytest.approx(NEUTRAL_WEATHER_FACTOR)
    assert first_row["weather_wind_factor"] == pytest.approx(0.0)
    assert first_row["weather_data_missing"] == pytest.approx(1.0)

    assert second_row["home_team_woba_7g"] == pytest.approx(0.401)
    assert second_row["away_starter_xfip_7s"] == pytest.approx(2.35)
    assert second_row["weather_composite"] == pytest.approx(1.08)


def test_resolve_training_years_backfills_shortened_season_with_previous_full_year() -> None:
    resolved_years = resolve_training_years(
        start_year=2019,
        end_year=2025,
        full_regular_seasons_target=7,
        season_row_counts={2018: 2430, 2019: 2430, 2020: 898, 2021: 2430, 2022: 2430, 2023: 2430, 2024: 2430, 2025: 2430},
        shortened_season_game_threshold=2000,
        allow_backfill_years=True,
    )

    assert resolved_years == [2018, 2019, 2021, 2022, 2023, 2024, 2025]


def test_resolve_training_years_does_not_backfill_unknown_years() -> None:
    resolved_years = resolve_training_years(
        start_year=2023,
        end_year=2025,
        full_regular_seasons_target=7,
        season_row_counts={2023: 2430, 2024: 2430, 2025: 2430},
        shortened_season_game_threshold=2000,
    )

    assert resolved_years == [2023, 2024, 2025]


def test_resolve_training_years_respects_strict_start_year_by_default() -> None:
    resolved_years = resolve_training_years(
        start_year=2021,
        end_year=2025,
        full_regular_seasons_target=7,
        season_row_counts={2018: 2430, 2019: 2430, 2020: 898, 2021: 2430, 2022: 2430, 2023: 2430, 2024: 2430, 2025: 2430},
        shortened_season_game_threshold=2000,
    )

    assert resolved_years == [2021, 2022, 2023, 2024, 2025]


def test_assert_training_data_is_complete_accepts_cached_validation_parquet(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "training_data_validation_fixture.parquet"
    fixture = _write_cached_training_validation_fixture(output_path)

    summary = assert_training_data_is_complete(output_path)

    assert summary.row_count == len(fixture) == 17_010
    assert summary.expected_row_range == (16_500, 17_500)
    assert summary.row_count_in_expected_range is True
    assert summary.target_null_counts == {"f5_ml_result": 0, "f5_rl_result": 0}
    assert summary.game_type_counts == {"R": 17_010}
    assert summary.non_regular_game_types == {}
    assert summary.seasons == _VALIDATION_FIXTURE_SEASONS


def test_assert_training_data_is_complete_rejects_unexpected_row_count(tmp_path: Path) -> None:
    output_path = tmp_path / "too_small_training_data.parquet"
    _write_cached_training_validation_fixture(output_path, rows_per_season=10)

    with pytest.raises(AssertionError, match="Row count 70 outside expected range 16500-17500"):
        assert_training_data_is_complete(output_path)


def test_assert_training_data_is_complete_rejects_nan_targets(tmp_path: Path) -> None:
    output_path = tmp_path / "training_data_with_nan_target.parquet"
    fixture = _write_cached_training_validation_fixture(output_path)
    fixture.loc[0, "f5_ml_result"] = pd.NA
    fixture.to_parquet(output_path, index=False)

    with pytest.raises(AssertionError, match="Target column f5_ml_result contains 1 NaN values"):
        assert_training_data_is_complete(output_path)


def test_assert_training_data_is_complete_rejects_non_regular_games(tmp_path: Path) -> None:
    output_path = tmp_path / "training_data_with_postseason_row.parquet"
    fixture = _write_cached_training_validation_fixture(output_path)
    fixture.loc[0, "game_type"] = "F"
    fixture.to_parquet(output_path, index=False)

    with pytest.raises(AssertionError, match="Found non-regular game types: {'F': 1}"):
        assert_training_data_is_complete(output_path)
