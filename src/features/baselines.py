from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.db import DEFAULT_DB_PATH, init_db
from src.models.features import GameFeatures


DEFAULT_WINDOWS: tuple[int, ...] = (30, 60)
DEFAULT_PYTHAGOREAN_EXPONENT = 1.83
TEAM_ENTITY_LEVEL = "team"


def calculate_pythagorean_win_percentage(
    runs_scored: float,
    runs_allowed: float,
    *,
    exponent: float = DEFAULT_PYTHAGOREAN_EXPONENT,
) -> float:
    """Calculate Pythagorean win percentage from runs scored and allowed."""

    if runs_scored < 0 or runs_allowed < 0:
        raise ValueError("runs scored and allowed must be non-negative")
    if exponent <= 0:
        raise ValueError("exponent must be positive")

    numerator = float(runs_scored) ** exponent
    denominator = numerator + float(runs_allowed) ** exponent
    if denominator == 0:
        return 0.5
    return numerator / denominator


def calculate_log5_probability(
    team_a_win_pct: float,
    team_b_win_pct: float,
    *,
    entity_level: str = TEAM_ENTITY_LEVEL,
) -> float:
    """Calculate team matchup probability using the Log5 formula."""

    if entity_level != TEAM_ENTITY_LEVEL:
        raise ValueError("Log5 is team-only and cannot be applied outside team-only matchups")

    if not 0.0 <= team_a_win_pct <= 1.0 or not 0.0 <= team_b_win_pct <= 1.0:
        raise ValueError("win percentages must be between 0 and 1")

    denominator = (
        team_a_win_pct * (1 - team_b_win_pct)
        + team_b_win_pct * (1 - team_a_win_pct)
    )
    if abs(denominator) < 1e-12:
        return 0.5

    return (team_a_win_pct * (1 - team_b_win_pct)) / denominator


def compute_baseline_features(
    game_date: str | date | datetime,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    exponent: float = DEFAULT_PYTHAGOREAN_EXPONENT,
) -> list[GameFeatures]:
    """Compute and persist rolling Pythagorean and Log5 baselines for games on a date."""

    target_day = _coerce_date(game_date)
    database_path = Path(db_path)
    init_db(database_path)

    games = _load_games_for_date(database_path, target_day)
    if games.empty:
        return []

    team_histories = _load_team_histories_before_date(database_path, target_day)
    as_of_timestamp = datetime.combine(target_day - timedelta(days=1), time.min, tzinfo=timezone.utc)

    features: list[GameFeatures] = []
    for game in games.to_dict(orient="records"):
        game_pk = int(game["game_pk"])
        home_team = str(game["home_team"])
        away_team = str(game["away_team"])
        home_history = team_histories.get(home_team, _empty_team_history())
        away_history = team_histories.get(away_team, _empty_team_history())

        for window in windows:
            home_full = _pythagorean_from_history(home_history, window=window, prefix="full", exponent=exponent)
            away_full = _pythagorean_from_history(away_history, window=window, prefix="full", exponent=exponent)
            home_f5 = _pythagorean_from_history(home_history, window=window, prefix="f5", exponent=exponent)
            away_f5 = _pythagorean_from_history(away_history, window=window, prefix="f5", exponent=exponent)
            home_log5 = calculate_log5_probability(home_full, away_full)
            away_log5 = 1.0 - home_log5

            features.extend(
                [
                    GameFeatures(
                        game_pk=game_pk,
                        feature_name=f"home_team_pythagorean_wp_{window}g",
                        feature_value=home_full,
                        window_size=int(window),
                        as_of_timestamp=as_of_timestamp,
                    ),
                    GameFeatures(
                        game_pk=game_pk,
                        feature_name=f"away_team_pythagorean_wp_{window}g",
                        feature_value=away_full,
                        window_size=int(window),
                        as_of_timestamp=as_of_timestamp,
                    ),
                    GameFeatures(
                        game_pk=game_pk,
                        feature_name=f"home_team_f5_pythagorean_wp_{window}g",
                        feature_value=home_f5,
                        window_size=int(window),
                        as_of_timestamp=as_of_timestamp,
                    ),
                    GameFeatures(
                        game_pk=game_pk,
                        feature_name=f"away_team_f5_pythagorean_wp_{window}g",
                        feature_value=away_f5,
                        window_size=int(window),
                        as_of_timestamp=as_of_timestamp,
                    ),
                    GameFeatures(
                        game_pk=game_pk,
                        feature_name=f"home_team_log5_{window}g",
                        feature_value=home_log5,
                        window_size=int(window),
                        as_of_timestamp=as_of_timestamp,
                    ),
                    GameFeatures(
                        game_pk=game_pk,
                        feature_name=f"away_team_log5_{window}g",
                        feature_value=away_log5,
                        window_size=int(window),
                        as_of_timestamp=as_of_timestamp,
                    ),
                ]
            )

    _persist_features(database_path, features)
    return features


def _coerce_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _load_games_for_date(db_path: Path, target_day: date) -> pd.DataFrame:
    with sqlite3.connect(db_path) as connection:
        return pd.read_sql_query(
            """
            SELECT game_pk, home_team, away_team
            FROM games
            WHERE substr(date, 1, 10) = ?
            ORDER BY game_pk
            """,
            connection,
            params=(target_day.isoformat(),),
        )


def _load_team_histories_before_date(db_path: Path, target_day: date) -> dict[str, pd.DataFrame]:
    with sqlite3.connect(db_path) as connection:
        games = pd.read_sql_query(
            """
            SELECT
                game_pk,
                date,
                home_team,
                away_team,
                final_home_score,
                final_away_score,
                f5_home_score,
                f5_away_score
            FROM games
            WHERE status = 'final'
              AND substr(date, 1, 10) < ?
            ORDER BY substr(date, 1, 10), game_pk
            """,
            connection,
            params=(target_day.isoformat(),),
        )

    if games.empty:
        return {}

    games["game_date"] = pd.to_datetime(games["date"], errors="coerce", utc=True)

    numeric_columns = [
        "final_home_score",
        "final_away_score",
        "f5_home_score",
        "f5_away_score",
    ]
    for column in numeric_columns:
        games[column] = pd.to_numeric(games[column], errors="coerce")

    home_rows = pd.DataFrame(
        {
            "team": games["home_team"],
            "game_date": games["game_date"],
            "game_pk": games["game_pk"],
            "full_runs_scored": games["final_home_score"],
            "full_runs_allowed": games["final_away_score"],
            "f5_runs_scored": games["f5_home_score"],
            "f5_runs_allowed": games["f5_away_score"],
        }
    )
    away_rows = pd.DataFrame(
        {
            "team": games["away_team"],
            "game_date": games["game_date"],
            "game_pk": games["game_pk"],
            "full_runs_scored": games["final_away_score"],
            "full_runs_allowed": games["final_home_score"],
            "f5_runs_scored": games["f5_away_score"],
            "f5_runs_allowed": games["f5_home_score"],
        }
    )
    history = pd.concat([home_rows, away_rows], ignore_index=True)
    history = history.sort_values(["team", "game_date", "game_pk"]).reset_index(drop=True)
    return {
        str(team): team_history.reset_index(drop=True)
        for team, team_history in history.groupby("team", sort=False)
    }


def _empty_team_history() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "team",
            "game_date",
            "game_pk",
            "full_runs_scored",
            "full_runs_allowed",
            "f5_runs_scored",
            "f5_runs_allowed",
        ]
    )


def _pythagorean_from_history(
    history: pd.DataFrame,
    *,
    window: int,
    prefix: str,
    exponent: float,
) -> float:
    if window <= 0:
        raise ValueError("window must be positive")

    runs_scored_column = f"{prefix}_runs_scored"
    runs_allowed_column = f"{prefix}_runs_allowed"
    if history.empty:
        return 0.5

    sample = history.dropna(subset=[runs_scored_column, runs_allowed_column]).tail(window)
    if sample.empty:
        return 0.5

    runs_scored = float(sample[runs_scored_column].sum())
    runs_allowed = float(sample[runs_allowed_column].sum())
    return calculate_pythagorean_win_percentage(
        runs_scored,
        runs_allowed,
        exponent=exponent,
    )


def _persist_features(db_path: Path, features: Sequence[GameFeatures]) -> None:
    if not features:
        return

    with sqlite3.connect(db_path) as connection:
        connection.executemany(
            """
            INSERT INTO features (game_pk, feature_name, feature_value, window_size, as_of_timestamp)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(game_pk, feature_name, window_size, as_of_timestamp)
            DO UPDATE SET feature_value = excluded.feature_value
            """,
            [
                (
                    feature.game_pk,
                    feature.feature_name,
                    feature.feature_value,
                    feature.window_size,
                    feature.as_of_timestamp.isoformat(),
                )
                for feature in features
            ],
        )
        connection.commit()
