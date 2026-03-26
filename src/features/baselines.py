from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.db import DEFAULT_DB_PATH, init_db, sqlite_connection
from src.features.bullpen import _resolve_season_start_date
from src.models.features import GameFeatures


DEFAULT_WINDOWS: tuple[int, ...] = (30, 60)
DEFAULT_RUN_RATE_WINDOWS: tuple[int, ...] = (7, 14)
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
    run_rate_windows: Sequence[int] = DEFAULT_RUN_RATE_WINDOWS,
    exponent: float = DEFAULT_PYTHAGOREAN_EXPONENT,
) -> list[GameFeatures]:
    """Compute and persist rolling Pythagorean and Log5 baselines for games on a date."""

    target_day = _coerce_date(game_date)
    database_path = Path(db_path)
    init_db(database_path)

    games = _load_games_for_date(database_path, target_day)
    if games.empty:
        return []

    season_start = _resolve_season_start_date(database_path, target_day.year)
    team_histories = _load_team_histories_before_date(
        database_path,
        target_day,
        season_start=season_start,
    )
    features = _build_baseline_features_for_games(
        games.to_dict(orient="records"),
        team_history_lookup=team_histories,
        target_day=target_day,
        windows=windows,
        run_rate_windows=run_rate_windows,
        exponent=exponent,
    )

    _persist_features(database_path, features)
    return features


def compute_baseline_features_for_schedule(
    schedule: pd.DataFrame,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    run_rate_windows: Sequence[int] = DEFAULT_RUN_RATE_WINDOWS,
    exponent: float = DEFAULT_PYTHAGOREAN_EXPONENT,
) -> list[GameFeatures]:
    """Compute and persist rolling baseline features for a schedule slice in one pass."""

    if schedule.empty:
        return []

    database_path = Path(db_path)
    init_db(database_path)

    working_schedule = schedule.copy()
    working_schedule["game_pk"] = pd.to_numeric(working_schedule["game_pk"], errors="coerce")
    working_schedule = working_schedule.dropna(subset=["game_pk", "game_date", "home_team", "away_team"]).copy()
    if working_schedule.empty:
        return []

    working_schedule["game_pk"] = working_schedule["game_pk"].astype(int)
    working_schedule["game_date"] = pd.to_datetime(working_schedule["game_date"], errors="coerce").dt.date
    working_schedule["home_team"] = working_schedule["home_team"].astype(str).str.strip().str.upper()
    working_schedule["away_team"] = working_schedule["away_team"].astype(str).str.strip().str.upper()
    working_schedule = working_schedule.dropna(subset=["game_date"]).copy()
    if working_schedule.empty:
        return []

    completed_games = _load_completed_games(database_path)
    full_history = _build_team_history_frame(completed_games)
    season_start_by_year = {
        int(season): _resolve_season_start_date(database_path, int(season))
        for season in sorted(
            {
                int(year)
                for year in pd.to_datetime(working_schedule["game_date"], errors="coerce").dt.year.dropna().tolist()
            }
        )
    }
    team_histories_by_season = _group_team_histories_by_season(
        full_history,
        season_start_by_year=season_start_by_year,
    )

    features: list[GameFeatures] = []
    for season, season_schedule in working_schedule.groupby(
        pd.to_datetime(working_schedule["game_date"], errors="coerce").dt.year,
        sort=True,
    ):
        season_int = int(season)
        target_dates = sorted({value for value in season_schedule["game_date"].dropna().tolist()})
        team_slices_by_day = _build_team_history_slices_for_dates(
            season_schedule=season_schedule,
            target_dates=target_dates,
            season_team_histories=team_histories_by_season.get(season_int, {}),
        )
        for target_day, day_games in season_schedule.groupby("game_date", sort=True):
            features.extend(
                _build_baseline_features_for_games(
                    day_games.to_dict(orient="records"),
                    team_history_lookup=team_slices_by_day.get(target_day, {}),
                    target_day=target_day,
                    windows=windows,
                    run_rate_windows=run_rate_windows,
                    exponent=exponent,
                )
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
    with sqlite_connection(db_path, builder_optimized=True) as connection:
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


def _load_team_histories_before_date(
    db_path: Path,
    target_day: date,
    *,
    season_start: date | None = None,
) -> dict[str, pd.DataFrame]:
    games = _load_completed_games(
        db_path,
        end_day_exclusive=target_day,
        season_start=season_start,
    )
    history = _build_team_history_frame(games)
    return {
        str(team): team_history.reset_index(drop=True)
        for team, team_history in history.groupby("team", sort=False)
    }


def _load_completed_games(
    db_path: Path,
    *,
    end_day_exclusive: date | None = None,
    season_start: date | None = None,
) -> pd.DataFrame:
    query = """
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
    """
    params: list[Any] = []
    if season_start is not None:
        query += " AND substr(date, 1, 10) >= ?"
        params.append(season_start.isoformat())
    if end_day_exclusive is not None:
        query += " AND substr(date, 1, 10) < ?"
        params.append(end_day_exclusive.isoformat())
    query += " ORDER BY substr(date, 1, 10), game_pk"

    with sqlite_connection(db_path, builder_optimized=True) as connection:
        games = pd.read_sql_query(
            query,
            connection,
            params=tuple(params),
        )

    if games.empty:
        return games

    games["game_date"] = pd.to_datetime(games["date"], errors="coerce").dt.date

    numeric_columns = [
        "final_home_score",
        "final_away_score",
        "f5_home_score",
        "f5_away_score",
    ]
    for column in numeric_columns:
        games[column] = pd.to_numeric(games[column], errors="coerce")
    return games


def _build_team_history_frame(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return _empty_team_history()

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
    history["team"] = history["team"].astype(str).str.strip().str.upper()
    history["game_day_ordinal"] = history["game_date"].map(lambda value: value.toordinal() if pd.notna(value) else None)
    return history.sort_values(["team", "game_date", "game_pk"]).reset_index(drop=True)


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
            "game_day_ordinal",
        ]
    )


def _build_baseline_features_for_games(
    games: Sequence[dict[str, Any]],
    *,
    team_history_lookup: dict[str, pd.DataFrame],
    target_day: date,
    windows: Sequence[int],
    run_rate_windows: Sequence[int],
    exponent: float,
) -> list[GameFeatures]:
    as_of_timestamp = datetime.combine(target_day - timedelta(days=1), time.min, tzinfo=timezone.utc)
    features: list[GameFeatures] = []

    for game in games:
        game_pk = int(game["game_pk"])
        home_team = str(game["home_team"]).strip().upper()
        away_team = str(game["away_team"]).strip().upper()
        home_history = team_history_lookup.get(home_team, _empty_team_history())
        away_history = team_history_lookup.get(away_team, _empty_team_history())

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
                        feature_name=f"home_team_pythagorean_wp_{int(window)}g",
                        feature_value=home_full,
                        window_size=int(window),
                        as_of_timestamp=as_of_timestamp,
                    ),
                    GameFeatures(
                        game_pk=game_pk,
                        feature_name=f"away_team_pythagorean_wp_{int(window)}g",
                        feature_value=away_full,
                        window_size=int(window),
                        as_of_timestamp=as_of_timestamp,
                    ),
                    GameFeatures(
                        game_pk=game_pk,
                        feature_name=f"home_team_f5_pythagorean_wp_{int(window)}g",
                        feature_value=home_f5,
                        window_size=int(window),
                        as_of_timestamp=as_of_timestamp,
                    ),
                    GameFeatures(
                        game_pk=game_pk,
                        feature_name=f"away_team_f5_pythagorean_wp_{int(window)}g",
                        feature_value=away_f5,
                        window_size=int(window),
                        as_of_timestamp=as_of_timestamp,
                    ),
                    GameFeatures(
                        game_pk=game_pk,
                        feature_name=f"home_team_log5_{int(window)}g",
                        feature_value=home_log5,
                        window_size=int(window),
                        as_of_timestamp=as_of_timestamp,
                    ),
                    GameFeatures(
                        game_pk=game_pk,
                        feature_name=f"away_team_log5_{int(window)}g",
                        feature_value=away_log5,
                        window_size=int(window),
                        as_of_timestamp=as_of_timestamp,
                    ),
                ]
            )

        for window in run_rate_windows:
            features.extend(
                [
                    GameFeatures(
                        game_pk=game_pk,
                        feature_name=f"home_team_runs_scored_{int(window)}g",
                        feature_value=_rolling_mean_from_history(
                            home_history,
                            window=window,
                            column="full_runs_scored",
                            default=4.5,
                        ),
                        window_size=int(window),
                        as_of_timestamp=as_of_timestamp,
                    ),
                    GameFeatures(
                        game_pk=game_pk,
                        feature_name=f"home_team_runs_allowed_{int(window)}g",
                        feature_value=_rolling_mean_from_history(
                            home_history,
                            window=window,
                            column="full_runs_allowed",
                            default=4.5,
                        ),
                        window_size=int(window),
                        as_of_timestamp=as_of_timestamp,
                    ),
                    GameFeatures(
                        game_pk=game_pk,
                        feature_name=f"away_team_runs_scored_{int(window)}g",
                        feature_value=_rolling_mean_from_history(
                            away_history,
                            window=window,
                            column="full_runs_scored",
                            default=4.5,
                        ),
                        window_size=int(window),
                        as_of_timestamp=as_of_timestamp,
                    ),
                    GameFeatures(
                        game_pk=game_pk,
                        feature_name=f"away_team_runs_allowed_{int(window)}g",
                        feature_value=_rolling_mean_from_history(
                            away_history,
                            window=window,
                            column="full_runs_allowed",
                            default=4.5,
                        ),
                        window_size=int(window),
                        as_of_timestamp=as_of_timestamp,
                    ),
                ]
            )

    return features


def _group_team_histories_by_season(
    history: pd.DataFrame,
    *,
    season_start_by_year: dict[int, date],
) -> dict[int, dict[str, pd.DataFrame]]:
    if history.empty:
        return {}

    team_histories_by_season: dict[int, dict[str, pd.DataFrame]] = {}
    history_years = pd.to_datetime(history["game_date"], errors="coerce").dt.year
    for season, season_history in history.groupby(history_years, sort=True):
        if pd.isna(season):
            continue
        season_int = int(season)
        season_start = season_start_by_year.get(season_int)
        if season_start is not None:
            season_history = season_history.loc[season_history["game_date"] >= season_start].copy()
        if season_history.empty:
            continue
        team_histories_by_season[season_int] = {
            str(team): team_history.reset_index(drop=True)
            for team, team_history in season_history.groupby("team", sort=False)
        }
    return team_histories_by_season


def _build_team_history_slices_for_dates(
    *,
    season_schedule: pd.DataFrame,
    target_dates: Sequence[date],
    season_team_histories: dict[str, pd.DataFrame],
) -> dict[date, dict[str, pd.DataFrame]]:
    if season_schedule.empty or not target_dates:
        return {}

    team_codes = sorted(
        {
            str(team).strip().upper()
            for column in ("home_team", "away_team")
            for team in season_schedule[column].dropna().tolist()
        }
    )
    slices_by_day: dict[date, dict[str, pd.DataFrame]] = {}
    for target_day in target_dates:
        day_lookup: dict[str, pd.DataFrame] = {}
        target_ordinal = target_day.toordinal()
        for team in team_codes:
            history = season_team_histories.get(team)
            if history is None or history.empty:
                continue
            cutoff = history["game_day_ordinal"].to_numpy().searchsorted(target_ordinal, side="left")
            day_lookup[team] = history.iloc[:cutoff].reset_index(drop=True)
        slices_by_day[target_day] = day_lookup
    return slices_by_day


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


def _rolling_mean_from_history(
    history: pd.DataFrame,
    *,
    window: int,
    column: str,
    default: float,
) -> float:
    if window <= 0:
        raise ValueError("window must be positive")

    if history.empty:
        return float(default)

    sample = history.dropna(subset=[column]).tail(window)
    if sample.empty:
        return float(default)

    return float(sample[column].mean())


def _persist_features(db_path: Path, features: Sequence[GameFeatures]) -> None:
    if not features:
        return

    with sqlite_connection(db_path, builder_optimized=True) as connection:
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
