from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.clients.retrosheet_client import fetch_retrosheet_umpires
from src.config import _load_settings_yaml
from src.db import DEFAULT_DB_PATH, init_db, sqlite_connection
from src.models.features import GameFeatures


DEFAULT_WINDOWS: tuple[int, ...] = (30, 90)
DEFAULT_HOME_WIN_PCT = 0.54
DEFAULT_TOTAL_RUNS_AVG = 8.8
DEFAULT_F5_TOTAL_RUNS_AVG = 4.5

_TEAM_CODE_ALIASES = {team_code: team_code for team_code in _load_settings_yaml()["teams"].keys()}
_TEAM_CODE_ALIASES.update(
    {
        "AZ": "ARI",
        "ANA": "LAA",
        "ATH": "OAK",
        "CHA": "CWS",
        "CHN": "CHC",
        "CHW": "CWS",
        "KCR": "KC",
        "KCA": "KC",
        "LAN": "LAD",
        "NYA": "NYY",
        "NYN": "NYM",
        "SDP": "SD",
        "SDN": "SD",
        "SFG": "SF",
        "SFN": "SF",
        "SLN": "STL",
        "TBR": "TB",
        "TBA": "TB",
        "WAS": "WSH",
        "WSN": "WSH",
    }
)

_UmpireFetcher = Callable[..., pd.DataFrame]


def compute_umpire_features(
    game_date: str | date | datetime,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    refresh: bool = False,
    umpire_fetcher: _UmpireFetcher = fetch_retrosheet_umpires,
) -> list[GameFeatures]:
    """Compute plate-umpire historical tendency features for games on a target date."""

    target_day = _coerce_date(game_date)
    database_path = Path(db_path)
    init_db(database_path)

    games = _load_games_for_date(database_path, target_day)
    if games.empty:
        return []

    umpire_assignments = _load_umpire_assignments(target_day=target_day, refresh=refresh, umpire_fetcher=umpire_fetcher)
    if umpire_assignments.empty:
        return _persist_defaults(database_path, games, target_day, windows)

    current_assignments = umpire_assignments.loc[
        umpire_assignments["game_date"] == target_day
    ].copy()
    current_assignments = current_assignments.rename(columns={"plate_umpire": "current_plate_umpire"})

    games = games.merge(
        current_assignments[["game_date", "home_team", "away_team", "current_plate_umpire"]],
        how="left",
        left_on=["game_date", "home_team", "away_team"],
        right_on=["game_date", "home_team", "away_team"],
    )

    prior_games = _load_prior_games(database_path, target_day)
    if prior_games.empty:
        return _persist_defaults(database_path, games, target_day, windows)

    prior_umpire_games = prior_games.merge(
        umpire_assignments[["game_date", "home_team", "away_team", "plate_umpire"]],
        how="inner",
        on=["game_date", "home_team", "away_team"],
    )
    if prior_umpire_games.empty:
        return _persist_defaults(database_path, games, target_day, windows)

    prior_umpire_games = prior_umpire_games.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
    as_of_timestamp = datetime.combine(target_day - timedelta(days=1), time.min, tzinfo=timezone.utc)

    features: list[GameFeatures] = []
    for game in games.to_dict(orient="records"):
        game_pk = int(game["game_pk"])
        plate_umpire = _normalize_identifier(game.get("current_plate_umpire"))

        features.append(
            GameFeatures(
                game_pk=game_pk,
                feature_name="plate_umpire_known",
                feature_value=1.0 if plate_umpire else 0.0,
                window_size=None,
                as_of_timestamp=as_of_timestamp,
            )
        )

        umpire_history = (
            prior_umpire_games.loc[prior_umpire_games["plate_umpire"] == plate_umpire].copy()
            if plate_umpire
            else pd.DataFrame()
        )

        for window in windows:
            history_slice = umpire_history.tail(int(window))
            features.extend(
                [
                    GameFeatures(
                        game_pk=game_pk,
                        feature_name=f"plate_umpire_home_win_pct_{int(window)}g",
                        feature_value=_safe_mean(history_slice.get("home_win"), DEFAULT_HOME_WIN_PCT),
                        window_size=int(window),
                        as_of_timestamp=as_of_timestamp,
                    ),
                    GameFeatures(
                        game_pk=game_pk,
                        feature_name=f"plate_umpire_total_runs_avg_{int(window)}g",
                        feature_value=_safe_mean(
                            history_slice.get("total_runs"),
                            DEFAULT_TOTAL_RUNS_AVG,
                        ),
                        window_size=int(window),
                        as_of_timestamp=as_of_timestamp,
                    ),
                    GameFeatures(
                        game_pk=game_pk,
                        feature_name=f"plate_umpire_f5_total_runs_avg_{int(window)}g",
                        feature_value=_safe_mean(
                            history_slice.get("f5_total_runs"),
                            DEFAULT_F5_TOTAL_RUNS_AVG,
                        ),
                        window_size=int(window),
                        as_of_timestamp=as_of_timestamp,
                    ),
                    GameFeatures(
                        game_pk=game_pk,
                        feature_name=f"plate_umpire_sample_size_{int(window)}g",
                        feature_value=float(len(history_slice)),
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
    with sqlite_connection(db_path, builder_optimized=True) as connection:
        games = pd.read_sql_query(
            """
            SELECT game_pk, substr(date, 1, 10) AS game_date, home_team, away_team
            FROM games
            WHERE substr(date, 1, 10) = ?
            ORDER BY game_pk
            """,
            connection,
            params=(target_day.isoformat(),),
        )
    if not games.empty:
        games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce").dt.date
    return games


def _load_prior_games(db_path: Path, target_day: date) -> pd.DataFrame:
    with sqlite_connection(db_path, builder_optimized=True) as connection:
        games = pd.read_sql_query(
            """
            SELECT
                game_pk,
                substr(date, 1, 10) AS game_date,
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
        return games

    for column in ("final_home_score", "final_away_score", "f5_home_score", "f5_away_score"):
        games[column] = pd.to_numeric(games[column], errors="coerce")
    games["total_runs"] = games["final_home_score"].fillna(0) + games["final_away_score"].fillna(0)
    games["f5_total_runs"] = games["f5_home_score"].fillna(0) + games["f5_away_score"].fillna(0)
    games["home_win"] = (
        games["final_home_score"].fillna(-1) > games["final_away_score"].fillna(-1)
    ).astype(float)
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce").dt.date
    return games.dropna(subset=["game_date"]).reset_index(drop=True)


def _load_umpire_assignments(
    *,
    target_day: date,
    refresh: bool,
    umpire_fetcher: _UmpireFetcher,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in sorted({target_day.year - 1, target_day.year}):
        frame = umpire_fetcher(season=season, refresh=refresh)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["game_date", "home_team", "away_team", "plate_umpire"])

    assignments = pd.concat(frames, ignore_index=True)
    date_column = _first_existing_column(assignments, ("date", "game_date"))
    home_team_column = _first_existing_column(assignments, ("hometeam", "home_team"))
    away_team_column = _first_existing_column(assignments, ("visteam", "away_team"))
    plate_umpire_column = _first_existing_column(assignments, ("umphome", "plate_umpire"))
    if None in (date_column, home_team_column, away_team_column, plate_umpire_column):
        return pd.DataFrame(columns=["game_date", "home_team", "away_team", "plate_umpire"])

    normalized = pd.DataFrame(
        {
            "game_date": _normalize_dates(assignments[date_column]),
            "home_team": assignments[home_team_column].map(_normalize_team_code),
            "away_team": assignments[away_team_column].map(_normalize_team_code),
            "plate_umpire": assignments[plate_umpire_column].map(_normalize_identifier),
        }
    )
    normalized = normalized.dropna(subset=["game_date", "home_team", "away_team"])
    normalized = normalized.drop_duplicates(
        subset=["game_date", "home_team", "away_team", "plate_umpire"],
        keep="last",
    )
    return normalized.reset_index(drop=True)


def _normalize_dates(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    return parsed.dt.date


def _normalize_team_code(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().upper()
    return _TEAM_CODE_ALIASES.get(normalized)


def _normalize_identifier(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip().lower()
    return normalized or None


def _first_existing_column(dataframe: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for column in candidates:
        if column in dataframe.columns:
            return column
    return None


def _safe_mean(values: pd.Series | None, default: float) -> float:
    if values is None or len(values) == 0:
        return float(default)
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float(default)
    return float(numeric.mean())


def _persist_defaults(
    db_path: Path,
    games: pd.DataFrame,
    target_day: date,
    windows: Sequence[int],
) -> list[GameFeatures]:
    as_of_timestamp = datetime.combine(target_day - timedelta(days=1), time.min, tzinfo=timezone.utc)
    features: list[GameFeatures] = []
    for game_pk in games["game_pk"].astype(int).tolist():
        features.append(
            GameFeatures(
                game_pk=game_pk,
                feature_name="plate_umpire_known",
                feature_value=0.0,
                window_size=None,
                as_of_timestamp=as_of_timestamp,
            )
        )
        for window in windows:
            features.extend(
                [
                    GameFeatures(
                        game_pk=game_pk,
                        feature_name=f"plate_umpire_home_win_pct_{int(window)}g",
                        feature_value=DEFAULT_HOME_WIN_PCT,
                        window_size=int(window),
                        as_of_timestamp=as_of_timestamp,
                    ),
                    GameFeatures(
                        game_pk=game_pk,
                        feature_name=f"plate_umpire_total_runs_avg_{int(window)}g",
                        feature_value=DEFAULT_TOTAL_RUNS_AVG,
                        window_size=int(window),
                        as_of_timestamp=as_of_timestamp,
                    ),
                    GameFeatures(
                        game_pk=game_pk,
                        feature_name=f"plate_umpire_f5_total_runs_avg_{int(window)}g",
                        feature_value=DEFAULT_F5_TOTAL_RUNS_AVG,
                        window_size=int(window),
                        as_of_timestamp=as_of_timestamp,
                    ),
                    GameFeatures(
                        game_pk=game_pk,
                        feature_name=f"plate_umpire_sample_size_{int(window)}g",
                        feature_value=0.0,
                        window_size=int(window),
                        as_of_timestamp=as_of_timestamp,
                    ),
                ]
            )
    _persist_features(db_path, features)
    return features


def _persist_features(db_path: Path, features: Sequence[GameFeatures]) -> None:
    if not features:
        return

    rows = [
        (
            feature.game_pk,
            feature.feature_name,
            float(feature.feature_value),
            feature.window_size,
            feature.as_of_timestamp.isoformat(),
        )
        for feature in features
    ]

    with sqlite_connection(db_path, builder_optimized=True) as connection:
        connection.executemany(
            """
            INSERT OR REPLACE INTO features (
                game_pk,
                feature_name,
                feature_value,
                window_size,
                as_of_timestamp
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()
