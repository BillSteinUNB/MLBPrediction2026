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
RETROSHEET_HISTORY_START_SEASON = 2018
_DEFAULT_TEAM_TIMEZONE_OFFSET = -5

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
_TEAM_TIMEZONE_OFFSETS = {
    "ARI": -7,
    "ATL": -5,
    "BAL": -5,
    "BOS": -5,
    "CHC": -6,
    "CIN": -5,
    "CLE": -5,
    "COL": -7,
    "CWS": -6,
    "DET": -5,
    "HOU": -6,
    "KC": -6,
    "LAA": -8,
    "LAD": -8,
    "MIA": -5,
    "MIL": -6,
    "MIN": -6,
    "NYM": -5,
    "NYY": -5,
    "OAK": -8,
    "PHI": -5,
    "PIT": -5,
    "SD": -8,
    "SEA": -8,
    "SF": -8,
    "STL": -6,
    "TB": -5,
    "TEX": -6,
    "TOR": -5,
    "WSH": -5,
}

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
    target_day_ts = pd.Timestamp(target_day)

    umpire_assignments = _load_umpire_assignments(target_day=target_day, refresh=refresh, umpire_fetcher=umpire_fetcher)
    if umpire_assignments.empty:
        return _persist_defaults(database_path, games, target_day, windows)

    current_assignments = umpire_assignments.loc[
        umpire_assignments["game_date"] == target_day_ts
    ].copy()
    current_assignments = current_assignments.rename(columns={"plate_umpire": "current_plate_umpire"})

    games["game_date"] = _to_tz_naive_datetime_series(games["game_date"])
    current_assignments["game_date"] = _to_tz_naive_datetime_series(current_assignments["game_date"])

    games = games.merge(
        current_assignments[
            ["game_date", "home_team", "away_team", "matchup_sequence", "current_plate_umpire"]
        ],
        how="left",
        left_on=["game_date", "home_team", "away_team", "matchup_sequence"],
        right_on=["game_date", "home_team", "away_team", "matchup_sequence"],
    )
    games = _fill_missing_current_assignments(
        games,
        current_assignments,
        plate_umpire_column="current_plate_umpire",
    )

    prior_games = _load_prior_games(database_path, target_day)
    if prior_games.empty:
        return _persist_defaults(database_path, games, target_day, windows)

    prior_games["game_date"] = _to_tz_naive_datetime_series(prior_games["game_date"])
    umpire_assignments["game_date"] = _to_tz_naive_datetime_series(umpire_assignments["game_date"])
    prior_umpire_games = prior_games.merge(
        umpire_assignments[
            ["game_date", "home_team", "away_team", "matchup_sequence", "plate_umpire"]
        ],
        how="inner",
        on=["game_date", "home_team", "away_team", "matchup_sequence"],
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
    lower_bound = datetime.combine(target_day - timedelta(days=1), time.min, tzinfo=timezone.utc)
    upper_bound = datetime.combine(target_day + timedelta(days=2), time.min, tzinfo=timezone.utc)
    with sqlite_connection(db_path, builder_optimized=True) as connection:
        games = pd.read_sql_query(
            """
            SELECT game_pk, date AS game_datetime, home_team, away_team
            FROM games
            WHERE date >= ?
              AND date < ?
            ORDER BY date, game_pk
            """,
            connection,
            params=(lower_bound.isoformat(), upper_bound.isoformat()),
        )
    if not games.empty:
        games["game_datetime"] = _to_tz_naive_datetime_series(games["game_datetime"])
        games["game_date"] = _local_game_date_series(games["game_datetime"], games["home_team"])
        games = games.loc[games["game_date"].dt.date == target_day].reset_index(drop=True)
        games = _attach_matchup_sequence(games, order_columns=("game_datetime", "game_pk"))
    return games


def _load_prior_games(db_path: Path, target_day: date) -> pd.DataFrame:
    upper_bound = datetime.combine(target_day + timedelta(days=1), time.min, tzinfo=timezone.utc)
    with sqlite_connection(db_path, builder_optimized=True) as connection:
        games = pd.read_sql_query(
            """
            SELECT
                game_pk,
                date AS game_datetime,
                home_team,
                away_team,
                final_home_score,
                final_away_score,
                f5_home_score,
                f5_away_score
            FROM games
            WHERE status = 'final'
              AND date < ?
            ORDER BY date, game_pk
            """,
            connection,
            params=(upper_bound.isoformat(),),
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
    games["game_datetime"] = _to_tz_naive_datetime_series(games["game_datetime"])
    games["game_date"] = _local_game_date_series(games["game_datetime"], games["home_team"])
    games = games.dropna(subset=["game_date"]).reset_index(drop=True)
    games = games.loc[games["game_date"].dt.date < target_day].reset_index(drop=True)
    return _attach_matchup_sequence(games, order_columns=("game_date", "game_datetime", "game_pk"))


def _load_umpire_assignments(
    *,
    target_day: date,
    refresh: bool,
    umpire_fetcher: _UmpireFetcher,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in _retrosheet_seasons_for_target_day(target_day):
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
    if "matchup_sequence" in assignments.columns:
        normalized["matchup_sequence"] = pd.to_numeric(
            assignments["matchup_sequence"],
            errors="coerce",
        ).astype("Int64")
    normalized = normalized.dropna(subset=["game_date", "home_team", "away_team"])
    if "matchup_sequence" not in normalized.columns:
        normalized = normalized.drop_duplicates(
            subset=["game_date", "home_team", "away_team", "plate_umpire"],
            keep="last",
        )
        normalized = _attach_matchup_sequence(normalized, order_columns=("game_date",))
    else:
        missing_matchup_sequence = normalized["matchup_sequence"].isna()
        if missing_matchup_sequence.any():
            fallback = _attach_matchup_sequence(
                normalized.drop(columns=["matchup_sequence"]),
                order_columns=("game_date",),
            )
            normalized.loc[missing_matchup_sequence, "matchup_sequence"] = fallback.loc[
                missing_matchup_sequence,
                "matchup_sequence",
            ]
    normalized["matchup_sequence"] = (
        pd.to_numeric(normalized["matchup_sequence"], errors="coerce").fillna(1).astype(int)
    )
    normalized = normalized.drop_duplicates(
        subset=["game_date", "home_team", "away_team", "matchup_sequence"],
        keep="last",
    )
    return normalized.reset_index(drop=True)


def _normalize_dates(values: pd.Series) -> pd.Series:
    return _to_tz_naive_datetime_series(values).dt.normalize()


def _local_game_date_series(game_datetimes: pd.Series, home_teams: pd.Series) -> pd.Series:
    parsed_datetimes = _to_tz_naive_datetime_series(game_datetimes)
    normalized_teams = pd.Series(home_teams, copy=False).map(_normalize_team_code)
    timezone_offsets = (
        normalized_teams.map(_TEAM_TIMEZONE_OFFSETS).fillna(_DEFAULT_TEAM_TIMEZONE_OFFSET).astype(float)
    )
    local_datetimes = parsed_datetimes + pd.to_timedelta(timezone_offsets, unit="h")
    return local_datetimes.dt.normalize()


def _to_tz_naive_datetime_series(values: pd.Series) -> pd.Series:
    text = pd.Series(values, copy=False).astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")

    ymd_mask = text.str.fullmatch(r"\d{8}")
    if ymd_mask.any():
        parsed.loc[ymd_mask] = pd.to_datetime(
            text.loc[ymd_mask],
            errors="coerce",
            format="%Y%m%d",
        )

    fallback_mask = ~ymd_mask
    if fallback_mask.any():
        fallback = pd.to_datetime(
            text.loc[fallback_mask],
            errors="coerce",
            format="mixed",
            utc=True,
        ).dt.tz_localize(None)
        parsed.loc[fallback_mask] = fallback

    return parsed


def _retrosheet_seasons_for_target_day(target_day: date) -> list[int]:
    if target_day.year < RETROSHEET_HISTORY_START_SEASON:
        return []
    return [
        season
        for season in range(RETROSHEET_HISTORY_START_SEASON, target_day.year + 1)
        if season != 2020
    ]


def _attach_matchup_sequence(
    dataframe: pd.DataFrame,
    *,
    order_columns: Sequence[str],
) -> pd.DataFrame:
    if dataframe.empty:
        empty = dataframe.copy()
        empty["matchup_sequence"] = pd.Series(dtype="int64")
        return empty

    ordered = dataframe.copy()
    sort_columns = [column for column in order_columns if column in ordered.columns]
    if sort_columns:
        ordered = ordered.sort_values(sort_columns).reset_index(drop=True)
    ordered["matchup_sequence"] = (
        ordered.groupby(["game_date", "home_team", "away_team"]).cumcount() + 1
    )
    return ordered


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


def _fill_missing_current_assignments(
    games: pd.DataFrame,
    current_assignments: pd.DataFrame,
    *,
    plate_umpire_column: str,
) -> pd.DataFrame:
    if games.empty or plate_umpire_column not in games.columns:
        return games

    missing_mask = games[plate_umpire_column].isna()
    if not missing_mask.any():
        return games

    assignment_lookup = (
        current_assignments.groupby(["game_date", "home_team", "away_team"])[plate_umpire_column]
        .agg(_unique_non_empty_identifier)
        .reset_index()
    )
    assignment_lookup = assignment_lookup.dropna(subset=[plate_umpire_column])
    if assignment_lookup.empty:
        return games

    fallback = games.loc[missing_mask, ["game_pk", "game_date", "home_team", "away_team"]].merge(
        assignment_lookup,
        how="left",
        on=["game_date", "home_team", "away_team"],
    )
    fallback = fallback.set_index("game_pk")[plate_umpire_column]
    if fallback.empty:
        return games

    game_indexed = games.set_index("game_pk")
    fillable = fallback.notna()
    if fillable.any():
        game_indexed.loc[fallback.index[fillable], plate_umpire_column] = fallback.loc[fillable]
    return game_indexed.reset_index()


def _unique_non_empty_identifier(values: pd.Series) -> str | None:
    normalized_values = [_normalize_identifier(value) for value in values.tolist()]
    unique_values = sorted({value for value in normalized_values if value})
    if len(unique_values) != 1:
        return None
    return unique_values[0]


def _persist_defaults(
    db_path: Path,
    games: pd.DataFrame,
    target_day: date,
    windows: Sequence[int],
) -> list[GameFeatures]:
    as_of_timestamp = datetime.combine(target_day - timedelta(days=1), time.min, tzinfo=timezone.utc)
    features: list[GameFeatures] = []
    game_records = games.to_dict(orient="records")
    for game in game_records:
        game_pk = int(game["game_pk"])
        known_umpire = _normalize_identifier(game.get("current_plate_umpire"))
        features.append(
            GameFeatures(
                game_pk=game_pk,
                feature_name="plate_umpire_known",
                feature_value=1.0 if known_umpire else 0.0,
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
