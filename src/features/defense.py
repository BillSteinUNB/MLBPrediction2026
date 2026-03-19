from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.clients.statcast_client import (
    TEAM_LABEL_TO_CODE,
    fetch_catcher_framing,
    fetch_fielding_stats,
    fetch_team_game_logs,
)
from src.config import _load_settings_yaml
from src.db import DEFAULT_DB_PATH, init_db
from src.models.features import GameFeatures


DEFAULT_WINDOWS: tuple[int, ...] = (30, 60)
DEFAULT_DEFENSIVE_EFFICIENCY = 0.700
POSITION_IMPORTANCE_WEIGHTS: dict[str, float] = {
    "C": 1.30,
    "SS": 1.25,
    "2B": 1.15,
    "CF": 1.10,
    "3B": 1.00,
    "LF": 0.90,
    "RF": 0.90,
    "1B": 0.80,
}
POSITION_ALIASES: dict[str, str] = {
    "2": "C",
    "3": "1B",
    "4": "2B",
    "5": "3B",
    "6": "SS",
    "7": "LF",
    "8": "CF",
    "9": "RF",
}
FIELDING_VALUE_COLUMNS: tuple[str, ...] = ("drs", "oaa")
TEAM_CODES: tuple[str, ...] = tuple(sorted(_load_settings_yaml()["teams"].keys()))
DEFAULT_ABS_RETENTION_FACTOR = float(_load_settings_yaml()["abs_retention_factor"])


_FieldingFetcher = Callable[..., pd.DataFrame]
_FramingFetcher = Callable[..., pd.DataFrame]
_TeamLogsFetcher = Callable[..., pd.DataFrame]


def compute_defense_features(
    game_date: str | date | datetime,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    abs_retention_factor: float = DEFAULT_ABS_RETENTION_FACTOR,
    refresh: bool = False,
    fielding_fetcher: _FieldingFetcher = fetch_fielding_stats,
    framing_fetcher: _FramingFetcher = fetch_catcher_framing,
    team_logs_fetcher: _TeamLogsFetcher = fetch_team_game_logs,
) -> list[GameFeatures]:
    """Compute and persist lagged team defense features for games on a target date."""

    target_day = _coerce_date(game_date)
    database_path = Path(db_path)
    init_db(database_path)

    games = _load_games_for_date(database_path, target_day)
    if games.empty:
        return []

    defensive_efficiency_histories = _build_defensive_efficiency_histories(
        season=target_day.year,
        target_day=target_day,
        team_logs_fetcher=team_logs_fetcher,
        refresh=refresh,
    )
    games_played_lookup = {
        team: len(history)
        for team, history in defensive_efficiency_histories.items()
        if not history.empty
    }
    defensive_efficiency_baseline = _resolve_defensive_efficiency_baseline(
        defensive_efficiency_histories
    )

    fielding_frame = _safe_fetch_fielding_frame(
        season=target_day.year,
        fielding_fetcher=fielding_fetcher,
        refresh=refresh,
    )
    normalized_fielding = _normalize_fielding_frame(fielding_frame)
    fielding_histories = _build_metric_histories(
        normalized_fielding,
        target_day=target_day,
        value_columns=FIELDING_VALUE_COLUMNS,
        games_played_lookup=games_played_lookup,
    )
    name_team_lookup = _build_name_team_lookup(normalized_fielding)

    framing_frame = _safe_fetch_framing_frame(
        season=target_day.year,
        framing_fetcher=framing_fetcher,
        refresh=refresh,
    )
    normalized_framing = _normalize_framing_frame(framing_frame, name_team_lookup)
    framing_histories = _build_metric_histories(
        normalized_framing,
        target_day=target_day,
        value_columns=("raw_framing",),
        games_played_lookup=games_played_lookup,
    )

    as_of_timestamp = datetime.combine(
        target_day - timedelta(days=1),
        time.min,
        tzinfo=timezone.utc,
    )

    features: list[GameFeatures] = []
    for game in games.to_dict(orient="records"):
        abs_active = bool(game.get("is_abs_active", 1))
        for side_name, team_key in (("home", "home_team"), ("away", "away_team")):
            team = str(game[team_key])
            fielding_history = fielding_histories.get(team, _empty_metric_history(FIELDING_VALUE_COLUMNS))
            framing_history = framing_histories.get(team, _empty_metric_history(("raw_framing",)))
            defensive_efficiency_history = defensive_efficiency_histories.get(
                team,
                _empty_metric_history(("defensive_efficiency",)),
            )

            season_values = _build_feature_values(
                fielding_history=fielding_history,
                framing_history=framing_history,
                defensive_efficiency_history=defensive_efficiency_history,
                abs_retention_factor=abs_retention_factor,
                abs_active=abs_active,
                defensive_efficiency_baseline=defensive_efficiency_baseline,
                window=None,
            )
            features.extend(
                _to_feature_rows(
                    game_pk=int(game["game_pk"]),
                    side_name=side_name,
                    feature_values=season_values,
                    window_label="season",
                    window_size=None,
                    as_of_timestamp=as_of_timestamp,
                )
            )

            for window in windows:
                window_values = _build_feature_values(
                    fielding_history=fielding_history,
                    framing_history=framing_history,
                    defensive_efficiency_history=defensive_efficiency_history,
                    abs_retention_factor=abs_retention_factor,
                    abs_active=abs_active,
                    defensive_efficiency_baseline=defensive_efficiency_baseline,
                    window=window,
                )
                features.extend(
                    _to_feature_rows(
                        game_pk=int(game["game_pk"]),
                        side_name=side_name,
                        feature_values=window_values,
                        window_label=f"{window}g",
                        window_size=int(window),
                        as_of_timestamp=as_of_timestamp,
                    )
                )

    _persist_features(database_path, features)
    return features


def adjust_framing_runs(
    raw_framing_runs: float,
    *,
    retention_factor: float = DEFAULT_ABS_RETENTION_FACTOR,
    abs_active: bool = True,
) -> float:
    """Adjust catcher framing runs for ABS-active environments."""

    return float(raw_framing_runs) * (retention_factor if abs_active else 1.0)


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
            SELECT game_pk, date, home_team, away_team, is_abs_active
            FROM games
            WHERE substr(date, 1, 10) = ?
            ORDER BY game_pk
            """,
            connection,
            params=(target_day.isoformat(),),
        )


def _safe_fetch_fielding_frame(
    *,
    season: int,
    fielding_fetcher: _FieldingFetcher,
    refresh: bool,
) -> pd.DataFrame:
    try:
        return fielding_fetcher(season, refresh=refresh)
    except Exception:
        return pd.DataFrame()


def _safe_fetch_framing_frame(
    *,
    season: int,
    framing_fetcher: _FramingFetcher,
    refresh: bool,
) -> pd.DataFrame:
    try:
        return framing_fetcher(season, refresh=refresh)
    except Exception:
        return pd.DataFrame()


def _build_defensive_efficiency_histories(
    *,
    season: int,
    target_day: date,
    team_logs_fetcher: _TeamLogsFetcher,
    refresh: bool,
) -> dict[str, pd.DataFrame]:
    observations: list[dict[str, Any]] = []
    cutoff = pd.Timestamp(target_day)

    for offense_team in TEAM_CODES:
        try:
            logs = team_logs_fetcher(season, offense_team, refresh=refresh)
        except Exception:
            logs = pd.DataFrame()

        normalized = _normalize_offensive_game_logs(logs)
        if normalized.empty:
            continue

        prior_rows = normalized.loc[normalized["game_date"] < cutoff].copy()
        if prior_rows.empty:
            continue

        for row in prior_rows.to_dict(orient="records"):
            defensive_team = str(row.get("opponent") or "").strip().upper()
            if not defensive_team:
                continue
            observations.append(
                {
                    "team": defensive_team,
                    "game_date": row["game_date"],
                    "defensive_efficiency": float(row["defensive_efficiency"]),
                }
            )

    if not observations:
        return {}

    combined = pd.DataFrame(observations)
    histories: dict[str, pd.DataFrame] = {}
    for team, group in combined.groupby("team", dropna=True):
        histories[str(team)] = (
            group.groupby("game_date", as_index=False)["defensive_efficiency"]
            .mean()
            .sort_values("game_date")
            .reset_index(drop=True)
        )
    return histories


def _resolve_defensive_efficiency_baseline(
    histories: dict[str, pd.DataFrame],
) -> float:
    values = [
        frame["defensive_efficiency"]
        for frame in histories.values()
        if not frame.empty and "defensive_efficiency" in frame
    ]
    if not values:
        return DEFAULT_DEFENSIVE_EFFICIENCY

    combined = pd.concat(values, ignore_index=True)
    if combined.dropna().empty:
        return DEFAULT_DEFENSIVE_EFFICIENCY
    return float(combined.mean())


def _normalize_offensive_game_logs(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return _empty_metric_history(("opponent", "defensive_efficiency"))

    date_column = _first_column(dataframe, ("Date", "Offense_Date", "game_date"))
    opponent_column = _first_column(dataframe, ("Opp", "Opponent", "opponent", "Offense_Opp"))
    efficiency_column = _first_column(
        dataframe,
        ("DER", "defensive_efficiency", "DefEff", "def_efficiency"),
    )

    result = pd.DataFrame()
    result["game_date"] = pd.to_datetime(
        dataframe[date_column] if date_column is not None else pd.Series(dtype=object),
        errors="coerce",
    )
    result["opponent"] = (
        dataframe[opponent_column].map(_normalize_team_label)
        if opponent_column is not None
        else pd.Series(dtype=str)
    )

    if efficiency_column is not None:
        result["defensive_efficiency"] = pd.to_numeric(
            dataframe[efficiency_column],
            errors="coerce",
        )
    else:
        ab = _extract_numeric(dataframe, "AB", "Offense_AB")
        hits = _extract_numeric(dataframe, "H", "Offense_H")
        home_runs = _extract_numeric(dataframe, "HR", "Offense_HR")
        strikeouts = _extract_numeric(dataframe, "SO", "K", "Offense_SO", "Offense_K")
        sacrifice_flies = _extract_numeric(dataframe, "SF", "Offense_SF")
        balls_in_play = ab - strikeouts - home_runs + sacrifice_flies
        hits_in_play = (hits - home_runs).clip(lower=0)
        result["defensive_efficiency"] = _safe_ratio(
            numerator=(balls_in_play - hits_in_play).clip(lower=0),
            denominator=balls_in_play,
            default=DEFAULT_DEFENSIVE_EFFICIENCY,
        )

    result = result.dropna(subset=["game_date"]).copy()
    if "opponent" in result.columns:
        result = result.loc[result["opponent"].astype(str) != ""].copy()
    if "defensive_efficiency" in result.columns:
        result["defensive_efficiency"] = pd.to_numeric(
            result["defensive_efficiency"],
            errors="coerce",
        ).fillna(DEFAULT_DEFENSIVE_EFFICIENCY)

    return result.sort_values("game_date").reset_index(drop=True)


def _normalize_fielding_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return pd.DataFrame(columns=["team", "game_date", "player_name_key", "drs", "oaa"])

    name_column = _first_column(dataframe, ("Name", "name", "player_name", "player"))
    team_column = _first_column(dataframe, ("Team", "team", "Tm"))
    position_column = _first_column(dataframe, ("Pos", "position", "Position"))
    date_column = _first_column(dataframe, ("game_date", "date", "Date"))
    drs_column = _first_column(dataframe, ("DRS", "drs"))
    oaa_column = _first_column(dataframe, ("OAA", "oaa", "outs_above_average"))

    result = pd.DataFrame()
    result["player_name_key"] = (
        dataframe[name_column].map(_normalize_name)
        if name_column is not None
        else pd.Series(dtype=str)
    )
    result["team"] = (
        dataframe[team_column].map(_normalize_team_label)
        if team_column is not None
        else pd.Series(dtype=str)
    )
    positions = (
        dataframe[position_column].map(_normalize_position)
        if position_column is not None
        else pd.Series(dtype=str)
    )
    weights = positions.map(POSITION_IMPORTANCE_WEIGHTS).fillna(1.0)
    result["game_date"] = pd.to_datetime(
        dataframe[date_column] if date_column is not None else pd.Series(dtype=object),
        errors="coerce",
    )
    result["drs"] = _extract_numeric(dataframe, drs_column) * weights
    result["oaa"] = _extract_numeric(dataframe, oaa_column) * weights

    return result.loc[result["team"].astype(str) != ""].reset_index(drop=True)


def _build_name_team_lookup(dataframe: pd.DataFrame) -> dict[str, str]:
    if dataframe.empty:
        return {}

    valid_rows = dataframe.loc[
        (dataframe["player_name_key"].astype(str) != "") & (dataframe["team"].astype(str) != "")
    ]
    if valid_rows.empty:
        return {}

    deduped = valid_rows.drop_duplicates(subset=["player_name_key"], keep="last")
    return dict(zip(deduped["player_name_key"], deduped["team"], strict=False))


def _normalize_framing_frame(
    dataframe: pd.DataFrame,
    name_team_lookup: dict[str, str],
) -> pd.DataFrame:
    if dataframe.empty:
        return pd.DataFrame(columns=["team", "game_date", "raw_framing"])

    name_column = _first_column(dataframe, ("name", "Name", "player_name", "catcher", "player"))
    team_column = _first_column(dataframe, ("team", "Team", "Tm"))
    date_column = _first_column(dataframe, ("game_date", "date", "Date"))
    framing_column = _first_column(
        dataframe,
        (
            "runs_extra_strikes",
            "framing_runs",
            "framing",
            "strike_runs",
            "extra_strike_runs",
        ),
    )

    result = pd.DataFrame()
    result["player_name_key"] = (
        dataframe[name_column].map(_normalize_name)
        if name_column is not None
        else pd.Series(dtype=str)
    )
    if team_column is not None:
        result["team"] = dataframe[team_column].map(_normalize_team_label)
    else:
        result["team"] = result["player_name_key"].map(name_team_lookup).fillna("")

    result["game_date"] = pd.to_datetime(
        dataframe[date_column] if date_column is not None else pd.Series(dtype=object),
        errors="coerce",
    )
    result["raw_framing"] = _extract_numeric(dataframe, framing_column)
    return result.loc[result["team"].astype(str) != ""].reset_index(drop=True)


def _build_metric_histories(
    dataframe: pd.DataFrame,
    *,
    target_day: date,
    value_columns: Sequence[str],
    games_played_lookup: dict[str, int],
) -> dict[str, pd.DataFrame]:
    if dataframe.empty:
        return {}

    histories: dict[str, pd.DataFrame] = {}
    cutoff = pd.Timestamp(target_day)
    for team, group in dataframe.groupby("team", dropna=True):
        team_group = group.copy()
        has_dated_history = team_group["game_date"].notna().any()
        dated_rows = team_group.loc[
            team_group["game_date"].notna() & (team_group["game_date"] < cutoff)
        ].copy()

        if not dated_rows.empty:
            history = (
                dated_rows.groupby("game_date", as_index=False)[list(value_columns)]
                .sum()
                .sort_values("game_date")
                .reset_index(drop=True)
            )
            histories[str(team)] = history
            continue

        if has_dated_history:
            continue

        totals = team_group[list(value_columns)].sum(numeric_only=True)
        games_played = max(int(games_played_lookup.get(str(team), 0)), 1)
        histories[str(team)] = pd.DataFrame(
            {
                "game_date": [pd.Timestamp(target_day - timedelta(days=1))],
                **{column: [float(totals.get(column, 0.0)) / games_played] for column in value_columns},
            }
        )

    return histories


def _build_feature_values(
    *,
    fielding_history: pd.DataFrame,
    framing_history: pd.DataFrame,
    defensive_efficiency_history: pd.DataFrame,
    abs_retention_factor: float,
    abs_active: bool,
    defensive_efficiency_baseline: float,
    window: int | None,
) -> dict[str, float]:
    raw_framing = _history_mean(framing_history, "raw_framing", window=window, default=0.0)
    return {
        "drs": _history_mean(fielding_history, "drs", window=window, default=0.0),
        "oaa": _history_mean(fielding_history, "oaa", window=window, default=0.0),
        "defensive_efficiency": _history_mean(
            defensive_efficiency_history,
            "defensive_efficiency",
            window=window,
            default=defensive_efficiency_baseline,
        ),
        "adjusted_framing": adjust_framing_runs(
            raw_framing,
            retention_factor=abs_retention_factor,
            abs_active=abs_active,
        ),
    }


def _to_feature_rows(
    *,
    game_pk: int,
    side_name: str,
    feature_values: dict[str, float],
    window_label: str,
    window_size: int | None,
    as_of_timestamp: datetime,
) -> list[GameFeatures]:
    return [
        GameFeatures(
            game_pk=game_pk,
            feature_name=f"{side_name}_team_{metric}_{window_label}",
            feature_value=value,
            window_size=window_size,
            as_of_timestamp=as_of_timestamp,
        )
        for metric, value in feature_values.items()
    ]


def _persist_features(db_path: Path, features: Sequence[GameFeatures]) -> None:
    if not features:
        return

    rows = [
        (
            feature.game_pk,
            feature.feature_name,
            feature.feature_value,
            feature.window_size,
            feature.as_of_timestamp.isoformat(),
        )
        for feature in features
    ]

    with sqlite3.connect(db_path) as connection:
        connection.executemany(
            """
            DELETE FROM features
            WHERE game_pk = ?
              AND feature_name = ?
              AND as_of_timestamp = ?
              AND ((window_size IS NULL AND ? IS NULL) OR window_size = ?)
            """,
            [
                (game_pk, feature_name, as_of_timestamp, window_size, window_size)
                for game_pk, feature_name, _, window_size, as_of_timestamp in rows
            ],
        )
        connection.executemany(
            """
            INSERT INTO features (game_pk, feature_name, feature_value, window_size, as_of_timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()


def _history_mean(
    dataframe: pd.DataFrame,
    column: str,
    *,
    window: int | None,
    default: float,
) -> float:
    if dataframe.empty or column not in dataframe:
        return float(default)

    values = pd.to_numeric(dataframe[column], errors="coerce").dropna()
    if values.empty:
        return float(default)
    if window is not None:
        values = values.tail(window)
    return float(values.mean())


def _empty_metric_history(extra_columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=["game_date", *extra_columns])


def _first_column(dataframe: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    normalized_columns = {str(column).strip().lower(): str(column) for column in dataframe.columns}
    for candidate in candidates:
        match = normalized_columns.get(candidate.strip().lower())
        if match is not None:
            return match
    return None


def _extract_numeric(dataframe: pd.DataFrame, *candidates: str | None) -> pd.Series:
    valid_candidates = [candidate for candidate in candidates if candidate]
    if not valid_candidates:
        return pd.Series(0.0, index=dataframe.index, dtype=float)

    column = _first_column(dataframe, tuple(valid_candidates))
    if column is None:
        return pd.Series(0.0, index=dataframe.index, dtype=float)
    return pd.to_numeric(dataframe[column], errors="coerce").fillna(0.0)


def _safe_ratio(
    *,
    numerator: pd.Series,
    denominator: pd.Series,
    default: float,
) -> pd.Series:
    resolved_denominator = denominator.replace(0, pd.NA)
    ratio = (numerator / resolved_denominator).astype(float)
    return ratio.fillna(default)


def _normalize_position(value: object) -> str:
    if value is None:
        return ""

    text = str(value).strip().upper()
    if not text:
        return ""

    tokens = (
        text.replace("/", " ")
        .replace("-", " ")
        .replace(",", " ")
        .split()
    )
    for token in tokens:
        resolved = POSITION_ALIASES.get(token, token)
        if resolved in POSITION_IMPORTANCE_WEIGHTS:
            return resolved
    return POSITION_ALIASES.get(text, text)


def _normalize_team_label(value: object) -> str:
    if value is None:
        return ""

    label = " ".join(str(value).strip().upper().split())
    return TEAM_LABEL_TO_CODE.get(label, label)


def _normalize_name(value: object) -> str:
    if value is None:
        return ""

    text = str(value).strip()
    if not text:
        return ""
    if "," in text:
        last_name, first_name = [part.strip() for part in text.split(",", 1)]
        text = f"{first_name} {last_name}".strip()
    return " ".join(text.lower().split())
