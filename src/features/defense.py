from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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
from src.db import DEFAULT_DB_PATH, init_db, sqlite_connection
from src.features.adjustments.abs_adjustment import (
    DEFAULT_ABS_RETENTION_FACTOR,
    adjust_framing_for_abs,
    estimate_framing_retention_proxy,
)
from src.features.marcel_blend import blend_value, get_regression_weight
from src.models.features import GameFeatures


DEFAULT_WINDOWS: tuple[int, ...] = (30, 60)
DEFAULT_REGRESSION_WEIGHT = int(get_regression_weight("defense"))
DEFAULT_DEFENSIVE_EFFICIENCY = 0.700
DEFAULT_SNAPSHOT_GAMES_PLAYED = 162
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
DEFAULT_FIELDING_BASELINES: dict[str, float] = {"drs": 0.0, "oaa": 0.0}
DEFAULT_FRAMING_BASELINES: dict[str, float] = {"raw_framing": 0.0}


_FieldingFetcher = Callable[..., pd.DataFrame]
_FramingFetcher = Callable[..., pd.DataFrame]
_TeamLogsFetcher = Callable[..., pd.DataFrame]


def compute_defense_features(
    game_date: str | date | datetime,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    regression_weight: int = DEFAULT_REGRESSION_WEIGHT,
    abs_retention_factor: float = DEFAULT_ABS_RETENTION_FACTOR,
    refresh: bool = False,
    roster_turnover_by_team: Mapping[str, float] | None = None,
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

    prior_season = target_day.year - 1
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
    prior_defensive_efficiency_histories = _build_defensive_efficiency_histories(
        season=prior_season,
        target_day=date(target_day.year, 1, 1),
        team_logs_fetcher=team_logs_fetcher,
        refresh=refresh,
    )
    prior_games_played_lookup = {
        team: len(history)
        for team, history in prior_defensive_efficiency_histories.items()
        if not history.empty
    }
    defensive_efficiency_defaults = _resolve_metric_defaults(
        prior_defensive_efficiency_histories,
        value_columns=("defensive_efficiency",),
        fallback_defaults={"defensive_efficiency": DEFAULT_DEFENSIVE_EFFICIENCY},
    )
    prior_defensive_efficiency_baselines = _build_metric_baselines(
        prior_defensive_efficiency_histories,
        value_columns=("defensive_efficiency",),
        default_values=defensive_efficiency_defaults,
    )
    defensive_efficiency_baseline = defensive_efficiency_defaults["defensive_efficiency"]

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
        allow_snapshot_fallback=False,
    )

    prior_fielding_frame = _safe_fetch_fielding_frame(
        season=prior_season,
        fielding_fetcher=fielding_fetcher,
        refresh=refresh,
    )
    prior_normalized_fielding = _normalize_fielding_frame(prior_fielding_frame)
    prior_fielding_histories = _build_metric_histories(
        prior_normalized_fielding,
        target_day=target_day,
        value_columns=FIELDING_VALUE_COLUMNS,
        games_played_lookup=prior_games_played_lookup,
        allow_snapshot_fallback=True,
    )
    fielding_defaults = _resolve_metric_defaults(
        prior_fielding_histories,
        value_columns=FIELDING_VALUE_COLUMNS,
        fallback_defaults=DEFAULT_FIELDING_BASELINES,
    )
    prior_fielding_baselines = _build_metric_baselines(
        prior_fielding_histories,
        value_columns=FIELDING_VALUE_COLUMNS,
        default_values=fielding_defaults,
    )
    name_team_lookup = _build_name_team_lookup(normalized_fielding)
    prior_name_team_lookup = _build_name_team_lookup(prior_normalized_fielding)

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
        allow_snapshot_fallback=False,
    )

    prior_framing_frame = _safe_fetch_framing_frame(
        season=prior_season,
        framing_fetcher=framing_fetcher,
        refresh=refresh,
    )
    prior_normalized_framing = _normalize_framing_frame(
        prior_framing_frame,
        prior_name_team_lookup,
    )
    prior_framing_histories = _build_metric_histories(
        prior_normalized_framing,
        target_day=target_day,
        value_columns=("raw_framing",),
        games_played_lookup=prior_games_played_lookup,
        allow_snapshot_fallback=True,
    )
    framing_defaults = _resolve_metric_defaults(
        prior_framing_histories,
        value_columns=("raw_framing",),
        fallback_defaults=DEFAULT_FRAMING_BASELINES,
    )
    prior_framing_baselines = _build_metric_baselines(
        prior_framing_histories,
        value_columns=("raw_framing",),
        default_values=framing_defaults,
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
            fielding_baseline = prior_fielding_baselines.get(team, fielding_defaults)
            framing_baseline = prior_framing_baselines.get(team, framing_defaults)
            defensive_efficiency_team_baseline = prior_defensive_efficiency_baselines.get(
                team,
                {"defensive_efficiency": defensive_efficiency_baseline},
            )["defensive_efficiency"]
            season_games_played = _resolve_games_played(
                games_played_lookup.get(team, 0),
                fielding_history,
                framing_history,
                defensive_efficiency_history,
            )
            roster_turnover_pct = None if roster_turnover_by_team is None else roster_turnover_by_team.get(team)

            season_values = _build_feature_values(
                fielding_history=fielding_history,
                framing_history=framing_history,
                defensive_efficiency_history=defensive_efficiency_history,
                fielding_baseline=fielding_baseline,
                fielding_defaults=fielding_defaults,
                framing_baseline=framing_baseline,
                framing_defaults=framing_defaults,
                abs_retention_factor=abs_retention_factor,
                abs_active=abs_active,
                defensive_efficiency_baseline=defensive_efficiency_team_baseline,
                defensive_efficiency_league_average=defensive_efficiency_baseline,
                games_played=season_games_played,
                regression_weight=regression_weight,
                roster_turnover_pct=roster_turnover_pct,
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
                    fielding_baseline=fielding_baseline,
                    fielding_defaults=fielding_defaults,
                    framing_baseline=framing_baseline,
                    framing_defaults=framing_defaults,
                    abs_retention_factor=abs_retention_factor,
                    abs_active=abs_active,
                    defensive_efficiency_baseline=defensive_efficiency_team_baseline,
                    defensive_efficiency_league_average=defensive_efficiency_baseline,
                    games_played=season_games_played,
                    regression_weight=regression_weight,
                    roster_turnover_pct=roster_turnover_pct,
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


def compute_defense_features_for_schedule(
    schedule: pd.DataFrame,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    regression_weight: int = DEFAULT_REGRESSION_WEIGHT,
    abs_retention_factor: float = DEFAULT_ABS_RETENTION_FACTOR,
    refresh: bool = False,
    roster_turnover_lookup: dict[tuple[str, str], float] | None = None,
    fielding_fetcher: _FieldingFetcher = fetch_fielding_stats,
    framing_fetcher: _FramingFetcher = fetch_catcher_framing,
    team_logs_fetcher: _TeamLogsFetcher = fetch_team_game_logs,
) -> list[GameFeatures]:
    """Compute and persist lagged team defense features for a schedule slice."""

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
    if "is_abs_active" in working_schedule.columns:
        working_schedule["is_abs_active"] = pd.to_numeric(
            working_schedule["is_abs_active"],
            errors="coerce",
        ).fillna(1).astype(int)
    else:
        working_schedule["is_abs_active"] = 1
    working_schedule = working_schedule.dropna(subset=["game_date"]).copy()
    if working_schedule.empty:
        return []
    working_schedule = (
        working_schedule.sort_values(["game_date", "game_pk"], kind="mergesort")
        .drop_duplicates(subset=["game_pk"], keep="last")
        .reset_index(drop=True)
    )

    season_values = pd.to_datetime(working_schedule["game_date"], errors="coerce").dt.year.dropna().astype(int)
    if season_values.empty:
        return []

    unique_seasons = sorted(set(season_values.tolist()))
    log_seasons = sorted({*unique_seasons, *(season - 1 for season in unique_seasons)})
    normalized_team_logs_by_season = {
        season: _load_normalized_team_logs_for_season(
            season=season,
            team_logs_fetcher=team_logs_fetcher,
            refresh=refresh,
        )
        for season in log_seasons
    }
    defensive_efficiency_histories_by_season = {
        season: _build_defensive_efficiency_histories_from_preloaded_logs(
            normalized_team_logs_by_season.get(season, {})
        )
        for season in log_seasons
    }

    fetch_seasons = sorted({*unique_seasons, *(season - 1 for season in unique_seasons)})
    fielding_frames_by_season = {
        season: _safe_fetch_fielding_frame(
            season=season,
            fielding_fetcher=fielding_fetcher,
            refresh=refresh,
        )
        for season in fetch_seasons
    }
    framing_frames_by_season = {
        season: _safe_fetch_framing_frame(
            season=season,
            framing_fetcher=framing_fetcher,
            refresh=refresh,
        )
        for season in fetch_seasons
    }
    normalized_roster_turnover_lookup = _normalize_schedule_roster_turnover_lookup(roster_turnover_lookup)

    features: list[GameFeatures] = []
    for season, season_schedule in working_schedule.groupby(
        pd.to_datetime(working_schedule["game_date"], errors="coerce").dt.year,
        sort=True,
    ):
        season_int = int(season)
        prior_season = season_int - 1
        target_dates = sorted({value for value in season_schedule["game_date"].dropna().tolist()})
        if not target_dates:
            continue

        defensive_efficiency_histories = defensive_efficiency_histories_by_season.get(season_int, {})
        defensive_efficiency_slices_by_day = _build_metric_history_slices_for_dates(
            season_schedule=season_schedule,
            target_dates=target_dates,
            team_histories=defensive_efficiency_histories,
        )

        prior_defensive_efficiency_histories = defensive_efficiency_histories_by_season.get(prior_season, {})
        prior_games_played_lookup = {
            team: len(history)
            for team, history in prior_defensive_efficiency_histories.items()
            if not history.empty
        }
        defensive_efficiency_defaults = _resolve_metric_defaults(
            prior_defensive_efficiency_histories,
            value_columns=("defensive_efficiency",),
            fallback_defaults={"defensive_efficiency": DEFAULT_DEFENSIVE_EFFICIENCY},
        )
        prior_defensive_efficiency_baselines = _build_metric_baselines(
            prior_defensive_efficiency_histories,
            value_columns=("defensive_efficiency",),
            default_values=defensive_efficiency_defaults,
        )
        defensive_efficiency_league_average = defensive_efficiency_defaults["defensive_efficiency"]

        normalized_fielding = _normalize_fielding_frame(
            fielding_frames_by_season.get(season_int, pd.DataFrame())
        )
        fielding_histories = _build_full_metric_histories(
            normalized_fielding,
            value_columns=FIELDING_VALUE_COLUMNS,
        )
        fielding_slices_by_day = _build_metric_history_slices_for_dates(
            season_schedule=season_schedule,
            target_dates=target_dates,
            team_histories=fielding_histories,
        )

        prior_normalized_fielding = _normalize_fielding_frame(
            fielding_frames_by_season.get(prior_season, pd.DataFrame())
        )
        prior_fielding_histories = _build_metric_histories(
            prior_normalized_fielding,
            target_day=date(season_int, 1, 1),
            value_columns=FIELDING_VALUE_COLUMNS,
            games_played_lookup=prior_games_played_lookup,
            allow_snapshot_fallback=True,
        )
        fielding_defaults = _resolve_metric_defaults(
            prior_fielding_histories,
            value_columns=FIELDING_VALUE_COLUMNS,
            fallback_defaults=DEFAULT_FIELDING_BASELINES,
        )
        prior_fielding_baselines = _build_metric_baselines(
            prior_fielding_histories,
            value_columns=FIELDING_VALUE_COLUMNS,
            default_values=fielding_defaults,
        )
        name_team_lookup = _build_name_team_lookup(normalized_fielding)
        prior_name_team_lookup = _build_name_team_lookup(prior_normalized_fielding)

        normalized_framing = _normalize_framing_frame(
            framing_frames_by_season.get(season_int, pd.DataFrame()),
            name_team_lookup,
        )
        framing_histories = _build_full_metric_histories(
            normalized_framing,
            value_columns=("raw_framing",),
        )
        framing_slices_by_day = _build_metric_history_slices_for_dates(
            season_schedule=season_schedule,
            target_dates=target_dates,
            team_histories=framing_histories,
        )

        prior_normalized_framing = _normalize_framing_frame(
            framing_frames_by_season.get(prior_season, pd.DataFrame()),
            prior_name_team_lookup,
        )
        prior_framing_histories = _build_metric_histories(
            prior_normalized_framing,
            target_day=date(season_int, 1, 1),
            value_columns=("raw_framing",),
            games_played_lookup=prior_games_played_lookup,
            allow_snapshot_fallback=True,
        )
        framing_defaults = _resolve_metric_defaults(
            prior_framing_histories,
            value_columns=("raw_framing",),
            fallback_defaults=DEFAULT_FRAMING_BASELINES,
        )
        prior_framing_baselines = _build_metric_baselines(
            prior_framing_histories,
            value_columns=("raw_framing",),
            default_values=framing_defaults,
        )

        for target_day, day_games in season_schedule.groupby("game_date", sort=True):
            as_of_timestamp = datetime.combine(
                target_day - timedelta(days=1),
                time.min,
                tzinfo=timezone.utc,
            )
            day_fielding_histories = fielding_slices_by_day.get(target_day, {})
            day_framing_histories = framing_slices_by_day.get(target_day, {})
            day_defensive_efficiency_histories = defensive_efficiency_slices_by_day.get(target_day, {})

            for game in day_games.to_dict(orient="records"):
                abs_active = bool(game.get("is_abs_active", 1))
                for side_name, team_key in (("home", "home_team"), ("away", "away_team")):
                    team = str(game[team_key]).strip().upper()
                    fielding_history = day_fielding_histories.get(
                        team,
                        _empty_metric_history(FIELDING_VALUE_COLUMNS),
                    )
                    framing_history = day_framing_histories.get(
                        team,
                        _empty_metric_history(("raw_framing",)),
                    )
                    defensive_efficiency_history = day_defensive_efficiency_histories.get(
                        team,
                        _empty_metric_history(("defensive_efficiency",)),
                    )
                    fielding_baseline = prior_fielding_baselines.get(team, fielding_defaults)
                    framing_baseline = prior_framing_baselines.get(team, framing_defaults)
                    defensive_efficiency_team_baseline = prior_defensive_efficiency_baselines.get(
                        team,
                        {"defensive_efficiency": defensive_efficiency_league_average},
                    )["defensive_efficiency"]
                    season_games_played = _resolve_games_played(
                        len(defensive_efficiency_history),
                        fielding_history,
                        framing_history,
                        defensive_efficiency_history,
                    )
                    roster_turnover_pct = (
                        None
                        if normalized_roster_turnover_lookup is None
                        else normalized_roster_turnover_lookup.get((target_day.isoformat(), team))
                    )

                    season_values = _build_feature_values(
                        fielding_history=fielding_history,
                        framing_history=framing_history,
                        defensive_efficiency_history=defensive_efficiency_history,
                        fielding_baseline=fielding_baseline,
                        fielding_defaults=fielding_defaults,
                        framing_baseline=framing_baseline,
                        framing_defaults=framing_defaults,
                        abs_retention_factor=abs_retention_factor,
                        abs_active=abs_active,
                        defensive_efficiency_baseline=defensive_efficiency_team_baseline,
                        defensive_efficiency_league_average=defensive_efficiency_league_average,
                        games_played=season_games_played,
                        regression_weight=regression_weight,
                        roster_turnover_pct=roster_turnover_pct,
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
                            fielding_baseline=fielding_baseline,
                            fielding_defaults=fielding_defaults,
                            framing_baseline=framing_baseline,
                            framing_defaults=framing_defaults,
                            abs_retention_factor=abs_retention_factor,
                            abs_active=abs_active,
                            defensive_efficiency_baseline=defensive_efficiency_team_baseline,
                            defensive_efficiency_league_average=defensive_efficiency_league_average,
                            games_played=season_games_played,
                            regression_weight=regression_weight,
                            roster_turnover_pct=roster_turnover_pct,
                            window=window,
                        )
                        features.extend(
                            _to_feature_rows(
                                game_pk=int(game["game_pk"]),
                                side_name=side_name,
                                feature_values=window_values,
                                window_label=f"{int(window)}g",
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

    return adjust_framing_for_abs(
        raw_framing_runs,
        retention_factor=retention_factor,
        abs_active=abs_active,
    )


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


def _load_normalized_team_logs_for_season(
    *,
    season: int,
    team_logs_fetcher: _TeamLogsFetcher,
    refresh: bool,
) -> dict[str, pd.DataFrame]:
    normalized_logs: dict[str, pd.DataFrame] = {}
    for offense_team in TEAM_CODES:
        try:
            raw_logs = team_logs_fetcher(season, offense_team, refresh=refresh)
        except Exception:
            raw_logs = pd.DataFrame()
        normalized_logs[offense_team] = _normalize_offensive_game_logs(raw_logs)
    return normalized_logs


def _normalize_schedule_roster_turnover_lookup(
    roster_turnover_lookup: Mapping[tuple[str, str], float] | None,
) -> dict[tuple[str, str], float] | None:
    if not roster_turnover_lookup:
        return None

    normalized: dict[tuple[str, str], float] = {}
    for (game_date, team), value in roster_turnover_lookup.items():
        normalized[(_coerce_date(game_date).isoformat(), str(team).strip().upper())] = float(value)
    return normalized


def _build_defensive_efficiency_histories_from_preloaded_logs(
    normalized_team_logs: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    observations: list[dict[str, Any]] = []
    for normalized in normalized_team_logs.values():
        if normalized.empty:
            continue

        for row in normalized.to_dict(orient="records"):
            defensive_team = str(row.get("opponent") or "").strip().upper()
            game_date = row.get("game_date")
            if not defensive_team or pd.isna(game_date):
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
        history = (
            group.groupby("game_date", as_index=False)["defensive_efficiency"]
            .mean()
            .sort_values("game_date")
            .reset_index(drop=True)
        )
        histories[str(team)] = _append_game_day_ordinals(history)
    return histories


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

        game_dates = _to_tz_naive_datetime_series(normalized["game_date"])
        prior_rows = normalized.loc[game_dates < cutoff].copy()
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
    result["game_date"] = _to_tz_naive_datetime_series(
        dataframe[date_column] if date_column is not None else pd.Series(dtype=object),
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
        ab = _extract_numeric(dataframe, "AB", "Offense_AB", "Batting Stats_AB")
        hits = _extract_numeric(dataframe, "H", "Offense_H", "Batting Stats_H")
        home_runs = _extract_numeric(dataframe, "HR", "Offense_HR", "Batting Stats_HR")
        strikeouts = _extract_numeric(
            dataframe,
            "SO",
            "K",
            "Offense_SO",
            "Offense_K",
            "Batting Stats_SO",
        )
        sacrifice_flies = _extract_numeric(dataframe, "SF", "Offense_SF", "Batting Stats_SF")
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
    result["game_date"] = _to_tz_naive_datetime_series(
        dataframe[date_column] if date_column is not None else pd.Series(dtype=object),
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

    result["game_date"] = _to_tz_naive_datetime_series(
        dataframe[date_column] if date_column is not None else pd.Series(dtype=object),
    )
    result["raw_framing"] = _extract_numeric(dataframe, framing_column)
    return result.loc[result["team"].astype(str) != ""].reset_index(drop=True)


def _build_metric_histories(
    dataframe: pd.DataFrame,
    *,
    target_day: date,
    value_columns: Sequence[str],
    games_played_lookup: dict[str, int],
    allow_snapshot_fallback: bool,
) -> dict[str, pd.DataFrame]:
    if dataframe.empty:
        return {}

    histories: dict[str, pd.DataFrame] = {}
    cutoff = pd.Timestamp(target_day)
    for team, group in dataframe.groupby("team", dropna=True):
        team_group = group.copy()
        game_dates = _to_tz_naive_datetime_series(team_group["game_date"])
        team_group["game_date"] = game_dates
        has_dated_history = game_dates.notna().any()
        dated_rows = team_group.loc[
            game_dates.notna() & (game_dates < cutoff)
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

        if not allow_snapshot_fallback:
            continue

        totals = team_group[list(value_columns)].sum(numeric_only=True)
        games_played = max(int(games_played_lookup.get(str(team), DEFAULT_SNAPSHOT_GAMES_PLAYED)), 1)
        histories[str(team)] = pd.DataFrame(
            {
                "game_date": [pd.Timestamp(target_day - timedelta(days=1))],
                **{column: [float(totals.get(column, 0.0)) / games_played] for column in value_columns},
            }
        )

    return histories


def _build_full_metric_histories(
    dataframe: pd.DataFrame,
    *,
    value_columns: Sequence[str],
) -> dict[str, pd.DataFrame]:
    if dataframe.empty:
        return {}

    histories: dict[str, pd.DataFrame] = {}
    for team, group in dataframe.groupby("team", dropna=True):
        team_group = group.copy()
        game_dates = _to_tz_naive_datetime_series(team_group["game_date"])
        dated_rows = team_group.loc[game_dates.notna()].copy()
        if dated_rows.empty:
            continue

        history = (
            dated_rows.groupby("game_date", as_index=False)[list(value_columns)]
            .sum()
            .sort_values("game_date")
            .reset_index(drop=True)
        )
        histories[str(team)] = _append_game_day_ordinals(history)

    return histories


def _build_metric_history_slices_for_dates(
    *,
    season_schedule: pd.DataFrame,
    target_dates: Sequence[date],
    team_histories: Mapping[str, pd.DataFrame],
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
        target_ordinal = target_day.toordinal()
        day_lookup: dict[str, pd.DataFrame] = {}
        for team in team_codes:
            history = team_histories.get(team)
            if history is None or history.empty:
                continue
            cutoff = history["game_day_ordinal"].to_numpy().searchsorted(target_ordinal, side="left")
            day_lookup[team] = history.iloc[:cutoff].reset_index(drop=True)
        slices_by_day[target_day] = day_lookup

    return slices_by_day


def _resolve_metric_defaults(
    histories: Mapping[str, pd.DataFrame],
    *,
    value_columns: Sequence[str],
    fallback_defaults: Mapping[str, float],
) -> dict[str, float]:
    defaults: dict[str, float] = {}
    for column in value_columns:
        values = [
            frame[column]
            for frame in histories.values()
            if not frame.empty and column in frame
        ]
        if values:
            combined = pd.concat(values, ignore_index=True).dropna()
            if not combined.empty:
                defaults[column] = float(combined.mean())
                continue

        defaults[column] = float(fallback_defaults[column])

    return defaults


def _build_metric_baselines(
    histories: Mapping[str, pd.DataFrame],
    *,
    value_columns: Sequence[str],
    default_values: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    baselines: dict[str, dict[str, float]] = {}
    for team, history in histories.items():
        baselines[str(team)] = {
            column: _history_mean(
                history,
                column,
                window=None,
                default=float(default_values[column]),
            )
            for column in value_columns
        }
    return baselines


def _build_feature_values(
    *,
    fielding_history: pd.DataFrame,
    framing_history: pd.DataFrame,
    defensive_efficiency_history: pd.DataFrame,
    fielding_baseline: Mapping[str, float],
    fielding_defaults: Mapping[str, float],
    framing_baseline: Mapping[str, float],
    framing_defaults: Mapping[str, float],
    abs_retention_factor: float,
    abs_active: bool,
    defensive_efficiency_baseline: float,
    defensive_efficiency_league_average: float,
    games_played: int,
    regression_weight: int,
    roster_turnover_pct: float | None,
    window: int | None,
) -> dict[str, float]:
    current_raw_framing = _history_mean(
        framing_history,
        "raw_framing",
        window=window,
        default=float(framing_baseline.get("raw_framing", 0.0)),
    )
    blended_raw_framing = blend_value(
        current_raw_framing,
        games_played=games_played,
        metric_type="defense",
        prior_value=float(framing_baseline.get("raw_framing", 0.0)),
        league_average=float(framing_defaults.get("raw_framing", 0.0)),
        regression_weight=regression_weight,
        roster_turnover_pct=roster_turnover_pct,
    )
    adjusted_framing = adjust_framing_runs(
        blended_raw_framing,
        retention_factor=abs_retention_factor,
        abs_active=abs_active,
    )
    return {
        "drs": blend_value(
            _history_mean(
                fielding_history,
                "drs",
                window=window,
                default=float(fielding_baseline.get("drs", 0.0)),
            ),
            games_played=games_played,
            metric_type="defense",
            prior_value=float(fielding_baseline.get("drs", 0.0)),
            league_average=float(fielding_defaults.get("drs", 0.0)),
            regression_weight=regression_weight,
            roster_turnover_pct=roster_turnover_pct,
        ),
        "oaa": blend_value(
            _history_mean(
                fielding_history,
                "oaa",
                window=window,
                default=float(fielding_baseline.get("oaa", 0.0)),
            ),
            games_played=games_played,
            metric_type="defense",
            prior_value=float(fielding_baseline.get("oaa", 0.0)),
            league_average=float(fielding_defaults.get("oaa", 0.0)),
            regression_weight=regression_weight,
            roster_turnover_pct=roster_turnover_pct,
        ),
        "defensive_efficiency": blend_value(
            _history_mean(
                defensive_efficiency_history,
                "defensive_efficiency",
                window=window,
                default=defensive_efficiency_baseline,
            ),
            games_played=games_played,
            metric_type="defense",
            prior_value=defensive_efficiency_baseline,
            league_average=defensive_efficiency_league_average,
            regression_weight=regression_weight,
            roster_turnover_pct=roster_turnover_pct,
        ),
        "raw_framing": blended_raw_framing,
        "adjusted_framing": adjusted_framing,
        "framing_retention_proxy": estimate_framing_retention_proxy(
            blended_raw_framing,
            adjusted_framing_runs=adjusted_framing,
            abs_active=abs_active,
            retention_factor=abs_retention_factor,
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

    with sqlite_connection(db_path, builder_optimized=True) as connection:
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


def _resolve_games_played(base_games_played: int, *histories: pd.DataFrame) -> int:
    history_lengths = [len(history) for history in histories if not history.empty]
    return int(max([base_games_played, *history_lengths], default=0))


def _empty_metric_history(extra_columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=["game_date", *extra_columns])


def _append_game_day_ordinals(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty or "game_date" not in dataframe.columns:
        return dataframe.copy()

    history = dataframe.copy()
    history["game_day_ordinal"] = history["game_date"].map(
        lambda value: pd.Timestamp(value).date().toordinal() if pd.notna(value) else None
    )
    return history


def _to_tz_naive_datetime_series(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    if getattr(parsed.dt, "tz", None) is not None:
        return parsed.dt.tz_convert(None)
    return parsed


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
    ratio = pd.to_numeric(numerator / resolved_denominator, errors="coerce")
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
