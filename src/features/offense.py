from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.clients.statcast_client import (
    fetch_batting_stats,
    fetch_statcast_range,
    fetch_team_game_logs,
)
from src.db import DEFAULT_DB_PATH, init_db, sqlite_connection
from src.features.marcel_blend import blend_value, get_regression_weight
from src.models.features import GameFeatures


DEFAULT_WINDOWS: tuple[int, ...] = (7, 14, 30, 60)
DEFAULT_REGRESSION_WEIGHT = int(get_regression_weight("offense"))
DEFAULT_MIN_PERIODS = 1
METRICS: tuple[str, ...] = (
    "wrc_plus",
    "woba",
    "xwoba",
    "woba_minus_xwoba",
    "iso",
    "barrel_pct",
    "babip",
    "k_pct",
    "bb_pct",
)
WOBA_WEIGHTS = {
    "bb": 0.69,
    "hbp": 0.72,
    "single": 0.89,
    "double": 1.27,
    "triple": 1.62,
    "hr": 2.10,
}
LEAGUE_WOBA_BASELINE = 0.320
LEAGUE_WRC_PLUS_BASELINE = 100.0
LEAGUE_BARREL_PCT_BASELINE = 7.0
REPO_ROOT = Path(__file__).resolve().parents[2]
DERIVED_CACHE_ROOT = REPO_ROOT / "data" / "raw" / "derived" / "offense"
OFFENSE_STATCAST_CACHE_VERSION = 2


_TeamLogsFetcher = Callable[..., pd.DataFrame]
_BattingStatsFetcher = Callable[..., pd.DataFrame]
_OffenseStatcastFetcher = Callable[..., pd.DataFrame]


@dataclass(frozen=True, slots=True)
class _WindowMetric:
    value: float
    games_played: int


def compute_offensive_features(
    game_date: str | date | datetime,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    regression_weight: int = DEFAULT_REGRESSION_WEIGHT,
    min_periods: int = DEFAULT_MIN_PERIODS,
    refresh: bool = False,
    lineup_player_ids: Mapping[tuple[int, str], Sequence[int]] | None = None,
    roster_turnover_by_team: Mapping[str, float] | None = None,
    team_logs_fetcher: _TeamLogsFetcher = fetch_team_game_logs,
    batting_stats_fetcher: _BattingStatsFetcher = fetch_batting_stats,
    offense_statcast_fetcher: _OffenseStatcastFetcher | None = None,
) -> list[GameFeatures]:
    """Compute and persist lagged offensive features for games on a target date."""

    target_day = _coerce_date(game_date)
    database_path = Path(db_path)
    init_db(database_path)

    games = _load_games_for_date(database_path, target_day)
    if games.empty:
        return []

    resolved_offense_statcast_fetcher = (
        offense_statcast_fetcher or _fetch_season_offense_statcast_metrics
    )
    current_statcast_metrics = _normalize_statcast_offense_metrics(
        resolved_offense_statcast_fetcher(
            target_day.year,
            db_path=database_path,
            end_date=target_day - timedelta(days=1),
            refresh=refresh,
        )
    )
    prior_statcast_metrics = _normalize_statcast_offense_metrics(
        resolved_offense_statcast_fetcher(
            target_day.year - 1,
            db_path=database_path,
            end_date=None,
            refresh=refresh,
        )
    )

    teams = sorted({*games["home_team"].unique().tolist(), *games["away_team"].unique().tolist()})
    team_metrics, league_woba = _build_team_game_metrics(
        season=target_day.year,
        teams=teams,
        target_day=target_day,
        refresh=refresh,
        team_logs_fetcher=team_logs_fetcher,
        statcast_metrics=current_statcast_metrics,
    )
    prior_team_baselines = _build_prior_team_baselines(
        prior_season=target_day.year - 1,
        teams=teams,
        refresh=refresh,
        team_logs_fetcher=team_logs_fetcher,
        league_woba=league_woba,
        statcast_metrics=prior_statcast_metrics,
    )
    lineup_metric_lookup = _build_lineup_metric_lookup(
        lineup_player_ids=lineup_player_ids or {},
        season=target_day.year,
        target_day=target_day,
        windows=windows,
        league_woba=league_woba,
        refresh=refresh,
        batting_stats_fetcher=batting_stats_fetcher,
        statcast_metrics=current_statcast_metrics,
    )
    lineup_prior_lookup, lineup_all_first_year_lookup = _build_lineup_prior_metric_lookup(
        lineup_player_ids=lineup_player_ids or {},
        prior_season=target_day.year - 1,
        refresh=refresh,
        batting_stats_fetcher=batting_stats_fetcher,
        statcast_metrics=prior_statcast_metrics,
    )

    as_of_timestamp = datetime.combine(
        target_day - timedelta(days=1),
        time.min,
        tzinfo=timezone.utc,
    )

    features: list[GameFeatures] = []
    for game in games.to_dict(orient="records"):
        game_pk = int(game["game_pk"])
        for side_name, team_key in (("home", "home_team"), ("away", "away_team")):
            team = str(game[team_key])
            team_frame = team_metrics.get(team, pd.DataFrame())
            team_priors = prior_team_baselines.get(team, {})
            season_games_played = len(team_frame)
            roster_turnover_pct = (
                None if roster_turnover_by_team is None else roster_turnover_by_team.get(team)
            )
            for window in windows:
                current_window_metrics = _rolling_metrics_as_of(
                    team_frame=team_frame,
                    metric_names=(
                        "woba",
                        "xwoba",
                        "woba_minus_xwoba",
                        "iso",
                        "barrel_pct",
                        "babip",
                        "k_pct",
                        "bb_pct",
                    ),
                    window=window,
                    min_periods=min_periods,
                )

                current_team_values = {
                    metric: current_window_metrics[metric].value
                    for metric in (
                        "woba",
                        "xwoba",
                        "woba_minus_xwoba",
                        "iso",
                        "barrel_pct",
                        "babip",
                        "k_pct",
                        "bb_pct",
                    )
                }
                current_team_values["wrc_plus"] = _team_wrc_plus(
                    current_team_values["woba"],
                    league_woba,
                )

                team_feature_values: dict[str, float] = {}
                for metric in METRICS:
                    blended_value = _apply_metric_blend(
                        metric=metric,
                        current_value=current_team_values[metric],
                        prior_value=team_priors.get(metric, _default_metric_baseline(metric)),
                        games_played=season_games_played,
                        regression_weight=regression_weight,
                        roster_turnover_pct=roster_turnover_pct,
                    )
                    team_feature_values[metric] = blended_value
                    features.append(
                        GameFeatures(
                            game_pk=game_pk,
                            feature_name=f"{side_name}_team_{metric}_{window}g",
                            feature_value=blended_value,
                            window_size=window,
                            as_of_timestamp=as_of_timestamp,
                        )
                    )

                lineup_feature_values = team_feature_values.copy()
                lineup_metrics = lineup_metric_lookup.get((game_pk, team, window))
                if lineup_metrics is not None:
                    lineup_prior_metrics = lineup_prior_lookup.get((game_pk, team))
                    lineup_is_first_year = lineup_all_first_year_lookup.get((game_pk, team), False)
                    for metric in METRICS:
                        lineup_feature_values[metric] = _apply_metric_blend(
                            metric=metric,
                            current_value=lineup_metrics.get(metric, team_feature_values[metric]),
                            prior_value=(
                                lineup_prior_metrics.get(
                                    metric,
                                    team_priors.get(metric, _default_metric_baseline(metric)),
                                )
                                if lineup_prior_metrics is not None
                                else team_priors.get(metric, _default_metric_baseline(metric))
                            ),
                            games_played=season_games_played,
                            regression_weight=regression_weight,
                            roster_turnover_pct=roster_turnover_pct,
                            is_first_year=lineup_is_first_year,
                        )

                for metric in METRICS:
                    features.append(
                        GameFeatures(
                            game_pk=game_pk,
                            feature_name=f"{side_name}_lineup_{metric}_{window}g",
                            feature_value=lineup_feature_values[metric],
                            window_size=window,
                            as_of_timestamp=as_of_timestamp,
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
            SELECT game_pk, date AS game_date, home_team, away_team
            FROM games
            WHERE substr(date, 1, 10) = ?
            ORDER BY game_pk
            """,
            connection,
            params=(target_day.isoformat(),),
        )


def _load_season_games(
    db_path: Path,
    *,
    season: int,
    end_date: date | None,
) -> pd.DataFrame:
    query = """
        SELECT game_pk, date AS game_date
        FROM games
        WHERE substr(date, 1, 4) = ?
    """
    params: list[Any] = [str(season)]
    if end_date is not None:
        query += " AND substr(date, 1, 10) <= ?"
        params.append(end_date.isoformat())
    query += " ORDER BY date, game_pk"

    with sqlite_connection(db_path, builder_optimized=True) as connection:
        games = pd.read_sql_query(query, connection, params=params)

    if games.empty:
        return pd.DataFrame(columns=["game_pk", "game_date"])

    games["game_pk"] = pd.to_numeric(games["game_pk"], errors="coerce").astype("Int64")
    games["game_date"] = _to_tz_naive_datetime_series(games["game_date"])
    games = games.dropna(subset=["game_pk", "game_date"]).copy()
    if games.empty:
        return pd.DataFrame(columns=["game_pk", "game_date"])
    games["game_pk"] = games["game_pk"].astype(int)
    return games.reset_index(drop=True)


def _build_team_game_metrics(
    *,
    season: int,
    teams: Sequence[str],
    target_day: date,
    refresh: bool,
    team_logs_fetcher: _TeamLogsFetcher,
    statcast_metrics: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], float]:
    team_frames: dict[str, pd.DataFrame] = {}
    league_woba_numerator = 0.0
    league_woba_denominator = 0.0

    for team in teams:
        raw_logs = team_logs_fetcher(season, team, refresh=refresh)
        team_statcast = statcast_metrics.loc[statcast_metrics["team"] == team].copy()
        game_metrics = _compute_game_level_metrics(raw_logs, statcast_metrics=team_statcast)
        filtered = game_metrics.loc[game_metrics["game_date"].dt.date < target_day].reset_index(
            drop=True
        )
        team_frames[team] = filtered

        if not filtered.empty:
            league_woba_numerator += float(filtered["woba_numerator"].sum())
            league_woba_denominator += float(filtered["woba_denominator"].sum())

    league_woba = LEAGUE_WOBA_BASELINE
    if league_woba_denominator > 0:
        league_woba = league_woba_numerator / league_woba_denominator

    return team_frames, league_woba


def _build_prior_team_baselines(
    *,
    prior_season: int,
    teams: Sequence[str],
    refresh: bool,
    team_logs_fetcher: _TeamLogsFetcher,
    league_woba: float,
    statcast_metrics: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    baselines: dict[str, dict[str, float]] = {}

    for team in teams:
        raw_logs = team_logs_fetcher(prior_season, team, refresh=refresh)
        team_statcast = statcast_metrics.loc[statcast_metrics["team"] == team].copy()
        prior_frame = _compute_game_level_metrics(raw_logs, statcast_metrics=team_statcast)
        if prior_frame.empty:
            baselines[team] = {metric: _default_metric_baseline(metric) for metric in METRICS}
            continue

        team_woba = _series_mean(prior_frame["woba"])
        team_xwoba = _series_mean(prior_frame["xwoba"])
        baselines[team] = {
            "wrc_plus": _team_wrc_plus(team_woba, league_woba),
            "woba": team_woba,
            "xwoba": team_xwoba,
            "woba_minus_xwoba": team_woba - team_xwoba
            if pd.notna(team_woba) and pd.notna(team_xwoba)
            else 0.0,
            "iso": _series_mean(prior_frame["iso"]),
            "barrel_pct": _series_mean(prior_frame["barrel_pct"]),
            "babip": _series_mean(prior_frame["babip"]),
            "k_pct": _series_mean(prior_frame["k_pct"]),
            "bb_pct": _series_mean(prior_frame["bb_pct"]),
        }

    return baselines


def _build_lineup_metric_lookup(
    *,
    lineup_player_ids: Mapping[tuple[int, str], Sequence[int]],
    season: int,
    target_day: date,
    windows: Sequence[int],
    league_woba: float,
    refresh: bool,
    batting_stats_fetcher: _BattingStatsFetcher,
    statcast_metrics: pd.DataFrame,
) -> dict[tuple[int, str, int], dict[str, float]]:
    if not lineup_player_ids:
        return {}

    batting_stats = _normalize_batting_stats(
        batting_stats_fetcher(season, min_pa=0, refresh=refresh)
    )
    if not batting_stats.empty and "game_date" in batting_stats.columns:
        batting_stats = batting_stats.dropna(subset=["game_date"]).copy()
        batting_stats = batting_stats.loc[batting_stats["game_date"].dt.date < target_day].copy()
    else:
        batting_stats = pd.DataFrame(columns=["player_id", "game_date", "game_pk", "pa", *METRICS])

    batter_xwoba = _normalize_statcast_offense_metrics(statcast_metrics)
    if not batter_xwoba.empty:
        batter_xwoba = batter_xwoba.loc[batter_xwoba["game_date"].dt.date < target_day].copy()

    if batting_stats.empty and batter_xwoba.empty:
        return {}

    lookup: dict[tuple[int, str, int], dict[str, float]] = {}
    for key, player_ids in lineup_player_ids.items():
        if not player_ids:
            continue

        normalized_player_ids = [int(player_id) for player_id in player_ids]
        lineup_frame = batting_stats.loc[
            batting_stats["player_id"].isin(normalized_player_ids)
        ].copy()
        lineup_xwoba = batter_xwoba.loc[
            batter_xwoba["player_id"].isin(normalized_player_ids)
        ].copy()
        if lineup_frame.empty and lineup_xwoba.empty:
            continue

        for window in windows:
            lineup_metrics = _aggregate_lineup_window_metrics(
                lineup_frame,
                lineup_xwoba,
                player_ids=normalized_player_ids,
                window=window,
                league_woba=league_woba,
            )
            if lineup_metrics is not None:
                lookup[(key[0], key[1], window)] = lineup_metrics

    return lookup


def _build_lineup_prior_metric_lookup(
    *,
    lineup_player_ids: Mapping[tuple[int, str], Sequence[int]],
    prior_season: int,
    refresh: bool,
    batting_stats_fetcher: _BattingStatsFetcher,
    statcast_metrics: pd.DataFrame,
) -> tuple[dict[tuple[int, str], dict[str, float]], dict[tuple[int, str], bool]]:
    if not lineup_player_ids:
        return {}, {}

    try:
        prior_batting_stats = _normalize_batting_stats(
            batting_stats_fetcher(prior_season, min_pa=0, refresh=refresh)
        )
    except Exception:
        prior_batting_stats = pd.DataFrame(
            columns=["player_id", "game_date", "game_pk", "pa", *METRICS]
        )
    if prior_batting_stats.empty:
        prior_batting_stats = pd.DataFrame(
            columns=["player_id", "game_date", "game_pk", "pa", *METRICS]
        )

    prior_statcast_metrics = _normalize_statcast_offense_metrics(statcast_metrics)
    if prior_batting_stats.empty and prior_statcast_metrics.empty:
        return {}, {}

    lookup: dict[tuple[int, str], dict[str, float]] = {}
    all_first_year_lookup: dict[tuple[int, str], bool] = {}
    for key, player_ids in lineup_player_ids.items():
        if not player_ids:
            continue

        lineup_metrics, all_first_year = _aggregate_prior_lineup_metrics(
            prior_batting_stats,
            prior_statcast_metrics,
            player_ids=player_ids,
        )
        if lineup_metrics is None:
            continue

        lookup[key] = lineup_metrics
        all_first_year_lookup[key] = all_first_year

    return lookup, all_first_year_lookup


def _aggregate_lineup_window_metrics(
    lineup_frame: pd.DataFrame,
    lineup_statcast_frame: pd.DataFrame,
    *,
    player_ids: Sequence[int],
    window: int,
    league_woba: float,
) -> dict[str, float] | None:
    sampled_frames: list[pd.DataFrame] = []
    sampled_statcast_frames: list[pd.DataFrame] = []
    for player_id in player_ids:
        player_frame = lineup_frame.loc[lineup_frame["player_id"] == int(player_id)].copy()
        if not player_frame.empty:
            player_frame = player_frame.sort_values(["game_date", "game_pk"]).tail(window)
            sampled_frames.append(player_frame)

        player_statcast = lineup_statcast_frame.loc[
            lineup_statcast_frame["player_id"] == int(player_id)
        ].copy()
        if not player_statcast.empty:
            player_statcast = player_statcast.sort_values(["game_date", "game_pk"]).tail(window)
            sampled_statcast_frames.append(player_statcast)

    if not sampled_frames and not sampled_statcast_frames:
        return None

    metrics: dict[str, float] = {}
    if sampled_frames:
        sampled = pd.concat(sampled_frames, ignore_index=True)
        weights = sampled["pa"].where(sampled["pa"] > 0, 1.0).astype(float)
        if float(weights.sum()) > 0:
            metrics.update(
                {
                    metric: _weighted_metric_mean(
                        sampled[metric],
                        weights=weights,
                        default=_default_metric_baseline(metric),
                    )
                    for metric in ("woba", "iso", "babip", "k_pct", "bb_pct")
                }
            )

            derived_wrc_plus = _team_wrc_plus(metrics["woba"], league_woba)
            metrics["wrc_plus"] = _weighted_metric_mean(
                sampled["wrc_plus"],
                weights=weights,
                default=derived_wrc_plus,
            )

    if sampled_statcast_frames:
        sampled_statcast = pd.concat(sampled_statcast_frames, ignore_index=True)
        xwoba_weights = sampled_statcast["pa"].where(sampled_statcast["pa"] > 0, 1.0).astype(float)
        if float(xwoba_weights.sum()) > 0:
            metrics["xwoba"] = _weighted_metric_mean(
                sampled_statcast["xwoba"],
                weights=xwoba_weights,
                default=metrics.get("woba", _default_metric_baseline("xwoba")),
            )
        bbe_weights = sampled_statcast["bbe"].where(sampled_statcast["bbe"] > 0, 0.0).astype(float)
        if float(bbe_weights.sum()) > 0:
            metrics["barrel_pct"] = _weighted_metric_mean(
                sampled_statcast["barrel_pct"],
                weights=bbe_weights,
                default=_default_metric_baseline("barrel_pct"),
            )

    if "xwoba" not in metrics and "woba" in metrics:
        metrics["xwoba"] = metrics["woba"]
    if "woba" in metrics and "xwoba" in metrics:
        metrics["woba_minus_xwoba"] = metrics["woba"] - metrics["xwoba"]
    return metrics


def _aggregate_prior_lineup_metrics(
    prior_lineup_frame: pd.DataFrame,
    prior_statcast_frame: pd.DataFrame,
    *,
    player_ids: Sequence[int],
) -> tuple[dict[str, float] | None, bool]:
    first_year_count = 0
    actual_source_available = not prior_lineup_frame.empty
    statcast_source_available = not prior_statcast_frame.empty
    metric_rows: dict[str, list[float]] = {metric: [] for metric in METRICS}

    for player_id in player_ids:
        player_frame = prior_lineup_frame.loc[
            prior_lineup_frame["player_id"] == int(player_id)
        ].copy()
        player_statcast = prior_statcast_frame.loc[
            prior_statcast_frame["player_id"] == int(player_id)
        ].copy()
        if player_frame.empty and player_statcast.empty:
            first_year_count += 1
            if actual_source_available:
                for metric in ("wrc_plus", "woba", "iso", "babip", "k_pct", "bb_pct"):
                    metric_rows[metric].append(_default_metric_baseline(metric))
            if statcast_source_available or actual_source_available:
                metric_rows["xwoba"].append(_default_metric_baseline("xwoba"))
                metric_rows["barrel_pct"].append(_default_metric_baseline("barrel_pct"))
                if actual_source_available:
                    metric_rows["woba_minus_xwoba"].append(
                        _default_metric_baseline("woba_minus_xwoba")
                    )
            continue

        actual_woba: float | None = None
        if not player_frame.empty:
            player_row = player_frame.sort_values(["game_date", "game_pk"]).iloc[-1]
            for metric in ("wrc_plus", "woba", "iso", "babip", "k_pct", "bb_pct"):
                value = float(player_row.get(metric, _default_metric_baseline(metric)))
                metric_rows[metric].append(value)
                if metric == "woba":
                    actual_woba = value
        elif actual_source_available:
            for metric in ("wrc_plus", "woba", "iso", "babip", "k_pct", "bb_pct"):
                value = _default_metric_baseline(metric)
                metric_rows[metric].append(value)
                if metric == "woba":
                    actual_woba = value

        xwoba_value: float | None = None
        barrel_pct_value: float | None = None
        if not player_statcast.empty:
            expected_weights = (
                player_statcast["pa"].where(player_statcast["pa"] > 0, 1.0).astype(float)
            )
            xwoba_value = _weighted_metric_mean(
                player_statcast["xwoba"],
                weights=expected_weights,
                default=actual_woba
                if actual_woba is not None
                else _default_metric_baseline("xwoba"),
            )
            bbe_weights = (
                player_statcast["bbe"].where(player_statcast["bbe"] > 0, 0.0).astype(float)
            )
            if float(bbe_weights.sum()) > 0:
                barrel_pct_value = _weighted_metric_mean(
                    player_statcast["barrel_pct"],
                    weights=bbe_weights,
                    default=_default_metric_baseline("barrel_pct"),
                )
        elif actual_woba is not None:
            xwoba_value = actual_woba
        elif statcast_source_available:
            xwoba_value = _default_metric_baseline("xwoba")

        if xwoba_value is not None:
            metric_rows["xwoba"].append(xwoba_value)
        if barrel_pct_value is None and (statcast_source_available or actual_source_available):
            barrel_pct_value = _default_metric_baseline("barrel_pct")
        if barrel_pct_value is not None:
            metric_rows["barrel_pct"].append(barrel_pct_value)

        if actual_woba is not None and xwoba_value is not None:
            metric_rows["woba_minus_xwoba"].append(actual_woba - xwoba_value)
        elif actual_source_available and xwoba_value is not None:
            metric_rows["woba_minus_xwoba"].append(_default_metric_baseline("woba_minus_xwoba"))

    populated_metrics = {metric: values for metric, values in metric_rows.items() if values}
    if not populated_metrics:
        return None, False

    aggregated = {
        metric: float(pd.Series(values, dtype=float).mean())
        for metric, values in populated_metrics.items()
    }
    return aggregated, first_year_count == len(player_ids)


def _normalize_batting_stats(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return pd.DataFrame(columns=["player_id", "game_date", "game_pk", "pa", *METRICS])

    normalized = dataframe.copy()
    derived_metrics = _derive_offensive_metrics(normalized)
    player_id_column = _first_column(
        normalized, ("player_id", "playerid", "key_mlbam", "mlb_id", "id")
    )
    pa_column = _first_column(normalized, ("PA", "pa", "plate appearances", "_PA"))
    if player_id_column is None:
        return pd.DataFrame(columns=["player_id", "game_date", "game_pk", "pa", *METRICS])

    date_column = _first_column(
        normalized,
        ("game_date", "date", "Date", "as_of_timestamp", "as_of_date"),
    )

    column_map = {
        "player_id": player_id_column,
        "pa": pa_column,
        "game_pk": _first_column(normalized, ("game_pk", "gameid", "game_id")),
        "wrc_plus": _first_column(normalized, ("wRC+", "wrc_plus", "wRC")),
        "woba": _first_column(normalized, ("wOBA", "woba")),
        "xwoba": _first_column(normalized, ("xwOBA", "xwoba")),
        "woba_minus_xwoba": _first_column(
            normalized,
            ("woba_minus_xwoba", "woba_xwoba_gap", "wOBA_minus_xwOBA"),
        ),
        "iso": _first_column(normalized, ("ISO", "iso")),
        "barrel_pct": _first_column(
            normalized,
            ("barrel_pct", "barrel%", "barrel_rate", "barrel_batted_rate"),
        ),
        "babip": _first_column(normalized, ("BABIP", "babip")),
        "k_pct": _first_column(normalized, ("K%", "k_pct", "SO%")),
        "bb_pct": _first_column(normalized, ("BB%", "bb_pct")),
    }

    result = pd.DataFrame(index=normalized.index)
    result["game_date"] = (
        _to_tz_naive_datetime_series(normalized[date_column])
        if date_column is not None
        else pd.Series(pd.NaT, index=normalized.index, dtype="datetime64[ns]")
    )

    for output_column, source_column in column_map.items():
        if source_column is None:
            if output_column == "pa":
                result[output_column] = derived_metrics["pa"]
            elif output_column == "game_pk":
                result[output_column] = pd.Series(pd.NA, index=normalized.index, dtype="Int64")
            elif output_column == "wrc_plus":
                result[output_column] = pd.Series(float("nan"), index=normalized.index, dtype=float)
            elif output_column == "barrel_pct":
                result[output_column] = pd.Series(float("nan"), index=normalized.index, dtype=float)
            else:
                result[output_column] = derived_metrics[output_column]
            continue
        result[output_column] = _to_numeric(normalized[source_column])

    result = result.dropna(subset=["player_id"]).copy()
    result["player_id"] = result["player_id"].astype(int)
    result["game_pk"] = pd.to_numeric(result["game_pk"], errors="coerce").astype("Int64")
    result["pa"] = pd.to_numeric(result["pa"], errors="coerce").fillna(0.0)
    result["woba"] = pd.to_numeric(result["woba"], errors="coerce").fillna(
        _default_metric_baseline("woba")
    )
    result["xwoba"] = pd.to_numeric(result["xwoba"], errors="coerce").fillna(result["woba"])
    result["woba_minus_xwoba"] = pd.to_numeric(result["woba_minus_xwoba"], errors="coerce").fillna(
        result["woba"] - result["xwoba"]
    )
    result["barrel_pct"] = pd.to_numeric(result["barrel_pct"], errors="coerce")
    result = result.fillna(
        {
            "iso": _default_metric_baseline("iso"),
            "babip": _default_metric_baseline("babip"),
            "k_pct": _default_metric_baseline("k_pct"),
            "bb_pct": _default_metric_baseline("bb_pct"),
        }
    )
    return result.reset_index(drop=True)


def _derive_offensive_metrics(dataframe: pd.DataFrame) -> pd.DataFrame:
    ab = _extract_numeric(dataframe, "AB", "Offense_AB")
    hits = _extract_numeric(dataframe, "H", "Offense_H")
    doubles = _extract_numeric(dataframe, "2B", "Offense_2B")
    triples = _extract_numeric(dataframe, "3B", "Offense_3B")
    home_runs = _extract_numeric(dataframe, "HR", "Offense_HR")
    walks = _extract_numeric(dataframe, "BB", "Offense_BB")
    strikeouts = _extract_numeric(dataframe, "SO", "K", "Offense_SO", "Offense_K")
    hit_by_pitch = _extract_numeric(dataframe, "HBP", "Offense_HBP")
    sacrifice_flies = _extract_numeric(dataframe, "SF", "Offense_SF")
    sacrifice_hits = _extract_numeric(dataframe, "SH", "Offense_SH")

    singles = (hits - doubles - triples - home_runs).clip(lower=0)
    woba_numerator = (
        walks * WOBA_WEIGHTS["bb"]
        + hit_by_pitch * WOBA_WEIGHTS["hbp"]
        + singles * WOBA_WEIGHTS["single"]
        + doubles * WOBA_WEIGHTS["double"]
        + triples * WOBA_WEIGHTS["triple"]
        + home_runs * WOBA_WEIGHTS["hr"]
    )
    woba_denominator = ab + walks + hit_by_pitch + sacrifice_flies
    plate_appearances = ab + walks + hit_by_pitch + sacrifice_flies + sacrifice_hits
    total_bases = singles + doubles * 2 + triples * 3 + home_runs * 4

    return pd.DataFrame(
        {
            "pa": plate_appearances.astype(float),
            "woba": _series_from_value(woba_numerator / woba_denominator.replace(0, pd.NA)),
            "xwoba": _series_from_value(woba_numerator / woba_denominator.replace(0, pd.NA)),
            "woba_minus_xwoba": pd.Series(0.0, index=dataframe.index, dtype=float),
            "iso": _series_from_value((total_bases - hits) / ab.replace(0, pd.NA)),
            "babip": _series_from_value(
                (hits - home_runs)
                / (ab - strikeouts - home_runs + sacrifice_flies).replace(0, pd.NA)
            ),
            "k_pct": _series_from_value((strikeouts / plate_appearances.replace(0, pd.NA)) * 100),
            "bb_pct": _series_from_value((walks / plate_appearances.replace(0, pd.NA)) * 100),
        }
    )


def _compute_game_level_metrics(
    raw_logs: pd.DataFrame,
    *,
    statcast_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if raw_logs.empty:
        return _empty_game_metrics_frame()

    date_column = _first_column(raw_logs, ("Date", "Offense_Date", "game_date"))
    if date_column is None:
        return _empty_game_metrics_frame()

    ab = _extract_numeric(raw_logs, "AB", "Offense_AB", "Batting Stats_AB")
    hits = _extract_numeric(raw_logs, "H", "Offense_H", "Batting Stats_H")
    doubles = _extract_numeric(raw_logs, "2B", "Offense_2B", "Batting Stats_2B")
    triples = _extract_numeric(raw_logs, "3B", "Offense_3B", "Batting Stats_3B")
    home_runs = _extract_numeric(raw_logs, "HR", "Offense_HR", "Batting Stats_HR")
    walks = _extract_numeric(raw_logs, "BB", "Offense_BB", "Batting Stats_BB")
    strikeouts = _extract_numeric(
        raw_logs,
        "SO",
        "K",
        "Offense_SO",
        "Offense_K",
        "Batting Stats_SO",
    )
    hit_by_pitch = _extract_numeric(raw_logs, "HBP", "Offense_HBP", "Batting Stats_HBP")
    sacrifice_flies = _extract_numeric(raw_logs, "SF", "Offense_SF", "Batting Stats_SF")
    sacrifice_hits = _extract_numeric(raw_logs, "SH", "Offense_SH", "Batting Stats_SH")

    singles = (hits - doubles - triples - home_runs).clip(lower=0)
    woba_numerator = (
        walks * WOBA_WEIGHTS["bb"]
        + hit_by_pitch * WOBA_WEIGHTS["hbp"]
        + singles * WOBA_WEIGHTS["single"]
        + doubles * WOBA_WEIGHTS["double"]
        + triples * WOBA_WEIGHTS["triple"]
        + home_runs * WOBA_WEIGHTS["hr"]
    )
    woba_denominator = ab + walks + hit_by_pitch + sacrifice_flies
    plate_appearances = ab + walks + hit_by_pitch + sacrifice_flies + sacrifice_hits
    total_bases = singles + doubles * 2 + triples * 3 + home_runs * 4

    actual_woba = _series_from_value(woba_numerator / woba_denominator.replace(0, pd.NA))
    game_pk_column = _first_column(raw_logs, ("game_pk", "gameid", "game_id"))

    result = pd.DataFrame(
        {
            "game_pk": (
                pd.to_numeric(raw_logs[game_pk_column], errors="coerce").astype("Int64")
                if game_pk_column is not None
                else pd.Series(pd.NA, index=raw_logs.index, dtype="Int64")
            ),
            "game_date": _to_tz_naive_datetime_series(raw_logs[date_column]),
            "woba_numerator": woba_numerator,
            "woba_denominator": woba_denominator,
            "woba": actual_woba,
            "iso": _series_from_value((total_bases - hits) / ab.replace(0, pd.NA)),
            "babip": _series_from_value(
                (hits - home_runs)
                / (ab - strikeouts - home_runs + sacrifice_flies).replace(0, pd.NA)
            ),
            "k_pct": _series_from_value((strikeouts / plate_appearances.replace(0, pd.NA)) * 100),
            "bb_pct": _series_from_value((walks / plate_appearances.replace(0, pd.NA)) * 100),
        }
    )

    xwoba = _align_statcast_team_xwoba(
        raw_logs=raw_logs,
        game_dates=result["game_date"],
        statcast_metrics=statcast_metrics,
    ).fillna(actual_woba)
    barrel_pct = _align_statcast_team_metric(
        raw_logs=raw_logs,
        game_dates=result["game_date"],
        statcast_metrics=statcast_metrics,
        metric="barrel_pct",
    ).fillna(_default_metric_baseline("barrel_pct"))
    result["xwoba"] = _series_from_value(xwoba)
    result["woba_minus_xwoba"] = _series_from_value(result["woba"] - result["xwoba"])
    result["barrel_pct"] = _series_from_value(barrel_pct)
    result = result.dropna(subset=["game_date"]).sort_values(["game_date", "game_pk"]).reset_index(
        drop=True
    )
    return result


def _empty_game_metrics_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_pk": pd.Series(dtype="Int64"),
            "game_date": pd.to_datetime(pd.Series([], dtype="datetime64[ns]"), utc=False),
            "woba_numerator": pd.Series([], dtype=float),
            "woba_denominator": pd.Series([], dtype=float),
            "woba": pd.Series([], dtype=float),
            "xwoba": pd.Series([], dtype=float),
            "woba_minus_xwoba": pd.Series([], dtype=float),
            "iso": pd.Series([], dtype=float),
            "barrel_pct": pd.Series([], dtype=float),
            "babip": pd.Series([], dtype=float),
            "k_pct": pd.Series([], dtype=float),
            "bb_pct": pd.Series([], dtype=float),
        }
    )


def _empty_statcast_offense_metrics_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_pk": pd.Series(dtype="int64"),
            "game_date": pd.Series(dtype="datetime64[ns]"),
            "team": pd.Series(dtype="str"),
            "player_id": pd.Series(dtype="int64"),
            "pa": pd.Series(dtype="float64"),
            "bbe": pd.Series(dtype="float64"),
            "xwoba": pd.Series(dtype="float64"),
            "barrel_pct": pd.Series(dtype="float64"),
        }
    )


def _fetch_season_offense_statcast_metrics(
    season: int,
    *,
    db_path: str | Path,
    end_date: date | None,
    refresh: bool = False,
) -> pd.DataFrame:
    season_games = _load_season_games(Path(db_path), season=season, end_date=end_date)
    if season_games.empty:
        return _empty_statcast_offense_metrics_frame()

    cache_path = _offense_statcast_cache_path(season, season_games)
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    min_day = season_games["game_date"].min().date()
    max_day = season_games["game_date"].max().date()
    statcast_frame = fetch_statcast_range(min_day.isoformat(), max_day.isoformat(), refresh=refresh)
    metrics = _build_statcast_offense_metrics(season_games, statcast_frame)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(cache_path, index=False)
    return metrics


def _offense_statcast_cache_path(season: int, season_games: pd.DataFrame) -> Path:
    normalized = season_games.loc[:, ["game_pk", "game_date"]].copy()
    normalized["game_pk"] = pd.to_numeric(normalized["game_pk"], errors="coerce").astype("Int64")
    normalized["game_date"] = pd.to_datetime(normalized["game_date"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    normalized = normalized.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(normalized, index=False).values.tobytes()
    ).hexdigest()[:16]
    return DERIVED_CACHE_ROOT / (
        f"offense_statcast_metrics_v{OFFENSE_STATCAST_CACHE_VERSION}_{season}_{digest}.parquet"
    )


def _build_statcast_offense_metrics(
    season_games: pd.DataFrame,
    statcast_frame: pd.DataFrame,
) -> pd.DataFrame:
    if season_games.empty or statcast_frame.empty:
        return _empty_statcast_offense_metrics_frame()

    terminal = _collapse_plate_appearances(statcast_frame)
    game_pk_column = _first_column(terminal, ("game_pk",))
    batter_column = _first_column(terminal, ("batter", "player_id", "playerid"))
    team_column = _first_column(terminal, ("batting_team", "team"))
    xwoba_column = _first_column(terminal, ("estimated_woba_using_speedangle", "xwoba"))
    bb_type_column = _first_column(terminal, ("bb_type",))
    launch_speed_column = _first_column(terminal, ("launch_speed",))
    launch_angle_column = _first_column(terminal, ("launch_angle",))
    if game_pk_column is None or batter_column is None:
        return _empty_statcast_offense_metrics_frame()

    team_values = (
        terminal[team_column].fillna("").astype(str)
        if team_column is not None
        else _resolve_batting_team(terminal)
    )
    xwoba_values = (
        pd.to_numeric(terminal[xwoba_column], errors="coerce")
        if xwoba_column is not None
        else pd.Series(float("nan"), index=terminal.index, dtype=float)
    )
    bb_type = (
        terminal[bb_type_column].astype(str).str.strip()
        if bb_type_column is not None
        else pd.Series("", index=terminal.index, dtype=str)
    )
    launch_speed = (
        pd.to_numeric(terminal[launch_speed_column], errors="coerce")
        if launch_speed_column is not None
        else pd.Series(float("nan"), index=terminal.index, dtype=float)
    )
    launch_angle = (
        pd.to_numeric(terminal[launch_angle_column], errors="coerce")
        if launch_angle_column is not None
        else pd.Series(float("nan"), index=terminal.index, dtype=float)
    )
    batted_ball_mask = (
        terminal[bb_type_column].notna() & bb_type.ne("")
        if bb_type_column is not None
        else pd.Series(False, index=terminal.index, dtype=bool)
    )
    barrel_mask = (
        _is_barrel_mask(launch_speed=launch_speed, launch_angle=launch_angle) & batted_ball_mask
    )

    metrics = pd.DataFrame(
        {
            "game_pk": pd.to_numeric(terminal[game_pk_column], errors="coerce").astype("Int64"),
            "player_id": pd.to_numeric(terminal[batter_column], errors="coerce").astype("Int64"),
            "team": team_values.astype(str).str.strip().str.upper(),
            "xwoba": xwoba_values,
            "bbe": batted_ball_mask.astype(float),
            "barrel": barrel_mask.astype(float),
        }
    )
    metrics = metrics.dropna(subset=["game_pk", "player_id"]).copy()
    metrics = metrics.loc[metrics["team"].ne("")].copy()
    if metrics.empty:
        return _empty_statcast_offense_metrics_frame()

    season_lookup = season_games.loc[:, ["game_pk", "game_date"]].drop_duplicates().copy()
    season_lookup["game_pk"] = pd.to_numeric(season_lookup["game_pk"], errors="coerce").astype(
        "Int64"
    )
    season_lookup["game_date"] = pd.to_datetime(season_lookup["game_date"], errors="coerce")
    metrics = metrics.merge(season_lookup, on="game_pk", how="inner")
    metrics = metrics.dropna(subset=["game_date"]).copy()
    if metrics.empty:
        return _empty_statcast_offense_metrics_frame()

    grouped = (
        metrics.groupby(["game_pk", "game_date", "team", "player_id"], dropna=True)
        .agg(
            pa=("player_id", "size"),
            xwoba=("xwoba", "mean"),
            bbe=("bbe", "sum"),
            barrels=("barrel", "sum"),
        )
        .reset_index()
    )
    grouped["game_pk"] = grouped["game_pk"].astype(int)
    grouped["player_id"] = grouped["player_id"].astype(int)
    grouped["pa"] = grouped["pa"].astype(float)
    grouped["bbe"] = grouped["bbe"].astype(float)
    grouped["xwoba"] = grouped["xwoba"].astype(float)
    grouped["barrel_pct"] = _series_from_value(
        (grouped["barrels"] / grouped["bbe"].replace(0, pd.NA)) * 100.0
    )
    grouped = grouped.drop(columns=["barrels"])
    return grouped.sort_values(["game_date", "game_pk", "team", "player_id"]).reset_index(drop=True)


def _normalize_statcast_offense_metrics(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return _empty_statcast_offense_metrics_frame()

    normalized = dataframe.copy()
    game_pk_column = _first_column(normalized, ("game_pk",))
    game_date_column = _first_column(normalized, ("game_date", "date"))
    team_column = _first_column(normalized, ("team", "batting_team"))
    player_id_column = _first_column(normalized, ("player_id", "batter", "playerid"))
    pa_column = _first_column(normalized, ("pa", "plate_appearances"))
    xwoba_column = _first_column(normalized, ("xwoba", "estimated_woba_using_speedangle"))
    bbe_column = _first_column(normalized, ("bbe", "batted_ball_events"))
    barrel_pct_column = _first_column(
        normalized,
        ("barrel_pct", "barrel%", "barrel_rate", "barrel_batted_rate"),
    )
    if game_pk_column is None or game_date_column is None or player_id_column is None:
        return _empty_statcast_offense_metrics_frame()

    team_values = (
        normalized[team_column].fillna("").astype(str)
        if team_column is not None
        else _resolve_batting_team(normalized)
    )

    result = pd.DataFrame(
        {
            "game_pk": pd.to_numeric(normalized[game_pk_column], errors="coerce").astype("Int64"),
            "game_date": _to_tz_naive_datetime_series(normalized[game_date_column]),
            "team": team_values.astype(str).str.strip().str.upper(),
            "player_id": pd.to_numeric(normalized[player_id_column], errors="coerce").astype(
                "Int64"
            ),
            "pa": (
                pd.to_numeric(normalized[pa_column], errors="coerce")
                if pa_column is not None
                else pd.Series(1.0, index=normalized.index, dtype=float)
            ),
            "bbe": (
                pd.to_numeric(normalized[bbe_column], errors="coerce")
                if bbe_column is not None
                else pd.Series(0.0, index=normalized.index, dtype=float)
            ),
            "xwoba": (
                pd.to_numeric(normalized[xwoba_column], errors="coerce")
                if xwoba_column is not None
                else pd.Series(float("nan"), index=normalized.index, dtype=float)
            ),
            "barrel_pct": (
                pd.to_numeric(normalized[barrel_pct_column], errors="coerce")
                if barrel_pct_column is not None
                else pd.Series(float("nan"), index=normalized.index, dtype=float)
            ),
        }
    )
    result = result.dropna(subset=["game_pk", "game_date", "team", "player_id"]).copy()
    if result.empty:
        return _empty_statcast_offense_metrics_frame()

    result["game_pk"] = result["game_pk"].astype(int)
    result["player_id"] = result["player_id"].astype(int)
    result["pa"] = result["pa"].fillna(0.0).astype(float)
    result["bbe"] = result["bbe"].fillna(0.0).astype(float)
    result["xwoba"] = result["xwoba"].astype(float)
    result["barrel_pct"] = result["barrel_pct"].astype(float)
    return result.sort_values(["game_date", "game_pk", "team", "player_id"]).reset_index(drop=True)


def _align_statcast_team_xwoba(
    *,
    raw_logs: pd.DataFrame,
    game_dates: pd.Series,
    statcast_metrics: pd.DataFrame | None,
) -> pd.Series:
    return _align_statcast_team_metric(
        raw_logs=raw_logs,
        game_dates=game_dates,
        statcast_metrics=statcast_metrics,
        metric="xwoba",
    )


def _align_statcast_team_metric(
    *,
    raw_logs: pd.DataFrame,
    game_dates: pd.Series,
    statcast_metrics: pd.DataFrame | None,
    metric: str,
) -> pd.Series:
    if statcast_metrics is None or statcast_metrics.empty:
        return pd.Series(float("nan"), index=raw_logs.index, dtype=float)

    team_game_metrics = _aggregate_team_statcast_game_metrics(statcast_metrics)
    if team_game_metrics.empty or metric not in team_game_metrics.columns:
        return pd.Series(float("nan"), index=raw_logs.index, dtype=float)

    alignment = pd.DataFrame(
        {
            "_row_index": raw_logs.index,
            "game_pk": (
                pd.to_numeric(raw_logs["game_pk"], errors="coerce").astype("Int64")
                if "game_pk" in raw_logs.columns
                else pd.Series(pd.NA, index=raw_logs.index, dtype="Int64")
            ),
            "game_date": _to_tz_naive_datetime_series(game_dates),
        }
    )
    alignment = alignment.dropna(subset=["game_date"]).copy()
    if alignment.empty:
        return pd.Series(float("nan"), index=raw_logs.index, dtype=float)

    if alignment["game_pk"].notna().any():
        team_game_metrics = team_game_metrics.copy()
        team_game_metrics["game_pk"] = pd.to_numeric(
            team_game_metrics["game_pk"], errors="coerce"
        ).astype("Int64")
        merged = alignment.merge(
            team_game_metrics.loc[:, ["game_pk", metric]],
            on="game_pk",
            how="left",
        )
        aligned_metric = pd.Series(float("nan"), index=raw_logs.index, dtype=float)
        aligned_metric.loc[merged["_row_index"].astype(int)] = pd.to_numeric(
            merged[metric], errors="coerce"
        ).values
        return aligned_metric

    # Normalize both sides to tz-naive before merge to avoid datetime64[us] vs datetime64[us, UTC] conflict
    alignment["game_date"] = _to_tz_naive_datetime_series(alignment["game_date"])

    team_game_metrics = team_game_metrics.sort_values(["game_date", "game_pk"]).copy()
    team_game_metrics["game_date"] = _to_tz_naive_datetime_series(team_game_metrics["game_date"])

    alignment["_date_slot"] = alignment.groupby("game_date", sort=False).cumcount()
    team_game_metrics["_date_slot"] = team_game_metrics.groupby("game_date", sort=False).cumcount()

    merged = alignment.merge(
        team_game_metrics.loc[:, ["game_date", "_date_slot", metric]],
        on=["game_date", "_date_slot"],
        how="left",
    )
    aligned_metric = pd.Series(float("nan"), index=raw_logs.index, dtype=float)
    aligned_metric.loc[merged["_row_index"].astype(int)] = pd.to_numeric(
        merged[metric], errors="coerce"
    ).values
    return aligned_metric


def _aggregate_team_statcast_game_metrics(statcast_metrics: pd.DataFrame) -> pd.DataFrame:
    if statcast_metrics.empty:
        return pd.DataFrame(columns=["game_pk", "game_date", "xwoba", "barrel_pct"])

    rows: list[dict[str, float | int | pd.Timestamp]] = []
    for (game_pk, game_date), group in statcast_metrics.groupby(
        ["game_pk", "game_date"], dropna=True
    ):
        weights = group["pa"].where(group["pa"] > 0, 1.0).astype(float)
        bbe_weights = group["bbe"].where(group["bbe"] > 0, 0.0).astype(float)
        rows.append(
            {
                "game_pk": int(game_pk),
                "game_date": pd.Timestamp(game_date),
                "xwoba": _weighted_metric_mean(
                    group["xwoba"],
                    weights=weights,
                    default=_default_metric_baseline("xwoba"),
                ),
                "barrel_pct": _weighted_metric_mean(
                    group["barrel_pct"],
                    weights=bbe_weights,
                    default=_default_metric_baseline("barrel_pct"),
                ),
            }
        )
    return pd.DataFrame(rows)


def _collapse_plate_appearances(pitches: pd.DataFrame) -> pd.DataFrame:
    if pitches.empty:
        return pitches.copy()

    at_bat_column = _first_column(pitches, ("at_bat_number",))
    if at_bat_column is not None:
        sort_columns = [
            column
            for column in ("game_pk", at_bat_column, "pitch_number")
            if column in pitches.columns
        ]
        group_columns = [
            column for column in ("game_pk", at_bat_column) if column in pitches.columns
        ]
        terminal = pitches.sort_values(sort_columns).groupby(group_columns, as_index=False).tail(1)
        return terminal.reset_index(drop=True)

    events_column = _first_column(pitches, ("events",))
    if events_column is not None:
        terminal = pitches.loc[pitches[events_column].notna()].copy()
        if not terminal.empty:
            return terminal.reset_index(drop=True)

    return pitches.tail(1).reset_index(drop=True)


def _rolling_metrics_as_of(
    *,
    team_frame: pd.DataFrame,
    metric_names: Sequence[str],
    window: int,
    min_periods: int,
) -> dict[str, _WindowMetric]:
    if team_frame.empty:
        return {metric: _WindowMetric(float("nan"), 0) for metric in metric_names}

    sample = team_frame.tail(window)
    games_played = len(sample)
    if games_played < min_periods:
        return {metric: _WindowMetric(float("nan"), games_played) for metric in metric_names}

    return {
        metric: _WindowMetric(_series_mean(sample[metric]), games_played) for metric in metric_names
    }


def _apply_metric_blend(
    *,
    metric: str,
    current_value: float,
    prior_value: float,
    games_played: int,
    regression_weight: int,
    roster_turnover_pct: float | None,
    is_first_year: bool = False,
) -> float:
    return blend_value(
        current_value,
        games_played=games_played,
        metric_type="offense",
        prior_value=prior_value,
        league_average=_default_metric_baseline(metric),
        regression_weight=regression_weight,
        roster_turnover_pct=roster_turnover_pct,
        is_first_year=is_first_year,
    )


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
                    row.game_pk,
                    row.feature_name,
                    row.feature_value,
                    row.window_size,
                    row.as_of_timestamp.isoformat(),
                )
                for row in features
            ],
        )
        connection.commit()


def _first_column(dataframe: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    normalized_columns = {str(column).strip().lower(): str(column) for column in dataframe.columns}
    for candidate in candidates:
        match = normalized_columns.get(candidate.strip().lower())
        if match is not None:
            return match
    return None


def _resolve_batting_team(dataframe: pd.DataFrame) -> pd.Series:
    topbot_column = _first_column(dataframe, ("inning_topbot", "inning_top_bot"))
    home_team_column = _first_column(dataframe, ("home_team",))
    away_team_column = _first_column(dataframe, ("away_team",))
    if topbot_column is None or home_team_column is None or away_team_column is None:
        return pd.Series("", index=dataframe.index, dtype=str)

    half_inning = dataframe[topbot_column].astype(str).str.strip().str.lower()
    team = pd.Series("", index=dataframe.index, dtype=str)
    top_mask = half_inning.eq("top")
    bottom_mask = half_inning.eq("bot")
    team.loc[top_mask] = dataframe.loc[top_mask, away_team_column].astype(str)
    team.loc[bottom_mask] = dataframe.loc[bottom_mask, home_team_column].astype(str)
    return team.str.strip().str.upper()


def _extract_numeric(dataframe: pd.DataFrame, *candidates: str) -> pd.Series:
    column = _first_column(dataframe, candidates)
    if column is None:
        return pd.Series(0.0, index=dataframe.index, dtype=float)
    return _to_numeric(dataframe[column]).fillna(0.0)


def _to_numeric(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def _series_from_value(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def _series_mean(values: pd.Series) -> float:
    numeric = _to_numeric(values)
    if numeric.dropna().empty:
        return float("nan")
    return float(numeric.mean())


def _to_tz_naive_datetime_series(values: Any) -> pd.Series:
    if values is None:
        return pd.Series(dtype="datetime64[ns]")

    parsed = pd.to_datetime(values, errors="coerce", utc=True, format="mixed")
    return parsed.dt.tz_convert(None)


def _is_barrel(exit_velocity: float, launch_angle: float) -> bool:
    if pd.isna(exit_velocity) or pd.isna(launch_angle) or float(exit_velocity) < 98.0:
        return False
    min_angle = max(26.0 - (float(exit_velocity) - 98.0), 8.0)
    max_angle = min(30.0 + ((float(exit_velocity) - 98.0) * 1.2), 50.0)
    return min_angle <= float(launch_angle) <= max_angle


def _is_barrel_mask(*, launch_speed: pd.Series, launch_angle: pd.Series) -> pd.Series:
    exit_velocity = pd.to_numeric(launch_speed, errors="coerce")
    angle = pd.to_numeric(launch_angle, errors="coerce")
    min_angle = (26.0 - (exit_velocity - 98.0)).clip(lower=8.0)
    max_angle = (30.0 + ((exit_velocity - 98.0) * 1.2)).clip(upper=50.0)
    return exit_velocity.ge(98.0) & angle.ge(min_angle) & angle.le(max_angle)


def _weighted_metric_mean(values: pd.Series, *, weights: pd.Series, default: float) -> float:
    numeric = _to_numeric(values)
    valid = numeric.notna() & weights.notna()
    if not valid.any():
        return float(default)

    valid_values = numeric.loc[valid].astype(float)
    valid_weights = weights.loc[valid].astype(float)
    weight_sum = float(valid_weights.sum())
    if weight_sum <= 0:
        return float(default)
    return float((valid_values * valid_weights).sum() / weight_sum)


def _team_wrc_plus(team_woba: float, league_woba: float) -> float:
    if pd.isna(team_woba):
        return LEAGUE_WRC_PLUS_BASELINE
    if league_woba <= 0:
        return LEAGUE_WRC_PLUS_BASELINE
    return float((team_woba / league_woba) * LEAGUE_WRC_PLUS_BASELINE)


def _default_metric_baseline(metric: str) -> float:
    if metric == "wrc_plus":
        return LEAGUE_WRC_PLUS_BASELINE
    if metric in {"woba", "xwoba"}:
        return LEAGUE_WOBA_BASELINE
    if metric == "barrel_pct":
        return LEAGUE_BARREL_PCT_BASELINE
    if metric == "woba_minus_xwoba":
        return 0.0
    return 0.0
