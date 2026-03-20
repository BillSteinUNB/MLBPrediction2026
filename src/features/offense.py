from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.clients.statcast_client import fetch_batting_stats, fetch_team_game_logs
from src.db import DEFAULT_DB_PATH, init_db
from src.features.marcel_blend import blend_value, get_regression_weight
from src.models.features import GameFeatures


DEFAULT_WINDOWS: tuple[int, ...] = (7, 14, 30, 60)
DEFAULT_REGRESSION_WEIGHT = int(get_regression_weight("offense"))
DEFAULT_MIN_PERIODS = 1
METRICS: tuple[str, ...] = ("wrc_plus", "woba", "iso", "babip", "k_pct", "bb_pct")
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


_TeamLogsFetcher = Callable[..., pd.DataFrame]
_BattingStatsFetcher = Callable[..., pd.DataFrame]


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
) -> list[GameFeatures]:
    """Compute and persist lagged offensive features for games on a target date."""

    target_day = _coerce_date(game_date)
    database_path = Path(db_path)
    init_db(database_path)

    games = _load_games_for_date(database_path, target_day)
    if games.empty:
        return []

    teams = sorted({*games["home_team"].unique().tolist(), *games["away_team"].unique().tolist()})
    team_metrics, league_woba = _build_team_game_metrics(
        season=target_day.year,
        teams=teams,
        target_day=target_day,
        refresh=refresh,
        team_logs_fetcher=team_logs_fetcher,
    )
    prior_team_baselines = _build_prior_team_baselines(
        prior_season=target_day.year - 1,
        teams=teams,
        refresh=refresh,
        team_logs_fetcher=team_logs_fetcher,
        league_woba=league_woba,
    )
    lineup_metric_lookup = _build_lineup_metric_lookup(
        lineup_player_ids=lineup_player_ids or {},
        season=target_day.year,
        target_day=target_day,
        windows=windows,
        league_woba=league_woba,
        refresh=refresh,
        batting_stats_fetcher=batting_stats_fetcher,
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
            roster_turnover_pct = None if roster_turnover_by_team is None else roster_turnover_by_team.get(team)
            for window in windows:
                current_window_metrics = _rolling_metrics_as_of(
                    team_frame=team_frame,
                    metric_names=("woba", "iso", "babip", "k_pct", "bb_pct"),
                    window=window,
                    min_periods=min_periods,
                )

                current_team_values = {
                    metric: current_window_metrics[metric].value
                    for metric in ("woba", "iso", "babip", "k_pct", "bb_pct")
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
                    for metric in METRICS:
                        lineup_feature_values[metric] = _apply_metric_blend(
                            metric=metric,
                            current_value=lineup_metrics.get(metric, team_feature_values[metric]),
                            prior_value=team_priors.get(metric, _default_metric_baseline(metric)),
                            games_played=season_games_played,
                            regression_weight=regression_weight,
                            roster_turnover_pct=roster_turnover_pct,
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
    with sqlite3.connect(db_path) as connection:
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


def _build_team_game_metrics(
    *,
    season: int,
    teams: Sequence[str],
    target_day: date,
    refresh: bool,
    team_logs_fetcher: _TeamLogsFetcher,
) -> tuple[dict[str, pd.DataFrame], float]:
    team_frames: dict[str, pd.DataFrame] = {}
    league_woba_numerator = 0.0
    league_woba_denominator = 0.0

    for team in teams:
        raw_logs = team_logs_fetcher(season, team, refresh=refresh)
        game_metrics = _compute_game_level_metrics(raw_logs)
        filtered = game_metrics.loc[game_metrics["game_date"].dt.date < target_day].reset_index(drop=True)
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
) -> dict[str, dict[str, float]]:
    baselines: dict[str, dict[str, float]] = {}

    for team in teams:
        raw_logs = team_logs_fetcher(prior_season, team, refresh=refresh)
        prior_frame = _compute_game_level_metrics(raw_logs)
        if prior_frame.empty:
            baselines[team] = {metric: _default_metric_baseline(metric) for metric in METRICS}
            continue

        team_woba = _series_mean(prior_frame["woba"])
        baselines[team] = {
            "wrc_plus": _team_wrc_plus(team_woba, league_woba),
            "woba": team_woba,
            "iso": _series_mean(prior_frame["iso"]),
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
) -> dict[tuple[int, str, int], dict[str, float]]:
    if not lineup_player_ids:
        return {}

    batting_stats = _normalize_batting_stats(
        batting_stats_fetcher(season, min_pa=0, refresh=refresh)
    )
    if batting_stats.empty or "game_date" not in batting_stats.columns:
        return {}

    batting_stats = batting_stats.dropna(subset=["game_date"]).copy()
    batting_stats = batting_stats.loc[batting_stats["game_date"].dt.date < target_day].copy()
    if batting_stats.empty:
        return {}

    lookup: dict[tuple[int, str, int], dict[str, float]] = {}
    for key, player_ids in lineup_player_ids.items():
        if not player_ids:
            continue

        normalized_player_ids = [int(player_id) for player_id in player_ids]
        lineup_frame = batting_stats.loc[
            batting_stats["player_id"].isin(normalized_player_ids)
        ].copy()
        if lineup_frame.empty:
            continue

        for window in windows:
            lineup_metrics = _aggregate_lineup_window_metrics(
                lineup_frame,
                player_ids=normalized_player_ids,
                window=window,
                league_woba=league_woba,
            )
            if lineup_metrics is not None:
                lookup[(key[0], key[1], window)] = lineup_metrics

    return lookup


def _aggregate_lineup_window_metrics(
    lineup_frame: pd.DataFrame,
    *,
    player_ids: Sequence[int],
    window: int,
    league_woba: float,
) -> dict[str, float] | None:
    sampled_frames: list[pd.DataFrame] = []
    for player_id in player_ids:
        player_frame = lineup_frame.loc[lineup_frame["player_id"] == int(player_id)].copy()
        if player_frame.empty:
            continue
        player_frame = player_frame.sort_values("game_date").tail(window)
        if not player_frame.empty:
            sampled_frames.append(player_frame)

    if not sampled_frames:
        return None

    sampled = pd.concat(sampled_frames, ignore_index=True)
    weights = sampled["pa"].where(sampled["pa"] > 0, 1.0).astype(float)
    if float(weights.sum()) <= 0:
        return None

    metrics = {
        metric: _weighted_metric_mean(
            sampled[metric],
            weights=weights,
            default=_default_metric_baseline(metric),
        )
        for metric in ("woba", "iso", "babip", "k_pct", "bb_pct")
    }

    derived_wrc_plus = _team_wrc_plus(metrics["woba"], league_woba)
    metrics["wrc_plus"] = _weighted_metric_mean(
        sampled["wrc_plus"],
        weights=weights,
        default=derived_wrc_plus,
    )
    return metrics


def _normalize_batting_stats(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return pd.DataFrame(columns=["player_id", "game_date", "pa", *METRICS])

    normalized = dataframe.copy()
    derived_metrics = _derive_offensive_metrics(normalized)
    player_id_column = _first_column(normalized, ("player_id", "playerid", "key_mlbam", "mlb_id", "id"))
    pa_column = _first_column(normalized, ("PA", "pa", "plate appearances", "_PA"))
    if player_id_column is None:
        return pd.DataFrame(columns=["player_id", "game_date", "pa", *METRICS])

    date_column = _first_column(
        normalized,
        ("game_date", "date", "Date", "as_of_timestamp", "as_of_date"),
    )

    column_map = {
        "player_id": player_id_column,
        "pa": pa_column,
        "wrc_plus": _first_column(normalized, ("wRC+", "wrc_plus", "wRC")),
        "woba": _first_column(normalized, ("wOBA", "woba")),
        "iso": _first_column(normalized, ("ISO", "iso")),
        "babip": _first_column(normalized, ("BABIP", "babip")),
        "k_pct": _first_column(normalized, ("K%", "k_pct", "SO%")),
        "bb_pct": _first_column(normalized, ("BB%", "bb_pct")),
    }

    result = pd.DataFrame()
    if date_column is not None:
        result["game_date"] = pd.to_datetime(normalized[date_column], errors="coerce")

    for output_column, source_column in column_map.items():
        if source_column is None:
            if output_column == "pa":
                result[output_column] = derived_metrics["pa"]
            elif output_column == "wrc_plus":
                result[output_column] = pd.Series(float("nan"), index=normalized.index, dtype=float)
            else:
                result[output_column] = derived_metrics[output_column]
            continue
        result[output_column] = _to_numeric(normalized[source_column])

    result = result.dropna(subset=["player_id"]).copy()
    result["player_id"] = result["player_id"].astype(int)
    result = result.fillna(
        {
            "pa": 0.0,
            "woba": _default_metric_baseline("woba"),
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
            "iso": _series_from_value((total_bases - hits) / ab.replace(0, pd.NA)),
            "babip": _series_from_value(
                (hits - home_runs)
                / (ab - strikeouts - home_runs + sacrifice_flies).replace(0, pd.NA)
            ),
            "k_pct": _series_from_value((strikeouts / plate_appearances.replace(0, pd.NA)) * 100),
            "bb_pct": _series_from_value((walks / plate_appearances.replace(0, pd.NA)) * 100),
        }
    )


def _compute_game_level_metrics(raw_logs: pd.DataFrame) -> pd.DataFrame:
    if raw_logs.empty:
        return pd.DataFrame(
            columns=[
                "game_date",
                "woba_numerator",
                "woba_denominator",
                "woba",
                "iso",
                "babip",
                "k_pct",
                "bb_pct",
            ]
        )

    date_column = _first_column(raw_logs, ("Date", "Offense_Date", "game_date"))
    if date_column is None:
        return pd.DataFrame(
            columns=[
                "game_date",
                "woba_numerator",
                "woba_denominator",
                "woba",
                "iso",
                "babip",
                "k_pct",
                "bb_pct",
            ]
        )

    ab = _extract_numeric(raw_logs, "AB", "Offense_AB")
    hits = _extract_numeric(raw_logs, "H", "Offense_H")
    doubles = _extract_numeric(raw_logs, "2B", "Offense_2B")
    triples = _extract_numeric(raw_logs, "3B", "Offense_3B")
    home_runs = _extract_numeric(raw_logs, "HR", "Offense_HR")
    walks = _extract_numeric(raw_logs, "BB", "Offense_BB")
    strikeouts = _extract_numeric(raw_logs, "SO", "K", "Offense_SO", "Offense_K")
    hit_by_pitch = _extract_numeric(raw_logs, "HBP", "Offense_HBP")
    sacrifice_flies = _extract_numeric(raw_logs, "SF", "Offense_SF")
    sacrifice_hits = _extract_numeric(raw_logs, "SH", "Offense_SH")

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

    result = pd.DataFrame(
        {
            "game_date": pd.to_datetime(raw_logs[date_column], errors="coerce"),
            "woba_numerator": woba_numerator,
            "woba_denominator": woba_denominator,
            "woba": _series_from_value(woba_numerator / woba_denominator.replace(0, pd.NA)),
            "iso": _series_from_value((total_bases - hits) / ab.replace(0, pd.NA)),
            "babip": _series_from_value(
                (hits - home_runs)
                / (ab - strikeouts - home_runs + sacrifice_flies).replace(0, pd.NA)
            ),
            "k_pct": _series_from_value((strikeouts / plate_appearances.replace(0, pd.NA)) * 100),
            "bb_pct": _series_from_value((walks / plate_appearances.replace(0, pd.NA)) * 100),
        }
    )

    result = result.dropna(subset=["game_date"]).sort_values("game_date").reset_index(drop=True)
    return result


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
) -> float:
    return blend_value(
        current_value,
        games_played=games_played,
        metric_type="offense",
        prior_value=prior_value,
        league_average=_default_metric_baseline(metric),
        regression_weight=regression_weight,
        roster_turnover_pct=roster_turnover_pct,
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
    if metric == "woba":
        return LEAGUE_WOBA_BASELINE
    return 0.0
