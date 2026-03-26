from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.clients.lineup_client import fetch_confirmed_lineups
from src.clients.statcast_client import fetch_statcast_range
from src.db import DEFAULT_DB_PATH, init_db, sqlite_connection
from src.features.marcel_blend import blend_value, get_regression_weight
from src.models.features import GameFeatures
from src.models.lineup import Lineup


DEFAULT_WINDOWS: tuple[int, ...] = (7, 14, 30, 60)
DEFAULT_REGRESSION_WEIGHT = int(get_regression_weight("pitching"))
DEFAULT_MIN_PERIODS = 1
OPENER_IP_THRESHOLD = 3.0
RECENT_START_WINDOW = 5
FIP_CONSTANT = 3.2
LEAGUE_HR_FB_RATE = 0.11
FASTBALL_PITCH_TYPES = {"FA", "FC", "FF", "FI", "FO", "FS", "FT", "SI", "SF"}
FASTBALL_NAME_TOKENS = ("FASTBALL", "SINKER", "CUTTER", "SPLITTER")
REPO_ROOT = Path(__file__).resolve().parents[2]
DERIVED_CACHE_ROOT = REPO_ROOT / "data" / "raw" / "derived" / "pitching"
START_METRICS_CACHE_VERSION = 2
METRIC_CANDIDATES: dict[str, tuple[str, ...]] = {
    "xfip": ("xfip", "xFIP"),
    "xera": ("xera", "xERA"),
    "k_pct": ("k_pct", "K%", "k%"),
    "bb_pct": ("bb_pct", "BB%", "bb%"),
    "gb_pct": ("gb_pct", "GB%", "gb%"),
    "hr_fb_pct": ("hr_fb_pct", "HR/FB", "hr/fb", "hr_fb"),
    "avg_fastball_velocity": (
        "avg_fastball_velocity",
        "fastball_velocity",
        "FBv",
        "FBv_avg",
        "avg_fb_velocity",
    ),
    "pitch_mix_entropy": ("pitch_mix_entropy", "entropy"),
    "csw_pct": ("csw_pct", "CSW%", "csw%"),
    "innings_pitched": ("innings_pitched", "IP", "ip"),
    "pitch_count": ("pitch_count", "pitches", "Pitches", "pitch_count_total"),
}
METRICS: tuple[str, ...] = tuple(
    metric for metric in METRIC_CANDIDATES if metric not in {"innings_pitched", "pitch_count"}
)
DEFAULT_METRIC_BASELINES = {
    "xfip": 4.20,
    "xera": 4.10,
    "k_pct": 22.0,
    "bb_pct": 8.0,
    "gb_pct": 43.0,
    "hr_fb_pct": 11.0,
    "avg_fastball_velocity": 93.5,
    "pitch_mix_entropy": 1.50,
    "csw_pct": 29.0,
    "pitch_count": 90.0,
    "velo_delta": 0.0,
}


_LineupFetcher = Callable[[str | date | datetime], list[Lineup]]
_StartMetricsFetcher = Callable[..., pd.DataFrame]


@dataclass(frozen=True, slots=True)
class _PitchingContext:
    starter_id: int | None
    is_opener: bool
    uses_team_composite: bool
    is_first_year: bool
    current_history: pd.DataFrame
    prior_baseline: dict[str, float]


def compute_pitching_features(
    game_date: str | date | datetime,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    regression_weight: int = DEFAULT_REGRESSION_WEIGHT,
    min_periods: int = DEFAULT_MIN_PERIODS,
    refresh: bool = False,
    roster_turnover_by_team: Mapping[str, float] | None = None,
    lineup_fetcher: _LineupFetcher = fetch_confirmed_lineups,
    start_metrics_fetcher: _StartMetricsFetcher | None = None,
) -> list[GameFeatures]:
    """Compute and persist lagged pitching features for games on a target date."""

    target_day = _coerce_date(game_date)
    database_path = Path(db_path)
    init_db(database_path)

    games = _load_games_for_date(database_path, target_day)
    if games.empty:
        return []

    resolved_start_metrics_fetcher = start_metrics_fetcher or _fetch_season_start_metrics
    current_metrics = _normalize_start_metrics(
        resolved_start_metrics_fetcher(
            target_day.year,
            db_path=database_path,
            end_date=target_day - timedelta(days=1),
            refresh=refresh,
        )
    )
    prior_metrics = _normalize_start_metrics(
        resolved_start_metrics_fetcher(
            target_day.year - 1,
            db_path=database_path,
            end_date=None,
            refresh=refresh,
        )
    )

    lineups = _load_lineups_for_date(target_day, lineup_fetcher)
    current_metrics = current_metrics.loc[
        current_metrics["game_date"].dt.date < target_day
    ].reset_index(drop=True)

    pitcher_histories = _group_history(current_metrics, "pitcher_id")
    team_histories = _group_history(current_metrics, "team")
    prior_pitcher_baselines = _build_baselines(prior_metrics, "pitcher_id")
    prior_team_baselines = _build_baselines(prior_metrics, "team")
    fallback_baseline = _fallback_baseline(prior_metrics, current_metrics)

    as_of_timestamp = datetime.combine(
        target_day - timedelta(days=1),
        time.min,
        tzinfo=timezone.utc,
    )

    features: list[GameFeatures] = []
    for game in games.to_dict(orient="records"):
        game_pk = int(game["game_pk"])
        for side_name, team_key, starter_key in (
            ("home", "home_team", "home_starter_id"),
            ("away", "away_team", "away_starter_id"),
        ):
            team = str(game[team_key])
            starter_id = _coerce_int(game.get(starter_key))
            roster_turnover_pct = (
                None if roster_turnover_by_team is None else roster_turnover_by_team.get(team)
            )
            lineup = lineups.get((game_pk, team))
            if lineup is not None:
                starter_id = (
                    lineup.starting_pitcher_id or lineup.projected_starting_pitcher_id or starter_id
                )

            context = _build_pitching_context(
                team=team,
                starter_id=starter_id,
                lineup=lineup,
                pitcher_histories=pitcher_histories,
                team_histories=team_histories,
                prior_pitcher_baselines=prior_pitcher_baselines,
                prior_team_baselines=prior_team_baselines,
                fallback_baseline=fallback_baseline,
            )

            features.extend(
                [
                    GameFeatures(
                        game_pk=game_pk,
                        feature_name=f"{side_name}_starter_is_opener",
                        feature_value=float(context.is_opener),
                        window_size=None,
                        as_of_timestamp=as_of_timestamp,
                    ),
                    GameFeatures(
                        game_pk=game_pk,
                        feature_name=f"{side_name}_starter_uses_team_composite",
                        feature_value=float(context.uses_team_composite),
                        window_size=None,
                        as_of_timestamp=as_of_timestamp,
                    ),
                ]
            )

            for window in windows:
                sample = context.current_history.tail(window)
                starts_count = len(sample)
                season_starts_count = len(context.current_history)
                for metric in METRICS:
                    current_value = (
                        _series_mean(sample[metric])
                        if starts_count >= min_periods
                        else float("nan")
                    )
                    feature_value = _apply_metric_blend(
                        metric=metric,
                        current_value=current_value,
                        prior_value=context.prior_baseline.get(metric, fallback_baseline[metric]),
                        starts_count=season_starts_count,
                        regression_weight=regression_weight,
                        roster_turnover_pct=roster_turnover_pct,
                        is_first_year=context.is_first_year,
                    )
                    features.append(
                        GameFeatures(
                            game_pk=game_pk,
                            feature_name=f"{side_name}_starter_{metric}_{window}s",
                            feature_value=feature_value,
                            window_size=window,
                            as_of_timestamp=as_of_timestamp,
                        )
                    )
            velocity_short = _series_mean(context.current_history.tail(7)["avg_fastball_velocity"])
            velocity_long = _series_mean(context.current_history.tail(60)["avg_fastball_velocity"])
            velocity_delta = velocity_short - velocity_long
            if math.isnan(velocity_delta):
                velocity_delta = DEFAULT_METRIC_BASELINES["velo_delta"]
            features.append(
                GameFeatures(
                    game_pk=game_pk,
                    feature_name=f"{side_name}_starter_velo_delta_7v60s",
                    feature_value=float(velocity_delta),
                    window_size=None,
                    as_of_timestamp=as_of_timestamp,
                )
            )
            workload_features = _starter_workload_features(
                game_pk=game_pk,
                side_name=side_name,
                context=context,
                target_day=target_day,
                as_of_timestamp=as_of_timestamp,
            )
            features.extend(workload_features)

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
            SELECT game_pk, date, home_team, away_team, home_starter_id, away_starter_id
            FROM games
            WHERE substr(date, 1, 10) = ?
            ORDER BY game_pk
            """,
            connection,
            params=(target_day.isoformat(),),
        )


def _load_lineups_for_date(
    target_day: date,
    lineup_fetcher: _LineupFetcher,
) -> dict[tuple[int, str], Lineup]:
    try:
        lineups = lineup_fetcher(target_day.isoformat())
    except Exception:
        return {}

    return {(lineup.game_pk, lineup.team): lineup for lineup in lineups}


def _build_pitching_context(
    *,
    team: str,
    starter_id: int | None,
    lineup: Lineup | None,
    pitcher_histories: Mapping[int, pd.DataFrame],
    team_histories: Mapping[str, pd.DataFrame],
    prior_pitcher_baselines: Mapping[int, dict[str, float]],
    prior_team_baselines: Mapping[str, dict[str, float]],
    fallback_baseline: Mapping[str, float],
) -> _PitchingContext:
    pitcher_history = pitcher_histories.get(starter_id, _empty_start_metrics())
    team_history = team_histories.get(team, _empty_start_metrics())

    inferred_opener = _is_opener_from_history(pitcher_history)
    lineup_indicates_opener = lineup is not None and lineup.is_opener
    is_opener = lineup_indicates_opener or inferred_opener
    uses_team_composite = starter_id is None or is_opener
    is_first_year = False

    if uses_team_composite:
        current_history = team_history
        prior_baseline = dict(prior_team_baselines.get(team, fallback_baseline))
    else:
        current_history = pitcher_history
        is_first_year = starter_id not in prior_pitcher_baselines
        prior_baseline = dict(prior_pitcher_baselines.get(starter_id, fallback_baseline))

    return _PitchingContext(
        starter_id=starter_id,
        is_opener=is_opener,
        uses_team_composite=uses_team_composite,
        is_first_year=is_first_year,
        current_history=current_history,
        prior_baseline=prior_baseline,
    )


def _is_opener_from_history(history: pd.DataFrame) -> bool:
    if history.empty:
        return False

    recent_history = history.tail(RECENT_START_WINDOW)
    innings = recent_history["innings_pitched"].dropna()
    if innings.empty:
        return False

    return float(innings.mean()) < OPENER_IP_THRESHOLD


def _group_history(dataframe: pd.DataFrame, key_column: str) -> dict[Any, pd.DataFrame]:
    if dataframe.empty:
        return {}

    groups: dict[Any, pd.DataFrame] = {}
    for group_key, group in dataframe.groupby(key_column, dropna=True):
        groups[group_key] = group.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
    return groups


def _build_baselines(dataframe: pd.DataFrame, key_column: str) -> dict[Any, dict[str, float]]:
    if dataframe.empty:
        return {}

    baselines: dict[Any, dict[str, float]] = {}
    for group_key, group in dataframe.groupby(key_column, dropna=True):
        baselines[group_key] = {
            metric: _series_mean(group[metric])
            if not group.empty
            else DEFAULT_METRIC_BASELINES[metric]
            for metric in METRICS
        }
    return baselines


def _fallback_baseline(
    prior_metrics: pd.DataFrame, current_metrics: pd.DataFrame
) -> dict[str, float]:
    reference = prior_metrics if not prior_metrics.empty else current_metrics
    if reference.empty:
        return dict(DEFAULT_METRIC_BASELINES)
    return {
        metric: _series_mean(reference[metric])
        if not reference[metric].dropna().empty
        else DEFAULT_METRIC_BASELINES[metric]
        for metric in METRICS
    }


def _apply_metric_blend(
    *,
    metric: str,
    current_value: float,
    prior_value: float,
    starts_count: int,
    regression_weight: int,
    roster_turnover_pct: float | None,
    is_first_year: bool = False,
) -> float:
    return blend_value(
        current_value,
        games_played=starts_count,
        metric_type="pitching",
        prior_value=prior_value,
        league_average=DEFAULT_METRIC_BASELINES[metric],
        regression_weight=regression_weight,
        roster_turnover_pct=roster_turnover_pct,
        is_first_year=is_first_year,
    )


def _normalize_start_metrics(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return _empty_start_metrics()

    normalized = dataframe.copy()
    result = pd.DataFrame()
    result["game_pk"] = _to_numeric_series(normalized.get(_first_column(normalized, ("game_pk",))))
    result["game_date"] = _to_tz_naive_datetime_series(
        normalized.get(_first_column(normalized, ("game_date", "Date", "date")))
    )
    result["team"] = _string_series(
        normalized.get(_first_column(normalized, ("team", "Team")))
    ).str.upper()
    result["pitcher_id"] = _to_numeric_series(
        normalized.get(_first_column(normalized, ("pitcher_id", "pitcher", "player_id", "ID")))
    )

    for metric, candidates in METRIC_CANDIDATES.items():
        column = _first_column(normalized, candidates)
        series = _to_numeric_series(normalized.get(column))
        if metric.endswith("_pct"):
            series = _normalize_percent_series(series)
        result[metric] = series

    result = result.dropna(subset=["game_pk", "game_date", "team"]).copy()
    result["game_pk"] = result["game_pk"].astype(int)
    result["team"] = result["team"].str.strip().str.upper()
    if not result["pitcher_id"].dropna().empty:
        result.loc[result["pitcher_id"].notna(), "pitcher_id"] = result.loc[
            result["pitcher_id"].notna(), "pitcher_id"
        ].astype(int)

    for metric, baseline in DEFAULT_METRIC_BASELINES.items():
        if metric in result.columns:
            result[metric] = result[metric].astype(float)
            result[metric] = result[metric].fillna(baseline)

    result["innings_pitched"] = result["innings_pitched"].astype(float).fillna(0.0)
    result["pitch_count"] = (
        result["pitch_count"].astype(float).fillna(DEFAULT_METRIC_BASELINES["pitch_count"])
    )
    return result.sort_values(["game_date", "game_pk", "team"]).reset_index(drop=True)


def _empty_start_metrics() -> pd.DataFrame:
    data: dict[str, pd.Series] = {
        "game_pk": pd.Series(dtype="int64"),
        "game_date": pd.Series(dtype="datetime64[ns]"),
        "team": pd.Series(dtype="str"),
        "pitcher_id": pd.Series(dtype="float64"),
        "innings_pitched": pd.Series(dtype="float64"),
        "pitch_count": pd.Series(dtype="float64"),
    }
    for metric in METRICS:
        data[metric] = pd.Series(dtype="float64")
    return pd.DataFrame(data)


def _starter_workload_features(
    *,
    game_pk: int,
    side_name: str,
    context: _PitchingContext,
    target_day: date,
    as_of_timestamp: datetime,
) -> list[GameFeatures]:
    default_days_rest = 5.0
    default_pitch_count = DEFAULT_METRIC_BASELINES["pitch_count"]

    if context.uses_team_composite:
        days_rest = default_days_rest
        last_pitch_count = default_pitch_count
        cumulative_pitch_load = default_pitch_count
    else:
        history = context.current_history

        if not history.empty and "game_date" in history.columns:
            last_start_date = history["game_date"].iloc[-1]
            if pd.notna(last_start_date):
                _last_ts = pd.to_datetime(last_start_date)
                if _last_ts.tzinfo is not None:
                    _last_ts = _last_ts.tz_convert(None)
                days_rest = float((pd.Timestamp(target_day) - _last_ts).days)
                days_rest = max(1.0, min(days_rest, 30.0))
            else:
                days_rest = default_days_rest
        else:
            days_rest = default_days_rest

        if not history.empty and "pitch_count" in history.columns:
            last_pitch_count_value = history["pitch_count"].iloc[-1]
            if pd.notna(last_pitch_count_value) and float(last_pitch_count_value) > 0:
                last_pitch_count = float(last_pitch_count_value)
            else:
                last_pitch_count = default_pitch_count

            recent_counts = pd.to_numeric(history.tail(5)["pitch_count"], errors="coerce")
            recent_counts = recent_counts.loc[recent_counts > 0]
            cumulative_pitch_load = (
                float(recent_counts.mean()) if not recent_counts.empty else default_pitch_count
            )
        else:
            last_pitch_count = default_pitch_count
            cumulative_pitch_load = default_pitch_count

    return [
        GameFeatures(
            game_pk=game_pk,
            feature_name=f"{side_name}_starter_days_rest",
            feature_value=float(days_rest),
            window_size=None,
            as_of_timestamp=as_of_timestamp,
        ),
        GameFeatures(
            game_pk=game_pk,
            feature_name=f"{side_name}_starter_last_start_pitch_count",
            feature_value=float(last_pitch_count),
            window_size=None,
            as_of_timestamp=as_of_timestamp,
        ),
        GameFeatures(
            game_pk=game_pk,
            feature_name=f"{side_name}_starter_cumulative_pitch_load_5s",
            feature_value=float(cumulative_pitch_load),
            window_size=None,
            as_of_timestamp=as_of_timestamp,
        ),
    ]


def _fetch_season_start_metrics(
    season: int,
    *,
    db_path: str | Path,
    end_date: date | None,
    refresh: bool = False,
) -> pd.DataFrame:
    start_rows = _load_start_rows(Path(db_path), season=season, end_date=end_date)
    if start_rows.empty:
        return _empty_start_metrics()

    cache_path = _start_metrics_cache_path(season, start_rows)
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    min_day = start_rows["game_date"].min().date()
    max_day = start_rows["game_date"].max().date()
    statcast_frame = fetch_statcast_range(min_day.isoformat(), max_day.isoformat(), refresh=refresh)
    metrics = _build_start_metrics_from_statcast(start_rows, statcast_frame)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(cache_path, index=False)
    return metrics


def _load_start_rows(
    db_path: Path,
    *,
    season: int,
    end_date: date | None,
) -> pd.DataFrame:
    query = """
        SELECT game_pk, date, home_team, away_team, home_starter_id, away_starter_id
        FROM games
        WHERE substr(date, 1, 4) = ?
    """
    params: list[Any] = [str(season)]
    if end_date is not None:
        query += " AND substr(date, 1, 10) <= ?"
        params.append(end_date.isoformat())

    with sqlite_connection(db_path, builder_optimized=True) as connection:
        games = pd.read_sql_query(query, connection, params=params)

    if games.empty:
        return pd.DataFrame(columns=["game_pk", "game_date", "team", "pitcher_id"])

    start_rows: list[dict[str, Any]] = []
    for game in games.to_dict(orient="records"):
        game_pk = _coerce_int(game.get("game_pk"))
        game_date = pd.to_datetime(game.get("date"), errors="coerce")
        if game_pk is None or pd.isna(game_date):
            continue

        for team_key, pitcher_key in (
            ("home_team", "home_starter_id"),
            ("away_team", "away_starter_id"),
        ):
            pitcher_id = _coerce_int(game.get(pitcher_key))
            team = str(game.get(team_key) or "").strip().upper()
            if pitcher_id is None or not team:
                continue
            start_rows.append(
                {
                    "game_pk": game_pk,
                    "game_date": game_date,
                    "team": team,
                    "pitcher_id": pitcher_id,
                }
            )

    return pd.DataFrame(start_rows)


def _start_metrics_cache_path(season: int, start_rows: pd.DataFrame) -> Path:
    normalized = start_rows.loc[:, ["game_pk", "game_date", "team", "pitcher_id"]].copy()
    normalized["game_pk"] = pd.to_numeric(normalized["game_pk"], errors="coerce").astype("Int64")
    normalized["game_date"] = pd.to_datetime(normalized["game_date"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    normalized["team"] = normalized["team"].astype(str).str.strip().str.upper()
    normalized["pitcher_id"] = pd.to_numeric(normalized["pitcher_id"], errors="coerce").astype(
        "Int64"
    )
    normalized = normalized.sort_values(["game_date", "game_pk", "team", "pitcher_id"]).reset_index(
        drop=True
    )

    digest = hashlib.sha256(
        pd.util.hash_pandas_object(normalized, index=False).values.tobytes()
    ).hexdigest()[:16]
    return DERIVED_CACHE_ROOT / (
        f"start_metrics_v{START_METRICS_CACHE_VERSION}_{season}_{digest}.parquet"
    )


def _build_start_metrics_from_statcast(
    start_rows: pd.DataFrame,
    statcast_frame: pd.DataFrame,
) -> pd.DataFrame:
    if start_rows.empty:
        return _empty_start_metrics()

    if statcast_frame.empty:
        empty = _empty_start_metrics()
        empty = empty.reindex(range(len(start_rows))).reset_index(drop=True)
        empty["game_pk"] = start_rows["game_pk"].astype(int)
        empty["game_date"] = _to_tz_naive_datetime_series(start_rows["game_date"])
        empty["team"] = start_rows["team"].astype(str)
        empty["pitcher_id"] = start_rows["pitcher_id"].astype(int)
        return empty

    pitches = statcast_frame.copy()
    if "game_pk" not in pitches.columns or "pitcher" not in pitches.columns:
        return _empty_start_metrics()

    pitches["game_pk"] = _to_numeric_series(pitches["game_pk"]).astype("Int64")
    pitches["pitcher"] = _to_numeric_series(pitches["pitcher"]).astype("Int64")
    league_hr_fb_rate = _calculate_league_hr_fb_rate(pitches)

    rows: list[dict[str, Any]] = []
    for start in start_rows.to_dict(orient="records"):
        game_pk = int(start["game_pk"])
        pitcher_id = int(start["pitcher_id"])
        pitcher_pitches = pitches.loc[
            (pitches["game_pk"] == game_pk) & (pitches["pitcher"] == pitcher_id)
        ].copy()
        metrics = _compute_start_metrics(pitcher_pitches, league_hr_fb_rate)
        metrics.update(
            {
                "game_pk": game_pk,
                "game_date": start["game_date"],
                "team": start["team"],
                "pitcher_id": pitcher_id,
            }
        )
        rows.append(metrics)

    return pd.DataFrame(rows)


def _calculate_league_hr_fb_rate(statcast_frame: pd.DataFrame) -> float:
    terminal = _collapse_plate_appearances(statcast_frame)
    if terminal.empty or "bb_type" not in terminal.columns:
        return LEAGUE_HR_FB_RATE

    fly_balls = terminal["bb_type"].astype(str).str.lower().isin({"fly_ball", "popup"}).sum()
    home_runs = (
        terminal.get("events", pd.Series(dtype=object)).astype(str).str.lower().eq("home_run").sum()
    )
    if fly_balls <= 0:
        return LEAGUE_HR_FB_RATE
    return float(home_runs / fly_balls)


def _compute_start_metrics(pitches: pd.DataFrame, league_hr_fb_rate: float) -> dict[str, float]:
    if pitches.empty:
        return {
            **DEFAULT_METRIC_BASELINES,
            "innings_pitched": 0.0,
            "pitch_count": 0.0,
        }

    terminal = _collapse_plate_appearances(pitches)
    batters_faced = len(terminal)
    events = terminal.get("events", pd.Series(dtype=object)).astype(str).str.lower()
    bb_types = terminal.get("bb_type", pd.Series(dtype=object)).astype(str).str.lower()

    strikeouts = events.isin({"strikeout", "strikeout_double_play"}).sum()
    walks = events.isin({"walk", "intent_walk"}).sum()
    hit_by_pitch = events.eq("hit_by_pitch").sum()
    home_runs = events.eq("home_run").sum()
    fly_balls = bb_types.isin({"fly_ball", "popup"}).sum()
    balls_in_play = bb_types.isin({"ground_ball", "line_drive", "fly_ball", "popup"}).sum()
    ground_balls = bb_types.eq("ground_ball").sum()

    outs_recorded = sum(_event_outs(event_name) for event_name in events.tolist())
    innings_pitched = outs_recorded / 3 if outs_recorded else 0.0
    pitch_count = len(pitches)
    expected_home_runs = fly_balls * (
        league_hr_fb_rate if league_hr_fb_rate > 0 else LEAGUE_HR_FB_RATE
    )

    k_pct = _safe_pct(strikeouts, batters_faced)
    bb_pct = _safe_pct(walks, batters_faced)
    gb_pct = _safe_pct(ground_balls, balls_in_play)
    hr_fb_pct = _safe_pct(home_runs, fly_balls)

    avg_fastball_velocity = _average_fastball_velocity(pitches)
    pitch_mix_entropy = _pitch_mix_entropy(pitches)
    csw_pct = _csw_pct(pitches)
    xera = _estimate_xera(terminal)
    xfip = _estimate_xfip(
        strikeouts=strikeouts,
        walks=walks,
        hit_by_pitch=hit_by_pitch,
        expected_home_runs=expected_home_runs,
        innings_pitched=innings_pitched,
    )

    return {
        "xfip": xfip,
        "xera": xera,
        "k_pct": k_pct,
        "bb_pct": bb_pct,
        "gb_pct": gb_pct,
        "hr_fb_pct": hr_fb_pct,
        "avg_fastball_velocity": avg_fastball_velocity,
        "pitch_mix_entropy": pitch_mix_entropy,
        "csw_pct": csw_pct,
        "innings_pitched": innings_pitched,
        "pitch_count": float(pitch_count),
    }


def _collapse_plate_appearances(pitches: pd.DataFrame) -> pd.DataFrame:
    if pitches.empty:
        return pitches.copy()

    if "at_bat_number" in pitches.columns:
        sort_columns = [
            column for column in ("at_bat_number", "pitch_number") if column in pitches.columns
        ]
        terminal = (
            pitches.sort_values(sort_columns).groupby("at_bat_number", as_index=False).tail(1)
        )
        return terminal.reset_index(drop=True)

    if "events" in pitches.columns:
        terminal = pitches.loc[pitches["events"].notna()].copy()
        if not terminal.empty:
            return terminal.reset_index(drop=True)

    return pitches.tail(1).reset_index(drop=True)


def _event_outs(event_name: str) -> int:
    outs_map = {
        "double_play": 2,
        "field_out": 1,
        "fielders_choice_out": 1,
        "flyout": 1,
        "force_out": 1,
        "ground_into_double_play": 2,
        "grounded_into_double_play": 2,
        "groundout": 1,
        "lineout": 1,
        "double": 0,
        "field_error": 0,
        "fielders_choice": 0,
        "home_run": 0,
        "intent_walk": 0,
        "pop_out": 1,
        "sac_bunt": 1,
        "sac_bunt_double_play": 2,
        "sac_fly": 1,
        "single": 0,
        "strikeout": 1,
        "strikeout_double_play": 2,
        "triple": 0,
        "triple_play": 3,
        "walk": 0,
    }
    return outs_map.get(event_name, 0)


def _average_fastball_velocity(pitches: pd.DataFrame) -> float:
    pitch_type_column = _first_column(pitches, ("pitch_type",))
    pitch_name_column = _first_column(pitches, ("pitch_name",))
    velocity_column = _first_column(pitches, ("release_speed", "start_speed"))
    if velocity_column is None:
        return DEFAULT_METRIC_BASELINES["avg_fastball_velocity"]

    fastball_mask = pd.Series(False, index=pitches.index)
    if pitch_type_column is not None:
        fastball_mask |= (
            pitches[pitch_type_column].astype(str).str.upper().isin(FASTBALL_PITCH_TYPES)
        )
    if pitch_name_column is not None:
        fastball_mask |= (
            pitches[pitch_name_column]
            .astype(str)
            .str.upper()
            .str.contains(
                "|".join(FASTBALL_NAME_TOKENS),
                regex=True,
            )
        )

    fastball_velocities = _to_numeric_series(pitches.loc[fastball_mask, velocity_column]).dropna()
    if fastball_velocities.empty:
        fastball_velocities = _to_numeric_series(pitches[velocity_column]).dropna()
    if fastball_velocities.empty:
        return DEFAULT_METRIC_BASELINES["avg_fastball_velocity"]
    return float(fastball_velocities.mean())


def _pitch_mix_entropy(pitches: pd.DataFrame) -> float:
    pitch_type_column = _first_column(pitches, ("pitch_type", "pitch_name"))
    if pitch_type_column is None:
        return DEFAULT_METRIC_BASELINES["pitch_mix_entropy"]

    pitch_counts = pitches[pitch_type_column].astype(str).str.strip().value_counts()
    if pitch_counts.empty:
        return DEFAULT_METRIC_BASELINES["pitch_mix_entropy"]

    total = int(pitch_counts.sum())
    if total <= 0:
        return DEFAULT_METRIC_BASELINES["pitch_mix_entropy"]

    entropy = -sum(
        (count / total) * math.log2(count / total) for count in pitch_counts if count > 0
    )
    return float(entropy)


def _csw_pct(pitches: pd.DataFrame) -> float:
    description_column = _first_column(pitches, ("description",))
    if description_column is None:
        return DEFAULT_METRIC_BASELINES["csw_pct"]

    descriptions = pitches[description_column].astype(str).str.lower().str.strip()
    total_pitches = len(descriptions)
    if total_pitches <= 0:
        return DEFAULT_METRIC_BASELINES["csw_pct"]

    csw_count = descriptions.isin(
        {
            "called_strike",
            "swinging_strike",
            "swinging_strike_blocked",
        }
    ).sum()
    return float((csw_count / total_pitches) * 100.0)


def _estimate_xera(terminal: pd.DataFrame) -> float:
    xwoba_column = _first_column(terminal, ("estimated_woba_using_speedangle", "xwoba"))
    if xwoba_column is None:
        return DEFAULT_METRIC_BASELINES["xera"]

    xwoba = _to_numeric_series(terminal[xwoba_column]).dropna()
    if xwoba.empty:
        return DEFAULT_METRIC_BASELINES["xera"]

    mean_xwoba = float(xwoba.mean())
    return float(3.2 + ((mean_xwoba - 0.320) * 15.0))


def _estimate_xfip(
    *,
    strikeouts: int,
    walks: int,
    hit_by_pitch: int,
    expected_home_runs: float,
    innings_pitched: float,
) -> float:
    if innings_pitched <= 0:
        return DEFAULT_METRIC_BASELINES["xfip"]

    value = (
        (13.0 * expected_home_runs) + (3.0 * (walks + hit_by_pitch)) - (2.0 * strikeouts)
    ) / innings_pitched
    return float(value + FIP_CONSTANT)


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
                (game_pk, feature_name, as_of, window_size, window_size)
                for game_pk, feature_name, _, window_size, as_of in rows
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


def _first_column(dataframe: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    normalized_columns = {str(column).strip().lower(): str(column) for column in dataframe.columns}
    for candidate in candidates:
        match = normalized_columns.get(candidate.strip().lower())
        if match is not None:
            return match
    return None


def _to_numeric_series(values: Any) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(values, errors="coerce")


def _to_tz_naive_datetime_series(values: Any) -> pd.Series:
    if values is None:
        return pd.Series(dtype="datetime64[ns]")

    parsed = pd.to_datetime(values, errors="coerce", utc=True, format="mixed")
    return parsed.dt.tz_convert(None)


def _string_series(values: Any) -> pd.Series:
    if values is None:
        return pd.Series(dtype=str)
    return values.astype(str)


def _normalize_percent_series(values: pd.Series) -> pd.Series:
    non_null = values.dropna()
    if non_null.empty:
        return values.astype(float)
    if float(non_null.abs().max()) <= 1.5:
        return values.astype(float) * 100.0
    return values.astype(float)


def _series_mean(values: pd.Series) -> float:
    numeric = _to_numeric_series(values)
    if numeric.dropna().empty:
        return float("nan")
    return float(numeric.mean())


def _safe_pct(numerator: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return 0.0
    return float((numerator / denominator) * 100.0)


def _coerce_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
