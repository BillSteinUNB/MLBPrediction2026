from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Sequence
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.clients.statcast_client import fetch_statcast_range, fetch_team_game_logs
from src.db import DEFAULT_DB_PATH, init_db, sqlite_connection
from src.models.features import GameFeatures


DEFAULT_PITCH_COUNT_WINDOWS: tuple[int, ...] = (3, 5)
DEFAULT_IR_WINDOW = 30
DEFAULT_TOP_RELIEVER_COUNT = 5
DEFAULT_MIN_SEASON_DAYS = 3
DEFAULT_XFIP = 4.20
DEFAULT_AVG_REST_DAYS = float(DEFAULT_MIN_SEASON_DAYS)
AVAILABLE_REST_DAYS_THRESHOLD = 2
LEAGUE_HR_FB_RATE = 0.11
FIP_CONSTANT = 3.2
REPO_ROOT = Path(__file__).resolve().parents[2]
DERIVED_CACHE_ROOT = REPO_ROOT / "data" / "raw" / "derived" / "bullpen"
BULLPEN_METRICS_CACHE_VERSION = 1

_BullpenMetricsFetcher = Callable[..., pd.DataFrame]
_TeamLogsFetcher = Callable[..., pd.DataFrame]

logger = logging.getLogger(__name__)


def compute_bullpen_features(
    game_date: str | date | datetime,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    pitch_count_windows: Sequence[int] = DEFAULT_PITCH_COUNT_WINDOWS,
    ir_window: int = DEFAULT_IR_WINDOW,
    top_reliever_count: int = DEFAULT_TOP_RELIEVER_COUNT,
    min_season_days: int = DEFAULT_MIN_SEASON_DAYS,
    refresh: bool = False,
    bullpen_metrics_fetcher: _BullpenMetricsFetcher | None = None,
    team_logs_fetcher: _TeamLogsFetcher = fetch_team_game_logs,
) -> list[GameFeatures]:
    """Compute and persist lagged bullpen fatigue features for games on a target date."""

    target_day = _coerce_date(game_date)
    database_path = Path(db_path)
    init_db(database_path)

    games = _load_games_for_date(database_path, target_day)
    if games.empty:
        return []

    if bullpen_metrics_fetcher is None:
        metrics_frame = _fetch_season_bullpen_metrics(
            target_day.year,
            db_path=database_path,
            end_date=target_day - timedelta(days=1),
            refresh=refresh,
            team_logs_fetcher=team_logs_fetcher,
        )
    else:
        metrics_frame = bullpen_metrics_fetcher(
            target_day.year,
            db_path=database_path,
            end_date=target_day - timedelta(days=1),
            refresh=refresh,
        )

    bullpen_metrics = _normalize_bullpen_metrics(metrics_frame)
    bullpen_metrics = bullpen_metrics.loc[bullpen_metrics["game_date"].dt.date < target_day].reset_index(
        drop=True
    )
    season_day = _resolve_season_day(database_path, target_day, bullpen_metrics)

    as_of_timestamp = datetime.combine(
        target_day - timedelta(days=1),
        time.min,
        tzinfo=timezone.utc,
    )

    features: list[GameFeatures] = []
    for game in games.to_dict(orient="records"):
        game_pk = int(game["game_pk"])
        for side_name, team_key in (("home", "home_team"), ("away", "away_team")):
            team = str(game[team_key]).strip().upper()
            team_history = bullpen_metrics.loc[bullpen_metrics["team"] == team].copy()

            feature_values = _build_feature_values(
                team_history=team_history,
                target_day=target_day,
                pitch_count_windows=pitch_count_windows,
                ir_window=ir_window,
                top_reliever_count=top_reliever_count,
                min_season_days=min_season_days,
                season_day=season_day,
            )

            for feature_name, feature_value, window_size in feature_values:
                features.append(
                    GameFeatures(
                        game_pk=game_pk,
                        feature_name=f"{side_name}_team_{feature_name}",
                        feature_value=float(feature_value),
                        window_size=window_size,
                        as_of_timestamp=as_of_timestamp,
                    )
                )

    _persist_features(database_path, features)
    return features


def _build_feature_values(
    *,
    team_history: pd.DataFrame,
    target_day: date,
    pitch_count_windows: Sequence[int],
    ir_window: int,
    top_reliever_count: int,
    min_season_days: int,
    season_day: int,
) -> list[tuple[str, float, int | None]]:
    if season_day <= min_season_days or team_history.empty:
        return _default_feature_values(
            pitch_count_windows=pitch_count_windows,
            ir_window=ir_window,
            top_reliever_count=top_reliever_count,
        )

    top_relievers = _select_top_relievers(team_history, top_reliever_count)
    average_rest_days = _average_rest_days(top_relievers, target_day)
    availability_count = _available_reliever_count(top_relievers, target_day)
    bullpen_xfip = _weighted_mean(
        team_history["xfip"],
        weights=team_history["innings_pitched"],
        default=DEFAULT_XFIP,
    )
    ir_pct = _rolling_inherited_runner_pct(team_history, ir_window)

    features: list[tuple[str, float, int | None]] = [
        ("bullpen_avg_rest_days_top5", average_rest_days, None),
        (f"bullpen_ir_pct_{ir_window}g", ir_pct, ir_window),
        ("bullpen_xfip", bullpen_xfip, None),
        ("bullpen_high_leverage_available_count", availability_count, None),
    ]
    for window in pitch_count_windows:
        features.insert(
            len(features) - 4 if features else 0,
            (
                f"bullpen_pitch_count_{int(window)}d",
                _pitch_count_last_days(team_history, target_day, int(window)),
                int(window),
            ),
        )

    return features


def _default_feature_values(
    *,
    pitch_count_windows: Sequence[int],
    ir_window: int,
    top_reliever_count: int,
) -> list[tuple[str, float, int | None]]:
    defaults = [
        (f"bullpen_pitch_count_{int(window)}d", 0.0, int(window))
        for window in pitch_count_windows
    ]
    defaults.extend(
        [
            ("bullpen_avg_rest_days_top5", DEFAULT_AVG_REST_DAYS, None),
            (f"bullpen_ir_pct_{ir_window}g", 0.0, ir_window),
            ("bullpen_xfip", DEFAULT_XFIP, None),
            ("bullpen_high_leverage_available_count", float(top_reliever_count), None),
        ]
    )
    return defaults


def _pitch_count_last_days(team_history: pd.DataFrame, target_day: date, window: int) -> float:
    start_day = target_day - timedelta(days=window)
    mask = (team_history["game_date"].dt.date >= start_day) & (team_history["game_date"].dt.date < target_day)
    return float(team_history.loc[mask, "pitch_count"].sum())


def _select_top_relievers(team_history: pd.DataFrame, top_reliever_count: int) -> pd.DataFrame:
    if team_history.empty:
        return _empty_bullpen_metrics().iloc[0:0]

    usage = (
        team_history.groupby("pitcher_id", dropna=True, as_index=False)
        .agg(
            total_pitch_count=("pitch_count", "sum"),
            last_game_date=("game_date", "max"),
        )
        .sort_values(["total_pitch_count", "last_game_date", "pitcher_id"], ascending=[False, False, True])
        .head(top_reliever_count)
    )
    if usage.empty:
        return _empty_bullpen_metrics().iloc[0:0]

    selected_ids = usage["pitcher_id"].dropna().astype(int).tolist()
    return team_history.loc[team_history["pitcher_id"].isin(selected_ids)].copy()


def _average_rest_days(top_reliever_history: pd.DataFrame, target_day: date) -> float:
    rest_days = _rest_days_lookup(top_reliever_history, target_day)
    if not rest_days:
        return DEFAULT_AVG_REST_DAYS
    return float(sum(rest_days.values()) / len(rest_days))


def _available_reliever_count(top_reliever_history: pd.DataFrame, target_day: date) -> float:
    rest_days = _rest_days_lookup(top_reliever_history, target_day)
    if not rest_days:
        return float(DEFAULT_TOP_RELIEVER_COUNT)
    return float(sum(1 for days_rest in rest_days.values() if days_rest >= AVAILABLE_REST_DAYS_THRESHOLD))


def _rest_days_lookup(top_reliever_history: pd.DataFrame, target_day: date) -> dict[int, int]:
    if top_reliever_history.empty:
        return {}

    latest = (
        top_reliever_history.groupby("pitcher_id", dropna=True)["game_date"]
        .max()
        .dropna()
    )
    rest_days: dict[int, int] = {}
    for pitcher_id, appearance_date in latest.items():
        days_rest = (target_day - pd.Timestamp(appearance_date).date()).days - 1
        rest_days[int(pitcher_id)] = max(int(days_rest), 0)
    return rest_days


def _rolling_inherited_runner_pct(team_history: pd.DataFrame, window: int) -> float:
    if team_history.empty:
        return 0.0

    game_level = (
        team_history.groupby(["game_pk", "game_date"], as_index=False)
        .agg(
            inherited_runners=("inherited_runners", "max"),
            inherited_runners_scored=("inherited_runners_scored", "max"),
        )
        .sort_values(["game_date", "game_pk"])
        .tail(window)
    )
    inherited_runners = float(game_level["inherited_runners"].sum())
    if inherited_runners <= 0:
        return 0.0

    inherited_runners_scored = float(game_level["inherited_runners_scored"].sum())
    return float(inherited_runners_scored / inherited_runners)


def _weighted_mean(values: pd.Series, *, weights: pd.Series, default: float) -> float:
    numeric_values = pd.to_numeric(values, errors="coerce")
    numeric_weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    valid = numeric_values.notna() & numeric_weights.gt(0)
    if not valid.any():
        return float(default)
    numerator = float((numeric_values.loc[valid] * numeric_weights.loc[valid]).sum())
    denominator = float(numeric_weights.loc[valid].sum())
    if denominator <= 0:
        return float(default)
    return float(numerator / denominator)


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
            SELECT game_pk, date, home_team, away_team
            FROM games
            WHERE substr(date, 1, 10) = ?
            ORDER BY game_pk
            """,
            connection,
            params=(target_day.isoformat(),),
        )


def _resolve_season_day(db_path: Path, target_day: date, bullpen_metrics: pd.DataFrame) -> int:
    season_start_candidates: list[date] = []

    with sqlite_connection(db_path, builder_optimized=True) as connection:
        row = connection.execute(
            """
            SELECT MIN(substr(date, 1, 10))
            FROM games
            WHERE substr(date, 1, 4) = ?
            """,
            (str(target_day.year),),
        ).fetchone()
    if row and row[0]:
        season_start_candidates.append(date.fromisoformat(str(row[0])))

    if not bullpen_metrics.empty:
        season_start_candidates.append(pd.Timestamp(bullpen_metrics["game_date"].min()).date())

    if not season_start_candidates:
        return 1

    season_start = min(season_start_candidates)
    return max((target_day - season_start).days + 1, 1)


def _normalize_bullpen_metrics(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return _empty_bullpen_metrics()

    normalized = dataframe.copy()
    result = pd.DataFrame()
    result["game_pk"] = _to_numeric_series(normalized.get(_first_column(normalized, ("game_pk",))))
    result["game_date"] = _normalize_date_series(
        normalized.get(_first_column(normalized, ("game_date", "date", "Date")))
    )
    result["team"] = _string_series(
        normalized.get(_first_column(normalized, ("team", "Team")))
    ).str.strip().str.upper()
    result["pitcher_id"] = _to_numeric_series(
        normalized.get(_first_column(normalized, ("pitcher_id", "pitcher", "player_id", "ID")))
    )
    result["pitch_count"] = _to_numeric_series(
        normalized.get(_first_column(normalized, ("pitch_count", "pitches", "pitches_thrown")))
    ).fillna(0.0)
    result["innings_pitched"] = _to_numeric_series(
        normalized.get(_first_column(normalized, ("innings_pitched", "IP", "ip")))
    ).fillna(0.0)
    result["xfip"] = _to_numeric_series(
        normalized.get(_first_column(normalized, ("xfip", "xFIP")))
    ).fillna(DEFAULT_XFIP)
    result["inherited_runners"] = _to_numeric_series(
        normalized.get(
            _first_column(
                normalized,
                (
                    "inherited_runners",
                    "IR",
                    "ir",
                ),
            )
        )
    ).fillna(0.0)
    result["inherited_runners_scored"] = _to_numeric_series(
        normalized.get(
            _first_column(
                normalized,
                (
                    "inherited_runners_scored",
                    "IS",
                    "is",
                ),
            )
        )
    ).fillna(0.0)

    result = result.dropna(subset=["game_pk", "game_date", "team"]).copy()
    if result.empty:
        return _empty_bullpen_metrics()

    result["game_pk"] = result["game_pk"].astype(int)
    result.loc[result["pitcher_id"].notna(), "pitcher_id"] = result.loc[
        result["pitcher_id"].notna(), "pitcher_id"
    ].astype(int)
    return result.sort_values(["game_date", "game_pk", "team", "pitcher_id"]).reset_index(drop=True)


def _empty_bullpen_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_pk": pd.Series(dtype="int64"),
            "game_date": pd.Series(dtype="datetime64[ns]"),
            "team": pd.Series(dtype="str"),
            "pitcher_id": pd.Series(dtype="float64"),
            "pitch_count": pd.Series(dtype="float64"),
            "innings_pitched": pd.Series(dtype="float64"),
            "xfip": pd.Series(dtype="float64"),
            "inherited_runners": pd.Series(dtype="float64"),
            "inherited_runners_scored": pd.Series(dtype="float64"),
        }
    )


def _fetch_season_bullpen_metrics(
    season: int,
    *,
    db_path: Path,
    end_date: date,
    refresh: bool = False,
    team_logs_fetcher: _TeamLogsFetcher,
) -> pd.DataFrame:
    games = _load_season_games(db_path, season=season, end_date=end_date)
    if games.empty:
        return _empty_bullpen_metrics()

    cache_path = _bullpen_metrics_cache_path(season, games)
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    min_day = pd.Timestamp(games["game_date"].min()).date()
    max_day = pd.Timestamp(games["game_date"].max()).date()
    statcast_frame = fetch_statcast_range(min_day.isoformat(), max_day.isoformat(), refresh=refresh)
    relief_metrics = _build_relief_metrics_from_statcast(games, statcast_frame)
    if relief_metrics.empty:
        return _empty_bullpen_metrics()

    inherited_runner_lookup = _build_inherited_runner_lookup(
        games,
        season=season,
        refresh=refresh,
        team_logs_fetcher=team_logs_fetcher,
    )
    if inherited_runner_lookup.empty:
        relief_metrics["inherited_runners"] = 0.0
        relief_metrics["inherited_runners_scored"] = 0.0
        return relief_metrics

    merged = relief_metrics.merge(
        inherited_runner_lookup,
        on=["game_pk", "game_date", "team"],
        how="left",
    )
    merged["inherited_runners"] = pd.to_numeric(
        merged["inherited_runners"],
        errors="coerce",
    ).fillna(0.0)
    merged["inherited_runners_scored"] = pd.to_numeric(
        merged["inherited_runners_scored"],
        errors="coerce",
    ).fillna(0.0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(cache_path, index=False)
    return merged


def _load_season_games(db_path: Path, *, season: int, end_date: date) -> pd.DataFrame:
    with sqlite_connection(db_path, builder_optimized=True) as connection:
        games = pd.read_sql_query(
            """
            SELECT game_pk, date, home_team, away_team, home_starter_id, away_starter_id
            FROM games
            WHERE substr(date, 1, 4) = ?
              AND substr(date, 1, 10) <= ?
            ORDER BY date, game_pk
            """,
            connection,
            params=(str(season), end_date.isoformat()),
        )

    if games.empty:
        return pd.DataFrame(
            columns=["game_pk", "game_date", "home_team", "away_team", "home_starter_id", "away_starter_id"]
        )

    games["game_date"] = _normalize_date_series(games["date"])
    games["home_team"] = games["home_team"].astype(str).str.strip().str.upper()
    games["away_team"] = games["away_team"].astype(str).str.strip().str.upper()
    for starter_column in ("home_starter_id", "away_starter_id"):
        games[starter_column] = pd.to_numeric(games[starter_column], errors="coerce").astype("Int64")
    return games.dropna(subset=["game_pk", "game_date"]).reset_index(drop=True)


def _bullpen_metrics_cache_path(season: int, games: pd.DataFrame) -> Path:
    normalized = games.loc[
        :,
        ["game_pk", "game_date", "home_team", "away_team", "home_starter_id", "away_starter_id"],
    ].copy()
    normalized["game_pk"] = pd.to_numeric(normalized["game_pk"], errors="coerce").astype("Int64")
    normalized["game_date"] = pd.to_datetime(normalized["game_date"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    for team_column in ("home_team", "away_team"):
        normalized[team_column] = normalized[team_column].astype(str).str.strip().str.upper()
    for starter_column in ("home_starter_id", "away_starter_id"):
        normalized[starter_column] = pd.to_numeric(
            normalized[starter_column],
            errors="coerce",
        ).astype("Int64")
    normalized = normalized.sort_values(["game_date", "game_pk"]).reset_index(drop=True)

    digest = hashlib.sha256(
        pd.util.hash_pandas_object(normalized, index=False).values.tobytes()
    ).hexdigest()[:16]
    return DERIVED_CACHE_ROOT / (
        f"bullpen_metrics_v{BULLPEN_METRICS_CACHE_VERSION}_{season}_{digest}.parquet"
    )


def _build_relief_metrics_from_statcast(games: pd.DataFrame, statcast_frame: pd.DataFrame) -> pd.DataFrame:
    if games.empty or statcast_frame.empty:
        return _empty_bullpen_metrics()

    pitches = statcast_frame.copy()
    game_pk_column = _first_column(pitches, ("game_pk",))
    pitcher_column = _first_column(pitches, ("pitcher", "pitcher_id"))
    if game_pk_column is None or pitcher_column is None:
        return _empty_bullpen_metrics()

    pitches["game_pk"] = _to_numeric_series(pitches[game_pk_column]).astype("Int64")
    pitches["pitcher_id"] = _to_numeric_series(pitches[pitcher_column]).astype("Int64")
    pitches["game_date"] = _normalize_date_series(
        pitches.get(_first_column(pitches, ("game_date", "date", "Date")))
    )

    schedule = games[["game_pk", "game_date", "home_team", "away_team", "home_starter_id", "away_starter_id"]].copy()
    schedule["game_pk"] = pd.to_numeric(schedule["game_pk"], errors="coerce").astype("Int64")
    pitches = pitches.merge(schedule, on="game_pk", how="inner", suffixes=("", "_schedule"))
    if pitches.empty:
        return _empty_bullpen_metrics()

    pitches["game_date"] = pitches["game_date"].fillna(pitches["game_date_schedule"])
    pitches["team"] = _resolve_pitching_team(pitches)
    pitches = pitches.loc[pitches["team"].astype(str) != ""].copy()
    if pitches.empty:
        return _empty_bullpen_metrics()

    pitch_counts = (
        pitches.groupby(["game_pk", "team", "pitcher_id"], dropna=True)
        .size()
        .rename("pitch_count")
        .reset_index()
    )
    designated_starters = _designated_starters(games, pitch_counts)
    starters_lookup = designated_starters.set_index(["game_pk", "team"])["starter_id"]
    pitcher_keys = pd.MultiIndex.from_frame(pitch_counts[["game_pk", "team"]])
    pitch_counts["starter_id"] = starters_lookup.reindex(pitcher_keys).to_numpy()
    reliever_ids = pitch_counts.loc[
        pitch_counts["pitcher_id"].notna()
        & (pitch_counts["starter_id"].isna() | (pitch_counts["pitcher_id"] != pitch_counts["starter_id"])),
        ["game_pk", "team", "pitcher_id", "pitch_count"],
    ].copy()
    if reliever_ids.empty:
        return _empty_bullpen_metrics()

    relief_pitches = pitches.merge(
        reliever_ids[["game_pk", "team", "pitcher_id"]],
        on=["game_pk", "team", "pitcher_id"],
        how="inner",
    )
    if relief_pitches.empty:
        return _empty_bullpen_metrics()

    league_hr_fb_rate = _calculate_league_hr_fb_rate(relief_pitches)
    rows: list[dict[str, Any]] = []
    for keys, group in relief_pitches.groupby(["game_pk", "game_date", "team", "pitcher_id"], dropna=True):
        game_pk, game_date, team, pitcher_id = keys
        pitch_count = int(len(group))
        innings_pitched = _innings_pitched(group)
        xfip = _xfip_from_pitches(group, league_hr_fb_rate)
        rows.append(
            {
                "game_pk": int(game_pk),
                "game_date": pd.Timestamp(game_date).normalize(),
                "team": str(team).strip().upper(),
                "pitcher_id": int(pitcher_id),
                "pitch_count": float(pitch_count),
                "innings_pitched": innings_pitched,
                "xfip": xfip,
            }
        )

    return pd.DataFrame(rows)


def _resolve_pitching_team(pitches: pd.DataFrame) -> pd.Series:
    topbot_column = _first_column(pitches, ("inning_topbot", "inning_top_bot"))
    if topbot_column is None:
        return pd.Series("", index=pitches.index, dtype=str)

    half_inning = pitches[topbot_column].astype(str).str.strip().str.lower()
    team = pd.Series("", index=pitches.index, dtype=str)
    team.loc[half_inning.eq("top")] = pitches.loc[half_inning.eq("top"), "home_team"].astype(str)
    team.loc[half_inning.eq("bot")] = pitches.loc[half_inning.eq("bot"), "away_team"].astype(str)
    return team.str.strip().str.upper()


def _designated_starters(games: pd.DataFrame, pitch_counts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for game in games.to_dict(orient="records"):
        for team_key, starter_key in (("home_team", "home_starter_id"), ("away_team", "away_starter_id")):
            team = str(game[team_key]).strip().upper()
            starter_id = pd.to_numeric(game.get(starter_key), errors="coerce")
            if pd.isna(starter_id):
                team_counts = pitch_counts.loc[
                    (pitch_counts["game_pk"] == game["game_pk"]) & (pitch_counts["team"] == team)
                ]
                if team_counts.empty:
                    starter_value = pd.NA
                else:
                    starter_value = int(
                        team_counts.sort_values(["pitch_count", "pitcher_id"], ascending=[False, True]).iloc[0][
                            "pitcher_id"
                        ]
                    )
            else:
                starter_value = int(starter_id)
            rows.append({"game_pk": int(game["game_pk"]), "team": team, "starter_id": starter_value})
    return pd.DataFrame(rows)


def _build_inherited_runner_lookup(
    games: pd.DataFrame,
    *,
    season: int,
    refresh: bool,
    team_logs_fetcher: _TeamLogsFetcher,
) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame(columns=["game_pk", "game_date", "team", "inherited_runners", "inherited_runners_scored"])

    scheduled_games = _scheduled_team_games(games)
    if scheduled_games.empty:
        return pd.DataFrame(columns=["game_pk", "game_date", "team", "inherited_runners", "inherited_runners_scored"])

    lookups: list[pd.DataFrame] = []
    for team in scheduled_games["team"].drop_duplicates().tolist():
        try:
            raw_logs = team_logs_fetcher(season, team, log_type="pitching", refresh=refresh)
        except Exception:
            raw_logs = pd.DataFrame()

        normalized_logs = _normalize_inherited_runner_logs(raw_logs, team)
        if normalized_logs.empty:
            continue

        team_schedule = scheduled_games.loc[scheduled_games["team"] == team].copy()
        merged = team_schedule.merge(
            normalized_logs,
            on=["team", "game_date", "date_index"],
            how="left",
        )
        lookups.append(
            merged[["game_pk", "game_date", "team", "inherited_runners", "inherited_runners_scored"]]
        )

    if not lookups:
        return pd.DataFrame(columns=["game_pk", "game_date", "team", "inherited_runners", "inherited_runners_scored"])

    combined = pd.concat(lookups, ignore_index=True)
    combined["inherited_runners"] = pd.to_numeric(combined["inherited_runners"], errors="coerce").fillna(0.0)
    combined["inherited_runners_scored"] = pd.to_numeric(
        combined["inherited_runners_scored"],
        errors="coerce",
    ).fillna(0.0)
    return combined


def _scheduled_team_games(games: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for game in games.to_dict(orient="records"):
        for team_key in ("home_team", "away_team"):
            rows.append(
                {
                    "game_pk": int(game["game_pk"]),
                    "game_date": pd.Timestamp(game["game_date"]).normalize(),
                    "team": str(game[team_key]).strip().upper(),
                }
            )

    scheduled = pd.DataFrame(rows)
    scheduled = scheduled.sort_values(["team", "game_date", "game_pk"]).reset_index(drop=True)
    scheduled["date_index"] = scheduled.groupby(["team", "game_date"]).cumcount()
    return scheduled


def _normalize_inherited_runner_logs(raw_logs: pd.DataFrame, team: str) -> pd.DataFrame:
    if raw_logs.empty:
        return pd.DataFrame(columns=["team", "game_date", "date_index", "inherited_runners", "inherited_runners_scored"])

    date_column = _first_column(
        raw_logs,
        (
            "Date",
            "date",
            "game_date",
            "Pitching_Date",
            "pitching_date",
        ),
    )
    inherited_runners_column = _first_column(
        raw_logs,
        (
            "IR",
            "ir",
            "Inherited Runners",
            "Inherited_Runners",
            "pitching_ir",
            "Pitching_IR",
        ),
    )
    inherited_runners_scored_column = _first_column(
        raw_logs,
        (
            "IS",
            "is",
            "Inherited Runners Scored",
            "Inherited_Runners_Scored",
            "pitching_is",
            "Pitching_IS",
        ),
    )
    if date_column is None or inherited_runners_column is None or inherited_runners_scored_column is None:
        return pd.DataFrame(columns=["team", "game_date", "date_index", "inherited_runners", "inherited_runners_scored"])

    normalized = pd.DataFrame({"game_date": _normalize_date_series(raw_logs[date_column])})
    normalized["team"] = team
    normalized["inherited_runners"] = pd.to_numeric(raw_logs[inherited_runners_column], errors="coerce")
    normalized["inherited_runners_scored"] = pd.to_numeric(
        raw_logs[inherited_runners_scored_column],
        errors="coerce",
    )
    normalized = normalized.dropna(subset=["game_date"]).reset_index(drop=True)
    if normalized.empty:
        return pd.DataFrame(columns=["team", "game_date", "date_index", "inherited_runners", "inherited_runners_scored"])

    normalized["date_index"] = normalized.groupby(["team", "game_date"]).cumcount()
    normalized["inherited_runners"] = normalized["inherited_runners"].fillna(0.0)
    normalized["inherited_runners_scored"] = normalized["inherited_runners_scored"].fillna(0.0)
    return normalized


def _calculate_league_hr_fb_rate(pitches: pd.DataFrame) -> float:
    terminal = _collapse_plate_appearances(pitches)
    if terminal.empty:
        return LEAGUE_HR_FB_RATE

    bb_type_column = _first_column(terminal, ("bb_type",))
    events_column = _first_column(terminal, ("events",))
    if bb_type_column is None or events_column is None:
        return LEAGUE_HR_FB_RATE

    fly_balls = terminal[bb_type_column].astype(str).str.lower().isin({"fly_ball", "popup"}).sum()
    home_runs = terminal[events_column].astype(str).str.lower().eq("home_run").sum()
    if fly_balls <= 0:
        return LEAGUE_HR_FB_RATE
    return float(home_runs / fly_balls)


def _xfip_from_pitches(pitches: pd.DataFrame, league_hr_fb_rate: float) -> float:
    terminal = _collapse_plate_appearances(pitches)
    if terminal.empty:
        return DEFAULT_XFIP

    events_column = _first_column(terminal, ("events",))
    bb_type_column = _first_column(terminal, ("bb_type",))
    if events_column is None:
        return DEFAULT_XFIP

    events = terminal[events_column].astype(str).str.lower()
    bb_types = (
        terminal[bb_type_column].astype(str).str.lower()
        if bb_type_column is not None
        else pd.Series("", index=terminal.index, dtype=str)
    )
    strikeouts = int(events.isin({"strikeout", "strikeout_double_play"}).sum())
    walks = int(events.isin({"walk", "intent_walk"}).sum())
    hit_by_pitch = int(events.eq("hit_by_pitch").sum())
    fly_balls = int(bb_types.isin({"fly_ball", "popup"}).sum())
    innings_pitched = _innings_from_terminal_events(events)
    if innings_pitched <= 0:
        return DEFAULT_XFIP

    expected_home_runs = fly_balls * (league_hr_fb_rate if league_hr_fb_rate > 0 else LEAGUE_HR_FB_RATE)
    return float(
        (
            (13.0 * expected_home_runs) + (3.0 * (walks + hit_by_pitch)) - (2.0 * strikeouts)
        )
        / innings_pitched
        + FIP_CONSTANT
    )


def _innings_pitched(pitches: pd.DataFrame) -> float:
    terminal = _collapse_plate_appearances(pitches)
    if terminal.empty:
        return 0.0

    events_column = _first_column(terminal, ("events",))
    if events_column is None:
        return 0.0
    return _innings_from_terminal_events(terminal[events_column].astype(str).str.lower())


def _innings_from_terminal_events(events: pd.Series) -> float:
    outs_recorded = sum(_event_outs(event_name) for event_name in events.tolist())
    if outs_recorded <= 0:
        return 0.0
    return float(outs_recorded / 3)


def _collapse_plate_appearances(pitches: pd.DataFrame) -> pd.DataFrame:
    if pitches.empty:
        return pitches.copy()

    if "at_bat_number" in pitches.columns:
        sort_columns = [column for column in ("at_bat_number", "pitch_number") if column in pitches.columns]
        terminal = pitches.sort_values(sort_columns).groupby("at_bat_number", as_index=False).tail(1)
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
        "pop_out": 1,
        "sac_bunt": 1,
        "sac_bunt_double_play": 2,
        "sac_fly": 1,
        "strikeout": 1,
        "strikeout_double_play": 2,
        "triple_play": 3,
    }
    return outs_map.get(event_name, 0)


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
                for game_pk, feature_name, _feature_value, window_size, as_of_timestamp in rows
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


def _normalize_date_series(values: Any) -> pd.Series:
    if values is None:
        return pd.Series(dtype="datetime64[ns]")

    series = pd.to_datetime(values, errors="coerce")
    if getattr(series.dt, "tz", None) is not None:
        series = series.dt.tz_localize(None)
    return series.dt.normalize()


def _string_series(values: Any) -> pd.Series:
    if values is None:
        return pd.Series(dtype=str)
    return values.astype(str)
