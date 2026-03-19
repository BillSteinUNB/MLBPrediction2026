from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import httpx
import pandas as pd

from src.clients.statcast_client import (
    TEAM_GAME_LOG_CODES,
    TEAM_LABEL_TO_CODE,
    fetch_batting_stats,
    fetch_catcher_framing,
    fetch_fielding_stats,
    fetch_pitcher_stats,
)
from src.config import _load_settings_yaml
from src.features.adjustments.abs_adjustment import (
    DEFAULT_ABS_RETENTION_FACTOR,
    DEFAULT_STRIKEOUT_RATE_DELTA,
    DEFAULT_WALK_RATE_DELTA,
    is_abs_active,
)
from src.features.adjustments.park_factors import get_park_factors
from src.features.baselines import calculate_log5_probability, calculate_pythagorean_win_percentage


DEFAULT_OUTPUT_PATH = Path("data") / "training" / "training_data_2019_2025.parquet"
DEFAULT_WINDOWS: tuple[int, ...] = (7, 14, 30, 60)
DEFAULT_PYTHAGOREAN_WINDOWS: tuple[int, ...] = (30, 60)
DEFAULT_FULL_REGULAR_SEASONS_TARGET = 7
SHORTENED_SEASON_GAME_THRESHOLD = 2_000
DEFAULT_FULL_RUNS_BASELINE = 4.5
DEFAULT_F5_RUNS_BASELINE = 2.25
DEFAULT_DEFENSIVE_EFFICIENCY = 0.700
DEFAULT_BULLPEN_IR_PCT = 0.30
NEUTRAL_WEATHER_FACTOR = 1.0
SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"

_SETTINGS = _load_settings_yaml()
_TEAM_CODES = set(_SETTINGS["teams"].keys())
_TEAM_CODE_ALIASES = {value: key for key, value in TEAM_GAME_LOG_CODES.items()}
_TEAM_CODE_ALIASES.update(
    {
        "AZ": "ARI",
        "ATH": "OAK",
        "CHW": "CWS",
        "KCR": "KC",
        "SDP": "SD",
        "SFG": "SF",
        "TBR": "TB",
        "WSN": "WSH",
    }
)


ScheduleFetcher = Callable[[int], pd.DataFrame]
SeasonStatsFetcher = Callable[..., pd.DataFrame]


@dataclass(frozen=True, slots=True)
class TeamSnapshot:
    offense_wrc_plus: float = 100.0
    offense_woba: float = 0.320
    offense_iso: float = 0.170
    offense_babip: float = 0.300
    offense_k_pct: float = 22.0
    offense_bb_pct: float = 8.0
    starter_xfip: float = 4.20
    starter_xera: float = 4.10
    starter_k_pct: float = 22.0
    starter_bb_pct: float = 8.0
    starter_gb_pct: float = 43.0
    starter_hr_fb_pct: float = 11.0
    starter_avg_fastball_velocity: float = 93.5
    defense_drs: float = 0.0
    defense_oaa: float = 0.0
    defense_defensive_efficiency: float = DEFAULT_DEFENSIVE_EFFICIENCY
    defense_adjusted_framing: float = 0.0
    bullpen_ir_pct: float = DEFAULT_BULLPEN_IR_PCT
    bullpen_xfip: float = 4.20


@dataclass(frozen=True, slots=True)
class TrainingDataBuildResult:
    dataframe: pd.DataFrame
    output_path: Path
    metadata_path: Path
    data_version_hash: str
    build_timestamp: datetime
    requested_years: tuple[int, ...]
    effective_years: tuple[int, ...]


def resolve_training_years(
    *,
    start_year: int,
    end_year: int,
    full_regular_seasons_target: int = DEFAULT_FULL_REGULAR_SEASONS_TARGET,
    season_row_counts: Mapping[int, int] | None = None,
    shortened_season_game_threshold: int = SHORTENED_SEASON_GAME_THRESHOLD,
) -> list[int]:
    """Resolve effective training years, replacing shortened seasons with earlier full years."""

    if end_year < start_year:
        raise ValueError("end_year must be greater than or equal to start_year")

    requested_years = list(range(start_year, end_year + 1))
    row_counts = dict(season_row_counts or {})

    effective_years = [
        year
        for year in requested_years
        if row_counts.get(year, shortened_season_game_threshold) >= shortened_season_game_threshold
    ]

    backfill_year = start_year - 1
    while len(effective_years) < full_regular_seasons_target:
        if backfill_year not in effective_years and row_counts.get(
            backfill_year,
            shortened_season_game_threshold,
        ) >= shortened_season_game_threshold:
            effective_years.insert(0, backfill_year)
        backfill_year -= 1

    return sorted(effective_years)


def build_training_dataset(
    *,
    start_year: int = 2019,
    end_year: int = 2025,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    full_regular_seasons_target: int = DEFAULT_FULL_REGULAR_SEASONS_TARGET,
    shortened_season_game_threshold: int = SHORTENED_SEASON_GAME_THRESHOLD,
    refresh: bool = False,
    schedule_fetcher: ScheduleFetcher | None = None,
    batting_stats_fetcher: SeasonStatsFetcher = fetch_batting_stats,
    pitching_stats_fetcher: SeasonStatsFetcher = fetch_pitcher_stats,
    fielding_stats_fetcher: SeasonStatsFetcher = fetch_fielding_stats,
    framing_stats_fetcher: SeasonStatsFetcher = fetch_catcher_framing,
) -> TrainingDataBuildResult:
    """Build the anti-leakage-safe historical training dataset and persist it to parquet."""

    resolved_schedule_fetcher = schedule_fetcher or _fetch_regular_season_schedule
    requested_years = tuple(range(start_year, end_year + 1))

    schedules_by_year: dict[int, pd.DataFrame] = {}
    for year in requested_years:
        schedules_by_year[year] = _prepare_schedule_frame(resolved_schedule_fetcher(year))

    effective_years = tuple(
        resolve_training_years(
            start_year=start_year,
            end_year=end_year,
            full_regular_seasons_target=full_regular_seasons_target,
            season_row_counts={year: len(frame) for year, frame in schedules_by_year.items()},
            shortened_season_game_threshold=shortened_season_game_threshold,
        )
    )

    for year in effective_years:
        if year not in schedules_by_year:
            schedules_by_year[year] = _prepare_schedule_frame(resolved_schedule_fetcher(year))

    schedule = pd.concat(
        [schedules_by_year[year] for year in effective_years],
        ignore_index=True,
    )
    schedule = schedule.sort_values(["scheduled_start", "game_pk"]).reset_index(drop=True)

    snapshot_years = sorted({season - 1 for season in schedule["season"].unique().tolist()})
    snapshots_by_year = {
        season: _build_team_snapshots_for_season(
            season,
            refresh=refresh,
            batting_stats_fetcher=batting_stats_fetcher,
            pitching_stats_fetcher=pitching_stats_fetcher,
            fielding_stats_fetcher=fielding_stats_fetcher,
            framing_stats_fetcher=framing_stats_fetcher,
        )
        for season in snapshot_years
    }
    default_snapshot = _league_average_snapshot(snapshots_by_year)

    build_timestamp = datetime.now(UTC)
    dataset = _assemble_training_rows(
        schedule,
        snapshots_by_year=snapshots_by_year,
        default_snapshot=default_snapshot,
        build_timestamp=build_timestamp,
    )
    assert_training_data_is_leakage_free(dataset)
    _assert_targets_present(dataset)

    data_version_hash = _compute_data_version_hash(dataset)
    dataset = dataset.copy()
    dataset["data_version_hash"] = data_version_hash
    dataset["build_timestamp"] = build_timestamp.isoformat()

    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(resolved_output_path, index=False)

    metadata_path = resolved_output_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "requested_years": list(requested_years),
                "effective_years": list(effective_years),
                "shortened_seasons_skipped": [
                    year for year in requested_years if year not in effective_years
                ],
                "row_count": int(len(dataset)),
                "feature_column_count": int(len(_feature_columns(dataset))),
                "data_version_hash": data_version_hash,
                "build_timestamp": build_timestamp.isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return TrainingDataBuildResult(
        dataframe=dataset,
        output_path=resolved_output_path,
        metadata_path=metadata_path,
        data_version_hash=data_version_hash,
        build_timestamp=build_timestamp,
        requested_years=requested_years,
        effective_years=effective_years,
    )


def assert_training_data_is_leakage_free(dataframe: pd.DataFrame) -> None:
    """Assert that every training row has an as-of timestamp strictly before scheduled start."""

    if dataframe.empty:
        return

    as_of_timestamp = pd.to_datetime(dataframe["as_of_timestamp"], utc=True, errors="coerce")
    scheduled_start = pd.to_datetime(dataframe["scheduled_start"], utc=True, errors="coerce")
    violations = dataframe.loc[as_of_timestamp >= scheduled_start, ["game_pk", "scheduled_start"]]
    if not violations.empty:
        raise AssertionError(
            f"Anti-leakage assertion failed for {len(violations)} rows: "
            f"{violations['game_pk'].tolist()[:5]}"
        )


def _fetch_regular_season_schedule(year: int) -> pd.DataFrame:
    with httpx.Client(timeout=60.0) as client:
        response = client.get(
            SCHEDULE_URL,
            params={
                "sportId": 1,
                "startDate": f"{year}-01-01",
                "endDate": f"{year}-12-31",
                "gameType": "R",
                "hydrate": "linescore,probablePitcher,venue,team",
            },
        )
        response.raise_for_status()

    payload = response.json()
    rows: list[dict[str, Any]] = []
    for date_entry in payload.get("dates", []):
        for game in date_entry.get("games", []):
            row = _schedule_game_to_row(game)
            if row is not None:
                rows.append(row)

    return pd.DataFrame(rows)


def _schedule_game_to_row(game: Mapping[str, Any]) -> dict[str, Any] | None:
    game_type = str(game.get("gameType") or "").upper()
    if game_type != "R":
        return None

    detailed_state = str(game.get("status", {}).get("detailedState") or "")
    if detailed_state.lower() not in {"final", "game over", "completed early"}:
        return None

    innings = game.get("linescore", {}).get("innings") or []
    if len(innings) < 5:
        return None

    home_team = _normalize_team_code(
        game.get("teams", {}).get("home", {}).get("team", {}).get("abbreviation")
        or game.get("teams", {}).get("home", {}).get("team", {}).get("name")
    )
    away_team = _normalize_team_code(
        game.get("teams", {}).get("away", {}).get("team", {}).get("abbreviation")
        or game.get("teams", {}).get("away", {}).get("team", {}).get("name")
    )
    if home_team is None or away_team is None:
        return None

    scheduled_start = pd.Timestamp(game.get("gameDate"), tz="UTC")
    venue = str(game.get("venue", {}).get("name") or game.get("venue", {}).get("locationName") or home_team)
    park_factors = get_park_factors(team_code=home_team, venue=venue)

    f5_home_score = sum(_inning_runs(inning, "home") for inning in innings[:5])
    f5_away_score = sum(_inning_runs(inning, "away") for inning in innings[:5])
    final_home_score = _coerce_int(game.get("teams", {}).get("home", {}).get("score"))
    final_away_score = _coerce_int(game.get("teams", {}).get("away", {}).get("score"))
    if final_home_score is None or final_away_score is None:
        return None

    return {
        "game_pk": int(game["gamePk"]),
        "season": int(scheduled_start.year),
        "game_date": scheduled_start.date().isoformat(),
        "scheduled_start": scheduled_start.isoformat(),
        "home_team": home_team,
        "away_team": away_team,
        "venue": venue,
        "is_dome": bool(_SETTINGS["stadiums"].get(home_team, {}).get("is_dome", False)),
        "is_abs_active": bool(is_abs_active(venue)),
        "park_runs_factor": float(park_factors.runs),
        "park_hr_factor": float(park_factors.hr),
        "game_type": game_type,
        "status": detailed_state,
        "f5_home_score": int(f5_home_score),
        "f5_away_score": int(f5_away_score),
        "final_home_score": int(final_home_score),
        "final_away_score": int(final_away_score),
    }


def _prepare_schedule_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return pd.DataFrame(
            columns=[
                "game_pk",
                "season",
                "game_date",
                "scheduled_start",
                "home_team",
                "away_team",
                "venue",
                "is_dome",
                "is_abs_active",
                "park_runs_factor",
                "park_hr_factor",
                "game_type",
                "status",
                "f5_home_score",
                "f5_away_score",
                "final_home_score",
                "final_away_score",
            ]
        )

    schedule = dataframe.copy()
    schedule = schedule.loc[schedule.get("game_type", "R").astype(str).str.upper() == "R"].copy()
    schedule["scheduled_start"] = pd.to_datetime(schedule["scheduled_start"], utc=True, errors="coerce")
    schedule["game_date"] = pd.to_datetime(schedule.get("game_date", schedule["scheduled_start"]), errors="coerce")
    schedule["season"] = pd.to_numeric(schedule.get("season", schedule["scheduled_start"].dt.year), errors="coerce").astype(int)
    schedule["home_team"] = schedule["home_team"].map(_normalize_team_code)
    schedule["away_team"] = schedule["away_team"].map(_normalize_team_code)
    if "venue" not in schedule.columns:
        schedule["venue"] = schedule["home_team"].map(
            lambda team: _SETTINGS["stadiums"].get(team, {}).get("park_name", team)
        )
    if "is_dome" not in schedule.columns:
        schedule["is_dome"] = schedule["home_team"].map(
            lambda team: bool(_SETTINGS["stadiums"].get(team, {}).get("is_dome", False))
        )
    if "is_abs_active" not in schedule.columns:
        schedule["is_abs_active"] = schedule["venue"].map(lambda venue: bool(is_abs_active(venue)))
    if "park_runs_factor" not in schedule.columns:
        schedule["park_runs_factor"] = schedule.apply(
            lambda row: get_park_factors(
                team_code=row["home_team"],
                venue=row.get("venue"),
            ).runs,
            axis=1,
        )
    if "park_hr_factor" not in schedule.columns:
        schedule["park_hr_factor"] = schedule.apply(
            lambda row: get_park_factors(
                team_code=row["home_team"],
                venue=row.get("venue"),
            ).hr,
            axis=1,
        )
    for column in (
        "f5_home_score",
        "f5_away_score",
        "final_home_score",
        "final_away_score",
        "park_runs_factor",
        "park_hr_factor",
    ):
        schedule[column] = pd.to_numeric(schedule[column], errors="coerce")

    schedule["is_dome"] = schedule["is_dome"].astype(bool)
    schedule["is_abs_active"] = schedule["is_abs_active"].astype(bool)
    schedule = schedule.dropna(
        subset=[
            "game_pk",
            "scheduled_start",
            "home_team",
            "away_team",
            "f5_home_score",
            "f5_away_score",
            "final_home_score",
            "final_away_score",
        ]
    ).copy()
    schedule["game_pk"] = schedule["game_pk"].astype(int)
    schedule["game_date"] = schedule["game_date"].dt.date.astype(str)
    return schedule.sort_values(["scheduled_start", "game_pk"]).reset_index(drop=True)


def _build_team_snapshots_for_season(
    season: int,
    *,
    refresh: bool,
    batting_stats_fetcher: SeasonStatsFetcher,
    pitching_stats_fetcher: SeasonStatsFetcher,
    fielding_stats_fetcher: SeasonStatsFetcher,
    framing_stats_fetcher: SeasonStatsFetcher,
) -> dict[str, TeamSnapshot]:
    offense = _aggregate_offense_snapshot(batting_stats_fetcher(season, min_pa=0, refresh=refresh))
    pitching = _aggregate_pitching_snapshot(pitching_stats_fetcher(season, min_ip=0, refresh=refresh))
    fielding_frame = fielding_stats_fetcher(season, refresh=refresh)
    defense = _aggregate_fielding_snapshot(fielding_frame)
    framing = _aggregate_framing_snapshot(
        framing_stats_fetcher(season, refresh=refresh),
        name_team_lookup=_build_name_team_lookup(fielding_frame),
    )

    teams = {team for team in offense} | {team for team in pitching} | {team for team in defense} | {team for team in framing}
    snapshots: dict[str, TeamSnapshot] = {}
    for team in teams:
        offense_values = offense.get(team, {})
        pitching_values = pitching.get(team, {})
        defense_values = defense.get(team, {})
        framing_values = framing.get(team, {})
        snapshots[team] = TeamSnapshot(
            offense_wrc_plus=float(offense_values.get("wrc_plus", 100.0)),
            offense_woba=float(offense_values.get("woba", 0.320)),
            offense_iso=float(offense_values.get("iso", 0.170)),
            offense_babip=float(offense_values.get("babip", 0.300)),
            offense_k_pct=float(offense_values.get("k_pct", 22.0)),
            offense_bb_pct=float(offense_values.get("bb_pct", 8.0)),
            starter_xfip=float(pitching_values.get("xfip", 4.20)),
            starter_xera=float(pitching_values.get("xera", 4.10)),
            starter_k_pct=float(pitching_values.get("k_pct", 22.0)),
            starter_bb_pct=float(pitching_values.get("bb_pct", 8.0)),
            starter_gb_pct=float(pitching_values.get("gb_pct", 43.0)),
            starter_hr_fb_pct=float(pitching_values.get("hr_fb_pct", 11.0)),
            starter_avg_fastball_velocity=float(
                pitching_values.get("avg_fastball_velocity", 93.5)
            ),
            defense_drs=float(defense_values.get("drs", 0.0)),
            defense_oaa=float(defense_values.get("oaa", 0.0)),
            defense_defensive_efficiency=float(
                defense_values.get("defensive_efficiency", DEFAULT_DEFENSIVE_EFFICIENCY)
            ),
            defense_adjusted_framing=float(framing_values.get("adjusted_framing", 0.0)),
            bullpen_ir_pct=float(pitching_values.get("bullpen_ir_pct", DEFAULT_BULLPEN_IR_PCT)),
            bullpen_xfip=float(pitching_values.get("bullpen_xfip", pitching_values.get("xfip", 4.20))),
        )

    return snapshots


def _aggregate_offense_snapshot(dataframe: pd.DataFrame) -> dict[str, dict[str, float]]:
    if dataframe.empty:
        return {}

    frame = dataframe.copy()
    frame["team"] = frame.get("Team").map(_normalize_team_code)
    frame["pa"] = pd.to_numeric(frame.get("PA"), errors="coerce").fillna(0.0)
    if frame["team"].isna().all():
        return {}

    frame["wrc_plus"] = pd.to_numeric(frame.get("wRC+"), errors="coerce")
    frame["woba"] = pd.to_numeric(frame.get("wOBA"), errors="coerce")
    frame["iso"] = pd.to_numeric(frame.get("ISO"), errors="coerce")
    frame["babip"] = pd.to_numeric(frame.get("BABIP"), errors="coerce")
    frame["k_pct"] = _normalize_percentage_series(frame.get("K%"))
    frame["bb_pct"] = _normalize_percentage_series(frame.get("BB%"))

    return {
        str(team): {
            "wrc_plus": _weighted_average(group["wrc_plus"], group["pa"], default=100.0),
            "woba": _weighted_average(group["woba"], group["pa"], default=0.320),
            "iso": _weighted_average(group["iso"], group["pa"], default=0.170),
            "babip": _weighted_average(group["babip"], group["pa"], default=0.300),
            "k_pct": _weighted_average(group["k_pct"], group["pa"], default=22.0),
            "bb_pct": _weighted_average(group["bb_pct"], group["pa"], default=8.0),
        }
        for team, group in frame.dropna(subset=["team"]).groupby("team", dropna=True)
    }


def _aggregate_pitching_snapshot(dataframe: pd.DataFrame) -> dict[str, dict[str, float]]:
    if dataframe.empty:
        return {}

    frame = dataframe.copy()
    frame["team"] = frame.get("Team").map(_normalize_team_code)
    frame["weight"] = pd.to_numeric(frame.get("IP"), errors="coerce").fillna(0.0)
    frame["tbf"] = pd.to_numeric(frame.get("TBF"), errors="coerce").fillna(frame["weight"])
    if frame["team"].isna().all():
        return {}

    frame["xfip"] = pd.to_numeric(frame.get("xFIP"), errors="coerce")
    frame["xera"] = pd.to_numeric(frame.get("xERA"), errors="coerce")
    frame["k_pct"] = _normalize_percentage_series(frame.get("K%"))
    frame["bb_pct"] = _normalize_percentage_series(frame.get("BB%"))
    frame["gb_pct"] = _normalize_percentage_series(frame.get("GB%"))
    frame["hr_fb_pct"] = _normalize_percentage_series(frame.get("HR/FB"))
    frame["avg_fastball_velocity"] = pd.to_numeric(frame.get("FBv"), errors="coerce")

    return {
        str(team): {
            "xfip": _weighted_average(group["xfip"], group["weight"], default=4.20),
            "xera": _weighted_average(group["xera"], group["weight"], default=4.10),
            "k_pct": _weighted_average(group["k_pct"], group["tbf"], default=22.0),
            "bb_pct": _weighted_average(group["bb_pct"], group["tbf"], default=8.0),
            "gb_pct": _weighted_average(group["gb_pct"], group["tbf"], default=43.0),
            "hr_fb_pct": _weighted_average(group["hr_fb_pct"], group["tbf"], default=11.0),
            "avg_fastball_velocity": _weighted_average(
                group["avg_fastball_velocity"],
                group["weight"],
                default=93.5,
            ),
            "bullpen_xfip": _weighted_average(group["xfip"], group["weight"], default=4.20),
            "bullpen_ir_pct": DEFAULT_BULLPEN_IR_PCT,
        }
        for team, group in frame.dropna(subset=["team"]).groupby("team", dropna=True)
    }


def _aggregate_fielding_snapshot(dataframe: pd.DataFrame) -> dict[str, dict[str, float]]:
    if dataframe.empty:
        return {}

    frame = dataframe.copy()
    frame["team"] = frame.get("Team").map(_normalize_team_code)
    if frame["team"].isna().all():
        return {}

    frame["drs"] = pd.to_numeric(frame.get("DRS"), errors="coerce").fillna(0.0)
    frame["oaa"] = pd.to_numeric(frame.get("OAA"), errors="coerce").fillna(0.0)
    return {
        str(team): {
            "drs": float(group["drs"].sum() / 162.0),
            "oaa": float(group["oaa"].sum() / 162.0),
            "defensive_efficiency": DEFAULT_DEFENSIVE_EFFICIENCY,
        }
        for team, group in frame.dropna(subset=["team"]).groupby("team", dropna=True)
    }


def _aggregate_framing_snapshot(
    dataframe: pd.DataFrame,
    *,
    name_team_lookup: Mapping[str, str] | None = None,
) -> dict[str, dict[str, float]]:
    if dataframe.empty:
        return {}

    frame = dataframe.copy()
    team_series = frame.get("team") if "team" in frame.columns else frame.get("Team")
    if team_series is not None:
        frame["team"] = team_series.map(_normalize_team_code)
    else:
        name_column = None
        for candidate in ("name", "Name", "player_name", "catcher", "player"):
            if candidate in frame.columns:
                name_column = candidate
                break
        if name_column is None or not name_team_lookup:
            return {}
        frame["team"] = frame[name_column].map(_normalize_name).map(name_team_lookup)
    if frame["team"].isna().all():
        return {}

    framing_column = None
    for candidate in (
        "runs_extra_strikes",
        "framing_runs",
        "framing",
        "strike_runs",
        "extra_strike_runs",
    ):
        if candidate in frame.columns:
            framing_column = candidate
            break
    if framing_column is None:
        return {}

    frame["raw_framing"] = pd.to_numeric(frame[framing_column], errors="coerce").fillna(0.0)
    return {
        str(team): {
            "adjusted_framing": float(
                (group["raw_framing"].sum() / 162.0) * DEFAULT_ABS_RETENTION_FACTOR
            )
        }
        for team, group in frame.dropna(subset=["team"]).groupby("team", dropna=True)
    }


def _league_average_snapshot(
    snapshots_by_year: Mapping[int, Mapping[str, TeamSnapshot]],
) -> TeamSnapshot:
    snapshots = [snapshot for season in snapshots_by_year.values() for snapshot in season.values()]
    if not snapshots:
        return TeamSnapshot()

    payload = pd.DataFrame([asdict(snapshot) for snapshot in snapshots])
    return TeamSnapshot(**{column: float(payload[column].mean()) for column in payload.columns})


def _assemble_training_rows(
    schedule: pd.DataFrame,
    *,
    snapshots_by_year: Mapping[int, Mapping[str, TeamSnapshot]],
    default_snapshot: TeamSnapshot,
    build_timestamp: datetime,
) -> pd.DataFrame:
    histories: dict[str, list[dict[str, float]]] = {}
    rows: list[dict[str, Any]] = []

    for game in schedule.to_dict(orient="records"):
        season = int(game["season"])
        game_start = pd.Timestamp(game["scheduled_start"])
        if game_start.tzinfo is None:
            game_start = game_start.tz_localize("UTC")
        else:
            game_start = game_start.tz_convert("UTC")
        as_of_timestamp = (game_start.normalize() - pd.Timedelta(days=1)).isoformat()
        year_snapshots = snapshots_by_year.get(season - 1, {})

        row: dict[str, Any] = {
            "game_pk": int(game["game_pk"]),
            "season": season,
            "game_date": str(game["game_date"]),
            "scheduled_start": game_start.isoformat(),
            "as_of_timestamp": as_of_timestamp,
            "home_team": str(game["home_team"]),
            "away_team": str(game["away_team"]),
            "venue": str(game["venue"]),
            "game_type": str(game["game_type"]),
            "park_runs_factor": float(game["park_runs_factor"]),
            "park_hr_factor": float(game["park_hr_factor"]),
            "abs_active": float(bool(game["is_abs_active"])),
            "abs_walk_rate_delta": float(DEFAULT_WALK_RATE_DELTA if game["is_abs_active"] else 0.0),
            "abs_strikeout_rate_delta": float(
                DEFAULT_STRIKEOUT_RATE_DELTA if game["is_abs_active"] else 0.0
            ),
            "weather_temp_factor": NEUTRAL_WEATHER_FACTOR,
            "weather_air_density_factor": NEUTRAL_WEATHER_FACTOR,
            "weather_humidity_factor": NEUTRAL_WEATHER_FACTOR,
            "weather_wind_factor": 0.0,
            "weather_rain_risk": NEUTRAL_WEATHER_FACTOR,
            "weather_composite": NEUTRAL_WEATHER_FACTOR,
            "weather_data_missing": 1.0,
            "build_timestamp": build_timestamp.isoformat(),
        }

        for side_name, team_key, opponent_key in (
            ("home", "home_team", "away_team"),
            ("away", "away_team", "home_team"),
        ):
            team = str(game[team_key])
            opponent = str(game[opponent_key])
            team_history = histories.get(team, [])
            opponent_history = histories.get(opponent, [])
            snapshot = year_snapshots.get(team, default_snapshot)

            row[f"{side_name}_team_prior_games"] = float(len(team_history))
            row.update(_rolling_history_features(side_name, team_history))
            row.update(_snapshot_features(side_name, snapshot))
            row.update(_baseline_features(side_name, team_history, opponent_history))

        f5_home_score = int(game["f5_home_score"])
        f5_away_score = int(game["f5_away_score"])
        final_home_score = int(game["final_home_score"])
        final_away_score = int(game["final_away_score"])
        row.update(
            {
                "f5_home_score": f5_home_score,
                "f5_away_score": f5_away_score,
                "final_home_score": final_home_score,
                "final_away_score": final_away_score,
                "f5_margin": float(f5_home_score - f5_away_score),
                "final_margin": float(final_home_score - final_away_score),
                "f5_tied_after_5": int(f5_home_score == f5_away_score),
                "f5_ml_result": int(f5_home_score > f5_away_score),
                "f5_rl_result": int((f5_home_score - f5_away_score) >= 2),
            }
        )
        rows.append(row)

        histories.setdefault(str(game["home_team"]), []).append(
            {
                "full_runs_scored": float(final_home_score),
                "full_runs_allowed": float(final_away_score),
                "f5_runs_scored": float(f5_home_score),
                "f5_runs_allowed": float(f5_away_score),
                "win": float(final_home_score > final_away_score),
                "f5_win": float(f5_home_score > f5_away_score),
            }
        )
        histories.setdefault(str(game["away_team"]), []).append(
            {
                "full_runs_scored": float(final_away_score),
                "full_runs_allowed": float(final_home_score),
                "f5_runs_scored": float(f5_away_score),
                "f5_runs_allowed": float(f5_home_score),
                "win": float(final_away_score > final_home_score),
                "f5_win": float(f5_away_score > f5_home_score),
            }
        )

    dataset = pd.DataFrame(rows).sort_values(["scheduled_start", "game_pk"]).reset_index(drop=True)
    numeric_columns = dataset.select_dtypes(include=["number"]).columns
    dataset[numeric_columns] = dataset[numeric_columns].fillna(0.0)
    return dataset


def _rolling_history_features(side_name: str, history: Sequence[Mapping[str, float]]) -> dict[str, float]:
    features: dict[str, float] = {}
    for window in DEFAULT_WINDOWS:
        sample = history[-window:]
        features[f"{side_name}_offense_runs_scored_{window}g"] = _history_average(
            sample,
            "full_runs_scored",
            DEFAULT_FULL_RUNS_BASELINE,
        )
        features[f"{side_name}_pitching_runs_allowed_{window}g"] = _history_average(
            sample,
            "full_runs_allowed",
            DEFAULT_FULL_RUNS_BASELINE,
        )
        features[f"{side_name}_offense_f5_runs_scored_{window}g"] = _history_average(
            sample,
            "f5_runs_scored",
            DEFAULT_F5_RUNS_BASELINE,
        )
        features[f"{side_name}_pitching_f5_runs_allowed_{window}g"] = _history_average(
            sample,
            "f5_runs_allowed",
            DEFAULT_F5_RUNS_BASELINE,
        )
        features[f"{side_name}_team_win_pct_{window}g"] = _history_average(sample, "win", 0.5)
        features[f"{side_name}_team_f5_win_pct_{window}g"] = _history_average(
            sample,
            "f5_win",
            0.5,
        )
    return features


def _snapshot_features(side_name: str, snapshot: TeamSnapshot) -> dict[str, float]:
    return {
        f"{side_name}_team_wrc_plus_prior": snapshot.offense_wrc_plus,
        f"{side_name}_team_woba_prior": snapshot.offense_woba,
        f"{side_name}_team_iso_prior": snapshot.offense_iso,
        f"{side_name}_team_babip_prior": snapshot.offense_babip,
        f"{side_name}_team_k_pct_prior": snapshot.offense_k_pct,
        f"{side_name}_team_bb_pct_prior": snapshot.offense_bb_pct,
        f"{side_name}_starter_xfip_prior": snapshot.starter_xfip,
        f"{side_name}_starter_xera_prior": snapshot.starter_xera,
        f"{side_name}_starter_k_pct_prior": snapshot.starter_k_pct,
        f"{side_name}_starter_bb_pct_prior": snapshot.starter_bb_pct,
        f"{side_name}_starter_gb_pct_prior": snapshot.starter_gb_pct,
        f"{side_name}_starter_hr_fb_pct_prior": snapshot.starter_hr_fb_pct,
        f"{side_name}_starter_avg_fastball_velocity_prior": snapshot.starter_avg_fastball_velocity,
        f"{side_name}_starter_is_opener": 0.0,
        f"{side_name}_starter_uses_team_composite": 1.0,
        f"{side_name}_team_drs_prior": snapshot.defense_drs,
        f"{side_name}_team_oaa_prior": snapshot.defense_oaa,
        f"{side_name}_team_defensive_efficiency_prior": snapshot.defense_defensive_efficiency,
        f"{side_name}_team_adjusted_framing_prior": snapshot.defense_adjusted_framing,
        f"{side_name}_bullpen_ir_pct_prior": snapshot.bullpen_ir_pct,
        f"{side_name}_bullpen_xfip_prior": snapshot.bullpen_xfip,
        f"{side_name}_bullpen_pitch_count_3d": 0.0,
        f"{side_name}_bullpen_pitch_count_5d": 0.0,
        f"{side_name}_bullpen_avg_rest_days_top5": 2.0,
        f"{side_name}_bullpen_high_leverage_available_count": 5.0,
    }


def _baseline_features(
    side_name: str,
    team_history: Sequence[Mapping[str, float]],
    opponent_history: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    features: dict[str, float] = {}
    for window in DEFAULT_PYTHAGOREAN_WINDOWS:
        team_sample = team_history[-window:]
        opponent_sample = opponent_history[-window:]
        team_pythagorean = _pythagorean_from_history(team_sample, prefix="full")
        opponent_pythagorean = _pythagorean_from_history(opponent_sample, prefix="full")
        team_f5_pythagorean = _pythagorean_from_history(team_sample, prefix="f5")
        features[f"{side_name}_team_pythagorean_wp_{window}g"] = team_pythagorean
        features[f"{side_name}_team_f5_pythagorean_wp_{window}g"] = team_f5_pythagorean
        features[f"{side_name}_team_log5_{window}g"] = calculate_log5_probability(
            team_pythagorean,
            opponent_pythagorean,
        )
    return features


def _pythagorean_from_history(
    history: Sequence[Mapping[str, float]],
    *,
    prefix: str,
) -> float:
    if not history:
        return 0.5

    runs_scored = sum(float(entry[f"{prefix}_runs_scored"]) for entry in history)
    runs_allowed = sum(float(entry[f"{prefix}_runs_allowed"]) for entry in history)
    return calculate_pythagorean_win_percentage(runs_scored, runs_allowed)


def _history_average(
    history: Sequence[Mapping[str, float]],
    key: str,
    default: float,
) -> float:
    if not history:
        return float(default)
    return float(sum(float(entry[key]) for entry in history) / len(history))


def _assert_targets_present(dataframe: pd.DataFrame) -> None:
    if dataframe[["f5_ml_result", "f5_rl_result"]].isna().any().any():
        raise AssertionError("Target columns must not contain NaN values")


def _compute_data_version_hash(dataframe: pd.DataFrame) -> str:
    stable_columns = [
        column
        for column in dataframe.columns
        if column not in {"build_timestamp", "data_version_hash"}
    ]
    stable_payload = dataframe[stable_columns].sort_values(["scheduled_start", "game_pk"])
    encoded = stable_payload.to_json(orient="records", date_format="iso").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _feature_columns(dataframe: pd.DataFrame) -> list[str]:
    non_feature_columns = {
        "game_pk",
        "season",
        "game_date",
        "scheduled_start",
        "as_of_timestamp",
        "build_timestamp",
        "data_version_hash",
        "home_team",
        "away_team",
        "venue",
        "game_type",
        "status",
        "f5_home_score",
        "f5_away_score",
        "final_home_score",
        "final_away_score",
        "f5_margin",
        "final_margin",
        "f5_tied_after_5",
        "f5_ml_result",
        "f5_rl_result",
    }
    return [column for column in dataframe.columns if column not in non_feature_columns]


def _build_name_team_lookup(dataframe: pd.DataFrame) -> dict[str, str]:
    if dataframe.empty or "Name" not in dataframe.columns or "Team" not in dataframe.columns:
        return {}

    lookup_frame = pd.DataFrame(
        {
            "name": dataframe["Name"].map(_normalize_name),
            "team": dataframe["Team"].map(_normalize_team_code),
        }
    ).dropna(subset=["name", "team"])
    if lookup_frame.empty:
        return {}
    lookup_frame = lookup_frame.drop_duplicates(subset=["name"], keep="last")
    return dict(zip(lookup_frame["name"], lookup_frame["team"], strict=False))


def _normalize_team_code(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip().upper()
    if not text or text in {"TOT", "- - -"}:
        return None
    if text in _TEAM_CODES:
        return text
    if text in _TEAM_CODE_ALIASES:
        return _TEAM_CODE_ALIASES[text]
    return TEAM_LABEL_TO_CODE.get(text)


def _normalize_name(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if "," in text:
        last_name, first_name = [part.strip() for part in text.split(",", 1)]
        text = f"{first_name} {last_name}".strip()
    return " ".join(text.lower().split())


def _normalize_percentage_series(values: Any) -> pd.Series:
    series = pd.to_numeric(values, errors="coerce")
    valid = series.dropna()
    if valid.empty:
        return series
    if float(valid.abs().max()) <= 1.5:
        return series * 100.0
    return series


def _weighted_average(values: pd.Series, weights: pd.Series, *, default: float) -> float:
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


def _inning_runs(inning: Mapping[str, Any], side: str) -> int:
    return int(inning.get(side, {}).get("runs") or 0)


def _coerce_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
