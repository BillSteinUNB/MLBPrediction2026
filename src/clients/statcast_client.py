from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from pybaseball import (
    batting_stats,
    cache as pybaseball_cache,
    fielding_stats,
    pitching_stats,
    playerid_lookup as _pybaseball_playerid_lookup,
    statcast,
    statcast_catcher_framing,
    statcast_outs_above_average,
    team_game_logs,
)

from src.config import _load_settings_yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DATA_ROOT = REPO_ROOT / "data" / "raw"
DEFAULT_PYBASEBALL_CACHE_DIR = DEFAULT_RAW_DATA_ROOT / "pybaseball_cache"
UNQUALIFIED_LEADERBOARD_MINIMUM = 0
FIELDING_CACHE_VERSION = 2
CATCHER_FRAMING_CACHE_VERSION = 2
OAA_POSITIONS: tuple[int, ...] = (3, 4, 5, 6, 7, 8, 9)
TEAM_GAME_LOG_CODES = {
    "ARI": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BOS": "BOS",
    "CHC": "CHC",
    "CIN": "CIN",
    "CLE": "CLE",
    "COL": "COL",
    "CWS": "CHW",
    "DET": "DET",
    "HOU": "HOU",
    "KC": "KCR",
    "LAA": "LAA",
    "LAD": "LAD",
    "MIA": "MIA",
    "MIL": "MIL",
    "MIN": "MIN",
    "NYM": "NYM",
    "NYY": "NYY",
    "OAK": "ATH",
    "PHI": "PHI",
    "PIT": "PIT",
    "SD": "SDP",
    "SEA": "SEA",
    "SF": "SFG",
    "STL": "STL",
    "TB": "TBR",
    "TEX": "TEX",
    "TOR": "TOR",
    "WSH": "WSN",
}


def _build_team_label_to_code() -> dict[str, str]:
    teams = _load_settings_yaml()["teams"]
    team_name_to_code: dict[str, str] = {}
    for team_code, payload in teams.items():
        full_name = str(payload["full_name"]).strip()
        nickname = str(payload["nickname"]).strip()
        city = str(payload["city"]).strip()
        team_name_to_code[full_name.upper()] = team_code
        team_name_to_code[nickname.upper()] = team_code
        team_name_to_code[f"{city} {nickname}".upper()] = team_code

    team_name_to_code["OAKLAND ATHLETICS"] = "OAK"
    return team_name_to_code


TEAM_LABEL_TO_CODE = _build_team_label_to_code()


def enable_pybaseball_cache(
    cache_directory: str | Path = DEFAULT_PYBASEBALL_CACHE_DIR,
) -> Path:
    """Enable pybaseball caching with parquet files in a project-local directory."""

    cache_path = Path(cache_directory)
    cache_path.mkdir(parents=True, exist_ok=True)

    os.environ["PYBASEBALL_CACHE"] = str(cache_path)
    pybaseball_cache.config.cache_directory = str(cache_path)
    pybaseball_cache.config.cache_type = "parquet"
    pybaseball_cache.enable()
    pybaseball_cache.config.save()

    return cache_path


def fetch_statcast_range(
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    *,
    raw_data_root: str | Path = DEFAULT_RAW_DATA_ROOT,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch statcast data into daily parquet files and only pull missing dates."""

    raw_root = Path(raw_data_root)
    statcast_dir = raw_root / "statcast"
    enable_pybaseball_cache(raw_root / "pybaseball_cache")

    frames: list[pd.DataFrame] = []
    for query_date in _iterate_dates(start_date, end_date):
        parquet_path = statcast_dir / f"statcast_{query_date.isoformat()}.parquet"
        if parquet_path.exists() and not refresh:
            daily_frame = pd.read_parquet(parquet_path)
        else:
            fetched_frame = statcast(
                start_dt=query_date.isoformat(),
                end_dt=query_date.isoformat(),
                verbose=False,
                parallel=False,
            )
            daily_frame = _normalize_statcast_day(fetched_frame, query_date)
            _write_parquet(daily_frame, parquet_path)

        frames.append(daily_frame)

    return _combine_frames(frames)


def fetch_pitcher_stats(
    season: int,
    min_ip: int = 20,
    *,
    raw_data_root: str | Path = DEFAULT_RAW_DATA_ROOT,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch FanGraphs pitching data and persist it as parquet."""

    raw_root = Path(raw_data_root)
    parquet_path = raw_root / "fangraphs" / f"pitching_{season}_min_ip_{min_ip}.parquet"
    enable_pybaseball_cache(raw_root / "pybaseball_cache")

    if parquet_path.exists() and not refresh:
        return pd.read_parquet(parquet_path)

    dataframe = _stringify_columns(pitching_stats(season, qual=min_ip).copy())
    _write_parquet(dataframe, parquet_path)
    return dataframe


def fetch_batting_stats(
    season: int,
    min_pa: int = 50,
    *,
    raw_data_root: str | Path = DEFAULT_RAW_DATA_ROOT,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch FanGraphs batting data and persist it as parquet."""

    raw_root = Path(raw_data_root)
    parquet_path = raw_root / "fangraphs" / f"batting_{season}_min_pa_{min_pa}.parquet"
    enable_pybaseball_cache(raw_root / "pybaseball_cache")

    if parquet_path.exists() and not refresh:
        return pd.read_parquet(parquet_path)

    dataframe = _stringify_columns(batting_stats(season, qual=min_pa).copy())
    _write_parquet(dataframe, parquet_path)
    return dataframe


def fetch_fielding_stats(
    season: int,
    *,
    raw_data_root: str | Path = DEFAULT_RAW_DATA_ROOT,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch FanGraphs fielding data and merge Statcast OAA totals."""

    raw_root = Path(raw_data_root)
    parquet_path = raw_root / "fielding" / f"fielding_{season}.parquet"
    enable_pybaseball_cache(raw_root / "pybaseball_cache")
    expected_metadata = _fielding_cache_metadata()

    cached_frame = _load_versioned_parquet_cache(
        parquet_path,
        expected_metadata=expected_metadata,
        refresh=refresh,
    )
    if cached_frame is not None:
        return cached_frame

    fangraphs_frame = _stringify_columns(
        fielding_stats(season, qual=UNQUALIFIED_LEADERBOARD_MINIMUM).copy()
    )
    oaa_frame = _load_oaa_totals(season).rename(columns={"OAA": "statcast_oaa"})

    merged = fangraphs_frame.copy()
    if "Name" in merged.columns:
        merged["_player_name_key"] = merged["Name"].map(_normalize_name)
        existing_oaa = (
            merged["OAA"] if "OAA" in merged.columns else pd.Series(0.0, index=merged.index)
        )
        if "Team" in merged.columns and "team" in oaa_frame.columns:
            merged["_team_key"] = merged["Team"].astype(str).str.strip().str.upper()
            merged = merged.merge(
                oaa_frame,
                left_on=["_player_name_key", "_team_key"],
                right_on=["player_name_key", "team"],
                how="left",
            )
        else:
            merged = merged.merge(
                oaa_frame.drop(columns=["team"], errors="ignore"),
                left_on="_player_name_key",
                right_on="player_name_key",
                how="left",
            )
        merged["OAA"] = merged["statcast_oaa"].fillna(existing_oaa).fillna(0)
        merged = merged.drop(
            columns=["_player_name_key", "_team_key", "player_name_key", "team", "statcast_oaa"],
            errors="ignore",
        )
    else:
        merged["OAA"] = 0.0

    _write_parquet(merged, parquet_path)
    _write_cache_metadata(parquet_path, expected_metadata)
    return merged


def fetch_catcher_framing(
    season: int,
    *,
    raw_data_root: str | Path = DEFAULT_RAW_DATA_ROOT,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch Statcast catcher framing data and persist it as parquet."""

    raw_root = Path(raw_data_root)
    parquet_path = raw_root / "catcher_framing" / f"catcher_framing_{season}.parquet"
    enable_pybaseball_cache(raw_root / "pybaseball_cache")
    expected_metadata = _catcher_framing_cache_metadata()

    cached_frame = _load_versioned_parquet_cache(
        parquet_path,
        expected_metadata=expected_metadata,
        refresh=refresh,
    )
    if cached_frame is not None:
        return cached_frame

    dataframe = _stringify_columns(
        statcast_catcher_framing(
            season,
            min_called_p=UNQUALIFIED_LEADERBOARD_MINIMUM,
        ).copy()
    )
    _write_parquet(dataframe, parquet_path)
    _write_cache_metadata(parquet_path, expected_metadata)
    return dataframe


def fetch_team_game_logs(
    season: int,
    team: str,
    *,
    raw_data_root: str | Path = DEFAULT_RAW_DATA_ROOT,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch Baseball Reference batting game logs for a team and store them as parquet."""

    raw_root = Path(raw_data_root)
    resolved_team = TEAM_GAME_LOG_CODES.get(team.upper(), team.upper())
    parquet_path = raw_root / "team_game_logs" / f"{resolved_team}_{season}.parquet"
    enable_pybaseball_cache(raw_root / "pybaseball_cache")

    if parquet_path.exists() and not refresh:
        return pd.read_parquet(parquet_path)

    dataframe = team_game_logs(season, resolved_team, log_type="batting").copy()
    dataframe = _stringify_columns(dataframe)
    _write_parquet(dataframe, parquet_path)
    return dataframe


def _iterate_dates(
    start_date: str | date | datetime,
    end_date: str | date | datetime,
) -> Iterable[date]:
    start_day = _coerce_date(start_date)
    end_day = _coerce_date(end_date)
    if end_day < start_day:
        raise ValueError("end_date must be on or after start_date")

    total_days = (end_day - start_day).days + 1
    for offset in range(total_days):
        yield start_day + timedelta(days=offset)


def _coerce_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _normalize_statcast_day(dataframe: pd.DataFrame, query_date: date) -> pd.DataFrame:
    normalized = dataframe.copy()
    normalized = _stringify_columns(normalized)

    if normalized.empty:
        return pd.DataFrame({"game_date": pd.Series(dtype="datetime64[ns]")})

    if "game_date" in normalized.columns:
        game_dates = pd.to_datetime(normalized["game_date"], errors="coerce")
        normalized = normalized.loc[game_dates.dt.date == query_date].copy()
        normalized["game_date"] = pd.to_datetime(normalized["game_date"], errors="coerce")

    return normalized.reset_index(drop=True)


def _load_oaa_totals(season: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for position in OAA_POSITIONS:
        position_frame = statcast_outs_above_average(
            season,
            position,
            min_att=UNQUALIFIED_LEADERBOARD_MINIMUM,
        )
        extracted = _extract_oaa_frame(position_frame)
        if not extracted.empty:
            frames.append(extracted)

    if not frames:
        return pd.DataFrame(columns=["player_name_key", "team", "OAA"])

    combined = pd.concat(frames, ignore_index=True)
    return combined.groupby(["player_name_key", "team"], as_index=False, dropna=False)["OAA"].sum()


def _extract_oaa_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    normalized = _stringify_columns(dataframe.copy())
    if normalized.empty:
        return pd.DataFrame(columns=["player_name_key", "team", "OAA"])

    name_column = _first_matching_column(
        normalized.columns,
        ("name", "entity_name", "player_name", "last_name, first_name"),
    )
    team_column = _first_matching_column(
        normalized.columns,
        ("team", "team_name", "display_team_name"),
    )
    oaa_column = _first_matching_column(normalized.columns, ("outs_above_average", "oaa"))

    if name_column is None or oaa_column is None:
        return pd.DataFrame(columns=["player_name_key", "team", "OAA"])

    extracted = pd.DataFrame(
        {
            "player_name_key": normalized[name_column].map(_normalize_name),
            "OAA": pd.to_numeric(normalized[oaa_column], errors="coerce").fillna(0.0),
        }
    )

    if team_column is not None:
        extracted["team"] = normalized[team_column].map(_normalize_team_label)
    else:
        extracted["team"] = ""

    return extracted.loc[extracted["player_name_key"] != ""].reset_index(drop=True)


def _combine_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()

    non_empty_frames = [frame for frame in frames if not frame.empty]
    if not non_empty_frames:
        return frames[0].copy()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    sort_columns = [
        column
        for column in ("game_date", "game_pk", "at_bat_number", "pitch_number")
        if column in combined.columns
    ]
    if sort_columns:
        combined = combined.sort_values(sort_columns).reset_index(drop=True)
    return combined


def _fielding_cache_metadata() -> dict[str, object]:
    return {
        "cache_version": FIELDING_CACHE_VERSION,
        "fielding_qual": UNQUALIFIED_LEADERBOARD_MINIMUM,
        "oaa_min_att": UNQUALIFIED_LEADERBOARD_MINIMUM,
        "oaa_positions": list(OAA_POSITIONS),
    }


def _catcher_framing_cache_metadata() -> dict[str, object]:
    return {
        "cache_version": CATCHER_FRAMING_CACHE_VERSION,
        "min_called_p": UNQUALIFIED_LEADERBOARD_MINIMUM,
    }


def _load_versioned_parquet_cache(
    parquet_path: Path,
    *,
    expected_metadata: dict[str, object],
    refresh: bool,
) -> pd.DataFrame | None:
    if refresh or not parquet_path.exists() or not _cache_metadata_matches(parquet_path, expected_metadata):
        return None

    try:
        return pd.read_parquet(parquet_path)
    except (OSError, ValueError):
        return None


def _cache_metadata_matches(parquet_path: Path, expected_metadata: dict[str, object]) -> bool:
    metadata_path = _cache_metadata_path(parquet_path)
    if not metadata_path.exists():
        return False

    try:
        raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    return raw_metadata == expected_metadata


def _stringify_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    normalized = dataframe.copy()
    columns: list[str] = []
    seen: dict[str, int] = {}

    for column in normalized.columns:
        if isinstance(column, tuple):
            parts = [
                str(part).strip()
                for part in column
                if part is not None and str(part).strip() and not str(part).startswith("Unnamed")
            ]
            candidate = "_".join(parts)
        else:
            candidate = str(column).strip()

        candidate = candidate or "column"
        duplicate_count = seen.get(candidate, 0)
        seen[candidate] = duplicate_count + 1
        if duplicate_count:
            candidate = f"{candidate}_{duplicate_count}"

        columns.append(candidate)

    normalized.columns = columns
    return normalized


def _cache_metadata_path(parquet_path: Path) -> Path:
    return parquet_path.with_suffix(".metadata.json")


def _write_cache_metadata(parquet_path: Path, metadata: dict[str, object]) -> None:
    metadata_path = _cache_metadata_path(parquet_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_parquet(dataframe: pd.DataFrame, parquet_path: Path) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(parquet_path, index=False)


def _normalize_name(value: object) -> str:
    if value is None:
        return ""

    text = str(value).strip()
    if "," in text:
        last_name, first_name = [part.strip() for part in text.split(",", 1)]
        text = f"{first_name} {last_name}".strip()

    return " ".join(text.lower().split())


def _normalize_team_label(value: object) -> str:
    if value is None:
        return ""

    label = " ".join(str(value).strip().upper().split())
    return TEAM_LABEL_TO_CODE.get(label, label)


def _first_matching_column(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    normalized_map = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in normalized_map:
            return normalized_map[candidate.lower()]
    return None


def lookup_player_ids(
    last_name: str,
    first_name: str | None = None,
) -> pd.DataFrame:
    """Cross-reference player IDs across MLB, FanGraphs, and Baseball Reference.

    Wraps pybaseball's ``playerid_lookup`` and returns a DataFrame containing
    ``key_mlbam`` (MLB ID), ``key_fangraphs``, ``key_bbref``, ``key_retro``,
    along with name and career date-range columns.

    Args:
        last_name: Player's last name.
        first_name: Player's first name (optional; narrows results).

    Returns:
        DataFrame with cross-referenced player IDs.
    """
    if first_name:
        return _pybaseball_playerid_lookup(last_name, first_name)
    return _pybaseball_playerid_lookup(last_name)
