from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Final

import pandas as pd
import requests

from src.clients.statcast_client import DEFAULT_RAW_DATA_ROOT
from src.db import DEFAULT_DB_PATH, init_db, sqlite_connection


RETROSHEET_BASE_URL: Final = "https://retrosheet.org/downloads"
RETROSHEET_GAME_LOG_URL_TEMPLATE: Final = "https://www.retrosheet.org/gamelogs/gl{season}.zip"
RETROSHEET_GAME_LOG_TABLE: Final = "retrosheet_game_logs"
HTTP_TIMEOUT: Final = 60.0

_PUBLIC_DATASETS: Final[dict[str, tuple[str, str]]] = {
    "gameinfo": (f"{RETROSHEET_BASE_URL}/gameinfo.zip", "gameinfo.csv"),
    "teamstats": (f"{RETROSHEET_BASE_URL}/teamstats.zip", "teamstats.csv"),
    "allplayers": (f"{RETROSHEET_BASE_URL}/allplayers.zip", "allplayers.csv"),
}

_GAME_LOG_USECOLS: Final = [0, 1, 3, 6, 9, 10, 16, 77, 78]
_GAME_LOG_READ_COLUMNS: Final = [
    "date",
    "doubleheader_code",
    "visteam",
    "hometeam",
    "vruns",
    "hruns",
    "site",
    "umphome",
    "umphome_name",
]
_GAME_LOG_FETCH_COLUMNS: Final = [
    "season",
    "date",
    "matchup_sequence",
    "doubleheader_code",
    "visteam",
    "hometeam",
    "site",
    "vruns",
    "hruns",
    "umphome",
    "umphome_name",
]


class RetrosheetClientError(RuntimeError):
    """Base exception for Retrosheet download and parsing failures."""


def fetch_retrosheet_gameinfo(
    *,
    raw_data_root: str | Path = DEFAULT_RAW_DATA_ROOT,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch and cache Retrosheet gameinfo rows."""

    return _fetch_public_dataset("gameinfo", raw_data_root=raw_data_root, refresh=refresh)


def fetch_retrosheet_teamstats(
    *,
    raw_data_root: str | Path = DEFAULT_RAW_DATA_ROOT,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch and cache Retrosheet teamstats rows."""

    return _fetch_public_dataset("teamstats", raw_data_root=raw_data_root, refresh=refresh)


def fetch_retrosheet_allplayers(
    *,
    raw_data_root: str | Path = DEFAULT_RAW_DATA_ROOT,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch and cache Retrosheet allplayers rows."""

    return _fetch_public_dataset("allplayers", raw_data_root=raw_data_root, refresh=refresh)


def fetch_retrosheet_starting_lineups(
    *,
    season: int | None = None,
    raw_data_root: str | Path = DEFAULT_RAW_DATA_ROOT,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return normalized starting lineup rows derived from Retrosheet teamstats."""

    frame = fetch_retrosheet_teamstats(raw_data_root=raw_data_root, refresh=refresh)
    if frame.empty:
        return pd.DataFrame()

    lineup_columns = [
        "gid",
        "team",
        "date",
        "opp",
        "site",
        "vishome",
        "season",
        *[f"start_l{index}" for index in range(1, 10)],
        *[f"start_f{index}" for index in range(1, 11)],
    ]
    available_columns = [column for column in lineup_columns if column in frame.columns]

    lineups = frame.copy()
    if "stattype" in lineups.columns:
        lineups = lineups.loc[lineups["stattype"].astype(str).str.lower() == "value"].copy()
    if season is not None and "season" in lineups.columns:
        lineups = lineups.loc[pd.to_numeric(lineups["season"], errors="coerce") == int(season)].copy()

    lineups = lineups[available_columns].reset_index(drop=True)
    return lineups


def fetch_retrosheet_game_logs(
    *,
    season: int | None,
    db_path: str | Path | None = None,
    raw_data_root: str | Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch and cache normalized Retrosheet yearly game logs in SQLite."""

    resolved_db_path = _resolve_retrosheet_db_path(db_path=db_path, raw_data_root=raw_data_root)
    init_db(resolved_db_path)

    if season is None:
        return _load_cached_game_logs(resolved_db_path, season=None)
    if int(season) == 2020:
        return _empty_game_logs_frame()

    if refresh:
        _replace_cached_game_logs(resolved_db_path, int(season), _download_game_logs(int(season)))

    cached = _load_cached_game_logs(resolved_db_path, season=int(season))
    if not cached.empty:
        return cached

    _replace_cached_game_logs(resolved_db_path, int(season), _download_game_logs(int(season)))
    return _load_cached_game_logs(resolved_db_path, season=int(season))


def fetch_retrosheet_umpires(
    *,
    season: int | None = None,
    db_path: str | Path | None = None,
    raw_data_root: str | Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return normalized umpire assignments derived from Retrosheet game logs."""

    frame = fetch_retrosheet_game_logs(
        season=season,
        db_path=db_path,
        raw_data_root=raw_data_root,
        refresh=refresh,
    )
    if frame.empty:
        return pd.DataFrame()

    umpire_columns = [
        "date",
        "season",
        "matchup_sequence",
        "visteam",
        "hometeam",
        "site",
        "umphome",
        "umphome_name",
    ]
    available_columns = [column for column in umpire_columns if column in frame.columns]
    return frame[available_columns].copy().reset_index(drop=True)


def _resolve_retrosheet_db_path(
    *,
    db_path: str | Path | None,
    raw_data_root: str | Path | None,
) -> Path:
    if db_path is not None:
        return Path(db_path)
    if raw_data_root is not None:
        return Path(raw_data_root) / "mlb.db"
    return Path(DEFAULT_DB_PATH)


def _load_cached_game_logs(db_path: Path, *, season: int | None) -> pd.DataFrame:
    query = f"""
        SELECT
            season,
            game_date AS date,
            matchup_sequence,
            doubleheader_code,
            away_team AS visteam,
            home_team AS hometeam,
            site,
            away_score AS vruns,
            home_score AS hruns,
            plate_umpire_id AS umphome,
            plate_umpire_name AS umphome_name
        FROM {RETROSHEET_GAME_LOG_TABLE}
    """
    params: tuple[object, ...] = ()
    if season is not None:
        query += " WHERE season = ?"
        params = (int(season),)
    query += " ORDER BY season, date, matchup_sequence, row_order"

    with sqlite_connection(db_path, builder_optimized=True) as connection:
        frame = pd.read_sql_query(query, connection, params=params)

    if frame.empty:
        return _empty_game_logs_frame()
    return _stringify_object_columns(frame)


def _replace_cached_game_logs(db_path: Path, season: int, frame: pd.DataFrame) -> None:
    with sqlite_connection(db_path, builder_optimized=True) as connection:
        connection.execute(
            f"DELETE FROM {RETROSHEET_GAME_LOG_TABLE} WHERE season = ?",
            (int(season),),
        )
        if not frame.empty:
            rows = [
                (
                    int(row["season"]),
                    str(row["game_date"]),
                    str(row["away_team"]),
                    str(row["home_team"]),
                    int(row["matchup_sequence"]),
                    str(row["doubleheader_code"]),
                    _db_scalar(row.get("site")),
                    _db_scalar(row.get("away_score")),
                    _db_scalar(row.get("home_score")),
                    _db_scalar(row.get("plate_umpire_id")),
                    _db_scalar(row.get("plate_umpire_name")),
                    int(row["row_order"]),
                )
                for row in frame.to_dict(orient="records")
            ]
            connection.executemany(
                f"""
                INSERT INTO {RETROSHEET_GAME_LOG_TABLE} (
                    season,
                    game_date,
                    away_team,
                    home_team,
                    matchup_sequence,
                    doubleheader_code,
                    site,
                    away_score,
                    home_score,
                    plate_umpire_id,
                    plate_umpire_name,
                    row_order
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        connection.commit()


def _download_game_logs(season: int) -> pd.DataFrame:
    url = RETROSHEET_GAME_LOG_URL_TEMPLATE.format(season=season)
    try:
        response = requests.get(url, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RetrosheetClientError(
            f"Failed to download Retrosheet game log archive for {season}: {url}"
        ) from exc

    try:
        archive = zipfile.ZipFile(io.BytesIO(response.content))
    except zipfile.BadZipFile as exc:
        raise RetrosheetClientError(
            f"Retrosheet game log archive was not a valid zip file for {season}: {url}"
        ) from exc

    member_name = _resolve_game_log_member_name(archive, season)
    try:
        with archive.open(member_name) as zipped_file:
            text_stream = io.TextIOWrapper(zipped_file, encoding="utf-8", errors="replace")
            frame = pd.read_csv(
                text_stream,
                header=None,
                usecols=_GAME_LOG_USECOLS,
                names=_GAME_LOG_READ_COLUMNS,
                low_memory=False,
            )
    except KeyError as exc:
        raise RetrosheetClientError(
            f"Retrosheet archive {url} did not contain expected member {member_name}"
        ) from exc

    return _normalize_game_logs_for_cache(frame, season)


def _resolve_game_log_member_name(archive: zipfile.ZipFile, season: int) -> str:
    expected_name = f"gl{season}.txt"
    if expected_name in archive.namelist():
        return expected_name
    lowered = {name.lower(): name for name in archive.namelist()}
    if expected_name in lowered:
        return lowered[expected_name]
    raise RetrosheetClientError(
        f"Retrosheet game log archive for {season} did not contain {expected_name}"
    )


def _normalize_game_logs_for_cache(frame: pd.DataFrame, season: int) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "game_date",
                "away_team",
                "home_team",
                "matchup_sequence",
                "doubleheader_code",
                "site",
                "away_score",
                "home_score",
                "plate_umpire_id",
                "plate_umpire_name",
                "row_order",
            ]
        )

    normalized = frame.copy()
    normalized["game_date"] = pd.to_datetime(
        normalized["date"].astype("string").str.strip(),
        format="%Y%m%d",
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    normalized["away_team"] = normalized["visteam"].astype("string").str.strip().str.upper()
    normalized["home_team"] = normalized["hometeam"].astype("string").str.strip().str.upper()
    normalized["doubleheader_code"] = (
        normalized["doubleheader_code"].astype("string").fillna("0").str.strip().replace("", "0")
    )
    normalized["site"] = normalized["site"].astype("string").str.strip()
    normalized["away_score"] = pd.to_numeric(normalized["vruns"], errors="coerce").astype("Int64")
    normalized["home_score"] = pd.to_numeric(normalized["hruns"], errors="coerce").astype("Int64")
    normalized["plate_umpire_id"] = normalized["umphome"].astype("string").str.strip()
    normalized["plate_umpire_name"] = normalized["umphome_name"].astype("string").str.strip()
    normalized["season"] = int(season)
    normalized["row_order"] = range(1, len(normalized) + 1)
    normalized = normalized.dropna(subset=["game_date", "away_team", "home_team"]).copy()

    sort_keys = pd.to_numeric(normalized["doubleheader_code"], errors="coerce").fillna(0)
    normalized = normalized.assign(_doubleheader_sort=sort_keys)
    normalized = normalized.sort_values(
        ["game_date", "home_team", "away_team", "_doubleheader_sort", "row_order"]
    ).reset_index(drop=True)
    normalized["matchup_sequence"] = (
        normalized.groupby(["game_date", "home_team", "away_team"]).cumcount() + 1
    )

    return normalized[
        [
            "season",
            "game_date",
            "away_team",
            "home_team",
            "matchup_sequence",
            "doubleheader_code",
            "site",
            "away_score",
            "home_score",
            "plate_umpire_id",
            "plate_umpire_name",
            "row_order",
        ]
    ].reset_index(drop=True)


def _empty_game_logs_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_GAME_LOG_FETCH_COLUMNS)


def _db_scalar(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def _fetch_public_dataset(
    dataset_name: str,
    *,
    raw_data_root: str | Path,
    refresh: bool,
) -> pd.DataFrame:
    if dataset_name not in _PUBLIC_DATASETS:
        raise ValueError(f"Unsupported Retrosheet dataset: {dataset_name}")

    url, member_name = _PUBLIC_DATASETS[dataset_name]
    parquet_path = Path(raw_data_root) / "retrosheet" / f"{dataset_name}.parquet"
    if parquet_path.exists() and not refresh:
        return pd.read_parquet(parquet_path)

    dataframe = _download_zip_csv(url, member_name)
    dataframe = _stringify_object_columns(dataframe)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(parquet_path, index=False)
    return dataframe


def _download_zip_csv(url: str, member_name: str) -> pd.DataFrame:
    try:
        response = requests.get(url, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RetrosheetClientError(f"Failed to download Retrosheet dataset: {url}") from exc

    try:
        archive = zipfile.ZipFile(io.BytesIO(response.content))
    except zipfile.BadZipFile as exc:
        raise RetrosheetClientError(f"Retrosheet dataset was not a valid zip archive: {url}") from exc

    try:
        with archive.open(member_name) as zipped_file:
            return pd.read_csv(zipped_file, low_memory=False)
    except KeyError as exc:
        raise RetrosheetClientError(
            f"Retrosheet archive {url} did not contain expected member {member_name}"
        ) from exc


def _stringify_object_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe

    normalized = dataframe.copy()
    for column in normalized.columns:
        if pd.api.types.is_object_dtype(normalized[column]):
            normalized[column] = normalized[column].astype("string")
    return normalized
