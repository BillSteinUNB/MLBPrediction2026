from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Final

import pandas as pd
import requests

from src.clients.statcast_client import DEFAULT_RAW_DATA_ROOT


RETROSHEET_BASE_URL: Final = "https://retrosheet.org/downloads"
HTTP_TIMEOUT: Final = 60.0

_PUBLIC_DATASETS: Final[dict[str, tuple[str, str]]] = {
    "gameinfo": (f"{RETROSHEET_BASE_URL}/gameinfo.zip", "gameinfo.csv"),
    "teamstats": (f"{RETROSHEET_BASE_URL}/teamstats.zip", "teamstats.csv"),
    "allplayers": (f"{RETROSHEET_BASE_URL}/allplayers.zip", "allplayers.csv"),
}


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


def fetch_retrosheet_umpires(
    *,
    season: int | None = None,
    raw_data_root: str | Path = DEFAULT_RAW_DATA_ROOT,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return normalized umpire assignments derived from Retrosheet gameinfo."""

    frame = fetch_retrosheet_gameinfo(raw_data_root=raw_data_root, refresh=refresh)
    if frame.empty:
        return pd.DataFrame()

    umpire_columns = [
        "gid",
        "date",
        "season",
        "visteam",
        "hometeam",
        "site",
        "umphome",
        "ump1b",
        "ump2b",
        "ump3b",
        "umplf",
        "umprf",
    ]
    available_columns = [column for column in umpire_columns if column in frame.columns]
    umpires = frame[available_columns].copy()
    if season is not None and "season" in umpires.columns:
        umpires = umpires.loc[pd.to_numeric(umpires["season"], errors="coerce") == int(season)].copy()
    return umpires.reset_index(drop=True)


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
