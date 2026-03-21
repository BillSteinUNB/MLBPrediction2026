from __future__ import annotations

import io
from pathlib import Path
from typing import Final, Sequence

import pandas as pd
import requests

from src.clients.statcast_client import DEFAULT_RAW_DATA_ROOT


CHADWICK_REGISTER_BASE_URL: Final = (
    "https://raw.githubusercontent.com/chadwickbureau/register/master/data"
)
CHADWICK_REGISTER_SHARDS: Final[tuple[str, ...]] = tuple("0123456789abcdef")
HTTP_TIMEOUT: Final = 60.0


class ChadwickClientError(RuntimeError):
    """Base exception for Chadwick register download and parsing failures."""


def fetch_chadwick_register(
    *,
    shards: Sequence[str] | None = None,
    raw_data_root: str | Path = DEFAULT_RAW_DATA_ROOT,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch and cache one or more Chadwick register people shards."""

    resolved_shards = _normalize_shards(shards)
    frames: list[pd.DataFrame] = []
    for shard in resolved_shards:
        parquet_path = Path(raw_data_root) / "chadwick" / "register" / f"people-{shard}.parquet"
        if parquet_path.exists() and not refresh:
            frame = pd.read_parquet(parquet_path)
        else:
            url = f"{CHADWICK_REGISTER_BASE_URL}/people-{shard}.csv"
            frame = _download_csv(url)
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(parquet_path, index=False)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_chadwick_names(
    *,
    raw_data_root: str | Path = DEFAULT_RAW_DATA_ROOT,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch and cache the Chadwick alternate names table."""

    parquet_path = Path(raw_data_root) / "chadwick" / "register" / "names.parquet"
    if parquet_path.exists() and not refresh:
        return pd.read_parquet(parquet_path)

    frame = _download_csv(f"{CHADWICK_REGISTER_BASE_URL}/names.csv")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path, index=False)
    return frame


def lookup_chadwick_register(
    *,
    mlbam_id: int | str | None = None,
    retrosheet_id: str | None = None,
    fangraphs_id: int | str | None = None,
    bbref_id: str | None = None,
    last_name: str | None = None,
    first_name: str | None = None,
    raw_data_root: str | Path = DEFAULT_RAW_DATA_ROOT,
) -> pd.DataFrame:
    """Lookup Chadwick register rows by one or more common identifiers."""

    filters: list[tuple[str, str]] = []
    if mlbam_id is not None:
        filters.append(("key_mlbam", str(mlbam_id)))
    if retrosheet_id is not None:
        filters.append(("key_retro", str(retrosheet_id)))
    if fangraphs_id is not None:
        filters.append(("key_fangraphs", str(fangraphs_id)))
    if bbref_id is not None:
        filters.append(("key_bbref", str(bbref_id)))
    if last_name is not None:
        filters.append(("name_last", str(last_name)))
    if first_name is not None:
        filters.append(("name_first", str(first_name)))

    if not filters:
        raise ValueError("At least one Chadwick lookup filter is required")

    cached_shards = _discover_cached_register_shards(raw_data_root)
    selected_shards = cached_shards if cached_shards else None
    register = _apply_lookup_filters(
        fetch_chadwick_register(
            shards=selected_shards,
            raw_data_root=raw_data_root,
        ),
        filters,
    )
    if (
        register.empty
        and cached_shards
        and set(cached_shards) != set(CHADWICK_REGISTER_SHARDS)
    ):
        register = _apply_lookup_filters(
            fetch_chadwick_register(
                raw_data_root=raw_data_root,
            ),
            filters,
        )

    return register.reset_index(drop=True)


def _apply_lookup_filters(
    register: pd.DataFrame,
    filters: Sequence[tuple[str, str]],
) -> pd.DataFrame:
    filtered = register.copy()
    for column, value in filters:
        if column not in filtered.columns:
            return pd.DataFrame(columns=filtered.columns)
        series = filtered[column].fillna("").astype(str)
        if column in {"name_last", "name_first"}:
            filtered = filtered.loc[series.str.casefold() == value.casefold()].copy()
        else:
            filtered = filtered.loc[series == value].copy()

    return filtered


def _normalize_shards(shards: Sequence[str] | None) -> tuple[str, ...]:
    if shards is None:
        return CHADWICK_REGISTER_SHARDS

    normalized: list[str] = []
    for shard in shards:
        candidate = str(shard).strip().lower()
        if candidate not in CHADWICK_REGISTER_SHARDS:
            raise ValueError(f"Unsupported Chadwick shard: {shard}")
        normalized.append(candidate)

    return tuple(dict.fromkeys(normalized))


def _download_csv(url: str) -> pd.DataFrame:
    try:
        response = requests.get(url, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ChadwickClientError(f"Failed to download Chadwick register data: {url}") from exc

    return pd.read_csv(io.StringIO(response.text), dtype=str)


def _discover_cached_register_shards(raw_data_root: str | Path) -> list[str]:
    register_dir = Path(raw_data_root) / "chadwick" / "register"
    if not register_dir.exists():
        return []

    discovered: list[str] = []
    for parquet_path in sorted(register_dir.glob("people-*.parquet")):
        shard = parquet_path.stem.removeprefix("people-")
        if shard in CHADWICK_REGISTER_SHARDS:
            discovered.append(shard)
    return discovered
