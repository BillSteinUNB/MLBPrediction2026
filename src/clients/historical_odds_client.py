from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from src.clients.historical_f5_acquirer import DEFAULT_MAX_ABS_AMERICAN_ODDS
from src.clients.odds_client import _ensure_game_row, _persist_snapshots
from src.db import DEFAULT_DB_PATH, init_db
from src.models.odds import OddsSnapshot


_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "game_pk": ("game_pk",),
    "commence_time": ("commence_time", "scheduled_start", "date", "game_date", "event_date"),
    "home_team": ("home_team", "home_team_name", "home", "team_home"),
    "away_team": ("away_team", "away_team_name", "away", "team_away"),
    "market_type": ("market_type", "market", "bet_type"),
    "home_odds": (
        "home_odds",
        "home_ml",
        "home_moneyline",
        "moneyline_home",
        "odds_home",
        "closing_home_odds",
    ),
    "away_odds": (
        "away_odds",
        "away_ml",
        "away_moneyline",
        "moneyline_away",
        "odds_away",
        "closing_away_odds",
    ),
    "book_name": ("book_name", "sportsbook", "book"),
    "fetched_at": ("fetched_at", "snapshot_time", "captured_at"),
    "home_point": ("home_point", "home_spread", "point_home", "spread_home"),
    "away_point": ("away_point", "away_spread", "point_away", "spread_away"),
}

_MARKET_ALIASES = {
    "f5_ml": "f5_ml",
    "f5 moneyline": "f5_ml",
    "first five moneyline": "f5_ml",
    "moneyline_1st5": "f5_ml",
    "f5_rl": "f5_rl",
    "f5 run line": "f5_rl",
    "first five run line": "f5_rl",
    "spread_1st5": "f5_rl",
    "f5_total": "f5_total",
    "f5 total": "f5_total",
    "first five total": "f5_total",
    "totals_1st5": "f5_total",
    "full_game_ml": "full_game_ml",
    "full game moneyline": "full_game_ml",
    "moneyline_full": "full_game_ml",
    "full_game_rl": "full_game_rl",
    "full game run line": "full_game_rl",
    "spread_full": "full_game_rl",
    "full_game_total": "full_game_total",
    "full game total": "full_game_total",
    "totals_full": "full_game_total",
}
_TEAM_CODE_ALIASES = {
    "AZ": "ARI",
    "ARI": "ARI",
    "ATH": "OAK",
    "ATL": "ATL",
    "BAL": "BAL",
    "BOS": "BOS",
    "CWS": "CWS",
    "CHW": "CWS",
    "CHA": "CWS",
    "CHC": "CHC",
    "CHN": "CHC",
    "CIN": "CIN",
    "CLE": "CLE",
    "COL": "COL",
    "DET": "DET",
    "HOU": "HOU",
    "KC": "KC",
    "KCR": "KC",
    "KCA": "KC",
    "LAA": "LAA",
    "ANA": "LAA",
    "LAD": "LAD",
    "LAN": "LAD",
    "MIA": "MIA",
    "MIL": "MIL",
    "MIN": "MIN",
    "NYM": "NYM",
    "NYN": "NYM",
    "NYY": "NYY",
    "NYA": "NYY",
    "OAK": "OAK",
    "PHI": "PHI",
    "PIT": "PIT",
    "SD": "SD",
    "SDP": "SD",
    "SDN": "SD",
    "SEA": "SEA",
    "SF": "SF",
    "SFG": "SF",
    "SFN": "SF",
    "STL": "STL",
    "SLN": "STL",
    "TB": "TB",
    "TBR": "TB",
    "TBA": "TB",
    "TEX": "TEX",
    "TOR": "TOR",
    "WSH": "WSH",
    "WAS": "WSH",
    "WSN": "WSH",
}
_TIMEZONE_ABBREVIATION_OFFSETS = {
    "EDT": "-04:00",
    "EST": "-05:00",
    "CDT": "-05:00",
    "CST": "-06:00",
    "MDT": "-06:00",
    "MST": "-07:00",
    "PDT": "-07:00",
    "PST": "-08:00",
}
MarketTypeLiteral = Literal[
    "f5_ml",
    "f5_rl",
    "f5_total",
    "full_game_ml",
    "full_game_rl",
    "full_game_total",
]
SnapshotSelectionLiteral = Literal["latest", "opening"]


def import_historical_odds(
    *,
    source_path: str | Path,
    db_path: str | Path = DEFAULT_DB_PATH,
    default_market_type: str = "f5_ml",
    default_book_name: str = "historical",
    max_abs_odds: int = DEFAULT_MAX_ABS_AMERICAN_ODDS,
) -> int:
    """Import normalized historical odds rows into odds_snapshots."""

    frame = _load_source_dataframe(source_path)
    if frame.empty:
        return 0

    commence_column = _resolve_column(frame, "commence_time")
    game_pk_column = _resolve_column(frame, "game_pk")
    home_team_column = _resolve_column(frame, "home_team")
    away_team_column = _resolve_column(frame, "away_team")
    home_odds_column = _resolve_column(frame, "home_odds")
    away_odds_column = _resolve_column(frame, "away_odds")
    has_required_game_lookup_columns = None not in (
        commence_column,
        home_team_column,
        away_team_column,
    )
    if not has_required_game_lookup_columns and game_pk_column is None:
        raise ValueError(
            "Historical odds file must include game_pk or all of commence_time, home_team, and away_team"
        )
    if None in (home_odds_column, away_odds_column):
        raise ValueError("Historical odds file is missing one or more required odds columns")

    market_type_column = _resolve_column(frame, "market_type")
    book_name_column = _resolve_column(frame, "book_name")
    fetched_at_column = _resolve_column(frame, "fetched_at")
    home_point_column = _resolve_column(frame, "home_point")
    away_point_column = _resolve_column(frame, "away_point")

    database_path = init_db(db_path)
    snapshots: list[OddsSnapshot] = []
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        for row in frame.to_dict(orient="records"):
            commence_time = _coerce_timestamp(row.get(commence_column)) if commence_column else None
            game_pk_value = row.get(game_pk_column) if game_pk_column else None
            parsed_game_pk = (
                _coerce_nullable_int(game_pk_value) if game_pk_column is not None else None
            )
            if parsed_game_pk is not None:
                game_pk = parsed_game_pk
            else:
                assert commence_time is not None
                home_team_name = str(row[home_team_column]).strip()
                away_team_name = str(row[away_team_column]).strip()
                game_pk = _ensure_game_row(
                    connection,
                    event_id=f"historical-{home_team_name}-{away_team_name}-{commence_time.isoformat()}",
                    home_team_name=home_team_name,
                    away_team_name=away_team_name,
                    commence_time=commence_time,
                )
                if game_pk is None:
                    continue

            market_type = _normalize_market_type(
                row.get(market_type_column),
                default_market_type=default_market_type,
            )
            home_odds = int(row[home_odds_column])
            away_odds = int(row[away_odds_column])
            if abs(home_odds) > int(max_abs_odds) or abs(away_odds) > int(max_abs_odds):
                continue
            fetched_at_value = row.get(fetched_at_column) if fetched_at_column else None
            fetched_at = (
                _coerce_timestamp(fetched_at_value)
                if fetched_at_column
                and fetched_at_value is not None
                and not pd.isna(fetched_at_value)
                else commence_time
            )
            if fetched_at is None:
                raise ValueError("Historical odds file must include commence_time or fetched_at")
            book_value = row.get(book_name_column) if book_name_column else None
            book_name = (
                str(book_value).strip()
                if book_name_column and book_value is not None and not pd.isna(book_value)
                else default_book_name
            )
            snapshots.append(
                OddsSnapshot(
                    game_pk=int(game_pk),
                    book_name=book_name or default_book_name,
                    market_type=market_type,
                    home_odds=home_odds,
                    away_odds=away_odds,
                    home_point=_coerce_nullable_float(row.get(home_point_column))
                    if home_point_column is not None
                    else None,
                    away_point=_coerce_nullable_float(row.get(away_point_column))
                    if away_point_column is not None
                    else None,
                    fetched_at=fetched_at,
                    is_frozen=True,
                )
            )

        _persist_snapshots(connection, snapshots)
        connection.commit()
    return len(snapshots)


def load_historical_odds_for_games(
    *,
    db_path: str | Path,
    game_pks: Sequence[int] | None = None,
    games_frame: pd.DataFrame | None = None,
    market_type: str = "f5_ml",
    book_name: str | None = None,
    max_abs_odds: int = DEFAULT_MAX_ABS_AMERICAN_ODDS,
    snapshot_selection: SnapshotSelectionLiteral = "latest",
) -> pd.DataFrame:
    """Load historical odds snapshots for the requested games."""

    if snapshot_selection not in {"latest", "opening"}:
        raise ValueError("snapshot_selection must be one of {'latest', 'opening'}")

    resolved_game_pks = [int(game_pk) for game_pk in (game_pks or [])]
    if games_frame is not None and not games_frame.empty and "game_pk" in games_frame.columns:
        resolved_game_pks = [int(game_pk) for game_pk in games_frame["game_pk"].dropna().astype(int).tolist()]

    resolved_db_path = Path(db_path)
    connection_path = resolved_db_path if resolved_db_path.exists() else init_db(resolved_db_path)
    with sqlite3.connect(connection_path) as connection:
        schema_name = _detect_historical_odds_schema(connection)
        if schema_name == "canonical":
            return _load_canonical_historical_odds_for_games(
                connection=connection,
                game_pks=resolved_game_pks,
                games_frame=games_frame,
                market_type=market_type,
                book_name=book_name,
                max_abs_odds=max_abs_odds,
                snapshot_selection=snapshot_selection,
            )
        return _load_old_scraper_historical_odds_for_games(
            connection=connection,
            games_frame=games_frame,
            market_type=market_type,
            book_name=book_name,
            max_abs_odds=max_abs_odds,
            snapshot_selection=snapshot_selection,
        )


def _load_canonical_historical_odds_for_games(
    *,
    connection: sqlite3.Connection,
    game_pks: Sequence[int],
    games_frame: pd.DataFrame | None,
    market_type: str,
    book_name: str | None,
    max_abs_odds: int,
    snapshot_selection: SnapshotSelectionLiteral,
) -> pd.DataFrame:
    if not game_pks:
        return _empty_loaded_odds_frame()

    placeholders = ",".join("?" for _ in game_pks)
    query = f"""
        SELECT
            o.game_pk,
            o.book_name,
            o.market_type,
            o.home_odds,
            o.away_odds,
            o.home_point,
            o.away_point,
            o.fetched_at,
            g.date AS scheduled_start
        FROM odds_snapshots AS o
        LEFT JOIN games AS g
            ON g.game_pk = o.game_pk
        WHERE o.market_type = ?
          AND o.game_pk IN ({placeholders})
          AND ABS(o.home_odds) <= ?
          AND ABS(o.away_odds) <= ?
          {"" if book_name is None else "AND o.book_name = ?"}
    """
    params: list[Any] = [
        _coerce_market_type_literal(market_type),
        *[int(game_pk) for game_pk in game_pks],
        int(max_abs_odds),
        int(max_abs_odds),
    ]
    if book_name:
        params.append(book_name)

    frame = pd.read_sql_query(query, connection, params=params)
    if frame.empty:
        return _empty_loaded_odds_frame()
    frame["fetched_at"] = pd.to_datetime(frame["fetched_at"], utc=True, errors="coerce", format="mixed")
    frame["scheduled_start"] = pd.to_datetime(
        frame["scheduled_start"],
        utc=True,
        errors="coerce",
        format="mixed",
    )
    canonical_cutoffs = _resolve_canonical_request_cutoffs(
        connection,
        game_pks=game_pks,
        games_frame=games_frame,
    )
    if not canonical_cutoffs.empty:
        frame = frame.merge(canonical_cutoffs, on="game_pk", how="left")
    else:
        frame["pregame_cutoff"] = pd.NaT
    frame["pregame_cutoff"] = pd.to_datetime(
        frame["pregame_cutoff"],
        utc=True,
        errors="coerce",
        format="mixed",
    ).where(
        pd.to_datetime(frame["pregame_cutoff"], utc=True, errors="coerce", format="mixed").notna(),
        frame["scheduled_start"],
    )
    frame = frame.loc[
        frame["fetched_at"].notna()
        & frame["pregame_cutoff"].notna()
        & (frame["fetched_at"] <= frame["pregame_cutoff"])
    ].copy()
    if frame.empty:
        return _empty_loaded_odds_frame()
    ascending = snapshot_selection == "opening"
    frame = frame.sort_values(["game_pk", "fetched_at"], ascending=[True, ascending])
    frame = frame.drop_duplicates(subset=["game_pk"], keep="first").copy()
    frame = frame.drop(columns=["scheduled_start", "pregame_cutoff"], errors="ignore")
    frame["total_point"] = pd.Series(np.nan, index=frame.index, dtype=float)
    frame["over_odds"] = pd.Series(np.nan, index=frame.index, dtype=float)
    frame["under_odds"] = pd.Series(np.nan, index=frame.index, dtype=float)
    frame["source_schema"] = "canonical"
    return frame


def _load_old_scraper_historical_odds_for_games(
    *,
    connection: sqlite3.Connection,
    games_frame: pd.DataFrame | None,
    market_type: str,
    book_name: str | None,
    max_abs_odds: int,
    snapshot_selection: SnapshotSelectionLiteral,
) -> pd.DataFrame:
    if games_frame is None or games_frame.empty:
        return _empty_loaded_odds_frame()

    requested = _prepare_requested_game_frame(games_frame)
    if requested.empty:
        return _empty_loaded_odds_frame()

    min_date = (requested["request_game_date"].min() - timedelta(days=2)).isoformat()
    max_date = (requested["request_game_date"].max() + timedelta(days=2)).isoformat()
    query = """
        SELECT
            event_id,
            game_date,
            COALESCE(NULLIF(commence_time_utc, ''), NULLIF(commence_time, '')) AS commence_time_utc,
            away_team,
            home_team,
            bookmaker AS book_name,
            market_type,
            side,
            point,
            price,
            COALESCE(NULLIF(fetched_at, ''), COALESCE(NULLIF(commence_time_utc, ''), NULLIF(commence_time, ''))) AS fetched_at,
            COALESCE(is_opening, 0) AS is_opening
        FROM odds
        WHERE market_type = ?
          AND game_date BETWEEN ? AND ?
    """
    params: list[Any] = [_coerce_market_type_literal(market_type), min_date, max_date]
    if book_name is not None:
        query += " AND bookmaker = ?"
        params.append(book_name)

    raw = pd.read_sql_query(query, connection, params=params)
    if raw.empty:
        return _empty_loaded_odds_frame()

    paired = _pair_old_scraper_market_rows(
        raw,
        market_type=_coerce_market_type_literal(market_type),
        max_abs_odds=max_abs_odds,
    )
    if paired.empty:
        return _empty_loaded_odds_frame()

    matched = _match_requested_games_to_old_scraper_snapshots(
        requested=requested,
        snapshots=paired,
        snapshot_selection=snapshot_selection,
        aggregate_consensus=book_name is None,
    )
    if matched.empty:
        return _empty_loaded_odds_frame()
    return matched


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import historical MLB odds into sqlite")
    parser.add_argument("--source", required=True)
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--market-type", default="f5_ml")
    parser.add_argument("--book-name", default="historical")
    args = parser.parse_args(argv)

    imported_row_count = import_historical_odds(
        source_path=args.source,
        db_path=args.db_path,
        default_market_type=args.market_type,
        default_book_name=args.book_name,
    )
    print(
        json.dumps(
            {
                "db_path": str(Path(args.db_path)),
                "source": str(Path(args.source)),
                "imported_row_count": imported_row_count,
                "market_type": args.market_type,
                "book_name": args.book_name,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _load_source_dataframe(source_path: str | Path) -> pd.DataFrame:
    path = Path(source_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported historical odds file type: {path.suffix}")


def _resolve_column(frame: pd.DataFrame, logical_name: str) -> str | None:
    for candidate in _COLUMN_ALIASES[logical_name]:
        if candidate in frame.columns:
            return candidate
    return None


def _coerce_timestamp(value: Any) -> datetime:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError(f"Unable to parse timestamp value: {value}")
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize(timezone.utc)
    else:
        parsed = parsed.tz_convert(timezone.utc)
    resolved = parsed.to_pydatetime()
    if not isinstance(resolved, datetime):
        raise ValueError(f"Unable to coerce timestamp value: {value}")
    return resolved


def _normalize_market_type(value: Any, *, default_market_type: str) -> MarketTypeLiteral:
    default_value = _coerce_market_type_literal(default_market_type)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default_value
    normalized = str(value).strip().lower()
    return _coerce_market_type_literal(_MARKET_ALIASES.get(normalized, default_value))


def _coerce_market_type_literal(value: str) -> MarketTypeLiteral:
    if value == "f5_ml":
        return "f5_ml"
    if value == "f5_rl":
        return "f5_rl"
    if value == "f5_total":
        return "f5_total"
    if value == "full_game_ml":
        return "full_game_ml"
    if value == "full_game_rl":
        return "full_game_rl"
    if value == "full_game_total":
        return "full_game_total"
    raise ValueError(f"Unsupported market type: {value}")


def _has_scalar_value(value: Any) -> bool:
    return value is not None and not bool(pd.isna(value))


def _coerce_nullable_int(value: Any) -> int | None:
    if not _has_scalar_value(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_nullable_float(value: Any) -> float | None:
    if not _has_scalar_value(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _detect_historical_odds_schema(connection: sqlite3.Connection) -> Literal["canonical", "old_scraper"]:
    if _table_exists(connection, "odds") and _table_exists(connection, "games"):
        return "old_scraper"
    if _table_exists(connection, "odds_snapshots"):
        return "canonical"
    raise ValueError("Historical odds db must contain either odds_snapshots or old scraper odds/games tables")


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _empty_loaded_odds_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "game_pk",
            "book_name",
            "market_type",
            "home_odds",
            "away_odds",
            "home_point",
            "away_point",
            "total_point",
            "over_odds",
            "under_odds",
            "fetched_at",
            "source_schema",
        ]
    )


def _prepare_requested_game_frame(games_frame: pd.DataFrame) -> pd.DataFrame:
    if games_frame.empty:
        return pd.DataFrame()
    requested = games_frame.copy().reset_index(drop=True)
    requested["_request_id"] = requested.index.astype(int)
    requested["game_pk"] = (
        pd.to_numeric(requested["game_pk"], errors="coerce").astype("Int64")
        if "game_pk" in requested.columns
        else pd.Series(pd.array([pd.NA] * len(requested), dtype="Int64"), index=requested.index)
    )
    if "scheduled_start" in requested.columns:
        requested["scheduled_start"] = pd.to_datetime(
            requested["scheduled_start"],
            utc=True,
            errors="coerce",
            format="mixed",
        )
    else:
        requested["scheduled_start"] = pd.Series(pd.NaT, index=requested.index, dtype="datetime64[ns, UTC]")
    if "as_of_timestamp" in requested.columns:
        requested["as_of_timestamp"] = pd.to_datetime(
            requested["as_of_timestamp"],
            utc=True,
            errors="coerce",
            format="mixed",
        )
    else:
        requested["as_of_timestamp"] = pd.Series(pd.NaT, index=requested.index, dtype="datetime64[ns, UTC]")
    requested["pregame_cutoff"] = _resolve_pregame_cutoff_series(
        scheduled_start=requested["scheduled_start"],
        as_of_timestamp=requested["as_of_timestamp"],
    )
    if "game_date" in requested.columns:
        requested["request_game_date"] = pd.to_datetime(
            requested["game_date"],
            utc=True,
            errors="coerce",
            format="mixed",
        ).dt.tz_convert(None).dt.date
    else:
        requested["request_game_date"] = pd.Series([None] * len(requested), index=requested.index)
    scheduled_dates = requested["scheduled_start"].dt.tz_convert(None).dt.date
    requested["request_game_date"] = requested["request_game_date"].where(
        requested["request_game_date"].notna(),
        scheduled_dates,
    )
    requested["home_team_norm"] = requested.get("home_team", pd.Series(index=requested.index)).map(
        _normalize_team_code
    )
    requested["away_team_norm"] = requested.get("away_team", pd.Series(index=requested.index)).map(
        _normalize_team_code
    )
    requested = requested.dropna(
        subset=["request_game_date", "home_team_norm", "away_team_norm"],
    ).copy()
    return requested


def _pair_old_scraper_market_rows(
    raw: pd.DataFrame,
    *,
    market_type: MarketTypeLiteral,
    max_abs_odds: int,
) -> pd.DataFrame:
    working = raw.copy()
    working["price"] = pd.to_numeric(working["price"], errors="coerce")
    working = working.loc[working["price"].notna()].copy()
    working = working.loc[working["price"].abs() <= int(max_abs_odds)].copy()
    if working.empty:
        return _empty_loaded_odds_frame()

    working["point"] = pd.to_numeric(working["point"], errors="coerce")
    working["commence_time_utc"] = _coerce_old_scraper_commence_times(
        game_dates=working["game_date"],
        commence_values=working["commence_time_utc"],
    )
    working["fetched_at"] = pd.to_datetime(
        working["fetched_at"],
        utc=True,
        errors="coerce",
        format="mixed",
    )
    working["snapshot_time"] = working["fetched_at"].where(
        working["fetched_at"].notna(),
        working["commence_time_utc"],
    )
    working["home_team_norm"] = working["home_team"].map(_normalize_team_code)
    working["away_team_norm"] = working["away_team"].map(_normalize_team_code)
    working = working.dropna(subset=["home_team_norm", "away_team_norm"]).copy()

    index_columns = [
        "event_id",
        "game_date",
        "commence_time_utc",
        "away_team",
        "home_team",
        "away_team_norm",
        "home_team_norm",
        "book_name",
        "market_type",
        "snapshot_time",
    ]
    prices = (
        working.pivot_table(index=index_columns, columns="side", values="price", aggfunc="last")
        .rename_axis(None, axis=1)
        .rename(columns=lambda column_name: f"price_{column_name}")
        .reset_index()
    )
    points = (
        working.pivot_table(index=index_columns, columns="side", values="point", aggfunc="last")
        .rename_axis(None, axis=1)
        .rename(columns=lambda column_name: f"point_{column_name}")
        .reset_index()
    )
    opening_flags = (
        working.pivot_table(index=index_columns, columns="side", values="is_opening", aggfunc="max")
        .rename_axis(None, axis=1)
        .rename(columns=lambda column_name: f"opening_{column_name}")
        .reset_index()
    )
    snapshots = prices.merge(
        points,
        on=index_columns,
        how="left",
    ).merge(
        opening_flags,
        on=index_columns,
        how="left",
    )
    snapshots["fetched_at"] = snapshots["snapshot_time"]

    if market_type in {"f5_ml", "full_game_ml", "f5_rl", "full_game_rl"}:
        snapshots = snapshots.rename(
            columns={
                "price_home": "home_odds",
                "price_away": "away_odds",
                "point_home": "home_point",
                "point_away": "away_point",
                "opening_home": "home_is_opening",
                "opening_away": "away_is_opening",
            }
        )
        snapshots = snapshots.loc[
            snapshots["home_odds"].notna() & snapshots["away_odds"].notna()
        ].copy()
        snapshots["total_point"] = np.nan
        snapshots["over_odds"] = np.nan
        snapshots["under_odds"] = np.nan
        snapshots["opening_rank"] = np.where(
            snapshots[["home_is_opening", "away_is_opening"]].fillna(0).max(axis=1) > 0,
            0,
            1,
        )
    else:
        snapshots = snapshots.rename(
            columns={
                "price_over": "over_odds",
                "price_under": "under_odds",
                "point_over": "over_point",
                "point_under": "under_point",
                "opening_over": "over_is_opening",
                "opening_under": "under_is_opening",
            }
        )
        snapshots = snapshots.loc[
            snapshots["over_odds"].notna() & snapshots["under_odds"].notna()
        ].copy()
        snapshots["total_point"] = snapshots["over_point"].where(
            snapshots["over_point"].notna(),
            snapshots["under_point"],
        )
        snapshots["home_odds"] = np.nan
        snapshots["away_odds"] = np.nan
        snapshots["home_point"] = np.nan
        snapshots["away_point"] = np.nan
        snapshots["opening_rank"] = np.where(
            snapshots[["over_is_opening", "under_is_opening"]].fillna(0).max(axis=1) > 0,
            0,
            1,
        )

    for column in ("home_odds", "away_odds", "home_point", "away_point", "total_point", "over_odds", "under_odds"):
        if column not in snapshots.columns:
            snapshots[column] = np.nan

    snapshots["source_schema"] = "old_scraper"
    return snapshots[
        [
            "event_id",
            "game_date",
            "commence_time_utc",
            "away_team",
            "home_team",
            "away_team_norm",
            "home_team_norm",
            "book_name",
            "market_type",
            "home_odds",
            "away_odds",
            "home_point",
            "away_point",
            "total_point",
            "over_odds",
            "under_odds",
            "fetched_at",
            "opening_rank",
            "source_schema",
        ]
    ]


def _match_requested_games_to_old_scraper_snapshots(
    *,
    requested: pd.DataFrame,
    snapshots: pd.DataFrame,
    snapshot_selection: SnapshotSelectionLiteral,
    aggregate_consensus: bool,
) -> pd.DataFrame:
    merged = requested.merge(
        snapshots,
        on=["home_team_norm", "away_team_norm"],
        how="left",
        suffixes=("_request", ""),
    )
    if merged.empty:
        return _empty_loaded_odds_frame()

    merged["commence_time_utc"] = pd.to_datetime(
        merged["commence_time_utc"],
        utc=True,
        errors="coerce",
        format="mixed",
    )
    merged["request_game_date_text"] = merged["request_game_date"].astype(str)
    merged["game_date"] = merged["game_date"].astype(str)
    merged["date_matches"] = merged["request_game_date_text"] == merged["game_date"]

    request_start_naive = merged["scheduled_start"].dt.tz_convert(None)
    candidate_start_naive = merged["commence_time_utc"].dt.tz_convert(None)
    merged["delta_hours"] = (
        (request_start_naive - candidate_start_naive).abs().dt.total_seconds() / 3600.0
    )
    merged["delta_hours"] = merged["delta_hours"].fillna(9999.0)
    merged = merged.loc[
        merged["event_id"].notna()
        & (merged["date_matches"] | (merged["delta_hours"] <= 36.0))
    ].copy()
    if merged.empty:
        return _empty_loaded_odds_frame()

    merged["fetched_at"] = pd.to_datetime(merged["fetched_at"], utc=True, errors="coerce", format="mixed")
    merged["snapshot_time"] = merged["fetched_at"]
    merged["effective_cutoff"] = pd.to_datetime(
        merged["pregame_cutoff"],
        utc=True,
        errors="coerce",
        format="mixed",
    ).where(
        pd.to_datetime(merged["pregame_cutoff"], utc=True, errors="coerce", format="mixed").notna(),
        merged["commence_time_utc"],
    )
    opening_rank = pd.to_numeric(merged["opening_rank"], errors="coerce").fillna(1)
    opening_snapshot_mask = opening_rank.eq(0)
    if snapshot_selection == "opening":
        # Old scraper backfills store fetch time as scrape time, not archived market time.
        # The dedicated opener column is still explicitly pregame, so keep opener rows
        # when present while leaving latest/closing selection strict on real timestamps.
        group_has_opening = opening_rank.groupby([merged["_request_id"], merged["event_id"]]).transform("min").eq(0)
        merged = merged.loc[~group_has_opening | opening_snapshot_mask].copy()
        opening_rank = pd.to_numeric(merged["opening_rank"], errors="coerce").fillna(1)
        opening_snapshot_mask = opening_rank.eq(0)

    eligibility_mask = (
        merged["fetched_at"].notna()
        & merged["effective_cutoff"].notna()
        & (merged["fetched_at"] <= merged["effective_cutoff"])
    )
    if snapshot_selection == "opening":
        eligibility_mask = eligibility_mask | opening_snapshot_mask

    merged = merged.loc[eligibility_mask].copy()
    if merged.empty:
        return _empty_loaded_odds_frame()

    if snapshot_selection == "opening":
        merged = merged.sort_values(
            ["_request_id", "event_id", "book_name", "opening_rank", "snapshot_time"],
            ascending=[True, True, True, True, True],
        )
    else:
        merged = merged.sort_values(
            ["_request_id", "event_id", "book_name", "snapshot_time", "opening_rank"],
            ascending=[True, True, True, False, True],
        )
    merged = merged.drop_duplicates(
        subset=["_request_id", "event_id", "book_name"],
        keep="first",
    ).copy()
    if merged.empty:
        return _empty_loaded_odds_frame()

    if aggregate_consensus:
        merged = (
            merged.groupby(["_request_id", "event_id"], as_index=False, dropna=False)
            .apply(_aggregate_old_scraper_consensus_snapshot)
            .reset_index(drop=True)
        )
        merged["book_name"] = "consensus"

    merged = merged.sort_values(
        ["_request_id", "date_matches", "delta_hours", "fetched_at"],
        ascending=[True, False, True, False],
    )
    matched = merged.drop_duplicates(subset=["_request_id"], keep="first").copy()
    if matched.empty:
        return _empty_loaded_odds_frame()

    matched["game_pk"] = matched["game_pk"].astype("Int64")
    result = matched[
        [
            "game_pk",
            "book_name",
            "market_type",
            "home_odds",
            "away_odds",
            "home_point",
            "away_point",
            "total_point",
            "over_odds",
            "under_odds",
            "fetched_at",
            "source_schema",
        ]
    ].copy()
    if result["game_pk"].isna().all():
        result = result.drop(columns=["game_pk"])
    else:
        result["game_pk"] = result["game_pk"].astype(int)
    return result


def _normalize_team_code(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    return _TEAM_CODE_ALIASES.get(text, text)


def _coerce_old_scraper_commence_times(
    *,
    game_dates: pd.Series,
    commence_values: pd.Series,
) -> pd.Series:
    parsed = pd.to_datetime(commence_values, utc=True, errors="coerce", format="mixed")
    unresolved_mask = parsed.isna() & commence_values.notna() & game_dates.notna()
    if unresolved_mask.any():
        combined = (
            game_dates.loc[unresolved_mask].astype(str).str.strip()
            + " "
            + commence_values.loc[unresolved_mask].astype(str).str.strip()
        )
        offset_resolved = combined.copy()
        for abbreviation, offset in _TIMEZONE_ABBREVIATION_OFFSETS.items():
            offset_resolved = offset_resolved.str.replace(
                f" {abbreviation}$",
                f" {offset}",
                regex=True,
            )
        reparsed = pd.to_datetime(offset_resolved, utc=True, errors="coerce", format="mixed")
        parsed.loc[unresolved_mask] = reparsed
    return parsed


def _resolve_canonical_request_cutoffs(
    connection: sqlite3.Connection,
    *,
    game_pks: Sequence[int],
    games_frame: pd.DataFrame | None,
) -> pd.DataFrame:
    if not game_pks:
        return pd.DataFrame(columns=["game_pk", "pregame_cutoff"])

    cutoff_frame = pd.DataFrame({"game_pk": [int(game_pk) for game_pk in game_pks]})
    cutoff_frame["pregame_cutoff"] = pd.Series(pd.NaT, index=cutoff_frame.index, dtype="datetime64[ns, UTC]")
    if games_frame is not None and not games_frame.empty and "game_pk" in games_frame.columns:
        request = pd.DataFrame(
            {
                "game_pk": pd.to_numeric(games_frame["game_pk"], errors="coerce").astype("Int64"),
            }
        )
        request["scheduled_start"] = _coerce_optional_timestamp_series(
            games_frame["scheduled_start"] if "scheduled_start" in games_frame.columns else None
        )
        request["as_of_timestamp"] = _coerce_optional_timestamp_series(
            games_frame["as_of_timestamp"] if "as_of_timestamp" in games_frame.columns else None
        )
        request["pregame_cutoff"] = _resolve_pregame_cutoff_series(
            scheduled_start=request["scheduled_start"],
            as_of_timestamp=request["as_of_timestamp"],
        )
        request = request.dropna(subset=["game_pk"]).drop_duplicates(subset=["game_pk"], keep="first")
        cutoff_frame = cutoff_frame.merge(
            request.loc[:, ["game_pk", "pregame_cutoff"]],
            on="game_pk",
            how="left",
            suffixes=("", "_request"),
        )
        cutoff_frame["pregame_cutoff"] = cutoff_frame["pregame_cutoff_request"].where(
            cutoff_frame["pregame_cutoff_request"].notna(),
            cutoff_frame["pregame_cutoff"],
        )
        cutoff_frame = cutoff_frame.drop(columns=["pregame_cutoff_request"], errors="ignore")

    missing_game_pks = cutoff_frame.loc[cutoff_frame["pregame_cutoff"].isna(), "game_pk"].astype(int).tolist()
    if missing_game_pks:
        placeholders = ",".join("?" for _ in missing_game_pks)
        games = pd.read_sql_query(
            f"SELECT game_pk, date AS scheduled_start FROM games WHERE game_pk IN ({placeholders})",
            connection,
            params=missing_game_pks,
        )
        if not games.empty:
            games["game_pk"] = pd.to_numeric(games["game_pk"], errors="coerce").astype("Int64")
            games["pregame_cutoff"] = _coerce_optional_timestamp_series(games["scheduled_start"])
            cutoff_frame = cutoff_frame.merge(
                games.loc[:, ["game_pk", "pregame_cutoff"]],
                on="game_pk",
                how="left",
                suffixes=("", "_game"),
            )
            cutoff_frame["pregame_cutoff"] = cutoff_frame["pregame_cutoff"].where(
                cutoff_frame["pregame_cutoff"].notna(),
                cutoff_frame["pregame_cutoff_game"],
            )
            cutoff_frame = cutoff_frame.drop(columns=["pregame_cutoff_game"], errors="ignore")
    return cutoff_frame


def _coerce_optional_timestamp_series(values: pd.Series | None) -> pd.Series:
    if values is None:
        return pd.Series(dtype="datetime64[ns, UTC]")
    return pd.to_datetime(values, utc=True, errors="coerce", format="mixed")


def _resolve_pregame_cutoff_series(
    *,
    scheduled_start: pd.Series,
    as_of_timestamp: pd.Series,
) -> pd.Series:
    scheduled = pd.to_datetime(scheduled_start, utc=True, errors="coerce", format="mixed")
    as_of = pd.to_datetime(as_of_timestamp, utc=True, errors="coerce", format="mixed")
    combined = pd.concat([scheduled, as_of], axis=1)
    return combined.min(axis=1)


def _aggregate_old_scraper_consensus_snapshot(group: pd.DataFrame) -> pd.Series:
    first_row = group.iloc[0]
    group_name = group.name
    request_id_from_group = None
    event_id_from_group = group_name
    if isinstance(group_name, tuple):
        if len(group_name) >= 1:
            request_id_from_group = group_name[0]
        if len(group_name) >= 2:
            event_id_from_group = group_name[1]
    payload: dict[str, Any] = {
        "_request_id": first_row.get("_request_id", request_id_from_group),
        "event_id": first_row["event_id"] if "event_id" in first_row.index else event_id_from_group,
        "game_pk": first_row.get("game_pk"),
        "date_matches": first_row.get("date_matches"),
        "delta_hours": first_row.get("delta_hours"),
        "fetched_at": pd.to_datetime(group["fetched_at"], utc=True, errors="coerce").max(),
        "market_type": first_row["market_type"],
        "game_date": first_row["game_date"],
        "commence_time_utc": first_row["commence_time_utc"],
        "away_team": first_row["away_team"],
        "home_team": first_row["home_team"],
        "away_team_norm": first_row["away_team_norm"],
        "home_team_norm": first_row["home_team_norm"],
        "home_odds": _average_american_odds(group.get("home_odds")),
        "away_odds": _average_american_odds(group.get("away_odds")),
        "over_odds": _average_american_odds(group.get("over_odds")),
        "under_odds": _average_american_odds(group.get("under_odds")),
        "home_point": _average_numeric(group.get("home_point")),
        "away_point": _average_numeric(group.get("away_point")),
        "total_point": _average_numeric(group.get("total_point")),
        "source_schema": first_row.get("source_schema", "old_scraper"),
    }
    return pd.Series(payload)


def _average_numeric(values: pd.Series | None) -> float | None:
    if values is None:
        return None
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def _average_american_odds(values: pd.Series | None) -> float | None:
    if values is None:
        return None
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    implied_probabilities = [_american_to_implied_probability(int(value)) for value in numeric.tolist()]
    return float(_implied_probability_to_american(sum(implied_probabilities) / len(implied_probabilities)))


def _american_to_implied_probability(odds: int) -> float:
    if odds > 0:
        return float(100.0 / (odds + 100.0))
    return float(abs(odds) / (abs(odds) + 100.0))


def _implied_probability_to_american(probability: float) -> int:
    clipped_probability = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    if clipped_probability >= 0.5:
        return int(round(-100.0 * clipped_probability / (1.0 - clipped_probability)))
    return int(round(100.0 * (1.0 - clipped_probability) / clipped_probability))


if __name__ == "__main__":
    raise SystemExit(main())
