from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

import pandas as pd

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
}
MarketTypeLiteral = Literal["f5_ml", "f5_rl"]


def import_historical_odds(
    *,
    source_path: str | Path,
    db_path: str | Path = DEFAULT_DB_PATH,
    default_market_type: str = "f5_ml",
    default_book_name: str = "historical",
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
    game_pks: Sequence[int],
    market_type: str = "f5_ml",
    book_name: str | None = None,
) -> pd.DataFrame:
    """Load latest historical odds snapshots for the requested games."""

    if not game_pks:
        return pd.DataFrame(
            columns=["game_pk", "book_name", "market_type", "home_odds", "away_odds", "fetched_at"]
        )

    placeholders = ",".join("?" for _ in game_pks)
    query = f"""
        SELECT o.game_pk, o.book_name, o.market_type, o.home_odds, o.away_odds, o.fetched_at
        FROM odds_snapshots AS o
        INNER JOIN (
            SELECT game_pk, market_type, MAX(fetched_at) AS fetched_at
            FROM odds_snapshots
            WHERE market_type = ?
              AND game_pk IN ({placeholders})
              {{book_filter_inner}}
            GROUP BY game_pk, market_type
        ) AS latest
            ON latest.game_pk = o.game_pk
           AND latest.market_type = o.market_type
           AND latest.fetched_at = o.fetched_at
        WHERE o.market_type = ?
          {"" if book_name is None else "AND o.book_name = ?"}
    """
    query = query.replace("{book_filter_inner}", "" if book_name is None else "AND book_name = ?")
    params: list[Any] = [market_type, *[int(game_pk) for game_pk in game_pks]]
    if book_name:
        params.append(book_name)
    params.append(market_type)
    if book_name:
        params.append(book_name)

    with sqlite3.connect(init_db(db_path)) as connection:
        frame = pd.read_sql_query(query, connection, params=params)
    return frame


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


if __name__ == "__main__":
    raise SystemExit(main())
