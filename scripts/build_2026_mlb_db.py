from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
from rich.console import Console


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db import init_db  # noqa: E402
from src.pipeline.daily import _default_schedule_fetcher  # noqa: E402


console = Console()
DEFAULT_SOURCE_DB = Path("data/mlb.db")
DEFAULT_TARGET_DB = Path("data/2026MLB.db")


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _normalize_games_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "game_pk",
                "date",
                "home_team",
                "away_team",
                "home_starter_id",
                "away_starter_id",
                "venue",
                "is_dome",
                "is_abs_active",
                "f5_home_score",
                "f5_away_score",
                "final_home_score",
                "final_away_score",
                "status",
            ]
        )
    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized = normalized.loc[normalized["date"].notna()].copy()
    normalized["date"] = normalized["date"].dt.date.astype(str)
    for starter_col in ("home_starter_id", "away_starter_id"):
        normalized[starter_col] = pd.to_numeric(normalized[starter_col], errors="coerce").astype("Int64")
    for score_col in ("f5_home_score", "f5_away_score", "final_home_score", "final_away_score"):
        normalized[score_col] = pd.to_numeric(normalized[score_col], errors="coerce").astype("Int64")
    normalized["is_dome"] = pd.to_numeric(normalized["is_dome"], errors="coerce").fillna(0).astype(int)
    normalized["is_abs_active"] = (
        pd.to_numeric(normalized["is_abs_active"], errors="coerce").fillna(1).astype(int)
    )
    normalized["status"] = normalized["status"].astype(str).str.lower()
    return normalized[
        [
            "game_pk",
            "date",
            "home_team",
            "away_team",
            "home_starter_id",
            "away_starter_id",
            "venue",
            "is_dome",
            "is_abs_active",
            "f5_home_score",
            "f5_away_score",
            "final_home_score",
            "final_away_score",
            "status",
        ]
    ].sort_values(["date", "game_pk"]).reset_index(drop=True)


def _load_source_2026_games(source_db: Path) -> pd.DataFrame:
    with sqlite3.connect(source_db) as connection:
        frame = pd.read_sql_query(
            """
            SELECT
                game_pk,
                date,
                home_team,
                away_team,
                home_starter_id,
                away_starter_id,
                venue,
                is_dome,
                is_abs_active,
                f5_home_score,
                f5_away_score,
                final_home_score,
                final_away_score,
                status
            FROM games
            WHERE date LIKE '2026-%'
            ORDER BY date, game_pk
            """,
            connection,
        )
    return _normalize_games_frame(frame)


def _fetch_schedule_gap(start_day: date, end_day: date) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    current = start_day
    while current <= end_day:
        schedule = _default_schedule_fetcher(current, "prod")
        if not schedule.empty:
            filtered = schedule.loc[
                pd.to_datetime(schedule["game_date"], errors="coerce").dt.date <= end_day
            ].copy()
            if not filtered.empty:
                filtered = filtered.rename(columns={"game_date": "date"})
                filtered["f5_home_score"] = pd.NA
                filtered["f5_away_score"] = pd.NA
                filtered["final_home_score"] = pd.NA
                filtered["final_away_score"] = pd.NA
                rows.append(
                    filtered[
                        [
                            "game_pk",
                            "date",
                            "home_team",
                            "away_team",
                            "home_starter_id",
                            "away_starter_id",
                            "venue",
                            "is_dome",
                            "is_abs_active",
                            "f5_home_score",
                            "f5_away_score",
                            "final_home_score",
                            "final_away_score",
                            "status",
                        ]
                    ]
                )
        current = current.fromordinal(current.toordinal() + 1)
    if not rows:
        return _normalize_games_frame(pd.DataFrame())
    return _normalize_games_frame(pd.concat(rows, ignore_index=True))


def _ensure_metadata_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS build_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )


def _write_games(target_db: Path, frame: pd.DataFrame, metadata: dict[str, str]) -> None:
    database_path = init_db(target_db)
    with sqlite3.connect(database_path) as connection:
        connection.execute("DELETE FROM games WHERE date LIKE '2026-%'")
        rows = [
            (
                int(row["game_pk"]),
                str(row["date"]),
                str(row["home_team"]),
                str(row["away_team"]),
                None if pd.isna(row["home_starter_id"]) else int(row["home_starter_id"]),
                None if pd.isna(row["away_starter_id"]) else int(row["away_starter_id"]),
                str(row["venue"]),
                int(row["is_dome"]),
                int(row["is_abs_active"]),
                None if pd.isna(row["f5_home_score"]) else int(row["f5_home_score"]),
                None if pd.isna(row["f5_away_score"]) else int(row["f5_away_score"]),
                None if pd.isna(row["final_home_score"]) else int(row["final_home_score"]),
                None if pd.isna(row["final_away_score"]) else int(row["final_away_score"]),
                str(row["status"]),
            )
            for row in frame.to_dict(orient="records")
        ]
        connection.executemany(
            """
            INSERT INTO games (
                game_pk, date, home_team, away_team, home_starter_id, away_starter_id, venue,
                is_dome, is_abs_active, f5_home_score, f5_away_score, final_home_score, final_away_score, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        _ensure_metadata_table(connection)
        connection.execute("DELETE FROM build_metadata")
        connection.executemany(
            "INSERT INTO build_metadata (key, value) VALUES (?, ?)",
            [(str(key), json.dumps(value) if not isinstance(value, str) else value) for key, value in metadata.items()],
        )
        connection.commit()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a dedicated 2026 games DB.")
    parser.add_argument("--source-db", default=str(DEFAULT_SOURCE_DB))
    parser.add_argument("--target-db", default=str(DEFAULT_TARGET_DB))
    parser.add_argument("--fill-through-today", action="store_true")
    args = parser.parse_args(argv)

    source_db = _resolve_path(args.source_db)
    target_db = _resolve_path(args.target_db)

    source_games = _load_source_2026_games(source_db)
    fetched_games = pd.DataFrame()
    if args.fill_through_today:
        earliest_missing_day = date(2026, 1, 1)
        if not source_games.empty:
            normalized_dates = pd.to_datetime(source_games["date"], errors="coerce").dropna()
            latest_local_day = None if normalized_dates.empty else normalized_dates.max().date()
            if latest_local_day is not None:
                earliest_missing_day = latest_local_day.fromordinal(latest_local_day.toordinal() + 1)
        today = datetime.now().date()
        if earliest_missing_day <= today:
            fetched_games = _fetch_schedule_gap(earliest_missing_day, today)

    combined = pd.concat([source_games, fetched_games], ignore_index=True) if not fetched_games.empty else source_games
    combined = _normalize_games_frame(combined.drop_duplicates(subset=["game_pk"], keep="last"))
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_db": str(source_db),
        "row_count": str(len(combined)),
        "source_row_count": str(len(source_games)),
        "fetched_row_count": str(len(fetched_games)),
        "date_min": "" if combined.empty else str(combined["date"].min()),
        "date_max": "" if combined.empty else str(combined["date"].max()),
        "fill_through_today": str(bool(args.fill_through_today)),
    }
    _write_games(target_db, combined, metadata)

    console.print(
        "[bold green]Built 2026 games DB[/bold green] "
        f"rows={len(combined)} source_rows={len(source_games)} fetched_rows={len(fetched_games)}"
    )
    console.print(f"db={target_db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
