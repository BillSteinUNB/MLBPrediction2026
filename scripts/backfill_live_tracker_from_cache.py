from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.db import DEFAULT_DB_PATH, init_db
from src.models.bet import BetDecision
from src.models.prediction import Prediction
from src.ops.live_season_tracker import (
    TRACKING_SOURCE_CACHE_BACKFILL,
    TRACKING_SOURCE_RETROSPECTIVE,
    _neutral_weather_fetcher,
    capture_daily_result,
    settle_tracked_games,
)
from src.pipeline.daily import DailyPipelineResult, GameProcessingResult, PipelineDependencies, run_daily_pipeline


def _build_game_result(payload: dict[str, Any]) -> GameProcessingResult:
    prediction_payload = payload.get("prediction")
    selected_payload = payload.get("selected_decision")
    forced_payload = payload.get("forced_decision")
    return GameProcessingResult(
        game_pk=int(payload["game_pk"]),
        matchup=str(payload["matchup"]),
        status=str(payload["status"]),
        game_status=payload.get("game_status"),
        is_completed=bool(payload.get("is_completed", False)),
        prediction=(
            Prediction.model_validate(prediction_payload) if isinstance(prediction_payload, dict) else None
        ),
        selected_decision=(
            BetDecision.model_validate(selected_payload) if isinstance(selected_payload, dict) else None
        ),
        forced_decision=(
            BetDecision.model_validate(forced_payload) if isinstance(forced_payload, dict) else None
        ),
        no_pick_reason=payload.get("no_pick_reason"),
        error_message=payload.get("error_message"),
        notified=bool(payload.get("notified", False)),
        paper_fallback=bool(payload.get("paper_fallback", False)),
        input_status=payload.get("input_status"),
        narrative=payload.get("narrative"),
    )


def _build_daily_result(payload: dict[str, Any]) -> DailyPipelineResult:
    return DailyPipelineResult(
        run_id=str(payload["run_id"]),
        pipeline_date=str(payload["pipeline_date"]),
        mode=str(payload["mode"]),
        dry_run=bool(payload["dry_run"]),
        model_version=str(payload["model_version"]),
        pick_count=int(payload["pick_count"]),
        no_pick_count=int(payload["no_pick_count"]),
        error_count=int(payload["error_count"]),
        notification_type=str(payload.get("notification_type") or "replay"),
        notification_payload=dict(payload.get("notification_payload") or {}),
        games=[_build_game_result(game_payload) for game_payload in payload.get("games", [])],
    )


def _missing_tracker_dates(
    connection: sqlite3.Connection,
    *,
    season: int,
    through_date: date,
) -> set[str]:
    tracker_dates = {
        str(row[0])
        for row in connection.execute(
            "SELECT DISTINCT pipeline_date FROM live_season_tracking WHERE season = ?",
            (season,),
        ).fetchall()
    }
    game_dates = {
        str(row[0])
        for row in connection.execute(
            """
            SELECT DISTINCT date
            FROM games
            WHERE date LIKE ?
              AND LENGTH(date) = 10
              AND date <= ?
            """,
            (f"{season}-%", through_date.isoformat()),
        ).fetchall()
    }
    return game_dates - tracker_dates


def _load_latest_cached_payload(
    connection: sqlite3.Connection,
    *,
    pipeline_date: str,
) -> dict[str, Any] | None:
    row = connection.execute(
        """
        SELECT payload_json
        FROM cached_slate_responses
        WHERE pipeline_date = ?
        ORDER BY refreshed_at DESC
        LIMIT 1
        """,
        (pipeline_date,),
    ).fetchone()
    if row is None or not row[0]:
        return None
    return json.loads(str(row[0]))


def _offline_games_frame(
    database_path: str | Path,
    *,
    season: int,
    on_date: date | None = None,
    before_date: date | None = None,
) -> pd.DataFrame:
    where_clauses = ["substr(date, 1, 4) = ?"]
    params: list[Any] = [str(season)]
    if on_date is not None:
        where_clauses.append("substr(date, 1, 10) = ?")
        params.append(on_date.isoformat())
    if before_date is not None:
        where_clauses.append("substr(date, 1, 10) < ?")
        params.append(before_date.isoformat())

    query = f"""
        SELECT
            game_pk,
            substr(date, 1, 10) AS game_date,
            home_team,
            away_team,
            home_starter_id,
            away_starter_id,
            venue,
            is_dome,
            is_abs_active,
            status,
            f5_home_score,
            f5_away_score,
            final_home_score,
            final_away_score
        FROM games
        WHERE {' AND '.join(where_clauses)}
        ORDER BY substr(date, 1, 10), game_pk
    """
    with sqlite3.connect(database_path) as connection:
        frame = pd.read_sql_query(query, connection, params=params)
    if frame.empty:
        return frame
    frame["season"] = int(season)
    frame["scheduled_start"] = frame["game_date"].astype(str) + "T19:05:00+00:00"
    frame["game_type"] = "R"
    return frame


def _offline_schedule_fetcher(database_path: str | Path):
    def _fetch(target_date: date, _mode: str) -> pd.DataFrame:
        return _offline_games_frame(database_path, season=target_date.year, on_date=target_date)

    return _fetch


def _offline_history_fetcher(database_path: str | Path):
    def _fetch(season: int, before_date: date) -> pd.DataFrame:
        return _offline_games_frame(database_path, season=season, before_date=before_date)

    return _fetch


def backfill_live_tracker_from_cache(
    *,
    season: int = 2026,
    db_path: str | Path = DEFAULT_DB_PATH,
    requested_dates: set[str] | None = None,
    settle_after: bool = False,
    through_date: date | None = None,
) -> dict[str, Any]:
    database_path = init_db(db_path)
    resolved_through_date = through_date or (datetime.now().date() - timedelta(days=1))
    with sqlite3.connect(database_path) as connection:
        if requested_dates is None:
            requested_dates = _missing_tracker_dates(
                connection,
                season=season,
                through_date=resolved_through_date,
            )

    replayed_rows = 0
    replayed_dates: list[str] = []
    cache_replayed_dates: list[str] = []
    retrospective_dates: list[str] = []
    failed_dates: list[dict[str, str]] = []

    for pipeline_date in sorted(requested_dates):
        try:
            with sqlite3.connect(database_path) as connection:
                payload = _load_latest_cached_payload(connection, pipeline_date=pipeline_date)
            if payload is not None:
                result = _build_daily_result(payload)
                replayed_rows += capture_daily_result(
                    result=result,
                    db_path=database_path,
                    tracking_source=TRACKING_SOURCE_CACHE_BACKFILL,
                )
                replayed_dates.append(str(result.pipeline_date))
                cache_replayed_dates.append(str(result.pipeline_date))
                continue

            result = run_daily_pipeline(
                target_date=pipeline_date,
                mode="backtest",
                dry_run=True,
                db_path=database_path,
                dependencies=PipelineDependencies(
                    schedule_fetcher=_offline_schedule_fetcher(database_path),
                    history_fetcher=_offline_history_fetcher(database_path),
                    lineups_fetcher=lambda _target_date: [],
                    weather_fetcher=_neutral_weather_fetcher,
                ),
            )
            replayed_rows += capture_daily_result(
                result=result,
                db_path=database_path,
                tracking_source=TRACKING_SOURCE_RETROSPECTIVE,
            )
            replayed_dates.append(str(result.pipeline_date))
            retrospective_dates.append(str(result.pipeline_date))
        except Exception as exc:
            failed_dates.append({"pipeline_date": pipeline_date, "error": str(exc)})

    settled_rows = 0
    if settle_after:
        for pipeline_date in replayed_dates:
            settled_rows += settle_tracked_games(
                pipeline_date=pipeline_date,
                season=season,
                db_path=database_path,
                refresh_schedule=False,
            )
    return {
        "season": int(season),
        "requested_dates": sorted(requested_dates) if requested_dates is not None else None,
        "replayed_dates": replayed_dates,
        "cache_replayed_dates": cache_replayed_dates,
        "retrospective_dates": retrospective_dates,
        "failed_dates": failed_dates,
        "replayed_rows": int(replayed_rows),
        "settled_rows": int(settled_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill live season tracker rows from frozen cached slate payloads."
    )
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--date",
        dest="dates",
        action="append",
        default=None,
        help="Specific pipeline date to backfill. Repeat for multiple dates.",
    )
    parser.add_argument(
        "--settle-after",
        action="store_true",
        help="Run settle_tracked_games after replaying cached dates.",
    )
    parser.add_argument(
        "--through-date",
        type=date.fromisoformat,
        default=None,
        help="Latest completed date to consider when auto-finding missing tracker days.",
    )
    args = parser.parse_args()

    requested_dates = set(args.dates) if args.dates else None
    summary = backfill_live_tracker_from_cache(
        season=args.season,
        db_path=args.db_path,
        requested_dates=requested_dates,
        settle_after=args.settle_after,
        through_date=args.through_date,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
