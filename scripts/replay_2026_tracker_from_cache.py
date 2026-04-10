from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from src.db import DEFAULT_DB_PATH, init_db
from src.models.bet import BetDecision
from src.models.prediction import Prediction
from src.ops.live_season_tracker import capture_daily_result, settle_tracked_games
from src.pipeline.daily import DailyPipelineResult, GameProcessingResult


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


def _load_latest_cached_payloads(
    connection: sqlite3.Connection,
    *,
    season: int,
) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT csr.payload_json
        FROM cached_slate_responses AS csr
        JOIN (
            SELECT pipeline_date, MAX(refreshed_at) AS refreshed_at
            FROM cached_slate_responses
            WHERE pipeline_date LIKE ?
            GROUP BY pipeline_date
        ) AS latest
          ON latest.pipeline_date = csr.pipeline_date
         AND latest.refreshed_at = csr.refreshed_at
        ORDER BY csr.pipeline_date ASC
        """,
        (f"{season}-%",),
    ).fetchall()
    return [json.loads(str(row[0])) for row in rows]


def replay_tracker_from_cache(
    *,
    season: int = 2026,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, int]:
    database_path = init_db(db_path)
    with sqlite3.connect(database_path) as connection:
        payloads = _load_latest_cached_payloads(connection, season=season)
        connection.execute("DELETE FROM live_season_tracking WHERE season = ?", (season,))
        connection.commit()

    replayed_rows = 0
    replayed_dates = 0
    for payload in payloads:
        result = _build_daily_result(payload)
        replayed_rows += capture_daily_result(result=result, db_path=database_path)
        replayed_dates += 1

    settled_rows = settle_tracked_games(season=season, db_path=database_path)
    return {
        "season": int(season),
        "replayed_dates": int(replayed_dates),
        "replayed_rows": int(replayed_rows),
        "settled_rows": int(settled_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild 2026 live season tracking from cached slate payloads.")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    summary = replay_tracker_from_cache(season=args.season, db_path=args.db_path)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
