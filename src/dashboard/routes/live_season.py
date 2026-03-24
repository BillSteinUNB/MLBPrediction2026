from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Query

from src.dashboard.schemas import LiveSeasonGameResponse, LiveSeasonSummaryResponse
from src.ops.live_season_tracker import build_live_season_summary, list_tracked_games

router = APIRouter(prefix="/api/live-season", tags=["live-season"])


@router.get("/summary", response_model=LiveSeasonSummaryResponse)
async def get_live_season_summary(
    season: int = Query(default=2026, ge=2000, le=2100),
    db_path: str = Query(
        default=str(Path("data") / "mlb.db"),
        description="SQLite path used by live season tracking.",
    ),
) -> LiveSeasonSummaryResponse:
    summary = build_live_season_summary(season=season, db_path=db_path)
    return LiveSeasonSummaryResponse.model_validate(summary.to_dict())


@router.get("/games", response_model=list[LiveSeasonGameResponse])
async def get_live_season_games(
    season: int = Query(default=2026, ge=2000, le=2100),
    pipeline_date: str | None = Query(default=None),
    db_path: str = Query(
        default=str(Path("data") / "mlb.db"),
        description="SQLite path used by live season tracking.",
    ),
) -> list[LiveSeasonGameResponse]:
    rows = list_tracked_games(season=season, pipeline_date=pipeline_date, db_path=db_path)
    normalized_rows = []
    for row in rows:
        normalized = dict(row)
        normalized["paper_fallback"] = bool(normalized.get("paper_fallback"))
        normalized["is_play_of_day"] = bool(normalized.get("is_play_of_day"))
        normalized_rows.append(normalized)
    return [LiveSeasonGameResponse.model_validate(row) for row in normalized_rows]
