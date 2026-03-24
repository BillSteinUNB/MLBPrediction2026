from __future__ import annotations

from datetime import date as date_type, datetime
from pathlib import Path

from fastapi import APIRouter, Query

from src.dashboard.schemas import SlateResponse
from src.pipeline.daily import run_daily_pipeline

router = APIRouter(prefix="/api/slate", tags=["slate"])


@router.get("", response_model=SlateResponse)
async def get_slate(
    pipeline_date: str | None = Query(
        default=None,
        description="Target slate date in YYYY-MM-DD format. Defaults to today.",
    ),
    db_path: str = Query(
        default=str(Path("data") / "mlb.db"),
        description="SQLite path used by the daily pipeline.",
    ),
) -> SlateResponse:
    resolved_date = pipeline_date or date_type.today().isoformat()
    result = run_daily_pipeline(
        target_date=resolved_date,
        mode="prod",
        dry_run=True,
        db_path=db_path,
    )
    payload = result.to_dict()
    payload.pop("notification_payload", None)
    return SlateResponse.model_validate(payload)
