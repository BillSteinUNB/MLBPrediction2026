from __future__ import annotations

import os
import subprocess
from datetime import date as date_type, datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from src.dashboard.schemas import MacSyncResponse, SlateResponse
from src.pipeline.daily import load_cached_slate_response, run_daily_pipeline

router = APIRouter(prefix="/api/slate", tags=["slate"])


def _mac_sync_enabled() -> bool:
    return os.getenv("MAC_SYNC_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}


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
    refresh: bool = Query(
        default=False,
        description="When true, rebuild the slate instead of using the cached response.",
    ),
) -> SlateResponse:
    resolved_date = pipeline_date or date_type.today().isoformat()
    payload = None
    if not refresh:
        payload = load_cached_slate_response(
            pipeline_date=resolved_date,
            mode="prod",
            dry_run=True,
            db_path=db_path,
        )
    if payload is None:
        result = run_daily_pipeline(
            target_date=resolved_date,
            mode="prod",
            dry_run=True,
            db_path=db_path,
        )
        payload = result.to_dict()
        payload.pop("notification_payload", None)
    return SlateResponse.model_validate(payload)


@router.post("/pull-from-mac", response_model=MacSyncResponse)
async def pull_from_mac(
    pipeline_date: str | None = Query(
        default=None,
        description="Target slate date in YYYY-MM-DD format. Defaults to today.",
    ),
) -> MacSyncResponse:
    if not _mac_sync_enabled():
        raise HTTPException(
            status_code=503,
            detail="Mac sync is currently disabled on this host.",
        )

    sync_script = Path("scripts") / "sync_mac_odds_and_refresh.ps1"
    if not sync_script.exists():
        raise HTTPException(status_code=500, detail=f"Sync script not found: {sync_script}")

    mac_host = os.getenv("MAC_SYNC_HOST")
    mac_user = os.getenv("MAC_SYNC_USER", "bill")
    remote_repo_path = os.getenv("MAC_SYNC_REMOTE_REPO_PATH", "/Users/bill/Code/MLBTracker")
    if not mac_host:
        raise HTTPException(
            status_code=400,
            detail="MAC_SYNC_HOST is not configured on the backend host.",
        )

    resolved_date = pipeline_date or date_type.today().isoformat()
    command = [
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(sync_script),
        "-MacHost",
        mac_host,
        "-MacUser",
        mac_user,
        "-RemoteRepoPath",
        remote_repo_path,
        "-PipelineDate",
        resolved_date,
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=900,
        cwd=Path.cwd(),
    )
    output = "\n".join(
        part.strip()
        for part in (completed.stdout, completed.stderr)
        if part and part.strip()
    ).strip()
    if completed.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=output or "Mac sync failed.",
        )
    return MacSyncResponse(
        ok=True,
        message="Pulled latest Mac data and refreshed local slate.",
        output=output or None,
    )
