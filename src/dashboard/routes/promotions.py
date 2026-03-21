"""Routes for promotion endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from src.dashboard.adapters import ExperimentDataAdapter
from src.dashboard.dependencies import get_experiments_dir, get_models_dir
from src.dashboard.schemas import Promotion, PromotionRequest, RunSummary

router = APIRouter(prefix="/api/promotions", tags=["promotions"])


def get_adapter(
    models_dir: Path = Depends(get_models_dir),
    experiments_dir: Path = Depends(get_experiments_dir),
) -> ExperimentDataAdapter:
    """Dependency: Get data adapter instance."""
    return ExperimentDataAdapter(models_dir=models_dir, experiments_dir=experiments_dir)


def _run_exists(run_id: str, models_dir: Path, adapter: ExperimentDataAdapter) -> bool:
    """Check if a run exists by summary_path."""
    all_runs = adapter.get_all_runs(models_dir)
    return any(run.summary_path == run_id for run in all_runs)


@router.get("")
async def list_promotions(
    adapter: ExperimentDataAdapter = Depends(get_adapter),
    experiments_dir: Path = Depends(get_experiments_dir),
) -> list[Promotion]:
    """Get list of all promotions.

    Returns:
        List of Promotion objects. Empty list if promotions.json file missing.
    """
    return adapter.read_promotions(experiments_dir)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_promotion(
    request: PromotionRequest,
    adapter: ExperimentDataAdapter = Depends(get_adapter),
    models_dir: Path = Depends(get_models_dir),
    experiments_dir: Path = Depends(get_experiments_dir),
) -> Promotion:
    """Create a new promotion for a run.

    Args:
        request: PromotionRequest containing run_id, target_stage, and optional reason
        adapter: Data adapter instance
        models_dir: Models directory path
        experiments_dir: Experiments directory path

    Returns:
        Created Promotion object with generated promotion_id and timestamp

    Raises:
        HTTPException 404: If the run_id does not exist
    """
    # Validate that the run exists
    if not _run_exists(request.run_id, models_dir, adapter):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run with id '{request.run_id}' not found",
        )

    # Create promotion record
    promotion = Promotion(
        promotion_id=str(uuid.uuid4()),
        run_id=request.run_id,
        from_stage="development",  # Default from_stage
        to_stage=request.target_stage,
        promoted_timestamp=datetime.now(timezone.utc).isoformat(),
        metadata={"reason": request.reason} if request.reason else None,
    )

    # Write to adapter
    created = adapter.write_promotion(experiments_dir, promotion)

    return created
