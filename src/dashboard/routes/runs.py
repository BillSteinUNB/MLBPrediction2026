"""Routes for run endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from src.dashboard.adapters import ExperimentDataAdapter
from src.dashboard.dependencies import get_models_dir
from src.dashboard.schemas import RunDetail, RunSummary

router = APIRouter(prefix="/api/runs", tags=["runs"])


def get_adapter(models_dir: Path = Depends(get_models_dir)) -> ExperimentDataAdapter:
    """Dependency: Get data adapter instance."""
    return ExperimentDataAdapter(models_dir=models_dir)


@router.get("")
async def list_runs(
    holdout_season: Optional[int] = Query(None, description="Filter by holdout season"),
    target_column: Optional[str] = Query(None, description="Filter by target column"),
    variant: Optional[str] = Query(None, description="Filter by variant"),
    adapter: ExperimentDataAdapter = Depends(get_adapter),
    models_dir: Path = Depends(get_models_dir),
) -> list[RunSummary]:
    """Get list of all runs with optional filtering.

    Args:
        holdout_season: Optional filter by holdout season (e.g., 2024)
        target_column: Optional filter by target column (e.g., f5_ml_result)
        variant: Optional filter by variant (e.g., base)
        adapter: Data adapter instance
        models_dir: Models directory path

    Returns:
        List of RunSummary objects matching filters
    """
    # Get all runs from adapter
    all_runs = adapter.get_all_runs(models_dir)

    # Apply filters in-memory
    filtered_runs = all_runs

    if holdout_season is not None:
        filtered_runs = [r for r in filtered_runs if r.holdout_season == holdout_season]

    if target_column is not None:
        filtered_runs = [r for r in filtered_runs if r.target_column == target_column]

    if variant is not None:
        filtered_runs = [r for r in filtered_runs if r.variant == variant]

    return filtered_runs


@router.get("/detail")
async def get_run_detail(
    summary_path: str = Query(..., description="Path to run summary JSON file"),
    adapter: ExperimentDataAdapter = Depends(get_adapter),
    models_dir: Path = Depends(get_models_dir),
) -> RunDetail:
    """Get detailed run information for a specific run.

    Args:
        summary_path: Path to the run summary JSON file
        adapter: Data adapter instance
        models_dir: Models directory path

    Returns:
        RunDetail with feature importance, reliability diagram, quality gates, etc.

    Raises:
        HTTPException(404): If summary_path not found
    """
    detail = adapter.get_run_detail(models_dir, summary_path)
    if detail is None:
        raise HTTPException(
            status_code=404,
            detail=f"Run detail not found for summary_path: {summary_path}",
        )
    return detail
