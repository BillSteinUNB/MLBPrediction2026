"""Lane grouping endpoint for dashboard."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Query

from src.dashboard.adapters import ExperimentDataAdapter
from src.dashboard.dependencies import get_experiments_dir, get_models_dir
from src.dashboard.schemas import Lane, RunSummary

router = APIRouter(prefix="/api/lanes", tags=["lanes"])


@router.get("")
def get_lanes(
    models_dir: Path = Depends(get_models_dir),
    experiments_dir: Path = Depends(get_experiments_dir),
) -> list[Lane]:
    """Get lane groupings.

    Returns a list of lanes, where each lane groups runs by (holdout_season, target_column, variant).
    Each lane contains:
    - lane_id: unique identifier for the lane
    - model_name: name of the model from the latest run in the lane
    - variant: variant identifier
    - best_run: the best performing run in the lane (by ROC-AUC, accuracy, log-loss, then brier)
    - latest_run: the most recent run in the lane (by timestamp)

    Args:
        models_dir: Directory containing model artifacts (from dependency injection)
        experiments_dir: Directory containing experiment metadata (from dependency injection)

    Returns:
        List of Lane objects, sorted by lane_id.
    """
    adapter = ExperimentDataAdapter(models_dir=models_dir, experiments_dir=experiments_dir)
    runs = adapter.get_all_runs(models_dir=models_dir)
    lanes = adapter.get_lanes(runs)
    return lanes


@router.get("/{lane_id}/runs", response_model=list[RunSummary])
def get_lane_runs(
    lane_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    models_dir: Path = Depends(get_models_dir),
    experiments_dir: Path = Depends(get_experiments_dir),
) -> list[RunSummary]:
    adapter = ExperimentDataAdapter(models_dir=models_dir, experiments_dir=experiments_dir)
    runs = adapter.get_all_runs(models_dir=models_dir)
    return adapter.get_lane_runs(runs, lane_id=lane_id, skip=skip, limit=limit)
