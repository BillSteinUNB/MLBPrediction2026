"""Overview endpoint for dashboard."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends

from src.dashboard.adapters import ExperimentDataAdapter
from src.dashboard.dependencies import get_experiments_dir, get_models_dir
from src.dashboard.schemas import OverviewResponse

router = APIRouter(prefix="/api", tags=["overview"])


@router.get("/overview", response_model=OverviewResponse)
def get_overview(
    models_dir: Path = Depends(get_models_dir),
    experiments_dir: Path = Depends(get_experiments_dir),
) -> OverviewResponse:
    """Get dashboard overview with key metrics.

    Returns:
        OverviewResponse containing:
        - total_runs: Count of all runs
        - active_lanes: Count of unique model/variant combinations
        - best_run: Run with best roc_auc overall
        - latest_run: Most recent run by timestamp
        - recent_runs: Top improvements and regressions (up to 10 combined)
    """
    adapter = ExperimentDataAdapter(models_dir=models_dir, experiments_dir=experiments_dir)
    runs = adapter.get_all_runs(models_dir=models_dir)
    return adapter.get_overview(runs)
