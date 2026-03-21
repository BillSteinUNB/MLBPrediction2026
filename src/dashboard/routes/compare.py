"""Comparison endpoint for run analysis."""

from __future__ import annotations

from pathlib import Path

from fastapi import Depends, HTTPException

from src.dashboard.adapters import ExperimentDataAdapter
from src.dashboard.dependencies import get_models_dir
from src.dashboard.schemas import CompareResult


def create_compare_router():
    """Factory function to create compare router with dependencies."""
    from fastapi import APIRouter

    router = APIRouter()

    @router.get("/api/compare")
    def compare(
        run_a: str,
        run_b: str,
        models_dir: Path = Depends(get_models_dir),
    ) -> dict:
        """Compare two runs by their summary paths.

        Args:
            run_a: Summary path of first run (query param, relative or absolute)
            run_b: Summary path of second run (query param, relative or absolute)
            models_dir: Models directory (dependency)

        Returns:
            CompareResult with both RunSummary objects, metric deltas, same_lane flag

        Raises:
            HTTPException 404: If either run is not found
        """
        from src.dashboard.adapters import _normalize_path

        adapter = ExperimentDataAdapter(models_dir=models_dir)
        all_runs = adapter.get_all_runs(models_dir)

        # Resolve relative paths to absolute
        run_a_path = Path(str(run_a))
        run_b_path = Path(str(run_b))
        if not run_a_path.is_absolute():
            run_a_path = models_dir / run_a_path
        if not run_b_path.is_absolute():
            run_b_path = models_dir / run_b_path

        # Normalize paths for comparison
        run_a_normalized = _normalize_path(str(run_a_path))
        run_b_normalized = _normalize_path(str(run_b_path))

        # Find runs by normalized summary_path
        run_a_obj = next(
            (r for r in all_runs if r.summary_path == run_a_normalized),
            None,
        )
        run_b_obj = next(
            (r for r in all_runs if r.summary_path == run_b_normalized),
            None,
        )

        # 404 if either run not found
        if run_a_obj is None:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_a}")
        if run_b_obj is None:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_b}")

        # Compute comparison
        compare_result = adapter.compare_runs(run_a_obj, run_b_obj)

        # Check if runs share the same lane (holdout_season, target_column, variant)
        same_lane = (
            run_a_obj.holdout_season == run_b_obj.holdout_season
            and run_a_obj.target_column == run_b_obj.target_column
            and run_a_obj.variant == run_b_obj.variant
        )

        # Return as dict with extra same_lane field
        result_dict = compare_result.model_dump()
        result_dict["same_lane"] = same_lane

        return result_dict

    return router
