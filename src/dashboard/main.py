from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.dashboard.dependencies import get_experiments_dir, get_models_dir
from src.dashboard.routes import lanes_router
from src.dashboard.routes.compare import create_compare_router
from src.dashboard.routes.live_season import router as live_season_router
from src.dashboard.routes.overview import router as overview_router
from src.dashboard.routes.promotions import router as promotions_router
from src.dashboard.routes.runs import router as runs_router
from src.dashboard.routes.slate import router as slate_router

app = FastAPI(
    title="MLB Prediction Dashboard API",
    version="0.1.0",
    description="REST API for MLB prediction dashboard",
)

# CORS middleware: localhost origins only
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Include routers
app.include_router(overview_router)
app.include_router(lanes_router)
app.include_router(runs_router)
app.include_router(promotions_router)
app.include_router(create_compare_router())
app.include_router(slate_router)
app.include_router(live_season_router)


def _count_experiment_rows(experiments_dir: Path) -> int:
    """Count rows in experiment_metrics.csv if it exists."""
    metrics_file = experiments_dir / "experiment_metrics.csv"
    if not metrics_file.exists():
        return 0

    try:
        with open(metrics_file, encoding="utf-8") as f:
            # Skip header row, count remaining lines
            return max(0, len(f.readlines()) - 1)
    except Exception:
        return 0


@app.get("/api/health")
def health(
    models_dir: Path = Depends(get_models_dir),
    experiments_dir: Path = Depends(get_experiments_dir),
) -> dict:
    """Health check endpoint.

    Returns:
        - status: "ok" if service is running
        - data_available: True if models directory exists
        - run_count: Number of completed experiment runs
    """
    data_available = models_dir.exists()
    run_count = _count_experiment_rows(experiments_dir)

    return {
        "status": "ok",
        "data_available": data_available,
        "run_count": run_count,
    }
