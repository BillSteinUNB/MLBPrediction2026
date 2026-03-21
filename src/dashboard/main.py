from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.dashboard.dependencies import get_experiments_dir, get_models_dir

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
