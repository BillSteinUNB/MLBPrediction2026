from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def test_health_with_data(client: TestClient, models_dir: Path, experiments_dir: Path) -> None:
    """Test health endpoint when data is present."""
    # Create CSV with header + 2 data rows
    metrics_file = experiments_dir / "experiment_metrics.csv"
    metrics_file.write_text("epoch,loss,accuracy\n1,0.5,0.8\n2,0.3,0.9\n")

    response = client.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["data_available"] is True
    assert data["run_count"] == 2


def test_health_without_data(client: TestClient, models_dir: Path, experiments_dir: Path) -> None:
    """Test health endpoint when no data is present."""
    # Don't create any CSV file
    response = client.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["data_available"] is True  # models_dir exists from fixture
    assert data["run_count"] == 0


def test_health_without_models_dir(client: TestClient, temp_data_dir: Path) -> None:
    """Test health endpoint when models directory doesn't exist."""
    import os

    # Create a dir without models_dir
    experiments_path = temp_data_dir / "experiments"
    experiments_path.mkdir(parents=True, exist_ok=True)

    # Override env vars to point to non-existent models dir
    os.environ["DASHBOARD_MODELS_DIR"] = str(temp_data_dir / "nonexistent")
    os.environ["DASHBOARD_EXPERIMENTS_DIR"] = str(experiments_path)

    # Need fresh app instance with new env
    from src.dashboard.dependencies import DashboardSettings

    DashboardSettings.model_config["env_prefix"] = "DASHBOARD_"

    # Reimport to get fresh settings
    import importlib

    import src.dashboard.dependencies

    importlib.reload(src.dashboard.dependencies)
    import src.dashboard.main

    importlib.reload(src.dashboard.main)
    from src.dashboard.main import app

    client = TestClient(app)

    response = client.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["data_available"] is False
    assert data["run_count"] == 0
