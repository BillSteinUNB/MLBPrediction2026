from __future__ import annotations

import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def temp_data_dir() -> Generator[Path, None, None]:
    """Fixture providing a temporary data directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def models_dir(temp_data_dir: Path) -> Path:
    """Fixture providing a models directory."""
    models_path = temp_data_dir / "models"
    models_path.mkdir(parents=True, exist_ok=True)
    return models_path


@pytest.fixture
def experiments_dir(temp_data_dir: Path) -> Path:
    """Fixture providing an experiments directory."""
    experiments_path = temp_data_dir / "experiments"
    experiments_path.mkdir(parents=True, exist_ok=True)
    return experiments_path


@pytest.fixture(autouse=True)
def set_env_vars(models_dir: Path, experiments_dir: Path) -> Generator[None, None, None]:
    """Set environment variables for dashboard settings."""
    os.environ["DASHBOARD_MODELS_DIR"] = str(models_dir)
    os.environ["DASHBOARD_EXPERIMENTS_DIR"] = str(experiments_dir)

    # Reset module-level cache to force reload with new env vars
    import src.dashboard.dependencies

    src.dashboard.dependencies._settings = None

    yield

    # Cleanup
    if "DASHBOARD_MODELS_DIR" in os.environ:
        del os.environ["DASHBOARD_MODELS_DIR"]
    if "DASHBOARD_EXPERIMENTS_DIR" in os.environ:
        del os.environ["DASHBOARD_EXPERIMENTS_DIR"]

    # Reset cache again after test
    src.dashboard.dependencies._settings = None


@pytest.fixture
def client() -> TestClient:
    """Fixture providing a FastAPI test client."""
    # Import here to get fresh app with env vars set
    from src.dashboard.main import app

    return TestClient(app)
