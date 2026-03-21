"""Pytest configuration for dashboard E2E tests.

Provides fixtures to start backend (FastAPI) and frontend (Vite) servers
for Playwright smoke tests.
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Generator

import pytest
import requests
from playwright.sync_api import Browser, BrowserContext, Page


@pytest.fixture(scope="session")
def live_server() -> Generator[None, None, None]:
    """
    Start uvicorn backend (port 8000) and npm dev frontend (port 5173).
    Wait for health check, yield, then tear down.
    """
    # Resolve dashboard directory as absolute path
    project_root = Path(__file__).parent.parent.parent
    dashboard_dir = project_root / "dashboard"

    # Start backend: uvicorn on port 8000
    backend_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.dashboard.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(project_root),
    )

    # Start frontend: npm run dev on port 5173
    # Use shell=True on Windows to find npm.cmd
    frontend_proc = subprocess.Popen(
        "npm run dev",
        cwd=str(dashboard_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
    )

    # Wait for backend health check (max 30 seconds)
    backend_ready = False
    for _ in range(60):  # 60 * 0.5s = 30s
        try:
            resp = requests.get("http://localhost:8000/api/health", timeout=2)
            if resp.status_code == 200:
                backend_ready = True
                break
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(0.5)

    if not backend_ready:
        backend_proc.kill()
        frontend_proc.kill()
        pytest.fail("Backend health check failed after 30s")

    # Wait for frontend to be ready (max 30 seconds)
    frontend_ready = False
    for _ in range(60):
        try:
            resp = requests.get("http://localhost:5173", timeout=2)
            if resp.status_code == 200:
                frontend_ready = True
                break
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(0.5)

    if not frontend_ready:
        backend_proc.kill()
        frontend_proc.kill()
        pytest.fail("Frontend failed to start after 30s")

    # Both servers ready
    yield

    # Teardown
    backend_proc.terminate()
    frontend_proc.terminate()
    try:
        backend_proc.wait(timeout=5)
        frontend_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        backend_proc.kill()
        frontend_proc.kill()


@pytest.fixture(scope="function")
def context(browser: Browser, live_server: None) -> Generator[BrowserContext, None, None]:
    """
    Create a browser context for each test.
    Ensures servers are running via live_server dependency.
    """
    ctx = browser.new_context()
    yield ctx
    ctx.close()


@pytest.fixture(scope="function")
def page(context: BrowserContext) -> Generator[Page, None, None]:
    """
    Create a page for each test.
    """
    p = context.new_page()
    yield p
    p.close()
