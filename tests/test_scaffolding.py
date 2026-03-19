from __future__ import annotations

from importlib import import_module
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_required_scaffold_files_exist() -> None:
    for relative_path in ["pyproject.toml", ".gitignore", ".env.example", "src/__init__.py"]:
        assert (REPO_ROOT / relative_path).exists(), f"Missing required scaffold file: {relative_path}"


def test_env_example_contains_required_placeholders() -> None:
    env_contents = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")

    for key in ["ODDS_API_KEY", "OPENWEATHER_API_KEY", "DISCORD_WEBHOOK_URL"]:
        assert key in env_contents, f"Missing placeholder for {key}"


def test_gitignore_contains_required_entries() -> None:
    gitignore_contents = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")

    for entry in [".env", "data/*.db", "__pycache__/"]:
        assert entry in gitignore_contents, f"Missing gitignore entry: {entry}"


def test_src_package_importable() -> None:
    module = import_module("src")

    assert module is not None
