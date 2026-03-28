from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

RUFF_TARGETS = (
    "src/model/run_count_trainer.py",
    "src/model/data_builder.py",
    "src/model/single_model_profiles.py",
    "scripts/build_parquet.py",
    "scripts/train_run_count.py",
    "scripts/validate_modeling.py",
    "tests/test_run_count_trainer.py",
    "tests/test_data_builder.py",
    "tests/test_single_model_profiles.py",
)

FAST_PYTEST_TARGETS = (
    "tests/test_run_count_trainer.py",
    "tests/test_single_model_profiles.py",
    "tests/test_data_builder.py::test_derive_matchup_interaction_features_uses_lineup_and_starter_quality",
    "tests/test_data_builder.py::test_derive_temporal_delta_features_computes_lineup_and_starter_trends",
    "tests/test_data_builder.py::test_fill_missing_feature_values_uses_module_defaults_instead_of_dataset_means",
)

FULL_PYTEST_TARGETS = (
    "tests/test_run_count_trainer.py",
    "tests/test_data_builder.py",
    "tests/test_single_model_profiles.py",
)


def _run(command: list[str], *, label: str) -> None:
    started_at = perf_counter()
    print(f"\n[{label}] {' '.join(command)}")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    elapsed = perf_counter() - started_at
    print(f"[{label}] exit={completed.returncode} elapsed={elapsed:.1f}s")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run modeling validators with fast/full profiles")
    parser.add_argument(
        "--profile",
        choices=("fast", "full"),
        default="fast",
        help="Validation profile to run",
    )
    parser.add_argument(
        "--skip-lint",
        action="store_true",
        help="Skip Ruff and run pytest only",
    )
    parser.add_argument(
        "--tests-only",
        action="store_true",
        help="Alias for --skip-lint",
    )
    args = parser.parse_args(argv)

    resolved_python = PYTHON if PYTHON.exists() else Path(sys.executable)
    pytest_targets = FAST_PYTEST_TARGETS if args.profile == "fast" else FULL_PYTEST_TARGETS
    skip_lint = args.skip_lint or args.tests_only

    if not skip_lint:
        _run(
            [str(resolved_python), "-m", "ruff", "check", *RUFF_TARGETS],
            label=f"{args.profile}:lint",
        )

    _run(
        [str(resolved_python), "-m", "pytest", "-x", *pytest_targets],
        label=f"{args.profile}:pytest",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
