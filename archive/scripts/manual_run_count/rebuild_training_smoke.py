"""
Build the training parquet used by the run-count experiments.

Usage:
    .\.venv\Scripts\python.exe scripts\rebuild_training_smoke.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.data_builder import build_training_dataset

START_YEAR = 2025
END_YEAR = 2025
OUTPUT_PATH = Path("data/training/training_data_2025_smoke.parquet")


def main() -> None:
    result = build_training_dataset(
        start_year=START_YEAR,
        end_year=END_YEAR,
        output_path=OUTPUT_PATH,
        refresh=False,
    )
    print(f"Done. Rows={len(result.dataframe)} Output={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
