param(
    [int]$FeatureWorkers = 10
)

$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

$env:MLB_FEATURE_BUILD_WORKERS = $FeatureWorkers.ToString()

$code = @'
from pathlib import Path
from src.model.data_builder import build_training_dataset

result = build_training_dataset(
    start_year=2018,
    end_year=2025,
    output_path=Path(r"data/training/WorkingParquetModel-1.parquet"),
    refresh=True,
    refresh_raw_data=True,
)

print(f"Done. Rows={len(result.dataframe)}")
print(f"Parquet={result.output_path}")
print(f"Metadata={result.metadata_path}")
print(f"Hash={result.data_version_hash}")
'@

$code | & .\.venv\Scripts\python.exe -
