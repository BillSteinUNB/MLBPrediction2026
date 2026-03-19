## Isotonic calibration worker notes

- `src/model/calibration.py` now trains dedicated 10% chronological calibration splits before the 2025 holdout and saves combined `CalibratedStackingModel` joblib bundles.
- CLI-created bundles needed a pickle stability fix: `CalibratedStackingModel.__module__` is forced to `src.model.calibration` so `python -m src.model.calibration` artifacts can be loaded later.
- `.factory/services.yaml` command `pipeline.calibrate_eval` now points at `python -m src.model.calibration`.
- Real-data verification on `data/training/training_data_2019_2025.parquet` with `--search-iterations 1 --time-series-splits 3` produced:
  - F5 ML calibrated Brier `0.2479`, ECE `0.0316`, max reliability gap `0.1548`
  - F5 RL calibrated Brier `0.2179`, ECE `0.0159`, max reliability gap `0.1738`
- A fuller search (`search_iterations=15`, `time_series_splits=5`) did not fix the gap and slightly worsened ML Brier to `0.2504`.
- Current blocker for the milestone is calibration curve quality on real data, not the persistence/inference implementation.
