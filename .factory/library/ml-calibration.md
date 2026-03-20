## ML calibration notes

- `src/model/calibration.py` now supports `identity`, `isotonic`, `platt`, and `blend`
  probability calibrators on top of the stacking bundle. The current default is
  `DEFAULT_CALIBRATION_METHOD = "platt"`.
- CLI-created calibration bundles pin `sys.modules.setdefault("src.model.calibration", ...)`
  and set the custom calibrator/model classes' `__module__` to `src.model.calibration`
  so `python -m src.model.calibration` artifacts remain reloadable via `joblib` in a
  fresh interpreter.
- `.factory/services.yaml` command `pipeline.calibrate_eval` points at
  `python -m src.model.calibration`.
- Verified 2025 holdout metrics on `data/training/training_data_2019_2025.parquet` with
  the default CLI path passed all milestone gates:
  - F5 ML: Brier `0.24643230692917847`, ECE `0.016723177235313788`, max reliability gap `0.022495723306924953`
  - F5 RL: Brier `0.21759412185983892`, ECE `0.016525026496998555`, max reliability gap `0.03153923322817026`
- Current known caveat: if a single-class calibration slice forces an internal fallback to
  `IdentityProbabilityCalibrator`, the run-level JSON/CLI summary still reports the
  requested method instead of the fallback actually shipped.
