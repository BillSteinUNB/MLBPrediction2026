# Archive

This directory holds superseded manual-training wrappers that were kept for reference instead of being deleted.

Current archived manual run-count wrappers:
- `scripts/manual_run_count/train_single_model.py`
- `scripts/manual_run_count/rebuild_and_train_single_model.py`
- `scripts/manual_run_count/rebuild_training_smoke.py`
- `scripts/manual_run_count/build_working_parquet_model_1.ps1`

The current manual workflow is:
- build parquet: [`scripts/build_parquet.py`](/C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/scripts/build_parquet.py)
- train from existing parquet: [`scripts/train_run_count.py`](/C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/scripts/train_run_count.py)
