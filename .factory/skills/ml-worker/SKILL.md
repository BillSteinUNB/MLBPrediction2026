---
name: ml-worker
description: Handles ML pipeline including training data builder, model training, stacking, calibration, and backtesting
---

# ML Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Use this worker for features involving:
- Historical training data builder with anti-leakage enforcement
- XGBoost model training with hyperparameter tuning
- Logistic regression stacking meta-learner
- Isotonic calibration layer
- Walk-forward backtesting framework

## Work Procedure

1. **Read existing context**: Check `src/db.py` for data access, `src/features/` for feature modules, `.factory/library/` for model patterns, `config/settings.yaml` for training parameters.

2. **Write tests first (TDD)**:
   - Anti-leakage tests: verify training data has no future data
   - Model tests: verify TimeSeriesSplit usage, not random k-fold
   - Calibration tests: verify Brier score threshold (< 0.25), ECE (< 0.05)
   - Backtest tests: verify reproducibility (same inputs → same outputs)
   - Run tests to confirm they FAIL (red phase)

3. **Implement**:
   - Training data: ~17,000 rows, no NaN in targets, no spring training/postseason
   - Use TimeSeriesSplit for temporal cross-validation (respect time order)
   - Stacking: use out-of-fold predictions (cross_val_predict) to prevent leakage
   - Calibration: use dedicated holdout set (not training data)
   - Backtest: 6-month train / 1-month test windows, slide monthly

4. **Verify tests pass (green phase)**:
   - Run `pytest tests/test_model*.py tests/test_backtest*.py -v --tb=short`
   - All calibration quality gates must pass

5. **Manual verification**:
   - Train model on subset of data, verify accuracy > 50%
   - Run short backtest, verify Brier < 0.25
   - Check reproducibility: run twice, compare outputs

6. **Run linters**:
   - `python -m py_compile src/model/*.py src/backtest/*.py`

## Example Handoff

```json
{
  "salientSummary": "Implemented XGBoost F5 ML/RL classifiers with TimeSeriesSplit cross-validation, hyperparameter tuning, and model serialization. Models achieve 58% accuracy on holdout. All 8 tests pass.",
  "whatWasImplemented": "Created src/model/xgboost_trainer.py with train_f5_models() returning fitted XGBClassifier instances. Hyperparameter search over max_depth [3-8], n_estimators [100-500], learning_rate [0.01-0.1]. TimeSeriesSplit(n_splits=5) used for CV. Models saved to data/models/f5_ml_model_v1.joblib and f5_rl_model_v1.joblib. Feature importance extracted.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {
        "command": "pytest tests/test_xgboost_trainer.py -v",
        "exitCode": 0,
        "observation": "8 tests passed: temporal CV, model training, serialization, metrics"
      },
      {
        "command": "python -c \"import joblib; m = joblib.load('data/models/f5_ml_model_v1.joblib'); print(type(m).__name__)\"",
        "exitCode": 0,
        "observation": "XGBClassifier loaded successfully"
      }
    ],
    "interactiveChecks": [
      {
        "action": "Verified TimeSeriesSplit used instead of random k-fold",
        "observed": "Code shows TimeSeriesSplit import and usage in GridSearchCV"
      }
    ]
  },
  "tests": {
    "added": [
      {
        "file": "tests/test_xgboost_trainer.py",
        "cases": [
          { "name": "test_temporal_cross_validation", "verifies": "VAL-MODEL-001" },
          { "name": "test_both_models_trained", "verifies": "VAL-MODEL-002" }
        ]
      }
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- Training data not available (features not computed)
- Feature columns expected but missing from dataset
- Model performance significantly below expectations (< 50% accuracy)
- Calibration fails to meet thresholds despite tuning
- Backtest reveals data quality issues
