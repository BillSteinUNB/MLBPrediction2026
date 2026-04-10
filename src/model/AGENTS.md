# src/model — ML Training & Inference

**21 Python files.** Model training, stacking, calibration, MCMC run-count, artifact loading.

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Train XGBoost base model | `xgboost_trainer.py` | CLI: `python -m src.model.xgboost_trainer --training-data <path>` |
| Stacking (LR on XGBoost probs) | `stacking.py` | CLI: `python -m src.model.stacking --training-data <path>` |
| Platt calibration | `calibration.py` | CLI: `python -m src.model.calibration --training-data <path>` |
| Build training DataFrame | `data_builder.py` | Assembles features + labels from raw data |
| Load saved artifacts at runtime | `artifact_runtime.py` | joblib-based loader for model objects |
| Run-count regression | `run_count_trainer.py` | Separate model for total runs prediction |
| Run distribution / MCMC | `mcmc_engine.py`, `mcmc_feature_builder.py`, `mcmc_pricing.py` | Bayesian run distribution modeling |
| Win classification | `win_classifier_trainer.py` | Binary win/loss classifier |
| Margin pricing | `margin_pricing.py`, `margin_trainer.py` | Run margin prediction |
| Direct run-line model | `direct_rl_trainer.py` | End-to-end RL trainer |
| Market recalibration | `market_recalibration.py` | Post-hoc market alignment |
| Model promotion | `promotion.py` | Promotes experiment to production |
| Run distribution analysis | `run_distribution_trainer.py`, `run_distribution_metrics.py` | Run total distribution fitting |
| Run research features | `run_research_features.py` | Experimental features for run-count |
| Score pricing | `score_pricing.py` | Exact score probability estimation |
| Single model profiles | `single_model_profiles.py` | Per-model performance profiles |

## TRAINING FLOW

```
data_builder.py → xgboost_trainer.py → stacking.py → calibration.py
                                                          ↓
                                                 artifact_runtime.py (loading)
```

1. `data_builder.py` assembles features + labels → parquet
2. `xgboost_trainer.py` trains base model → saves to `data/models/<experiment>/`
3. `stacking.py` trains LR stacker on XGBoost probabilities + baselines
4. `calibration.py` applies Platt scaling on 10% holdout
5. Quality gates: Brier < 0.25, ECE < 0.05

## ARTIFACTS

- Saved to `data/models/<experiment_name>/` (joblib format)
- Loaded at inference by `artifact_runtime.py`
- Experiment metadata tracked by `src/ops/experiment_tracker.py`

## CONVENTIONS

- Every trainer has a `main()` with argparse CLI (`python -m src.model.<module>`)
- Training data path passed via `--training-data` flag
- Models saved via joblib, not pickle
- Temporal cross-validation (no random split — respects date ordering)
- Anti-leakage: features use only data available before game time
