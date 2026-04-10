# src/ops — Operational Tooling

**15 Python files.** Error handling, logging, experiment tracking, AutoResearch ops, reporting.

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| CircuitBreaker, retry, graceful degradation | `error_handler.py` | Core error utilities used by all clients + pipeline |
| Logging configuration | `logging_config.py` | configure_logging() with sentinel-marked handlers |
| Experiment metadata | `experiment_tracker.py` | Track model runs, hyperparams, metrics |
| Experiment reports | `experiment_report.py` | Generate experiment comparison reports |
| Performance tracking | `performance_tracker.py` | Track betting performance over time |
| Live season tracking | `live_season_tracker.py` | Current season result tracking |
| AutoResearch ops | `autoresearch.py` | CLI for overnight research automation |
| Edge bucket analysis | `edge_bucket_report.py` | Bucket-level edge performance |
| Feature drift detection | `feature_drift_report.py` | Monitor feature distribution changes |
| Market strategy sweep | `market_strategy_sweep.py` | Parameter sweep for market strategies |
| Model regime scanning | `model_regime_scan.py` | Detect model performance regimes |
| RL v2 evaluation | `rl_v2_evaluator.py` | Run-line model evaluation |
| Run count dual view | `run_count_dual_view.py` | Dual-model run count comparison |
| Run count walk-forward | `run_count_walk_forward.py` | Walk-forward validation for run counts |

## ERROR HANDLING PATTERN

```
CircuitBreaker (stateful, per-service)
    ↕
retry decorator (exponential backoff)
    ↕
call_with_graceful_degradation (fallback values)
    ↕
notify_fatal_error (Discord alert on pipeline death)
```

- Pipeline uses named circuit breakers: `_SCHEDULE_CIRCUIT`, `_HISTORY_CIRCUIT`, `_LINEUPS_CIRCUIT`, `_ODDS_CIRCUIT`
- All external fetches must be wrapped

## LOGGING

- Call `configure_logging()` at process start
- Uses sentinel `_HANDLER_SENTINEL` to avoid duplicate handlers
- `logging.getLogger(__name__)` per module with %-format strings
- `exc_info=True` for error-level logs where traceback needed

## NOTES

- `autoresearch.py` has `__main__` guard for CLI invocation
- Experiment artifacts stored under `data/autoresearch/`
- No CI/CD — ops scripts run manually or via scheduler
