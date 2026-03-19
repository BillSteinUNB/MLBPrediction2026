---
name: pipeline-worker
description: Handles daily pipeline orchestration, automation setup, testing, and operations
---

# Pipeline Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Use this worker for features involving:
- Daily pipeline orchestrator (ingest → predict → edge → notify)
- Cross-platform scheduler setup (Windows Task Scheduler + cron)
- pytest suite creation
- Performance tracker and CLV logging
- Error handling, retry logic, failure alerts

## Work Procedure

1. **Read existing context**: Check all `src/` modules for available components, `src/db.py` for storage, `.factory/library/` for patterns, `config/settings.yaml` for configuration.

2. **Write tests first (TDD)**:
   - Pipeline tests: verify end-to-end execution with dry-run
   - Missing data tests: verify graceful degradation
   - Scheduler tests: verify task creation
   - Error handling tests: verify retry and circuit breaker
   - Run tests to confirm they FAIL (red phase)

3. **Implement**:
   - Pipeline steps: ingest → validate → feature → predict → edge → size → store → notify → freeze
   - Missing data: explicit "NO PICK (reason)" row, never silent skip
   - Dry-run mode: print Discord payload without sending
   - Retry: exponential backoff (2/4/8 seconds), max 3 attempts
   - Circuit breaker: disable after 5 consecutive failures
   - Logging: daily rotation, 30-day retention

4. **Verify tests pass (green phase)**:
   - Run `pytest tests/ -v --tb=short`
   - All tests must pass

5. **Manual verification**:
   - Run `python -m src.pipeline.daily --date YYYY-MM-DD --mode backtest --dry-run`
   - Verify exit code 0, check SQLite for predictions
   - Test missing odds scenario
   - Test early-season scenario

6. **Run linters**:
   - `python -m py_compile src/pipeline/*.py src/ops/*.py`

## Example Handoff

```json
{
  "salientSummary": "Implemented daily pipeline orchestrator with 9 sequential steps, error handling per-game, and dry-run mode. Pipeline exits 0 with predictions stored in SQLite. All 10 tests pass.",
  "whatWasImplemented": "Created src/pipeline/daily.py with run_daily_pipeline() orchestrating: schedule fetch, lineup fetch, odds fetch, feature compute, model predict, edge calc, Kelly size, SQLite store, Discord notify. Each game processed independently - one failure doesn't crash pipeline. Dry-run mode prints JSON to console. Missing odds produces 'NO PICK (odds unavailable)'.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {
        "command": "pytest tests/test_pipeline.py -v",
        "exitCode": 0,
        "observation": "10 tests passed: end-to-end, missing data, dry-run, error isolation"
      },
      {
        "command": "python -m src.pipeline.daily --date 2025-09-15 --mode backtest --dry-run",
        "exitCode": 0,
        "observation": "Pipeline completed, JSON payload printed, SQLite updated"
      }
    ],
    "interactiveChecks": [
      {
        "action": "Verified missing odds scenario: removed odds data, ran pipeline",
        "observed": "Console showed 'NO PICK (odds unavailable)' for affected games, other games processed normally"
      }
    ]
  },
  "tests": {
    "added": [
      {
        "file": "tests/test_pipeline.py",
        "cases": [
          { "name": "test_pipeline_dry_run", "verifies": "VAL-PIPE-001" },
          { "name": "test_missing_odds_no_pick", "verifies": "VAL-PIPE-002" },
          { "name": "test_partial_success_on_error", "verifies": "VAL-PIPE-003" },
          { "name": "test_dry_run_no_webhook", "verifies": "VAL-PIPE-004" }
        ]
      }
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- Critical module not implemented (features, model, edge calculator)
- Database schema incompatible with pipeline needs
- Configuration missing required values
- Test reveals fundamental issue with pipeline design
