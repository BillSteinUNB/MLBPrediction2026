# Remaining Work

**Mission ID:** `83c0c194-72d1-4821-8b08-68a3497c3590`
**Generated:** 2026-03-20
**Status:** 57/62 assertions passed (92%)

This document tracks all incomplete items from the MLB F5 Betting Prediction System mission.

---

## Pending Validation Assertions

### VAL-AUTO-001: Scheduler task creation
**Status:** `pending`
**Milestone:** automation-testing
**Description:** Windows Task Scheduler task created successfully with correct schedule and command.
**Evidence Required:** schtasks query output showing task exists
**Blocking Issues:**
- Workers timed out before fix validation could complete

### VAL-AUTO-002: Environment validation
**Status:** `pending`
**Milestone:** automation-testing
**Description:** `.env` setup script validates API keys with test requests before saving.
**Evidence Required:** Terminal output showing validation success/failure per key
**Note:** Implementation exists in `scripts/setup_env.py`, needs user-testing validation

### VAL-AUTO-003: Retry with exponential backoff
**Status:** `pending`
**Milestone:** automation-testing
**Description:** API calls retry up to 3 times with 2/4/8 second delays before failing.
**Evidence Required:** Log output showing retry attempts and timing
**Blocking Issues:**
- Circuit breaker wraps already-retrying fetch helpers (fix-t31 needed)

### VAL-AUTO-004: Circuit breaker activation
**Status:** `pending`
**Milestone:** automation-testing
**Description:** After 5 consecutive API failures, circuit breaker activates and prevents further attempts.
**Evidence Required:** Log output showing circuit breaker activation
**Blocking Issues:**
- Circuit breaker trips after 5 exhausted retry batches instead of 5 consecutive failures (fix-t31 needed)

### VAL-AUTO-005: Log rotation
**Status:** `pending`
**Milestone:** automation-testing
**Description:** Log files rotate daily with 30-day retention.
**Evidence Required:** File listing showing dated log files
**Note:** Implementation exists in `src/ops/logging_config.py`, needs user-testing validation

### VAL-CROSS-004: Backtest to production consistency
**Status:** `pending` (deferred)
**Milestone:** cross-area
**Description:** Features computed during backtest use same logic as production daily pipeline, ensuring training and inference parity.
**Evidence Required:** pytest test comparing feature values from both paths
**Reason Deferred:** Requires both `src/backtest/run.py` and `src/pipeline/daily.py` CLI entrypoints to be validated together. Currently blocked by automation-testing fixes.

---

## Pending Fix Features

### fix-t28-real-training-tests
**Status:** `pending`
**Milestone:** automation-testing
**Skill:** pipeline-worker
**Preconditions:** t28-data-integrity-tests
**Issue:** Integrity/anti-leakage tests validate synthetic fixture, not real training-data builder or cached dataset.
**Required Fix:** Point tests at actual cached training artifact or builder path for real regression coverage.
**Files Affected:**
- `tests/pipeline/test_data_integrity.py`
- `tests/model/test_antileak.py`

### fix-t30-clv-bookmaker-refresh
**Status:** `pending`
**Milestone:** automation-testing
**Skill:** pipeline-worker
**Preconditions:** t30-performance-tracker
**Issue:** CLV uses wrong sportsbook closing line (no book_name stored) and CLV doesn't refresh when later closing snapshot arrives.
**Required Fix:** 
1. Persist bookmaker identity with each tracked bet/performance row
2. Add automatic close-line refresh flow that reruns CLV sync after authoritative closing snapshot is stored
**Files Affected:**
- `src/ops/performance_tracker.py`
- `src/engine/bankroll.py` (placement flow)
- `tests/ops/test_performance_tracker.py`

### fix-t31-breaker-retry-composition
**Status:** `pending`
**Milestone:** automation-testing
**Skill:** pipeline-worker
**Preconditions:** t31-error-handling
**Issue:** Circuit breaker wraps already-retrying fetch helpers, trips after 5 exhausted retry batches instead of 5 consecutive failures.
**Required Fix:** Recompose resilience stack so breaker sees each underlying failed attempt (or increments per raw failure) before retry batching completes.
**Files Affected:**
- `src/ops/error_handler.py`
- `src/pipeline/daily.py` (fetch wrappers)
- `tests/ops/test_error_handler.py`
- `tests/pipeline/test_pipeline_error_handling.py`

---

## Non-Blocking Issues (Documented but Not Fixed)

### init.sh / init_windows PowerShell Quoting
**Severity:** non-blocking
**Issue:** `.factory/services.yaml` `init_windows` command uses malformed quoting (`init_db(''data/mlb.db'')`) causing SyntaxError.
**Workaround:** Workers use direct `.venv\Scripts\python.exe` invocations on Windows.
**Suggested Fix:** Change to `init_db('data/mlb.db')` or add a tested Windows-safe command.

### Double-Persisted Odds Snapshots
**Severity:** non-blocking
**Issue:** Production runs persist fetched odds twice (in `fetch_mlb_odds()` and `_persist_odds_snapshots()`).
**Location:** `src/pipeline/daily.py`
**Suggested Fix:** Make persistence responsibility of only one layer.

### Settle-Without-Place Bankroll Inflation
**Severity:** non-blocking
**Issue:** `update_bankroll(action='settle', ...)` can create settled bet without matching pending bet, potentially inflating bankroll if misused.
**Location:** `src/engine/bankroll.py`
**Suggested Fix:** Reject settle-without-place calls or require matching pending bet.

### pybaseball team_game_logs Pandas 3 Compatibility
**Severity:** non-blocking
**Issue:** `pybaseball.team_game_logs()` fails with pandas 3 due to `errors="ignore"` deprecation.
**Workaround:** Training data builder avoids this API, uses cached/alternative sources.
**Suggested Fix:** Patch upstream or pin pandas version.

### ResourceWarning for Unclosed SQLite Connections
**Severity:** non-blocking
**Issue:** Test suite emits ResourceWarning for unclosed sqlite3.Connection objects.
**Suggested Fix:** Audit helpers/modules to use context managers for connections.

---

## How to Resume

1. **Read mission context:**
   - `{missionDir}/validation-contract.md` - All assertions
   - `{missionDir}/validation-state.json` - Current assertion status
   - `{missionDir}/features.json` - Feature completion status

2. **Resume automation-testing fixes:**
   ```
   # Start mission runner
   droid mission resume 83c0c194-72d1-4821-8b08-68a3497c3590
   ```

3. **Run remaining fixes manually (if workers timeout):**
   ```bash
   # Fix t28: Real training tests
   pytest tests/pipeline/test_data_integrity.py tests/model/test_antileak.py -v
   
   # Fix t30: CLV bookmaker
   pytest tests/ops/test_performance_tracker.py -v
   
   # Fix t31: Breaker/retry composition  
   pytest tests/ops/test_error_handler.py tests/pipeline/test_pipeline_error_handling.py -v
   ```

4. **Validate automation assertions:**
   ```bash
   # Scheduler
   python scripts/setup_scheduler.py --test
   
   # Environment validation
   python scripts/setup_env.py
   
   # Log rotation
   ls -la archive/logs/
   ```

---

## Summary Statistics

| Category | Count |
|----------|-------|
| Total Assertions | 62 |
| Passed | 57 |
| Pending | 5 |
| Deferred | 1 |
| Blocking Fixes | 3 |
| Non-Blocking Issues | 5 |
| Total Tests | 210+ |
| Test Coverage | 84% |

---

## Files to Update After Fixes

After completing the pending fixes, update these files:

1. **`validation-state.json`** - Mark VAL-AUTO-* assertions as passed
2. **`.sisyphus/plans/mlb-prediction-2026.md`** - Update milestone status
3. **`README.md`** - Remove "Known Limitations" or move to "Resolved"
4. **This file** - Archive or remove once all items addressed
