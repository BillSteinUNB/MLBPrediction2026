---
name: feature-worker
description: Handles multi-window rolling feature engineering, adjustments, and baselines with anti-leakage enforcement
---

# Feature Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Use this worker for features involving:
- Offensive rolling features (wRC+, wOBA, ISO, BABIP, K%, BB%)
- Pitching rolling features (xFIP, xERA, velocity, pitch mix entropy)
- Defense features (DRS, OAA, catcher framing with ABS depreciation)
- Bullpen fatigue features (PC L3/L5, rest days, IR%)
- Park factor and ABS zone adjustments
- Weather adjustment engine
- Pythagorean WP and Log5 baselines
- Marcel early-season blending

## Work Procedure

1. **Read existing context**: Check `src/db.py` for feature storage schema, `.factory/library/` for established patterns, `config/settings.yaml` for thresholds and windows.

2. **Write tests first (TDD)**:
   - Anti-leakage tests: verify rolling window excludes current game
   - Value range tests: verify reasonable bounds (wRC+ 50-200, xFIP 2.0-6.0)
   - Adjustment tests: verify ABS exception handling, park factor application
   - Formula tests: verify Pythagorean, Log5, Marcel blend math
   - Run tests to confirm they FAIL (red phase)

3. **Implement**:
   - Every feature row MUST have `as_of_timestamp` set to day BEFORE game
   - Rolling windows count by games/starts (not calendar days for pitchers)
   - Use Marcel blend when games_played < threshold (30 for offense, 15 for pitching)
   - ABS exceptions: Mexico City, Field of Dreams, Little League Classic
   - Sutter Health Park uses 2025 MLB factors (1.25 runs, 1.30 HR)

4. **Verify tests pass (green phase)**:
   - Run `pytest tests/test_feature_*.py -v --tb=short`
   - Anti-leakage tests must verify at least 100 random games

5. **Manual verification**:
   - Pick a specific team/date and manually compute rolling average
   - Verify feature value matches manual calculation
   - Check early-season date shows Marcel blend active

6. **Run linters**:
   - `python -m py_compile src/features/*.py`

## Example Handoff

```json
{
  "salientSummary": "Implemented offensive rolling features (wRC+, wOBA, ISO, BABIP, K%, BB%) over 7/14/30/60 game windows with Marcel blend for early-season. Anti-leakage enforced via as_of_timestamp. All 12 tests pass.",
  "whatWasImplemented": "Created src/features/offense.py with compute_offensive_features() returning dict per team per date. Multi-window rolling averages computed using pandas.rolling(). Marcel blend activates when games_played < 30. Lineup-weighted versions included. All features stored with as_of_timestamp.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {
        "command": "pytest tests/test_offense_features.py -v",
        "exitCode": 0,
        "observation": "12 tests passed: rolling windows, Marcel blend, anti-leakage, lineup weighting"
      },
      {
        "command": "python -c \"from src.features.offense import compute_offensive_features; f = compute_offensive_features('NYY', '2025-07-15'); print(f['wRC_plus_30g'])\"",
        "exitCode": 0,
        "observation": "Value 112.3, reasonable for Yankees"
      }
    ],
    "interactiveChecks": [
      {
        "action": "Manually verified anti-leakage: computed features for 2025-07-15, confirmed 30-game window only includes games through 2025-07-14",
        "observed": "Latest game in window was 2025-07-14, as_of_timestamp = 2025-07-14"
      }
    ]
  },
  "tests": {
    "added": [
      {
        "file": "tests/test_offense_features.py",
        "cases": [
          { "name": "test_rolling_windows_exist", "verifies": "VAL-FEAT-001" },
          { "name": "test_rolling_excludes_current_game", "verifies": "VAL-FEAT-002" },
          { "name": "test_marcel_blend_early_season", "verifies": "VAL-FEAT-003" }
        ]
      }
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- Source data table/column needed doesn't exist in database
- Configuration threshold needed but not in settings.yaml
- Prior-year data needed for Marcel but not available
- Feature definition ambiguous and can't be resolved from plan
