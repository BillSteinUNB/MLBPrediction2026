---
name: engine-worker
description: Handles decision engine components including edge calculation, bankroll management, and settlement logic
---

# Engine Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Use this worker for features involving:
- Edge calculator with de-vig, implied probability, EV calculation
- Quarter Kelly bankroll manager with kill-switch
- F5 settlement rules module
- Early-season Marcel blend integration
- Discord webhook formatter and notifier

## Work Procedure

1. **Read existing context**: Check `src/db.py` for bets/bankroll schema, `.factory/library/` for established patterns, `config/settings.yaml` for thresholds (edge_min=0.03, kelly_fraction=0.25, max_drawdown=0.30).

2. **Write tests first (TDD)**:
   - Odds conversion tests: positive, negative, even odds
   - De-vig tests: probabilities sum to 1.0
   - Edge calculation tests: known inputs → expected outputs
   - Kelly tests: stake calculation, 5% cap, kill-switch
   - Settlement tests: all F5 ML/RL scenarios, no-action triggers
   - Run tests to confirm they FAIL (red phase)

3. **Implement**:
   - American odds: negative → |odds|/(|odds|+100), positive → 100/(odds+100)
   - De-vig: fair_p = implied_p / sum(implied_p)
   - Edge threshold: only recommend if edge >= 0.03
   - Quarter Kelly: stake = bankroll × full_kelly × 0.25, capped at 5%
   - Kill-switch: stop betting if drawdown >= 30%
   - Settlement: tie = push, <5 innings = no_action, starter scratch = no_action

4. **Verify tests pass (green phase)**:
   - Run `pytest tests/test_edge*.py tests/test_bankroll*.py tests/test_settlement*.py -v --tb=short`
   - All financial logic tests must pass with exact expected values

5. **Manual verification**:
   - Calculate edge manually for known odds/model prob
   - Simulate bankroll with series of wins/losses
   - Test kill-switch activation

6. **Run linters**:
   - `python -m py_compile src/engine/*.py src/notifications/*.py`

## Example Handoff

```json
{
  "salientSummary": "Implemented edge calculator with de-vig, EV, and 3% threshold. Quarter Kelly bankroll manager with 5% cap and 30% drawdown kill-switch. All 15 financial tests pass.",
  "whatWasImplemented": "Created src/engine/edge_calculator.py with calculate_edge() returning edge_pct, ev, is_positive_ev. Created src/engine/bankroll.py with calculate_kelly_stake() and update_bankroll(). De-vig produces probabilities summing to 1.0. Kill-switch triggers at 30% drawdown. 5% bankroll cap enforced.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {
        "command": "pytest tests/test_edge_calculator.py tests/test_bankroll.py -v",
        "exitCode": 0,
        "observation": "15 tests passed: odds conversion, de-vig, edge, Kelly, kill-switch"
      },
      {
        "command": "python -c \"from src.engine.edge_calculator import american_to_implied, devig; h = american_to_implied(-150); a = american_to_implied(+130); fair_h, fair_a = devig(h, a); print(f'{fair_h:.4f} + {fair_a:.4f} = {fair_h+fair_a:.4f}')\"",
        "exitCode": 0,
        "observation": "0.5488 + 0.4512 = 1.0000"
      }
    ],
    "interactiveChecks": [
      {
        "action": "Simulated bankroll drop to 69% of peak, verified kill-switch activates",
        "observed": "is_killswitch_active() returned True at 31% drawdown"
      }
    ]
  },
  "tests": {
    "added": [
      {
        "file": "tests/test_edge_calculator.py",
        "cases": [
          { "name": "test_negative_odds_conversion", "verifies": "VAL-ODDS-001" },
          { "name": "test_devig_sums_to_one", "verifies": "VAL-ODDS-002" },
          { "name": "test_edge_threshold_filtering", "verifies": "VAL-ODDS-005" }
        ]
      },
      {
        "file": "tests/test_bankroll.py",
        "cases": [
          { "name": "test_quarter_kelly_calculation", "verifies": "VAL-BANK-001" },
          { "name": "test_five_percent_cap", "verifies": "VAL-BANK-002" },
          { "name": "test_killswitch_triggers", "verifies": "VAL-BANK-003" }
        ]
      }
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- Model probabilities not available (model not trained)
- Odds data structure different than expected
- Settlement edge case not covered by plan
- Discord webhook URL not configured
