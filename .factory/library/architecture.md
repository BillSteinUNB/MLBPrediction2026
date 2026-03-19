# Architecture

Architectural decisions, patterns, and code structure.

**What belongs here:** Design decisions, code organization patterns, key algorithms, data flow.

---

## System Overview

This is a **batch processing system** (not a web service). The pipeline runs on-demand or on a schedule, processes data, and outputs predictions via Discord webhook.

```
┌─────────────────────────────────────────────────────────────┐
│                    Daily Pipeline                            │
├─────────────────────────────────────────────────────────────┤
│  1. Ingest: Schedule, Lineups, Odds, Weather                │
│  2. Feature: Compute all features with anti-leakage          │
│  3. Predict: XGBoost → LR Stacking → Isotonic Calibration    │
│  4. Edge: De-vig odds, compare vs model prob                 │
│  5. Size: Quarter Kelly bankroll management                  │
│  6. Store: SQLite persistence                                │
│  7. Notify: Discord webhook                                  │
└─────────────────────────────────────────────────────────────┘
```

## Code Organization

```
src/
├── clients/          # External API clients (Odds, Statcast, MLB, Weather)
├── features/         # Feature engineering modules
│   └── adjustments/  # Park, ABS, weather adjustments
├── model/            # ML pipeline (training, stacking, calibration)
├── engine/           # Decision engine (edge, bankroll, settlement)
├── pipeline/         # Daily orchestrator
├── notifications/    # Discord webhook
├── ops/              # Scheduling, logging, error handling
├── models/           # Pydantic data models
├── config.py         # Configuration loader
└── db.py             # SQLite schema and operations
```

## Key Architectural Decisions

### Anti-Leakage Enforcement
Every feature row has `as_of_timestamp` set to the day BEFORE the game. This is enforced at:
- Database level: NOT NULL constraint
- Feature computation level: rolling windows exclude current game
- Test level: 100+ random games verified per test run

### Stacked Ensemble
1. **XGBoost** (base learner): Captures non-linear feature interactions
2. **Logistic Regression** (meta-learner): Calibrates XGBoost outputs
3. **Isotonic Regression** (calibrator): Maps to perfectly calibrated probabilities

Training uses out-of-fold predictions to prevent leakage.

### F5-Specific Design
- Features computed for first 5 innings only
- Pythagorean WP uses F5 runs, not full-game
- Settlement rules handle F5-specific outcomes (tie = push)

### Sabermetrics Only
Traditional stats (ERA, W-L, batting average) are **never used**. Only advanced metrics:
- Offense: wRC+, wOBA, ISO, BABIP
- Pitching: xFIP, xERA (proxy for SIERA)
- Defense: DRS, OAA

### Marcel Early-Season Handling
When insufficient current-year data (< 30 games offense, < 15 starts pitching):
```
blended = (current × games_played + prior × regression_weight) / (games_played + regression_weight)
```

### ABS Strike Zone Adjustment
- League-wide: +4% walk rate, -3% strikeout rate
- Catcher framing: ×0.75 retention factor
- Exception venues (Mexico City, Field of Dreams, Little League Classic): no ABS effect

## Data Flow

```
External APIs → Parquet Cache → SQLite DB → Feature Computation → Model Prediction → Edge Calculation → Discord
     │                              │
     └────── pybaseball ────────────┘
            Statcast data
```

## Threading and Concurrency

- Feature computation: Sequential per game (anti-leakage requirement)
- API calls: Sequential (rate limits on free tiers)
- No parallel processing needed (single-threaded batch system)
