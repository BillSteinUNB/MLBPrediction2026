# MLB F5 Betting Model — Full Audit Report

**Date**: 2026-03-31  
**Scope**: Full-stack audit of prediction pipeline. No code changes made.

---

## Summary

Holdout R² is stuck at 3-5% (historical best 5.16%). This audit found **3 critical bugs**, **4 high-impact performance drags**, **4 moderate issues**, and confirmed **5 areas are clean**. The critical bugs alone could explain a significant portion of the gap between training and production performance.

---

## 🔴 CRITICAL — Fix Immediately

### 1. Settlement Metadata Overwrite
**File**: `src/engine/settlement.py` lines 95-121  
**Confidence**: HIGH — read the code directly

`_load_pending_bets` reconstructs bet objects with `model_probability=0.5`. Every settled bet has its model probability overwritten to a coin flip, corrupting:
- ROI calculations
- CLV tracking
- Kelly sizing feedback
- Any metric that depends on knowing what the model actually predicted

You literally cannot measure whether your model's probabilities were good.

---

### 2. Timezone Offset Bug
**File**: `src/model/data_builder.py` ~line 2955  
**Confidence**: HIGH — read the code directly

Home team's timezone offset is used for **both** home and away teams when computing fatigue/travel features. Away team gets the wrong offset for every single game. These are potentially high-signal features that are currently noise.

---

### 3. Team Platoon Splits Ignore Target Date
**File**: `src/features/offense.py`  
**Confidence**: MEDIUM-HIGH — the wrapper deleting target_day is confirmed in code, but I couldn't verify how the `team_platoon_splits` DB table is populated

`build_db_backed_team_batting_splits_fetcher` receives `target_day` but **explicitly deletes it** (`del target_day`). It returns season-level splits from the `team_platoon_splits` table with no date filtering. If the DB table contains current-season aggregates computed using all games (including future ones relative to a target date), this leaks future information during training. Safe for prior seasons, risky for same-season.

---

## 🟠 HIGH — Significant Performance Drag

### 4. Run-Count Models Silently Overwrite ML/RL Probs
**File**: `src/pipeline/daily.py` lines 341-366  
**Confidence**: MEDIUM-HIGH — the overwrite is confirmed, but I didn't fully trace whether the f5_rl market path corrects this downstream

When run-count models produce valid projections, they **overwrite both** `ml_home_probability` AND `rl_home_probability` with the run-count-derived moneyline probability. The RL probability should be derived from the run-count model's spread distribution, not set equal to the moneyline. Your RL bets may be using ML-derived probabilities compared against posted RL lines — a fundamental mismatch.

---

### 5. Market Shrink is Aggressive
**File**: `src/model/market_recalibration.py`  
**Confidence**: MEDIUM — this is a judgment call, not a bug

Base shrink multiplier = 0.70 (high-edge = 0.60). This pulls every prediction 30-58% toward the market price. With holdout R² of 3-5%, this much shrink likely erases most of your detectable edge. You may have more signal than you think, but the shrink layer is burying it. Consider testing lower values.

---

### 6. Feature Selection Outside CV
**File**: `src/model/run_count_trainer.py` lines 456-488  
**Confidence**: HIGH — confirmed location and mechanism

Feature selection (correlation-based filtering) runs on the **entire dataset before cross-validation**. The model "sees" all data when choosing which features matter, then evaluates on folds — inflating CV scores. Backtest metrics are optimistically biased. The 5.16% best R² may overstate true out-of-sample performance.

---

### 7. Grouped Feature Selection Suppresses Deltas
**File**: `src/model/run_count_trainer.py`  
**Confidence**: HIGH — confirmed in config

The best historical run (5.16% R²) used `"flat"` selection mode. The current default is `"grouped"`, which allocates only 12 slots to the "delta" bucket (home-away differentials — the most predictive category). The allocation is short=24, medium=28, delta=12, context=16. You had a better configuration and moved away from it.

---

## 🟡 MODERATE — Worth Addressing

### 8. Calibration Promotes on log_loss, Not Expected Value
**File**: `src/model/promotion.py`  
**Confidence**: HIGH

Model variant selection uses log_loss as the promotion criterion. For a **betting** model, the right metric is expected value (EV) — you care about profitable edge detection at the tails, not pure probability accuracy. A model with worse log_loss but better edge calibration could be significantly more profitable.

---

### 9. Weather Adjustments Don't Validate Timestamp
**File**: `src/features/adjustments/weather.py`  
**Confidence**: HIGH

`compute_weather_adjustment` accepts a `WeatherData` object (which contains `forecast_time` and `fetched_at` fields) but never validates that the data is a pre-game forecast vs. post-game observation. In practice the live pipeline fetches forecasts, but there's no safety net.

---

### 10. SQL Loaders Use Fragile Inclusive Convention
**Files**: `src/features/pitching.py`, `src/features/bullpen.py`, `src/features/offense.py`  
**Confidence**: HIGH

All SQL loaders use `<= end_date` and rely on callers passing `end_date = target_day - 1`. If any future caller forgets the -1, target-day rows leak in silently. Currently correct but one wrong call creates invisible leakage.

---

### 11. Near-Constant Features Wasting Model Capacity
**Confidence**: HIGH — empirically verified with queries against `training_data_2018_2025.parquet` (17,006 rows × 373 columns)

These features add noise and waste feature selection slots:
- `abs_active`: 99.99% at 1.0 (only 2 games differ)
- `abs_walk_rate_delta`: 99.99% at 0.04
- `abs_strikeout_rate_delta`: 99.99% at -0.03
- `weather_precip_probability`: 100% at 0.0
- `weather_data_missing`: 100% at 0.0

**Important correction**: Earlier claims that ALL weather and platoon features were constant were **wrong**. Empirical verification proved:
- `weather_temp_factor`: 940 unique values, only 3.5% at default
- `weather_composite`: 16,320 unique values, only 3.3% at default
- `home_team_woba_vs_opposing_hand`: 374 unique values, 0% at zero
- All other platoon features have meaningful variance

---

## ✅ CLEAN — No Issues Found

### 12. Temporal Leakage in Core Features — SAFE
Four independent agents verified all feature modules. Every module accepts `game_date`/`target_day`, filters with strict `< target_day`, and computes rolling windows on pre-filtered frames:

- **Offense** (`src/features/offense.py`): `searchsorted(side="left")` for exclusive cutoffs, `game_date.dt.date < target_day` pandas filters. Rolling via `tail()` on pre-filtered frames.
- **Pitching** (`src/features/pitching.py`): Passes `end_date = target_day - 1` to fetchers, then pandas filter `< target_day`. Rolling on sorted, filtered histories.
- **Bullpen** (`src/features/bullpen.py`): Schedule mode loads full-season data but accumulates only `metric_day < target_day`. Pitch-count windows use explicit `>= start_day AND < target_day`.
- **Defense** (`src/features/defense.py`): Has `as_of_timestamp` guards.
- **Umpires** (`src/features/umpires.py`): Has `as_of_timestamp` guards.
- **Baselines** (`src/features/baselines.py`): Has `as_of_timestamp` guards.

### 13. Stacking — SAFE
Temporal OOF stacking generates out-of-fold predictions in time order. No leakage.

### 14. Payout Calculation — SAFE (Earlier Claim Overstated)
`AmericanOdds` validator in `src/models/_base.py` rejects `abs(odds) < 100` with a `ValidationError`. Odds +1 to +99 would **crash**, not silently compute wrong payouts. This is a safety guard, not a silent bug.

### 15. Park Factors — SAFE
Static config from `settings.yaml`. No dynamic computation from game data.

### 16. Data Builder Caller Contract — SAFE
`data_builder.py` passes either `schedule` (bulk) or `game_date` (per-day) to feature functions. Feature modules handle their own temporal filtering internally. Final `assert_training_data_is_leakage_free()` validates `as_of_timestamp < scheduled_start`.

---

## Areas NOT Fully Investigated

For transparency, these are areas I ran out of time on:

- **`team_platoon_splits` DB population** — I confirmed the wrapper ignores target_day, but couldn't find the code that writes to that table. If it's only populated post-season, there's no leak.
- **f5_rl market downstream path** — I confirmed the overwrite at lines 341-366 but didn't fully trace whether the RL probability gets separately corrected for the RL-specific market path.
- **Data integrity/entity resolution** — Whether games are matched correctly across data sources (Statcast, odds, retrosheet).
- **Training data target vs actual posted lines** — The code has `_attach_historical_runline_targets` but I didn't verify the data quality of the historical lines.
- **OddsScraper data quality** — Whether scraped historical odds are clean and complete.

---

## Priority Fix Order

| Priority | Fix | Expected Impact |
|----------|-----|-----------------|
| **1** | Settlement metadata overwrite | Enables actual model performance measurement |
| **2** | Timezone offset bug | Fixes away-team fatigue/travel features |
| **3** | Platoon splits temporal guard | Prevents potential same-season leakage |
| **4** | RL probability from run-count spread | Fixes RL bet edge calculation |
| **5** | Feature selection inside CV | Removes optimistic CV bias |
| **6** | Switch back to flat selection mode | Recovers the 5.16% R² configuration |
| **7** | Reduce market shrink | May unmask hidden edge |
| **8** | Promote on EV not log_loss | Better model variant for betting |
| **9** | Drop near-constant features | Frees feature selection slots |
| **10** | Add weather timestamp validation | Safety net against observed weather |

Items 1-3 are bugs. Items 4-7 are the biggest likely drag on holdout R². Items 8-10 are quality-of-life improvements.

---

## Audit Methodology

- 13 explore agents ran across: odds client, settlement/bankroll, TimeSeriesSplit, hyperparameters, inference path, home/away alignment, train/live parity, feature selection, stacking/calibration, offense temporal guards, pitching/bullpen temporal guards, adjustments temporal guards, data_builder caller contract
- Oracle verification identified overstated claims → empirical corrections made (weather features NOT all constant, platoon features NOT all constant, payout bug overstated)
- Empirical queries against `data/training/training_data_2018_2025.parquet` confirmed near-constant features and disproved earlier false claims
- Target/label alignment investigated (fixed -1.5 hardcoded label vs `_attach_historical_runline_targets`)
- Model precedence traced through `daily.py`
