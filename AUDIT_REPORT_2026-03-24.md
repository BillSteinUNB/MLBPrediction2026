# MLB F5 Betting System — Comprehensive Audit Report

**Date:** March 24, 2026  
**Scope:** Read-only audit of entire ML pipeline — no code was modified  
**Method:** 6 parallel specialist agents (4 codebase explorers + 2 research librarians), direct analysis of 20+ source files, benchmarking against academic papers, professional syndicate approaches, and open-source models

---

## Executive Summary

After examining **every core module** across the system (walk-forward, calibration, stacking, XGBoost training, edge calculation, betting strategy, feature engineering, data building, and the daily pipeline) plus benchmarking against industry best practices, I identified **4 critical, 7 high, and 6 medium-severity issues**.

The system has solid engineering foundations — proper temporal splits, anti-leak assertions, de-vig math, comprehensive feature engineering, a walk-forward framework. But several fundamental methodology problems would prevent it from generating real edges in live betting.

**The headline finding**: The +24.4% backtest ROI is an artifact of the evaluation methodology, not a genuine edge. The model is being evaluated against its own features, creating a circular feedback loop that inflates apparent performance.

---

## CRITICAL ISSUES (System-Breaking)

### 🔴 C1. The Model Competes Against Itself ("Synthetic Odds" Problem)

**Location:** `src/backtest/walk_forward.py` lines 993–1081 (`_resolve_market_pricing`)

When no historical odds database is provided, the walk-forward backtest generates **synthetic market odds from the Log5 feature**:

```python
# walk_forward.py line 1000-1005
default_home_fair = np.clip(
    pd.to_numeric(test_frame["home_team_log5_30g"], errors="coerce").to_numpy(dtype=float),
    0.02, 0.98,
)
```

Then edge is computed as:

```python
# walk_forward.py line 809
edge_home = model_home_prob - market_home_fair_prob
```

**The problem**: `home_team_log5_30g` is the **#1 most important feature** in the model (4–8% importance across all experiments per the ANALYSIS doc). The model literally learns from Log5, outputs something slightly different, and then that difference is called an "edge."

- The model is betting home when `prediction > Log5` and away when `prediction < Log5`
- This isn't finding market inefficiencies — it's measuring how much the model deviates from one of its own inputs
- The +24.4% ROI and 84% bet rate are artifacts of this circularity

**Why this matters**: Real sportsbook lines are set by sharp market makers using information the model doesn't have. A genuine edge must be measured against real market prices. The synthetic odds are essentially the model's prior, not an independent market.

**Impact**: All backtest ROI numbers without historical odds are **meaningless**. Brier scores remain valid (they measure probability accuracy), but profitability claims cannot be trusted.

---

### 🔴 C2. Strategy Parameters Optimized In-Sample

**Location:** `src/ops/market_strategy_sweep.py` lines 96–117 and `config/autoresearch_walk_forward.yaml`

The strategy sweep loads training data and optimizes edge thresholds, staking modes, and multipliers on the **same data**:

```yaml
# autoresearch_walk_forward.yaml
search_space:
  edge-threshold:
    values: [0.04, 0.05, 0.06, 0.07]
  staking-mode:
    values: [flat, edge_scaled, edge_bucketed]
  max-bet-size-units:
    values: [1.5, 2.0, 2.5, 3.0]
```

The `market_strategy_sweep.py` loads `ml_training_data` (lines 96–106) and evaluates sampled parameter sets on that same frame. The `autoresearch.py` driver runs 12+ trials per config, each sampling different parameter combinations.

**Why this is devastating**: With 4 × 3 × 4 × 4 = 192 strategy combinations across ~5 configs, you're testing **~960+ parameter sets** on historical data. Finding one that shows +ROI is guaranteed by chance alone. Research shows **89% of strategies that look profitable in standard backtests fail under proper walk-forward validation** (Frontier Ledger, 2025).

**The fix**: Strategy parameters must be optimized only within walk-forward train windows and evaluated on unseen test windows. Never optimize betting thresholds on the data you're measuring ROI with.

---

### 🔴 C3. Training Uses Closing Lines, Inference Uses Live Lines (Train/Serve Mismatch)

**Location:**
- `src/clients/historical_odds_client.py` lines 195–211 (`load_historical_odds_for_games`)
- `src/clients/historical_f5_acquirer.py` line ~440 (closing frame generation)
- `src/pipeline/daily.py` lines 1032–1051 (live odds fetching)

Training data assembly calls `load_historical_odds_for_games` which selects `MAX(fetched_at)` per game — giving the **closing line** (latest snapshot). But the daily inference pipeline fetches **current live odds** (which are typically opening or mid-day lines).

**Why this matters**: Closing lines incorporate sharp bettor information that wasn't available when the model would actually place bets. A model trained to predict against closing lines but evaluated against opening lines has a systematic information asymmetry:

- **Training**: Model sees closing-line-based targets (sharper, more efficient)
- **Inference**: Model bets against live odds available at the time of the pipeline run (which are typically earlier than eventual closing lines)
- This could create either optimistic OR pessimistic bias depending on how features interact with the target

Professional syndicates track CLV (Closing Line Value) as the gold standard because beating closing lines consistently by even 1–2% indicates genuine edge. If your model is trained on closing lines, it's essentially learning to be the market rather than beat it.

---

### 🔴 C4. The Walk-Forward ROI (+24.4%) Is Meaningless

**Location:** `EXPERIMENT_REVIEW_2026-03-21.md` — Walk-Forward results section

Combining C1, C2, and the experiment data:

| Red Flag | Value | Expected for Real System |
|----------|-------|--------------------------|
| ROI | +24.4% | 3–6% for elite bettors |
| Bet Rate | 84% of games | 10–30% (selective) |
| Win Rate | 44.7% (needs +130 avg) | Context-dependent |
| Sample | 3 months, 1013 bets | 12+ months, 1000+ bets minimum |
| Odds Source | Synthetic (from Log5) | Real sportsbook lines |
| Vig Assumed | 4% | 5–8% for F5 markets |

**Research benchmark**: Professional bettors sustain 3–6% ROI long-term. Returns above 10% are "extremely rare and usually indicate limited scale or unacceptable risk." The +24.4% is a 4–8x outlier from what the sharpest bettors in the world achieve.

---

## HIGH-SEVERITY ISSUES

### 🟠 H1. Calibration Consistently Hurts — Yet System Historically Used It

**Location:** `src/model/calibration.py` line 47, experiment results

Your own data shows calibration hurts in **7 out of 8 measurements**:

| Market | Exp 1 | Exp 2 | Exp 3 | Exp 4 |
|--------|-------|-------|-------|-------|
| F5 ML | -0.0011 | -0.0012 | +0.00002 | -0.0028 |
| F5 RL | -0.0022 | -0.0029 | -0.0007 | -0.0025 |

The default `DEFAULT_CALIBRATION_METHOD = "identity"` (line 47) is now correct. But the walk-forward autoresearch configs and historical experiments used Platt calibration, which compresses the probability range into 2 bins and destroys discrimination. The stacking layer is already producing well-calibrated probabilities (ECE 1.4–3.4%), so additional calibration adds noise.

Research confirms: *"Calibrating with only 100-200 matches is risky. You'll get unreliable scaling."* The calibration holdout of ~1,650 games is borderline for isotonic regression and insufficient for reliable Platt scaling.

---

### 🟠 H2. Stacking Layer is Counterproductive for F5 ML

**Location:** `src/model/stacking.py` lines 40–44

Only 3 meta-features feed the LR meta-learner:

```python
DEFAULT_RAW_META_FEATURE_COLUMNS = (
    "home_team_f5_pythagorean_wp_30g",
    "home_team_log5_30g",
    "park_runs_factor",
)
```

Stacking hurts F5 ML in all 4 experiments (Δ Brier: -0.0003, -0.0002, -0.0002, -0.0008). The meta-learner essentially passes through XGBoost probability with slight Pythagorean/Log5 blending — adding noise rather than corrective signal.

**Missing**: Away-team baselines, pitcher quality differentials, offensive differentials. The stacking layer could add value with richer meta-features, but with only 3 home-team-centric features it's doing more harm than good.

---

### 🟠 H3. Hyperparameter Search is Stuck in a Corner

**Location:** `src/model/xgboost_trainer.py` lines 36–40

Every single experiment converges to `max_depth=6, n_estimators=100, learning_rate=0.01` — the most conservative corner of the search space. With only 15 random search iterations across 150 combinations, the search has no power to explore.

**Critical missing parameters**:

| Parameter | Current Value | Recommended |
|-----------|--------------|-------------|
| `subsample` | 1.0 (hardcoded) | 0.7–0.9 |
| `colsample_bytree` | 1.0 (hardcoded) | 0.6–0.9 |
| `min_child_weight` | Not searched | 1–10 |
| `gamma` | Not searched | 0–5 |
| `reg_alpha` | Not searched | 0–1 |
| `reg_lambda` | Not searched | 1–10 |

With 224 features and ~2,400 training rows per walk-forward window, **no regularization from subsampling** makes overfitting nearly certain. Research recommends `subsample=0.7-0.9` and `colsample_bytree=0.6-0.9` as the first line of defense against overfitting.

---

### 🟠 H4. Feature Pivot Drops as_of_timestamp (Subtle Leakage Vector)

**Location:** `src/model/data_builder.py` lines ~1403–1424

The feature loading query:

```sql
SELECT game_pk, feature_name, feature_value FROM features ORDER BY game_pk, feature_name
```

This **omits `as_of_timestamp`**. The subsequent pivot uses `aggfunc="last"`, which means if multiple feature rows exist for the same (game_pk, feature_name) with different as_of timestamps, the pivot could pick a value from a later snapshot. While current feature writers appear correct, this is an undefended gap that could silently introduce leakage if features are ever recomputed.

---

### 🟠 H5. No Closing Line Value (CLV) Tracking in Evaluation

**Location:** `src/ops/performance_tracker.py` — CLV exists but not wired into model evaluation

CLV is the **gold standard** for evaluating betting models. Your system has a `sync_closing_lines_from_snapshots` function (performance_tracker.py lines 372–435), but it's only used post-hoc for individual bet tracking. There's no systematic evaluation of model predictions against closing lines across seasons.

Research consensus: *"If you don't beat the closing line, you will eventually bleed out. The closing line is the receipt that proves you got a discount."* (EdgeSlip, 2026)

*"CLV is the definitive metric for measuring betting skill... It is the only reliable predictor of future success and the truest benchmark for whether a bettor has edge."* (EdgeSlip, 2026)

---

### 🟠 H6. 224 Features With Concentration in ~20 (Overfitting Risk)

**Location:** Feature importance data from experiments

The model uses 224 features but importance is heavily concentrated:

- Top 5 features: ~25% of total importance
- Top 20 features: ~60% of importance
- Bottom 150 features: near-zero importance

Weather, ABS adjustments, adjusted framing, and many short-window defense metrics contribute nothing. These features add noise dimensions that increase overfitting risk without adding signal, especially with only 2,400 rows per walk-forward window. Research shows that **feature pruning consistently improves out-of-sample performance** when the feature-to-sample ratio is high.

---

### 🟠 H7. Multiple Inconsistent Edge Thresholds

**Location:** Multiple files

| Module | Default Threshold | File |
|--------|-------------------|------|
| Walk-forward | 3% | `walk_forward.py:70` |
| Strategy sweep baseline | 5% | `market_strategy_sweep.py:131` |
| Production ML | **7%** | `daily.py:68` |
| Production RL | 3% | `daily.py:71` |
| RL evaluator | 5% | `rl_v2_evaluator.py:52` |

Using a 3% threshold in backtests but 7% in production creates a disconnect where backtest performance doesn't predict production behavior. Research shows **edges below 3–5% in MLB are usually noise** given inherent variance. The production ML threshold of 7% is actually the most appropriate.

---

## MEDIUM-SEVERITY ISSUES

### 🟡 M1. No Embargo Gap in Walk-Forward Windows

**Location:** `walk_forward.py` lines 196–197

`train_end = test_start` with no gap. Features computed the day before a game could include same-day information (roster moves, injury announcements). Best practice: 1–7 day embargo between training and test periods.

---

### 🟡 M2. 2020 Season Included Inconsistently

**Location:** ANALYSIS_2026-03-21.md line 467–469

`training_data_2023_2025` includes 2020 (60-game season) while `training_data_2018_2025` correctly skips it. The shortened season has different statistical properties (smaller sample, unique rules like universal DH, 7-inning doubleheaders) and introduces noise.

---

### 🟡 M3. Kelly Input Ambiguity

**Location:** `market_strategy_sweep.py` lines 582–613 and 718–723

In the strategy sweep, Kelly receives market-shrunk probability (after `shrink_probability_toward_market`). In the walk-forward backtest, it receives calibrated probability directly. The Kelly criterion should use the probability you genuinely believe — mixing calibrated and shrunk inputs silently changes bet sizing behavior.

---

### 🟡 M4. Market Recalibration Can Mask a Weak Model

**Location:** `src/model/market_recalibration.py`

`shrink_probability_toward_market` pulls model probabilities toward market prices. If the model has no genuine edge, this just turns it into a noisy market follower. You should track raw model performance vs. recalibrated performance separately — if recalibrated vastly outperforms raw, the model isn't adding information.

| Scenario | Model Edge | After Shrinkage | Interpretation |
|----------|-----------|-----------------|----------------|
| Strong model | +8% | +6% | Real edge, slightly dampened |
| **Weak model** | +2% | +1.5% | **Appears profitable but edge is noise** |
| Random model | +0% | +0% | No edge visible |

---

### 🟡 M5. Direct RL Trainer is Misleadingly Named

**Location:** `src/model/direct_rl_trainer.py`

The "RL" trainer is actually standard supervised learning (XGBoost binary classification predicting `home_cover_at_posted_line`). It's not reinforcement learning. Research on actual RL for sports betting shows it's appropriate for sequential in-play decisions, not pre-game binary prediction. This naming confusion could lead to incorrect assumptions about what the model does and what its limitations are.

---

### 🟡 M6. The Bet Rate is Unrealistically High

The backtest bets on 84% of games (1,013 of 1,207). A real sharp model should be selective — **professional bettors typically bet 10–30% of available games**. An 84% bet rate with a 3% threshold strongly suggests the "edge" is noise from the synthetic odds calculation (C1). A model with genuine edge should be betting far fewer games at higher confidence.

---

## THINGS THE SYSTEM DOES WELL

Credit where due — the engineering quality is high in many areas:

| Area | Implementation | Assessment |
|------|----------------|------------|
| **Anti-leak assertions** | `assert_training_data_is_leakage_free()` in data_builder.py | ✅ Properly checks as_of < scheduled_start |
| **Temporal CV** | `TimeSeriesSplit` in xgboost_trainer.py | ✅ Correct for time-series data |
| **Walk-forward per-window retraining** | Model retrained each fold | ✅ No single model reused across windows |
| **Calibration per-window** | Calibrators fitted on reserved training slice, not test | ✅ No calibration leakage across folds |
| **Stacking uses OOF predictions** | `_generate_temporal_oof_probabilities` in stacking.py | ✅ Proper OOF procedure |
| **De-vig math** | `devig_probabilities` in odds_client.py | ✅ Correct proportional de-vig |
| **EV calculation** | `expected_value` in edge_calculator.py | ✅ Correct payout math |
| **Edge audit logging** | `_log_edge_calculation` to SQLite | ✅ Good for debugging |
| **Kill-switch** | Bankroll drawdown protection in bankroll.py | ✅ Risk management |
| **No closing lines as features** | Closing data only used post-hoc for CLV/reporting | ✅ No closing-line leakage |
| **Settlement logic** | F5 rules with NO_ACTION for incomplete games | ✅ Handles edge cases |
| **Feature imputation consistency** | Same `_default_feature_fill_value()` for train and inference | ✅ No train/serve skew here |

---

## WHAT A TOP-TIER MLB BETTING SYSTEM LOOKS LIKE

Based on research across academic papers, professional syndicate approaches, and successful open-source models:

| Dimension | Your System | Best Practice |
|-----------|-------------|---------------|
| **Primary evaluation metric** | Backtest ROI on synthetic odds | CLV against Pinnacle/sharp book closing lines |
| **Edge threshold** | 3% (backtest) / 7% (prod) | 5–7% minimum, consistent everywhere |
| **Bet selectivity** | 84% of games | 10–30% of games |
| **Expected ROI** | +24.4% (backtest) | 3–6% (world-class) |
| **Validation period** | 3-month walk-forward | 2+ full seasons, 1000+ bets minimum |
| **Calibration approach** | Platt (historically) | Identity or isotonic on large samples only |
| **Feature count** | 224 (mostly noise) | 20–50 curated features |
| **XGB regularization** | subsample=1.0, colsample=1.0 | subsample=0.7–0.9, colsample=0.6–0.9 |
| **Strategy tuning** | In-sample sweep | Nested walk-forward (inner loop) |
| **Market data for backtest** | Synthetic or closing | Timestamped opening lines from sharp books |
| **Sample size for validation** | ~1,000 bets, 3 months | 5,000+ bets, 2+ seasons |
| **Performance retention (WFE)** | Unknown | Expect <30% of in-sample returns |
| **Brier score target** | ~0.2448–0.2487 | <0.245 (marginal improvement matters) |

### Realistic Performance Benchmarks from Research

| Market Type | Achievable Accuracy | Breakeven (at -110) | Typical Sharp Edge |
|-------------|--------------------|-----------------------|--------------------|
| Moneyline | 55–60% | 52.4% | 2–5% EV |
| Run Line | 52–55% | 52.4% | 1–3% EV |
| Totals | 53–57% | 52.4% | 2–4% EV |
| First 5 Innings | 54–58% | 52.4% | 3–6% EV |

### Professional Bettor ROI Expectations

| Level | Expected ROI | Notes |
|-------|--------------|-------|
| Recreational | -8% to -10% | House edge dominates |
| Aspiring Sharp | 0% to 2% | Learning curve |
| Consistent Winner | 3% to 6% | Disciplined, well-calibrated |
| Elite | 5% to 10% | Extremely rare, high volume required |

---

## PRIORITIZED REMEDIATION ROADMAP

### Phase 1: Fix the Evaluation (Without This, Nothing Else Matters)

| # | Action | Why | Files |
|---|--------|-----|-------|
| 1 | **Backtest only against real historical odds** — never use synthetic Log5-derived odds | Synthetic odds create circular edge measurement | `walk_forward.py` |
| 2 | **Use opening lines, not closing** — or at minimum, document and be consistent | Closing lines contain information not available at bet time | `historical_odds_client.py`, `historical_f5_acquirer.py` |
| 3 | **Add systematic CLV tracking** — measure model predictions against sharp book closing lines | Only reliable predictor of future success | `performance_tracker.py` |
| 4 | **Increase sample size** — run walk-forward across 2+ full seasons (2021–2025) | 3 months is insufficient for statistical conclusions | `walk_forward.py` config |
| 5 | **Fix strategy parameter optimization** — use nested walk-forward (optimize within train windows only) | In-sample optimization guarantees overfit results | `market_strategy_sweep.py`, `autoresearch.py` |

### Phase 2: Fix the Model

| # | Action | Why | Files |
|---|--------|-----|-------|
| 6 | **Add XGBoost regularization** — `subsample=0.8`, `colsample_bytree=0.7`, `min_child_weight=3` | No subsampling = overfitting with 224 features | `xgboost_trainer.py` |
| 7 | **Prune features** — drop bottom 150 by importance, keep ~50–70 curated features | Noise dimensions increase overfitting risk | `data_builder.py`, feature modules |
| 8 | **Drop Platt calibration permanently** — use identity (already default) | Hurts in 7/8 measurements | `calibration.py` config |
| 9 | **Reconsider stacking for F5 ML** — either add richer meta-features or skip it | Hurts in all 4 experiments | `stacking.py` |
| 10 | **Increase hyperparameter search** — 50+ iterations, add regularization params to search space | 15 iterations over 150 combos has no search power | `xgboost_trainer.py` |

### Phase 3: Fix the Pipeline

| # | Action | Why | Files |
|---|--------|-----|-------|
| 11 | **Standardize edge thresholds** — one canonical config source | 3%/5%/7% inconsistency across modules | `settings.yaml`, all callers |
| 12 | **Fix feature pivot to preserve as_of_timestamp** — prevent subtle leakage | `aggfunc="last"` without timestamp is undefended | `data_builder.py` |
| 13 | **Add embargo gap to walk-forward** — minimum 1 day between train/test | Same-day information could leak | `walk_forward.py` |
| 14 | **Remove 2020 from all datasets** — or be explicitly consistent | Shortened season has different statistical properties | Dataset configs |
| 15 | **Separate raw vs. recalibrated performance tracking** | Need to know if model adds info or just follows market | `market_strategy_sweep.py` |

### Phase 4: Establish Real Benchmarks

| # | Target | Current | Notes |
|---|--------|---------|-------|
| 16 | Brier < 0.245 | 0.2448–0.2487 | Marginal improvements matter at this level |
| 17 | Positive CLV > 1% against Pinnacle closing | Not tracked | The only metric that matters for profitability |
| 18 | Bet rate < 30% of available games | 84% | Selectivity indicates genuine edge |
| 19 | ROI 3–6% with 1,000+ bet validation | +24.4% on synthetic odds | Expect massive drop with real odds |
| 20 | Walk-Forward Efficiency < 30% of in-sample | Unknown | Standard is <30% retention |

---

## Research Sources Consulted

| Topic | Key Sources |
|-------|-------------|
| Data Leakage | Journal of Big Data (2025), MachineLearningMastery |
| Backtest Overfitting | QuantFoundryLab (2026), Frontier Ledger (2025) |
| Walk-Forward Validation | SmartFinanceData (2026), Alpha Scientist |
| CLV Methodology | Bet Hero (2026), EdgeSlip (2026), Trademate Sports |
| Calibration | SportBot AI (2026), ExPrysm (2026) |
| MLB Edges | DailyMLBPicks (2026), Betstamp (2025), XCLSV |
| XGBoost Best Practices | TheLinuxCode XGBoost Guide, sklearn docs |
| RL in Betting | Stanford CS224R DQN NBA Study, betting forums |
| Successful Models | borsheng/Swing_Decision_Model, arjun-prabhakar/mlb_outcomes, Ali-m89/Sports_Prediction |

---

## Bottom Line

The system has strong engineering foundations — proper temporal splits, anti-leak assertions, de-vig math, comprehensive feature engineering, walk-forward framework. But the evaluation methodology has fundamental flaws that make current results unreliable.

The +24.4% ROI is almost certainly an artifact of competing against synthetic odds derived from the model's own features and optimizing strategy parameters in-sample.

**Before any model improvements**: fix the evaluation by backtesting exclusively against real historical odds, tracking CLV, and moving strategy optimization into a proper nested walk-forward loop. Only then will you know whether the model has genuine predictive power.

> *"The sweetest backtest result is the most dangerous signal. A quant's real skill isn't building strategies—it's doubting them."* — QuantFoundryLab, 2026
