# Experiment Results Review

**Generated:** 2026-03-21
**Runs Reviewed:** 4 holdout experiments + 1 walk-forward backtest

---

## Experiment Inventory

| # | Name | Dir | Holdout | Data Hash | Train Years | Train Rows | Holdout Rows |
|---|------|-----|---------|-----------|-------------|------------|--------------|
| 1 | Baseline | `holdout_2024` | 2024 | `16b1910a` | 2018-2023 (skip 2020) | 9,246 ML / 10,953 RL | 2,088 ML / 2,430 RL |
| 2 | Offense Fix | `2024-offense-fix` | 2024 | `0f74492f` | 2018-2025 (skip 2020) | 9,246 ML / 10,953 RL | 2,088 ML / 2,430 RL |
| 3 | Offense Fix + 2025 Holdout | `2025-offense-fix` | 2025 | `0f74492f` | 2018-2024 (skip 2020) | 11,127 ML / 13,140 RL | 2,048 ML / 2,434 RL |
| 4 | Offense Fix (2023+ only) | `2024-offense-fix-2023` | 2024 | `cb45d9d6` | 2023 + backfilled 2019-2022 | 8,099 ML / 9,574 RL | 2,088 ML / 2,430 RL |
| 5 | Walk-Forward Q2 2024 | `temp_backtest_q2_2024` | Apr-Jun 2024 | mixed | 12mo rolling | ~2,450/window | ~400/window |

---

## F5 Moneyline (ML) — Holdout Metrics Comparison

| Metric | #1 Baseline | #2 Offense Fix | #3 2025 Holdout | #4 2023+ Start |
|--------|-------------|----------------|-----------------|----------------|
| **XGBoost ROC-AUC** | 0.5854 | **0.5908** | 0.5708 | **0.5936** |
| **XGBoost Accuracy** | 55.46% | **55.99%** | 56.40% | **56.27%** |
| **XGBoost Log Loss** | 0.6826 | **0.6811** | 0.6830 | **0.6797** |
| **XGBoost CV Log Loss** | 0.6766 | **0.6735** | 0.6764 | 0.6787 |
| Best Params | d6/n100/lr0.01 | d6/n100/lr0.01 | d6/n100/lr0.01 | d6/n100/lr0.01 |
| Stacked Brier | 0.2450 | 0.2443 | 0.2452 | 0.2441 |
| Calibrated Brier | 0.2461 | 0.2455 | 0.2451 | **0.2469** |
| Stacking Δ Brier | -0.0003 (hurt) | -0.0002 (hurt) | -0.0002 (hurt) | -0.0008 (hurt) |
| Calibration Δ Brier | -0.0011 (hurt) | -0.0012 (hurt) | +0.00002 (neutral) | -0.0028 (hurt) |
| Cal. Quality Gates | ✅✅✅ | ✅✅✅ | ✅✅❌ (gap 0.21) | ✅✅✅ |

## F5 Run Line (RL) — Holdout Metrics Comparison

| Metric | #1 Baseline | #2 Offense Fix | #3 2025 Holdout | #4 2023+ Start |
|--------|-------------|----------------|-----------------|----------------|
| **XGBoost ROC-AUC** | 0.5861 | **0.5995** | 0.5647 | 0.5893 |
| **XGBoost Accuracy** | 67.57% | 67.53% | 67.63% | 67.57% |
| **XGBoost Log Loss** | 0.6211 | **0.6177** | 0.6244 | 0.6200 |
| **XGBoost CV Log Loss** | 0.6311 | **0.6273** | 0.6258 | 0.6276 |
| Best Params | d6/n100/lr0.01 | d6/n100/lr0.01 | d6/n100/lr0.01 | d6/n100/lr0.01 |
| Stacked Brier | 0.2150 | **0.2135** | 0.2161 | 0.2142 |
| Calibrated Brier | 0.2171 | 0.2164 | 0.2168 | 0.2167 |
| Stacking Δ Brier | +0.0003 (helped) | +0.0002 (helped) | +0.0004 (helped) | +0.0005 (helped) |
| Calibration Δ Brier | -0.0022 (hurt) | -0.0029 (hurt) | -0.0007 (hurt) | -0.0025 (hurt) |
| Cal. Quality Gates | ✅✅✅ | ✅✅✅ | ✅✅❌ (gap 0.23) | ✅✅✅ |

---

## Walk-Forward Backtest (Q2 2024, 3 Windows)

| Metric | Window 1 (Apr) | Window 2 (May) | Window 3 (Jun) | **Aggregate** |
|--------|---------------|----------------|----------------|---------------|
| Brier Score | 0.2486 | 0.2507 | 0.2468 | **0.2487** |
| Bet Count | 323 | 343 | 347 | **1,013** |
| Win/Loss | 120W / 161L | 133W / 165L | 141W / 161L | 394W / 487L |
| Push Count | 42 | 45 | 45 | 132 |
| ROI | +21.8% | +28.6% | +22.5% | **+24.4%** |
| Profit (units) | +70.55 | +98.23 | +78.03 | **+246.81** |
| Train Window | 12 months | 12 months | 12 months | |
| Train Rows | 2,476 | 2,457 | 2,454 | |
| Test Rows | 390 | 410 | 407 | |
| Scored (non-push) | 338 | 356 | 352 | |

Walk-forward config: `max_depth=3, n_estimators=120, lr=0.05, subsample=1.0, colsample=1.0, edge≥3%, vig=4%, 12mo train, 1mo test, Platt calibration (15% holdout)`

---

## Key Findings

### 1. Offense Fix Improved the Base Model Consistently

Comparing #1 (Baseline) → #2 (Offense Fix) on the same 2024 holdout:
- **F5 ML AUC**: 0.5854 → **0.5908** (+0.54 pts)
- **F5 RL AUC**: 0.5861 → **0.5995** (+1.34 pts)
- **F5 ML Brier (base XGB)**: 0.2448 → **0.2441** (better)
- **F5 RL Brier (base XGB)**: 0.2152 → **0.2137** (better)

The offense fix delivered real improvement at the XGBoost level.

### 2. Stacking Consistently Hurts F5 ML, Marginally Helps F5 RL

Across ALL 4 experiments:
- **F5 ML stacking Δ Brier**: -0.0003, -0.0002, -0.0002, -0.0008 → **always hurts**
- **F5 RL stacking Δ Brier**: +0.0003, +0.0002, +0.0004, +0.0005 → **always helps** (tiny)

The LR meta-learner with only 3 raw meta-features adds nothing for ML and barely helps RL. It's essentially passing through the XGBoost probability plus a slight Log5/Pythagorean blend.

### 3. Platt Calibration Hurts Every Single Run

Across ALL experiments, calibration Δ Brier is negative:
- **F5 ML**: -0.0011, -0.0012, +0.00002, -0.0028 → **hurts or neutral**
- **F5 RL**: -0.0022, -0.0029, -0.0007, -0.0025 → **always hurts**

The Platt calibration step is **destructive**. The stacking layer already produces well-calibrated probabilities (ECE is 1.4%–3.4% pre-calibration). Adding Platt on top compresses the probability range into just 2 bins (visible in every reliability diagram — after calibration, all predictions land in bins 4-5, destroying the probability spread that existed before).

### 4. Hyperparameter Search Always Lands on the Same Point

Every single run selected **`max_depth=6, n_estimators=100, learning_rate=0.01`**. This is the most conservative corner of the search space. The search space is too narrow and the RandomizedSearchCV with 15 iterations likely finds this early and can't explore further.

### 5. 2025 Holdout Shows Degradation

Run #3 (2025 holdout) shows noticeably worse AUC than 2024 holdout runs:
- **F5 ML AUC**: 0.5708 vs 0.5908 (2024 offense fix)
- **F5 RL AUC**: 0.5647 vs 0.5995 (2024 offense fix)

Possible explanations: 2025 season may be structurally different, model may need more recent data emphasis, or features from the training window are less predictive forward.

The 2025 holdout also **fails the reliability gap quality gate** (max gap 0.21 for ML, 0.23 for RL) — meaning calibration is poor on the most recent holdout season.

### 6. Shorter Training Window (2023+) Slightly Beats Longer (2018+) on 2024

Comparing #2 (2018-2023 train) vs #4 (2023 + backfill to 2019):
- **F5 ML AUC**: 0.5908 vs **0.5936** (+0.28 pts for shorter)
- **F5 RL AUC**: 0.5995 vs 0.5893 (-1.02 pts for shorter)

Mixed signal, but suggests recent data may be more predictive for ML while RL benefits from more history.

### 7. Walk-Forward ROI is Suspiciously High (+24.4%)

The 3-window Q2 2024 backtest shows +24.4% ROI on 1,013 bets (1 unit flat stakes). This is extremely high for sports betting. Key concerns:
- **Win rate**: 394W / 487L = 44.7% winners with average odds that must be >+130 equivalent
- **Market vig assumed at 4%**: Real market vig on F5 lines can be 5-8%
- **No real odds used**: The walk-forward generates synthetic odds from Log5 baselines, NOT from actual sportsbook lines. This massively overstates edge.
- **Betting on nearly every game**: 1,013 bets across 1,207 scored games = 84% bet rate at 3% edge threshold. A real model should be more selective.
- **Sample size**: Only 3 windows × 1 month each. Need at minimum 12+ windows for statistical significance.

### 8. Feature Importance Tells a Clear Story

Top features across ALL runs (remarkably stable):
1. **`home_team_log5_60g`** — Dominant feature (4-8% importance), team quality proxy
2. **`away_team_log5_60g`** / **`_30g`** — Same for away team
3. **`home_starter_k_pct_*`** — Starter strikeout rate across all windows
4. **`home_starter_xera_*`** — Expected ERA of home starter
5. **`away_starter_k_pct_*`** / **`xfip_*`** — Opposing starter quality
6. **`home_lineup_woba_7g`** / **`wrc_plus_7g`** — Short-window offense (only in offense-fix runs)

After the offense fix, lineup features (`home_lineup_woba_7g`, `away_lineup_woba_14g`) actually appear in the top-25, confirming the fix worked.

The model is **overwhelmingly driven by team quality proxies (Log5) and starter quality (K%, xERA, xFIP)**. Bullpen, defense, weather, and park factors contribute minimally.

---

## Concrete Recommendations Based on Evidence

### Immediate (from data)

1. **Drop Platt calibration** — Use `identity` or the raw stacking output. Calibration hurts in 7/8 measurements.

2. **Reconsider stacking for F5 ML** — The LR layer consistently worsens F5 ML. Either add more meta-features or skip stacking for ML and use raw XGBoost probabilities.

3. **Expand hyperparameter search** — Every run converges to the same `d6/n100/lr0.01`. Add `subsample`, `colsample_bytree`, `min_child_weight`, and increase search iterations to 50+. The walk-forward uses different params (`d3/n120/lr0.05`) — these differences are untested against each other.

4. **Don't trust the walk-forward ROI** — The +24.4% is based on synthetic odds derived from model baselines, not actual sportsbook lines. Need to run against historical Odds API data.

5. **Test training window sizes systematically** — The 2023+ start slightly beats 2018+ for ML. Run the holdout protocol with `start_year` in {2019, 2021, 2022, 2023} to find the optimal history depth.

### Near-term (from patterns)

6. **Prune low-importance features** — 224 features but importance is heavily concentrated in ~20. Features like `weather_*`, `abs_*`, `adjusted_framing_*`, many short-window defense metrics contribute near-zero importance. Dropping them may reduce overfitting.

7. **Add cross-team differential features to stacking** — Instead of just `home_team_log5_30g`, add `home_starter_k_pct_30s - away_starter_k_pct_30s`, `home_lineup_woba_30g - away_lineup_woba_30g`, etc.

8. **Run longer walk-forward** — The current Q2 backtest is only 3 windows. Extend to April-September 2024 (6 windows) for more robust evaluation.

---

## Training Data Summary

| Dataset | Hash | Years | Rows | Features | Built |
|---------|------|-------|------|----------|-------|
| `training_data_2018_2025` | `0f74492f` | 2018-2025 (skip 2020) | 17,035 | 224 | 2026-03-21 00:09 |
| `training_data_2023_2025` | `cb45d9d6` | 2019-2025 (incl. 2020!) | 15,502 | 224 | 2026-03-21 12:49 |

Note: `training_data_2023_2025` requested years 2023-2025 but backfilled to 2019 (7-season target). It includes 2020 (shortened season) — this differs from the 2018-2025 dataset which correctly skipped 2020. This inconsistency should be investigated.
