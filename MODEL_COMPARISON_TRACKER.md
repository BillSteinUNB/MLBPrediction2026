# MLB Run-Count Model: Training Run Comparison Tracker

**Purpose**: Track RMSE/R² improvements across training runs as phases are implemented.

---

## Baseline Run — `2026-run-count-100x5` (Pre-Phase Changes)

**Date**: March 25, 2026  
**Config**: 210 features, `reg:squarederror`, 100 Optuna trials × 5-fold CV, XGBoost only  
**Training**: 14,601 rows (2018-2024) | **Holdout**: 2,434 rows (2025)

| Model | CV RMSE | Holdout RMSE | Holdout R² | Holdout MAE | Pred Mean | Actual Mean |
|-------|---------|-------------|------------|------------|-----------|-------------|
| f5_home_runs | 2.4245 | 2.3537 | 0.0185 | 1.8386 | 2.550 | 2.620 |
| f5_away_runs | 2.3273 | 2.2971 | 0.0372 | 1.7869 | 2.376 | 2.384 |
| full_game_home_runs | 3.0643 | 3.0933 | 0.0216 | 2.3820 | 4.377 | 4.489 |
| full_game_away_runs | 3.1813 | 3.2801 | 0.0450 | 2.5646 | 4.395 | 4.404 |

**Naive Baselines** (predict-the-mean):

| Target | Naive RMSE | Model RMSE | Improvement |
|--------|-----------|------------|-------------|
| f5_home_score | 2.4595 | 2.3537 | 4.3% |
| f5_away_score | 2.3479 | 2.2971 | 2.2% |
| final_home_score | 3.1418 | 3.0933 | 1.5% |
| final_away_score | 3.2532 | 3.2801 | **-0.8% (worse!)** |

**Best Hyperparameters** (note heavy regularization = noisy features):

| Param | f5_home | f5_away | full_home | full_away |
|-------|---------|---------|-----------|-----------|
| max_depth | 3 | 4 | 3 | 5 |
| n_estimators | 300 | 500 | 200 | 400 |
| learning_rate | 0.0083 | 0.0054 | 0.0173 | 0.0076 |
| gamma | **3.59** | **3.64** | **3.44** | **2.33** |
| reg_alpha | **7.82** | 1.52 | 0.0002 | **3.71** |
| min_child_weight | 9 | 7 | 8 | 7 |

---

## Run 2 — `2026-run-count-full-refresh-150x5` (Post-Phase 1+2+3)

**Date**: March 25, 2026  
**Config**: ~80 features (pruned from 210), `count:poisson`, 150 Optuna trials × 5-fold CV, XGBoost+LightGBM blend (60/40)  
**New Features Added**: CSW%, velo_delta_7v60s, offense_vs_starter_woba_gap, day_after_night_game, timezone_crossings_east  
**Architecture Changes**: Poisson objective, LightGBM blend, Skellam for spreads, Negative Binomial for totals (overdispersion=2.3)  
**Training**: 14,601 rows (2018-2024) | **Holdout**: 2,434 rows (2025)

| Model | CV RMSE | Holdout RMSE | Holdout R² | Holdout MAE | Pred Mean | Actual Mean |
|-------|---------|-------------|------------|------------|-----------|-------------|
| f5_home_runs | 2.4251 | 2.3552 | 0.0172 | 1.8393 | 2.546 | 2.620 |
| f5_away_runs | 2.3293 | 2.2907 | 0.0426 | 1.7814 | 2.372 | 2.384 |
| full_game_home_runs | 3.0641 | 3.0950 | 0.0206 | 2.3849 | 4.401 | 4.489 |
| full_game_away_runs | 3.1842 | 3.2752 | 0.0479 | 2.5671 | 4.408 | 4.404 |

**vs Naive Baselines**:

| Target | Naive RMSE | Model RMSE | Improvement |
|--------|-----------|------------|-------------|
| f5_home_score | 2.3762 | 2.3552 | 0.88% |
| f5_away_score | 2.3416 | 2.2907 | 2.17% |
| final_home_score | 3.1277 | 3.0950 | 1.05% |
| final_away_score | 3.3577 | 3.2752 | **2.46%** |

**Improvement vs Baseline**:

| Model | Baseline RMSE | Run 2 RMSE | Delta | Baseline R² | Run 2 R² | R² Delta |
|-------|--------------|-----------|-------|-------------|---------|---------|
| f5_home_runs | 2.3537 | 2.3552 | +0.0015 (flat) | 1.85% | 1.72% | -0.13pp |
| f5_away_runs | 2.2971 | **2.2907** | **-0.0064** | 3.72% | **4.26%** | **+0.54pp** |
| full_game_home_runs | 3.0933 | 3.0950 | +0.0017 (flat) | 2.16% | 2.06% | -0.10pp |
| full_game_away_runs | 3.2801 | **3.2752** | **-0.0049** | 4.50% | **4.79%** | **+0.29pp** |

**Best Hyperparameters** (regularization collapsed — features are cleaner):

| Param | f5_home | f5_away | full_home | full_away |
|-------|---------|---------|-----------|-----------|
| max_depth | 3 | 4 | 3 | 4 |
| n_estimators | 1000 | 700 | 350 | 700 |
| learning_rate | 0.0069 | 0.0085 | 0.0175 | 0.0097 |
| gamma | **0.39** ✅ | **0.10** ✅ | **0.43** ✅ | **0.32** ✅ |
| reg_alpha | 9.97 | **0.10** ✅ | **0.016** ✅ | 1.18 ✅ |
| min_child_weight | 7 | 7 | 7 | 4 |

**Key Changes Checklist**:
- [x] Did gamma drop below 1.0? **YES — ALL 4 models.** From 2.3-3.6 → 0.1-0.4. Massive structural improvement.
- [ ] Did R² cross 0.05? **No.** Best is 4.79% (full_game_away). Still below 5% threshold.
- [ ] Did holdout RMSE beat naive by >5%? **No.** Best is 2.46% (full_game_away).
- [x] Did n_estimators increase? **YES.** f5_home: 300→1000. f5_away: 500→700. full_away: 400→700.

**New Features That Made the Top-80 Selection**:
- `velo_delta_7v60s` — in f5_home (#21 importance), f5_away (#20), full_game_away (selected)
- `offense_vs_starter_woba_gap` — in f5_home, f5_away, full_game_away (#23 importance)
- `is_day_after_night_game` — in f5_home, full_game_home (selected)
- `timezone_crossings_east` — in f5_home, f5_away, full_game_away (selected)

**Analysis**: The regularization collapse confirms the Phase 1 feature pruning worked — the model no longer needs to strangle itself. However, RMSE barely moved because the underlying signal in these features is limited. The away models showed modest improvement while home models stayed flat. The new Phase 2 features (velo_delta, matchup gap, travel) were selected but are mid-tier importance — they add breadth, not depth.

---

## Run 3 — Post-Prompts 1-6 (Pending)

**Date**: TBD  
**Config**: TBD (after xwOBA, barrel rate, starter rest, team batting splits, classifier are implemented)  
**New Features Expected**: xwOBA, woba_minus_xwoba, barrel_pct, starter_days_rest, starter_last_start_pitch_count, starter_cumulative_pitch_load_5s, team_woba_vs_LHP, team_woba_vs_RHP, team_woba_vs_opposing_hand

| Model | CV RMSE | Holdout RMSE | Holdout R² | Holdout MAE | Pred Mean | Actual Mean |
|-------|---------|-------------|------------|------------|-----------|-------------|
| f5_home_runs | | | | | | |
| f5_away_runs | | | | | | |
| full_game_home_runs | | | | | | |
| full_game_away_runs | | | | | | |

**Win Classifier Results** (new model type):

| Model | Accuracy | Log Loss | AUC-ROC | Naive Accuracy | Improvement |
|-------|----------|----------|---------|----------------|-------------|
| full_game_win | | | | | |
| f5_win | | | | | |

---

## Key Metrics Reference

| Target | Mean | Std Dev | Naive RMSE | Zeros % |
|--------|------|---------|------------|---------|
| f5_home_score | 2.659 | 2.460 | 2.4595 | 20.0% |
| f5_away_score | 2.425 | 2.348 | 2.3479 | 22.7% |
| final_home_score | 4.532 | 3.142 | 3.1418 | 5.9% |
| final_away_score | 4.481 | 3.253 | 3.2532 | 6.9% |

**Home/away score correlation**: ~0.00 (both F5 and full game — confirms correlation=0.0 is correct)

---

## What "Good" Looks Like

| Level | Holdout RMSE (F5) | R² | vs Naive | What it means |
|-------|-------------------|-----|----------|---------------|
| **Baseline** | 2.35 | 1.8% | 4.3% | Barely above guessing the mean |
| **Run 2 (current)** | 2.29-2.36 | 1.7-4.8% | 0.9-2.5% | Cleaner model, modest signal |
| **Meaningful** | < 2.25 | > 5% | > 8% | Model genuinely learns patterns |
| **Good** | < 2.15 | > 8% | > 12% | Competitive for betting edge |
| **Excellent** | < 2.05 | > 12% | > 16% | Strong predictive model |
