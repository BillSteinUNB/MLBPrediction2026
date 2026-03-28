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

**Timing**:

| Phase | Duration |
|-------|----------|
| Feature build | — |
| Optuna / training | — |
| Total | — |

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

**Timing**:

| Phase | Duration |
|-------|----------|
| Feature build | — |
| Optuna / training | — |
| Total | — |

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

## Run 3 — `2026-run-count-newfeatures-150x5` (Post-Prompts 1-6)

**Date**: March 26, 2026  
**Config**: ~103 features (pruned from expanded set), `count:poisson`, 150 Optuna trials × 5-fold CV, XGBoost+LightGBM blend (60/40)  
**New Features Added**: xwOBA, woba_minus_xwoba, barrel_pct, days_rest, last_start_pitch_count, cumulative_pitch_load_5s, team_woba_vs_LHP/RHP, offense_vs_starter_woba_gap  
**Training**: 14,601 rows (2018-2024) | **Holdout**: 2,434 rows (2025)

| Model | CV RMSE | Holdout RMSE | Holdout R² | Holdout MAE | Pred Mean | Actual Mean |
|-------|---------|-------------|------------|------------|-----------|-------------|
| f5_home_runs | 2.4240 | 2.3523 | 1.96% | | | |
| f5_away_runs | 2.3246 | 2.2880 | 4.48% | | | |
| full_game_home_runs | 3.0603 | 3.0846 | 2.71% | | | |
| full_game_away_runs | 3.1788 | 3.2688 | **5.16%** ← best so far | | | |

**Improvement vs Run 2**:

| Model | Run 2 RMSE | Run 3 RMSE | Delta | Run 2 R² | Run 3 R² | R² Delta |
|-------|-----------|-----------|-------|---------|---------|---------|
| f5_home_runs | 2.3552 | **2.3523** | **-0.0029** | 1.72% | **1.96%** | **+0.24pp** |
| f5_away_runs | 2.2907 | **2.2880** | **-0.0027** | 4.26% | **4.48%** | **+0.22pp** |
| full_game_home_runs | 3.0950 | **3.0846** | **-0.0104** | 2.06% | **2.71%** | **+0.65pp** |
| full_game_away_runs | 3.2752 | **3.2688** | **-0.0064** | 4.79% | **5.16%** | **+0.37pp** |

**Timing**:

| Phase | Duration |
|-------|----------|
| Feature build | 2h 10m |
| Optuna / training | — |
| Total | — |

---

## Run 4 — `2026-run-count-bugfix-250x7`

**Date**: March 26, 2026  
**Config**: 250 Optuna trials × 7-fold CV, `count:poisson`, XGBoost+LightGBM blend (60/40)  
**Changes**: Timezone bug fixes, woba_minus_xwoba repair attempt, vectorized feature build pipeline  
**Training**: 14,576 rows (2018-2024) | **Holdout**: 2,430 rows (2025)

| Model | Holdout RMSE | Holdout R² | Holdout MAE | Naive RMSE | RMSE Impr | Pred Mean | Actual Mean |
|-------|-------------|------------|------------|------------|-----------|-----------|-------------|
| f5_home_runs | 2.3604 | 1.42% | 1.8531 | 2.3779 | 0.73% | 2.595 | 2.620 |
| f5_away_runs | 2.3010 | 3.54% | 1.7944 | 2.3434 | 1.81% | 2.397 | 2.384 |
| full_game_home_runs | 3.1065 | 1.45% | 2.3982 | 3.1297 | 0.74% | 4.450 | 4.490 |
| full_game_away_runs | 3.3045 | 3.22% | 2.5982 | 3.3602 | 1.66% | 4.443 | 4.405 |

**Improvement vs Run 3**:

| Model | Run 3 R² | Run 4 R² | R² Delta | Run 3 RMSE | Run 4 RMSE | RMSE Delta |
|-------|---------|---------|---------|-----------|-----------|-----------|
| f5_home_runs | 1.96% | 1.42% | -0.54pp | 2.3523 | 2.3604 | +0.0081 |
| f5_away_runs | 4.48% | 3.54% | -0.94pp | 2.2880 | 2.3010 | +0.0130 |
| full_game_home_runs | 2.71% | 1.45% | -1.26pp | 3.0846 | 3.1065 | +0.0219 |
| full_game_away_runs | 5.16% | 3.22% | -1.94pp | 3.2688 | 3.3045 | +0.0357 |

**Best Hyperparameters**:

| Param | f5_home | f5_away | full_home | full_away |
|-------|---------|---------|-----------|-----------|
| max_depth | 4 | 3 | 3 | 3 |
| n_estimators (final) | 458 | 378 | 240 | 732 |
| learning_rate | 0.0073 | 0.0120 | 0.0223 | 0.0066 |
| subsample | 0.648 | 0.696 | 0.617 | 0.600 |

**Timing**:

| Phase | Duration |
|-------|----------|
| Feature build | (existing parquet reused) |
| Optuna / training | ~2h |
| Total | ~2h |

**Analysis**: All four models regressed vs Run 3 despite being a bugfix run. Root cause is **hyperparameter search variance** — with R² values of 1–5%, a ±1–2pp swing between runs is within expected noise from different Optuna trial outcomes. The intended fixes (timezone, woba_minus_xwoba) had weak underlying signal, and the Statcast data pipeline for `barrel_pct` and `woba_minus_xwoba` was later confirmed to still be broken at the source level. `full_game_home` early-stopped at only 240 trees, indicating very weak signal on that target. Run 4 is pre-feature-fix baseline; Run 5 is the first meaningful test.

**Notable**: `park_runs_factor` is the #1 or #2 feature in 3 of 4 models — confirming the model is hungry for dynamic environmental signal (weather). `home_lineup_woba_30g` showing top-5 importance despite being a constant-zero feature is evidence of overfitting to noise — resolved by the variance guard added in Run 5.

---

## Run 5 — `2026-run-count-features-fixed-250x7` (Pending)

**Date**: TBD  
**Config**: 250 Optuna trials × 7-fold CV, `reg:tweedie` (variance_power tuned by Optuna), XGBoost+LightGBM blend (60/40)  
**Changes vs Run 4**:
- Historical weather backfill via Open-Meteo (8 dead-constant features now have real variance)
- Team platoon splits from FanGraphs (wOBA vs LHP/RHP — 6 dead-constant features fixed)
- Retrosheet umpire data loaded (plate umpire known for ~70%+ of games)
- Bullpen ir_pct fixed (Statcast source)
- SIERA added to starter features (alongside xFIP/xERA)
- `reg:tweedie` objective replaces `count:poisson` (Var/Mean ≈ 2.2x in training data)
- 60g/60s window features re-enabled as Optuna candidates
- Variance guard on `_is_redundant_team_offense_feature` (keeps team-level when lineup is zero)
- `tweedie_variance_power` added to Optuna search space [1.3, 1.9]

**Training**: TBD | **Holdout**: 2025

| Model | Holdout RMSE | Holdout R² | Holdout MAE | Naive RMSE | RMSE Impr | Pred Mean | Actual Mean |
|-------|-------------|------------|------------|------------|-----------|-----------|-------------|
| f5_home_runs | | | | | | | |
| f5_away_runs | | | | | | | |
| full_game_home_runs | | | | | | | |
| full_game_away_runs | | | | | | | |

**Target thresholds**: f5_away > 6%, full_game_away > 7% R². If away models don't clear 6%, Statcast fixes (#3/#4) likely didn't fully land.

---

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

## Run 5 — `2026-run5-features-fixed-150x5`

**Date**: March 27, 2026
**Config**: 80 features, `reg:tweedie` (first use), 150×5, XGB+LGBM blend (60/40)
**Changes vs Run 4**: Real weather data, real woba_minus_xwoba, real barrel_pct, real SIERA. Platoon splits still broken (column name case bug: pipeline wrote `lhp` lowercase, parquet expected `LHP` uppercase — all platoon features still 0.320). 60g windows re-enabled.
**Training**: 14,576 rows | **Holdout**: 2,430 rows (2025)

| Model | Holdout RMSE | Holdout R² | Holdout MAE | n_est | lr | pred_mean | actual_mean |
|-------|-------------|------------|------------|-------|-----|-----------|-------------|
| f5_home_runs | 2.3570 | 1.71% | 1.8434 | 397 | 0.0072 | 2.565 | 2.620 |
| f5_away_runs | 2.2998 | 3.65% | 1.7900 | 446 | 0.0050 | 2.384 | 2.384 |
| full_game_home_runs | 3.1010 | 1.80% | 2.3894 | 337 | 0.0094 | 4.422 | 4.490 |
| full_game_away_runs | 3.3012 | 3.41% | 2.5861 | 236 | 0.0124 | 4.415 | 4.405 |

**Analysis**: Tweedie objective collapsed n_estimators to 236-446 (vs Run 3's 708). reg_alpha jumped to 7-9 (Optuna fighting Tweedie noise). Despite real feature data, all models worse than Run 3. Root cause: reg:tweedie gradient structure incompatible with this dataset size/type.

---

## Run 6 — `2026-run6-platoon-fix-500x10`

**Date**: March 27, 2026
**Config**: 80 features, `reg:tweedie`, 500×10, XGB+LGBM blend (60/40)
**Changes vs Run 5**: Platoon case bug fixed (LHP uppercase). 500×10 overnight search.
**Training**: 14,576 rows | **Holdout**: 2,430 rows (2025)

| Model | Holdout RMSE | Holdout R² | Holdout MAE | n_est | lr | pred_mean | actual_mean |
|-------|-------------|------------|------------|-------|-----|-----------|-------------|
| f5_home_runs | 2.3561 | 1.79% | 1.8436 | 235 | 0.0117 | 2.569 | 2.620 |
| f5_away_runs | 2.3015 | 3.50% | 1.7919 | 230 | 0.0081 | 2.387 | 2.384 |
| full_game_home_runs | 3.1006 | 1.82% | 2.3916 | 481 | 0.0056 | 4.438 | 4.490 |
| full_game_away_runs | 3.3002 | 3.47% | 2.5850 | 237 | 0.0145 | 4.411 | 4.405 |

**Analysis**: 10× more search (500×10 vs 150×5) made no difference — Tweedie objective is the bottleneck, not search depth. n_estimators still collapsed to 230-481. Confirmed: reg:tweedie must be reverted.

---

## Run 7 — `2026-run7-poisson-250x5`

**Date**: March 27, 2026
**Config**: 80 features, `count:poisson` (reverted), 250×5, 60g windows included
**Changes vs Run 6**: Reverted to count:poisson. New Optuna study (fresh trials).
**Training**: 14,576 rows | **Holdout**: 2,430 rows (2025)

| Model | Holdout RMSE | Holdout R² | Holdout MAE | n_est | lr | pred_mean | actual_mean |
|-------|-------------|------------|------------|-------|-----|-----------|-------------|
| f5_home_runs | 2.3568 | 1.73% | 1.8477 | 199 | 0.0247 | 2.586 | 2.620 |
| f5_away_runs | 2.2979 | 3.80% | 1.7913 | 300 | 0.0156 | 2.396 | 2.384 |
| full_game_home_runs | 3.1004 | 1.83% | 2.3925 | 549 | 0.0106 | 4.449 | 4.490 |
| full_game_away_runs | 3.3030 | 3.31% | 2.5920 | 237 | 0.0208 | 4.431 | 4.405 |

**Analysis**: Poisson restored reg_alpha to normal (0.001-0.16) but still below Run 3. Feature importance analysis revealed 60g windows displaced 7g short-term metrics. `away_lineup_xwoba_7g` (Run 3 importance 0.042) replaced by `weather_air_density_factor` (0.041). 80-feature budget can't hold both 7g form + 60g quality + new features simultaneously.

---

## Run 8 — `2026-run8-100feats-250x5`

**Date**: March 27, 2026
**Config**: **100 features** (bumped from 80), `count:poisson`, 250×5, 60g windows included
**Changes vs Run 7**: Feature count 80→100 to try to fit both 7g and 60g metrics.
**Training**: 14,576 rows | **Holdout**: 2,430 rows (2025)

| Model | Holdout RMSE | Holdout R² | Holdout MAE | n_est | lr | pred_mean | actual_mean |
|-------|-------------|------------|------------|-------|-----|-----------|-------------|
| f5_home_runs | 2.3570 | 1.71% | 1.8480 | 644 | 0.0078 | 2.588 | 2.620 |
| f5_away_runs | 2.3012 | 3.53% | 1.7979 | 407 | 0.0092 | 2.412 | 2.384 |
| full_game_home_runs | 3.1004 | 1.84% | 2.3913 | 648 | 0.0085 | 4.443 | 4.490 |
| full_game_away_runs | 3.3015 | 3.40% | 2.5898 | 199 | 0.0279 | 4.426 | 4.405 |

**Analysis**: 100 features added 32 new 60g window features and dropped 16 existing 7g/30g features. Mixed reg_alpha (9.99 on f5_home, 0.002 on f5_away) — Optuna finding inconsistent regularization, suggests feature noise. Still below Run 3. Reverted to 80 features and re-excluded 60g for Run 9.

**Current trainer config after Run 8**: 80 features, count:poisson, 60g excluded (same as Run 3 logic), real feature data in parquet.

---

## Summary Table: Runs 3–8

| Run | Config | f5_home | f5_away | fg_home | fg_away | Best model |
|-----|--------|---------|---------|---------|---------|-----------|
| **Run 3** | 80f, poisson, 150×5, broken data | 1.96% | 4.48% | 2.71% | **5.16%** | ← current best |
| Run 4 | 80f, poisson, 250×7, bugfix | 1.42% | 3.54% | 1.45% | 3.22% | regression |
| Run 5 | 80f, **tweedie**, 150×5, real data | 1.71% | 3.65% | 1.80% | 3.41% | tweedie kills n_est |
| Run 6 | 80f, **tweedie**, 500×10, real data | 1.79% | 3.50% | 1.82% | 3.47% | more search didn't help |
| Run 7 | 80f, poisson, 250×5, +60g | 1.73% | 3.80% | 1.83% | 3.31% | 60g crowds out 7g |
| Run 8 | **100f**, poisson, 250×5, +60g | 1.71% | 3.53% | 1.84% | 3.40% | wider but weaker |

**Run 9 hypothesis**: 80 features, count:poisson, 60g excluded (= Run 3 logic) + real feature data. If this beats Run 3, the new features genuinely help and a 500×10 overnight run is the next step.

---

## Run 9 - `2026-run9-away-only-500x5`

**Date**: March 27, 2026
**Config**: single-model research run (`full_game_away_runs` only), 80 features, `count:poisson`, 500x5, 60g windows excluded, bucketed selector (24 short-form / 36 medium-form / 20 context)
**Changes vs Run 8**: Re-excluded 60g/60s, restored broad Poisson search space, increased search to 500 trials, increased early stopping to 40, replaced flat Pearson top-80 with hard bucket quotas.
**Training**: 14,576 rows | **Holdout**: 2,430 rows (2025)

| Model | Holdout RMSE | Holdout R^2 | Holdout MAE | Naive RMSE | RMSE Impr | Pred Mean | Actual Mean |
|-------|-------------|------------|------------|------------|-----------|-----------|-------------|
| full_game_away_runs | 3.2982 | 3.59% | 2.5899 | 3.3602 | 1.84% | 4.429 | 4.405 |

**Improvement vs Recent Runs**:

| Compare To | Prior RMSE | Run 9 RMSE | RMSE Delta | Prior R^2 | Run 9 R^2 | R^2 Delta |
|-----------|-----------|-----------|-----------|---------|---------|---------|
| Run 7 | 3.3030 | **3.2982** | **-0.0048** | 3.31% | **3.59%** | **+0.28pp** |
| Run 8 | 3.3015 | **3.2982** | **-0.0032** | 3.40% | **3.59%** | **+0.19pp** |
| Run 3 | **3.2688** | 3.2982 | +0.0294 | **5.16%** | 3.59% | -1.57pp |

**Best Hyperparameters**:

| Param | full_game_away |
|-------|----------------|
| max_depth | 3 |
| n_estimators (requested/final) | 200 / 197 |
| learning_rate | 0.0315 |
| subsample | 0.642 |
| colsample_bytree | 0.750 |
| min_child_weight | 4 |
| gamma | 0.316 |
| reg_alpha | 0.00019 |
| reg_lambda | 3.53 |

**Feature Selection Diagnostics**:
- Bucket counts: short_form=24, medium_form=36, context=20
- Excluded candidates: 14-window=60, 60-window=68, redundant_team_offense=32
- Top importance mix: `home_starter_k_pct_30s`, `weather_air_density_factor`, `away_team_log5_30g`, `home_team_bullpen_xfip`, `home_team_log5_30g`
- Notable omission: the 7g away lineup features survived selection, but they did not rise to the top of model importance the way they did in Run 3

**Analysis**: This run rules out a simple "60g caused the whole regression" story. Removing 60g and spending 500 trials was not enough. The hard 24/36/20 quota system appears to be over-constraining the feature set and still leads Optuna toward short ensembles (200 trees) with mediocre CV. Result: slight improvement over Runs 7-8, but nowhere near Run 3. The next control should be **flat Pearson top-80 with 60g excluded** on the same current parquet and same 500x5 budget.

## Run 11 - `2026-run11-away-poisson-deviance-500x5`

**Date**: March 27, 2026
**Config**: single-model research run (`full_game_away_runs` only), 80 features, `count:poisson`, **Poisson deviance CV + Poisson-aligned early stopping**, 500x5, same rebuilt parquet as Run 10, same bucketed selector
**Changes vs Run 10**: switched Optuna ranking from RMSE to Poisson deviance, forced a Run-3-shaped search box (`n_estimators` 500-1000, lower learning rates, depth 3-5), kept the current 24/36/20 bucket selector.
**Training**: 14,576 rows | **Holdout**: 2,430 rows (2025)

| Model | Holdout RMSE | Holdout R² | Holdout MAE | Holdout Poisson Dev | Naive RMSE | RMSE Impr | Pred Mean | Actual Mean |
|-------|-------------|------------|------------|---------------------|------------|-----------|-----------|-------------|
| full_game_away_runs | 3.2997 | 3.50% | 2.5913 | 2.5163 | 3.3602 | 1.80% | 4.431 | 4.405 |

**Improvement vs Recent Runs**:

| Compare To | Prior RMSE | Run 11 RMSE | RMSE Delta | Prior R² | Run 11 R² | R² Delta |
|-----------|-----------|------------|-----------|---------|---------|---------|
| Run 10 | 3.3024 | **3.2997** | **-0.0027** | 3.34% | **3.50%** | **+0.16pp** |
| Run 9 | **3.2982** | 3.2997 | +0.0015 | **3.59%** | 3.50% | -0.09pp |
| Run 3 | **3.2688** | 3.2997 | +0.0309 | **5.16%** | 3.50% | -1.66pp |

**Best Hyperparameters**:

| Param | full_game_away |
|-------|----------------|
| max_depth | 4 |
| n_estimators (requested/final) | 500 / 497 |
| learning_rate | 0.0100 |
| subsample | 0.750 |
| colsample_bytree | 0.600 |
| min_child_weight | 5 |
| gamma | 0.300 |
| reg_alpha | 0.010 |
| reg_lambda | 0.25 |

**Analysis**: The trainer-side change **did fix the pathological short-ensemble behavior**: the run moved from the old `200 -> 197` regime (Runs 9-10) to a much healthier `500 -> 497` profile with depth 4 and learning rate 0.01, much closer to Run 3's shape. However, the holdout result barely moved and still missed Run 9 slightly. Top importance remains dominated by `log5`, weather, bullpen xFIP, and 30g pitching/context, while the away 7g lineup features still do not return to Run 3-style prominence. Conclusion: **metric alignment improved trainer behavior, but feature representation/selection remains the bottleneck**. The next change should target the feature layer (most likely temporal delta features or a less blunt selector), not more search.

## Run 10 — `2026-run10-away-only-250x5`

**Date**: March 27, 2026
**Config**: single-model research run (`full_game_away_runs` only), 80 features, `count:poisson`, 250×5, 60g windows excluded, bucketed selector (24 short-form / 36 medium-form / 20 context)
**Changes vs Run 9**: Same config as Run 9 but with 250 trials instead of 500, testing if reduced search still captures good hyperparameters.
**Training**: 14,576 rows | **Holdout**: 2,430 rows (2025)

| Model | Holdout RMSE | Holdout R² | Holdout MAE | Naive RMSE | RMSE Impr | Pred Mean | Actual Mean |
|-------|-------------|------------|------------|------------|-----------|-----------|-------------|
| full_game_away_runs | 3.3024 | 3.34% | 2.5917 | 3.3602 | 1.72% | 4.418 | 4.405 |

**Improvement vs Recent Runs**:

| Compare To | Prior RMSE | Run 10 RMSE | RMSE Delta | Prior R² | Run 10 R² | R² Delta |
|-----------|-----------|------------|-----------|---------|---------|---------|
| Run 9 | **3.2982** | 3.3024 | +0.0042 | **3.59%** | 3.34% | -0.25pp |

**Best Hyperparameters**:

| Param | full_game_away |
|-------|----------------|
| max_depth | 3 |
| n_estimators (requested/final) | 200 / 197 |
| learning_rate | 0.0297 |
| subsample | 0.601 |
| colsample_bytree | 0.939 |
| min_child_weight | 7 |
| gamma | 0.090 |
| reg_alpha | 7.03 |
| reg_lambda | 3.10 |

**Analysis**: Short ensemble behavior returned (200 → 197 trees) with high reg_alpha (7.03), similar to Run 9's pathology. The 250-trial budget found a different local optimum that performed slightly worse than Run 9. Confirms that search variance is significant and larger budgets (500+) are needed for stability.

---

## Run 12 — `2026-run12-away-deltas-poisson-parallel-500x5`

**Date**: March 27, 2026
**Config**: single-model research run (`full_game_away_runs` only), 80 features, `count:poisson`, 500×5, **parallel Optuna (2 workers)**, temporal delta features added
**Changes vs Run 11**: Added temporal delta features (7g vs 30g differences), switched to parallel Optuna search with 2 workers for faster iteration.
**Training**: 14,576 rows | **Holdout**: 2,430 rows (2025)

**Note**: Run 12 artifacts were not retained; see Run 12.5 for the follow-up with refined delta features.

---

## Run 12.5 — `2026-run12.5-away-deltas-poisson-parallel-500x5`

**Date**: March 27, 2026
**Config**: single-model research run (`full_game_away_runs` only), 80 features, `count:poisson`, **500×5 parallel (2 workers)**, Poisson deviance CV metric, refined delta features
**Changes vs Run 11**: Added temporal delta features (velocity deltas, form deltas, matchup deltas), parallel Optuna search, maintained Poisson deviance objective from Run 11.
**Training**: 14,576 rows | **Holdout**: 2,430 rows (2025)

| Model | Holdout RMSE | Holdout R² | Holdout MAE | Holdout Poisson Dev | Naive RMSE | RMSE Impr | Pred Mean | Actual Mean |
|-------|-------------|------------|------------|---------------------|------------|-----------|-----------|-------------|
| full_game_away_runs | 3.3026 | 3.33% | 2.5937 | 2.5209 | 3.3602 | 1.71% | 4.430 | 4.405 |

**Improvement vs Recent Runs**:

| Compare To | Prior RMSE | Run 12.5 RMSE | RMSE Delta | Prior R² | Run 12.5 R² | R² Delta |
|-----------|-----------|--------------|-----------|---------|-----------|---------|
| Run 11 | 3.2997 | 3.3026 | +0.0029 | 3.50% | 3.33% | -0.17pp |
| Run 10 | 3.3024 | 3.3026 | +0.0002 | 3.34% | 3.33% | -0.01pp |
| Run 9 | **3.2982** | 3.3026 | +0.0044 | **3.59%** | 3.33% | -0.26pp |
| **Run 3** | **3.2688** | 3.3026 | +0.0338 | **5.16%** | 3.33% | **-1.83pp** |

**Best Hyperparameters**:

| Param | full_game_away |
|-------|----------------|
| max_depth | 3 |
| n_estimators (requested/final) | 700 / 700 |
| learning_rate | 0.0075 |
| subsample | 0.65 |
| colsample_bytree | 0.70 |
| min_child_weight | 5 |
| gamma | 0.0 |
| reg_alpha | 0.0001 |
| reg_lambda | 0.25 |

**Key Metrics**:
- CV Poisson Deviance: 2.3429
- Holdout Poisson Deviance: 2.5209
- Naive Poisson Deviance: 2.6045
- Poisson Deviance Improvement: 3.21%

**Analysis**: Despite healthy ensemble size (700 trees, no early stopping) and parallel search with 500 trials, the delta features did **not** improve holdout performance. R² remains stuck at ~3.3%, well below Run 3's 5.16%. The model is learning training patterns (good CV deviance) but not generalizing. The temporal delta features may be introducing noise or the feature selection bucket quotas are still over-constraining the signal. **Conclusion**: The bottleneck is not search depth or objective alignment—it's feature representation/selection.

---

## What "Good" Looks Like

| Level | Holdout RMSE (F5) | R² | vs Naive | What it means |
|-------|-------------------|-----|----------|---------------|
| **Baseline** | 2.35 | 1.8% | 4.3% | Barely above guessing the mean |
| **Run 2 (current)** | 2.29-2.36 | 1.7-4.8% | 0.9-2.5% | Cleaner model, modest signal |
| **Meaningful** | < 2.25 | > 5% | > 8% | Model genuinely learns patterns |
| **Good** | < 2.15 | > 8% | > 12% | Competitive for betting edge |
| **Excellent** | < 2.05 | > 12% | > 16% | Strong predictive model |
