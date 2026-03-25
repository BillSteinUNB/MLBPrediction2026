# MLB Betting Model: Comprehensive Audit Report
**Date**: March 25, 2026  
**Scope**: Full codebase audit + external research comparison  
**Code Changes**: None (analysis only)

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [What Your Model Currently Does](#2-what-your-model-currently-does)
   - [Bet Types Supported](#21-bet-types-supported)
   - [Feature Engineering (All ~224 Features)](#22-feature-engineering)
   - [ML Pipeline Architecture](#23-ml-pipeline-architecture)
   - [Decision Engine](#24-decision-engine)
   - [Walk-Forward Backtesting](#25-walk-forward-backtesting)
   - [Known Issues from Prior Audits](#26-known-issues)
3. [What's Missing (Gap Analysis)](#3-whats-missing)
   - [Critical Gaps](#31-critical-gaps)
   - [High-Priority Gaps](#32-high-priority-gaps)
   - [Medium-Priority Gaps](#33-medium-priority-gaps)
4. [Research: What Profitable Models Do](#4-research-what-profitable-models-do)
   - [Features Used by Top Models](#41-features-used-by-top-models)
   - [Model Architectures That Work](#42-model-architectures-that-work)
   - [What Makes Models Actually Profitable](#43-what-makes-models-actually-profitable)
   - [Notable Open-Source MLB Models](#44-notable-open-source-mlb-models)
5. [Recommended Target Architecture](#5-recommended-target-architecture)
6. [Prioritized Roadmap](#6-prioritized-roadmap)
7. [Quick-Reference Comparison Table](#7-quick-reference-comparison-table)

---

## 1. Executive Summary

Your system is a **First-5-innings-only** MLB betting model supporting two markets: **F5 Moneyline** and **F5 Run Line**. The engineering foundation is solid — temporal train/test splits, anti-leakage assertions, de-vig math, walk-forward backtesting, Kelly sizing, and drawdown protection are all well-implemented.

However, compared to what profitable MLB betting models use, there are **critical gaps** in:
- **Market coverage**: No full-game bets, no totals (the highest-edge niche)
- **Matchup features**: No platoon splits, no batter-vs-pitcher data, no Stuff+ metrics
- **Weather**: Mostly defaults to neutral; precipitation probability not even parsed from the API
- **Evaluation methodology**: CLV (Closing Line Value) exists in code but isn't the primary metric; prior backtests used synthetic odds creating circular edge measurement

The +24.4% ROI from prior backtests is an artifact of methodology issues (synthetic odds, in-sample strategy optimization), not real expected performance. Realistic world-class MLB model performance is **3-6% ROI**.

**Bottom line**: The code quality is genuinely good. The gaps are in *what* is modeled, not *how* it's modeled. The biggest wins will come from fixing evaluation (CLV-first), adding platoon splits and better pitcher metrics, building a totals model, and expanding to full-game markets.

---

## 2. What Your Model Currently Does

### 2.1 Bet Types Supported

The system supports **exactly 2 bet types**:

| Bet Type | Market Key | Status |
|----------|-----------|--------|
| First 5 Innings Moneyline | `f5_ml` | Full pipeline (model → edge → sizing → placement → settlement) |
| First 5 Innings Run Line | `f5_rl` | Full pipeline (model → edge → sizing → placement → settlement) |

**Evidence**: `MarketType = Literal["f5_ml", "f5_rl"]` in `src/models/_base.py`. The `edge_calculations` SQLite table has a `CHECK` constraint enforcing only these two values. No full-game moneyline, no totals, no team totals, no player props exist anywhere in the system.

**What's fetched but NOT modeled**: The `odds_client.py` fetches full-game moneyline and spreads via `fetch_mlb_full_game_odds_context()`, but these are **only** used to estimate F5 ML odds when real F5 odds aren't available (via a `shrink_factor=0.78` conversion). The full-game data is never modeled directly.

### 2.2 Feature Engineering

**Total**: ~224 numeric features across 10 categories, computed per game for home/away teams.

#### Offense Features (`src/features/offense.py`)
| Feature | Windows | Source | Notes |
|---------|---------|--------|-------|
| `wrc_plus` | 7g, 14g, 30g, 60g | Statcast team game logs | Weighted Runs Created+ |
| `woba` | 7g, 14g, 30g, 60g | Statcast batting stats | Weighted On-Base Average |
| `iso` | 7g, 14g, 30g, 60g | Statcast batting stats | Isolated Power |
| `babip` | 7g, 14g, 30g, 60g | Statcast batting stats | Batting Avg on Balls in Play |
| `k_pct` | 7g, 14g, 30g, 60g | Statcast batting stats | Strikeout Rate |
| `bb_pct` | 7g, 14g, 30g, 60g | Statcast batting stats | Walk Rate |

**Lineup-level aggregation**: If `lineup_player_ids` are supplied, these same metrics are computed per-batter and aggregated across the lineup weighted by plate appearances, producing `{home|away}_lineup_{metric}_{window}g` features. Falls back to team-level when lineups unavailable.

**What's NOT here**: No handedness info, no platoon splits vs pitcher hand, no BvP history.

#### Pitching Features (`src/features/pitching.py`)
| Feature | Windows | Source | Notes |
|---------|---------|--------|-------|
| `xfip` | 7s, 14s, 30s, 60s | Statcast pitch-level | Expected FIP (HR-normalized) |
| `xera` | 7s, 14s, 30s, 60s | Statcast pitch-level | Expected ERA |
| `k_pct` | 7s, 14s, 30s, 60s | Statcast pitch-level | Strikeout rate |
| `bb_pct` | 7s, 14s, 30s, 60s | Statcast pitch-level | Walk rate |
| `gb_pct` | 7s, 14s, 30s, 60s | Statcast pitch-level | Ground ball rate |
| `hr_fb_pct` | 7s, 14s, 30s, 60s | Statcast pitch-level | HR per fly ball |
| `avg_fastball_velocity` | 7s, 14s, 30s, 60s | Statcast pitch-level | Average FB velo |
| `pitch_mix_entropy` | 7s, 14s, 30s, 60s | Statcast pitch-level | Pitch type diversity |
| `is_opener` | N/A | Lineup client | Opener detection |
| `uses_team_composite` | N/A | Lineup client | Falls back to team composite when starter history insufficient |

**What's NOT here**: No Stuff+, no CSW%, no SIERA, no velocity *trend* detection (decline vs baseline), no pitcher splits vs LHB/RHB, no pitch-type matchup features.

#### Defense Features (`src/features/defense.py`)
| Feature | Windows | Source | Notes |
|---------|---------|--------|-------|
| `drs` | season, 30g, 60g | Statcast fielding data | Defensive Runs Saved |
| `oaa` | season, 30g, 60g | Statcast fielding data | Outs Above Average |
| `defensive_efficiency` | season, 30g, 60g | Statcast fielding data | Balls in play converted to outs |
| `adjusted_framing` | season | Statcast framing data | Catcher framing (ABS-adjusted) |

#### Bullpen Features (`src/features/bullpen.py`)
| Feature | Windows | Source | Notes |
|---------|---------|--------|-------|
| `pitch_count` | 3d, 5d | Statcast pitch-level relief data | Recent workload |
| `avg_rest_days_top5` | Recent | Statcast pitch-level relief data | Top reliever availability |
| `ir_pct` | 30g | Statcast pitch-level relief data | Inherited runners scored % |
| `xfip` | 30g | Statcast pitch-level relief data | Bullpen quality metric |
| `high_leverage_available_count` | Current | Statcast pitch-level relief data | Key arms available |

#### Baseline Features (`src/features/baselines.py`)
| Feature | Windows | Source | Notes |
|---------|---------|--------|-------|
| `pythagorean_wp` | 30g, 60g | Internal games table | Expected W% from runs scored/allowed |
| `f5_pythagorean_wp` | 30g, 60g | Internal games table | Same but F5-specific |
| `log5` | 30g, 60g | Internal games table | Log5 win probability |

**Self-contained**: Computed purely from historical game scores stored locally. No external API needed.

#### Umpire Features (`src/features/umpires.py`)
| Feature | Windows | Source | Notes |
|---------|---------|--------|-------|
| `plate_umpire_known` | N/A | Retrosheet gameinfo | Binary: is umpire identified? |
| `plate_umpire_home_win_pct` | 30g, 90g | Retrosheet gameinfo | Umpire's home team win rate |
| `plate_umpire_total_runs_avg` | 30g, 90g | Retrosheet gameinfo | Mean total runs in umpire's games |
| `plate_umpire_f5_total_runs_avg` | 30g, 90g | Retrosheet gameinfo | Mean F5 total runs |
| `plate_umpire_sample_size` | 30g, 90g | Retrosheet gameinfo | Games available for umpire |

**Defaults applied** when Retrosheet data unavailable (which is common for live games early in season).

#### Weather Features (`src/features/adjustments/weather.py`)
| Feature | Source | Notes |
|---------|--------|-------|
| `weather_temp_factor` | OpenWeather forecast API | Linear factor per °F from 70°F baseline |
| `weather_air_density_factor` | OpenWeather forecast API | Lower density → ball carries farther |
| `weather_humidity_factor` | OpenWeather forecast API | Humidity adjustment |
| `weather_wind_factor` | OpenWeather forecast API | Signed projected wind (positive = blowing out to CF) |
| `weather_rain_risk` | **Humidity proxy** (NOT real precip data) | OpenWeather `pop` field is NOT parsed |
| `weather_composite` | Derived | Product of all weather factors |
| `weather_data_missing` | Flag | 1.0 when API unavailable or dome stadium |

**Critical weather issues**:
- Precipitation probability (`pop`) is available in the OpenWeather response but **never extracted**. Rain risk is computed as a humidity proxy.
- Wind gusts are not used, only average wind speed/direction.
- All factors are clamped to ±15% (`MIN_FACTOR=0.85, MAX_FACTOR=1.15`), which is very conservative.
- **Dome stadiums** return neutral defaults (70°F, 50% humidity, 0 wind).
- **No historical weather data** for training/backtesting (only 5-day forecast API).

#### Park Factor Features (`src/features/adjustments/park_factors.py`)
| Feature | Source | Notes |
|---------|--------|-------|
| `park_runs_factor` | `config/settings.yaml` | Static run factor per stadium |
| `park_hr_factor` | `config/settings.yaml` | Static HR factor per stadium |

#### ABS Adjustment Features (`src/features/adjustments/abs_adjustment.py`)
| Feature | Source | Notes |
|---------|--------|-------|
| `abs_active` | `config/settings.yaml` | Whether ABS (automated ball-strike) is active at park |
| `abs_walk_rate_delta` | `config/settings.yaml` | Expected walk rate change from ABS |
| `abs_strikeout_rate_delta` | `config/settings.yaml` | Expected strikeout rate change from ABS |

#### Marcel Blending (`src/features/marcel_blend.py`)
Features are blended with league-average priors using Marcel-style regression, weighting recent data more heavily. This prevents extreme values from small samples.

### 2.3 ML Pipeline Architecture

The system uses a **multi-layer model stack** for each bet type:

```
Layer 1: Base XGBoost Classifier
  ├── f5_ml_model: predicts f5_ml_result (binary: home wins F5)
  └── f5_rl_model: predicts f5_rl_result (binary: home covers RL)
         │
         ▼
Layer 2: Stacking Ensemble
  ├── Temporal OOF probabilities from XGBoost (warmup-augmented)
  ├── + 3 meta-features: pythagorean_wp_30g, log5_30g, park_runs_factor
  └── Logistic Regression meta-learner
         │
         ▼
Layer 3: Calibration
  ├── Identity (default) / Isotonic / Platt / Blend
  └── Dedicated calibration split (10% of pre-holdout data)
         │
         ▼
Layer 4: Market Recalibration
  └── Shrinks model probability toward de-vigged market probability
```

**Additional RL Models** (alternative paths for F5 Run Line):
| Model | File | Target | Notes |
|-------|------|--------|-------|
| Direct RL | `direct_rl_trainer.py` | `home_cover_at_posted_line` | XGBoost classifier using posted-line market features |
| Margin Model | `margin_trainer.py` | `f5_margin` (continuous) | XGBoost regressor → converts to cover probability via normal CDF + residual_std |

**Training Details**:
- **Hyperparameter tuning**: Optuna (100 iterations default) for base models; RandomizedSearchCV for direct RL/margin
- **CV strategy**: `TimeSeriesSplit` with 5 folds (temporal, not random)
- **Early stopping**: Temporal validation tail slice, 20 rounds default
- **Anti-leakage**: `assert_training_data_is_leakage_free()` checks `as_of_timestamp < scheduled_start`
- **Holdout**: Last season held out for final evaluation

**Model Promotion Logic** (`src/model/promotion.py`):
Selection priority: lowest holdout `log_loss` → lowest `brier` → highest `roc_auc` → highest `accuracy` → variant priority (base=0, calibrated=1, stacking=2). This is intentionally conservative (prefers base model unless stacking/calibration clearly improve).

### 2.4 Decision Engine

**End-to-end flow** (per game, per odds snapshot):

```
1. Build inference feature frame (src/pipeline/daily.py)
2. Load best model artifacts (base → stacking → calibrated, whichever was promoted)
3. Predict: model_probability for home/away for each market (f5_ml, f5_rl)
4. For RL: optionally blend legacy, direct_rl, margin_rl probabilities
5. Apply market recalibration (shrink toward de-vigged market probability)
6. Calculate edge:
   a. De-vig odds pair → fair_probability
   b. edge_pct = model_probability - fair_probability
   c. EV = expected_value(model_probability, odds_at_bet)
7. Filter: is_positive_ev when edge_pct >= threshold (configurable)
8. Kelly sizing:
   a. Full Kelly fraction = ((b * p) - (1 - p)) / b
   b. Apply fraction (quarter-Kelly default)
   c. Cap at MAX_BET_FRACTION = 5% of bankroll
9. Correlation handling: If ML + RL on same game/side, pick best single exposure
10. Kill-switch: Suppress all bets if drawdown >= 30%
11. Place bet: Update bankroll ledger, log to SQLite, freeze odds
12. Notify via Discord
```

**Settlement** (`src/engine/settlement.py`):
- **f5_ml**: Compare 5-inning home/away scores. Tied = PUSH. Starter scratched or <5 innings = NO_ACTION.
- **f5_rl**: If `line_at_bet` exists, compute `margin + line` → >0 WIN, <0 LOSS, =0 PUSH. Otherwise default ±1.5 heuristic.

### 2.5 Walk-Forward Backtesting

**File**: `src/backtest/walk_forward.py`

- **Window construction**: Monthly windows by default (6-month train, 1-month test). Modes: "rolling" or "anchored_expanding".
- **Per-window**: Trains fresh stacking + calibration model (not persisted globally). Uses same OOF/warmup strategy as production training.
- **Odds handling**: If historical odds DB provided, loads and de-vigs real odds. Otherwise creates synthetic market probs from `log5 + DEFAULT_MARKET_VIG` (this is a critical issue — see Known Issues).
- **Bankroll simulation**: Simulates flat/Kelly/edge-scaled/edge-bucketed staking across windows.
- **Output**: CSV/JSON of predictions, per-window metrics, aggregate ROI/drawdown.

### 2.6 Known Issues (from March 24, 2026 Audits)

| ID | Issue | Severity | Impact |
|----|-------|----------|--------|
| C1 | Synthetic odds from Log5 (circular edge measurement) | Critical | All ROI metrics unreliable when real odds not used |
| C2 | Strategy parameters optimized in-sample | Critical | Edge thresholds, Kelly fractions tuned on same data they're evaluated on |
| C3 | Closing vs opening line mismatch | High | Backtests may use closing lines but real bets placed at opening |
| C4 | +24.4% ROI is an artifact | Critical | Not representative of expected real performance |
| C5 | Stacking hurts F5 ML in all 4 experiments | High | Meta-learner with only 3 features doesn't add value |
| C6 | Calibration hurts in 7 of 8 measurements | High | Identity calibrator (no-op) consistently best |
| C7 | 224 features but only ~20 carry real importance | Medium | Massive overfitting risk at 10:1 feature-sample ratio |
| C8 | 84% bet rate (should be 10-30%) | Medium | Indicates insufficient selectivity / low edge thresholds |

---

## 3. What's Missing (Gap Analysis)

### 3.1 Critical Gaps

#### Gap 1: No Full-Game Moneyline Model
- **Current**: Only F5 markets modeled.
- **What's needed**: Full-game ML is the most liquid market. Your `odds_client.py` already fetches full-game odds — you just don't model them.
- **Impact**: Missing the largest betting market entirely.
- **To implement**: Add `MarketType "full_ml"`, train a full-game win probability model, add settlement logic using final scores, extend edge calculator CHECK constraint.

#### Gap 2: No Totals / Over-Under Model
- **Current**: Zero totals modeling anywhere in the codebase. Grep across all `.py` files for "totals", "over_under", "team_total", "game_total", "player_prop" returns nothing relevant.
- **What's needed**: Run-scoring models (Poisson regression is the standard approach) predict expected runs per team. This naturally gives: game totals, F5 totals, team totals.
- **Why this matters**: Weather and park factors have **significantly more predictive value** for totals than for sides. Your weather features are essentially wasted on a sides-only model.
- **Impact**: Missing the highest-edge niche market. F5 totals and team totals are where many profitable models focus because these markets are less efficiently priced.

#### Gap 3: No Platoon Split Features
- **Current**: **Zero** handedness information is used anywhere. The lineup client doesn't extract batter handedness. Pitching features don't include splits vs LHB/RHB. Offense features don't adjust for opposing pitcher handedness. The Chadwick register includes `bats` field but it's never pulled or used.
- **What research shows**: Platoon splits are among the **most persistent and well-documented** effects in baseball:
  - LHB vs RHP: +.025-.035 wOBA advantage
  - RHB vs LHP: +.015-.025 wOBA advantage
  - These effects are MORE reliable (higher signal-to-noise) than most other features currently in the model
- **Impact**: Missing one of the strongest, most well-established signals in baseball analytics.

#### Gap 4: No Batter vs Pitcher (BvP) Features
- **Current**: No BvP matchup data computed or persisted anywhere. Pitching features use pitcher's overall history, never performance against the specific opposing lineup.
- **What research shows**: BvP is tricky — small sample sizes are noise. But with proper weighting:
  - < 15 PA: Ignore entirely (weight = 0)
  - 15-50 PA: Minimal weight (0.1)
  - 50-100 PA: Moderate weight (0.3)
  - 100+ PA: Significant weight (0.5)
  - Always blend with platoon splits for base rate
- **Data sources available**: Statcast pitch-level data (already fetched for pitching features), Retrosheet historical PAs.

#### Gap 5: No Pitcher Stuff+ / CSW% / SIERA Metrics
- **Current**: xFIP, xERA, K%, BB%, GB%, avg fastball velocity, pitch mix entropy.
- **What's available and better**: Stuff+ (classification-based) outperforms ALL traditional metrics for predicting future pitcher performance:
  - Stuff+ correlation with future wOBA: r = 0.26-0.28
  - FIP correlation with future wOBA: r = 0.22
  - Prior wOBA correlation: r = 0.21
  - CSW% (Called Strikes + Whiffs) is also highly predictive: league avg ~28-30%, >32% = elite
  - SIERA (Skill-Interactive ERA) is the best descriptive ERA estimator, accounts for non-linear K/BB/GB interactions
- **Data source**: Baseball Savant (Statcast) has Stuff+ data. CSW% is computable from existing pitch-level data.

#### Gap 6: Incomplete Weather Implementation
**Specific issues**:
1. **Precipitation probability (`pop`) is NOT parsed** from OpenWeather API response — rain risk uses a humidity proxy
2. **Wind gusts** are not used, only average wind speed/direction
3. **No historical weather data** for training/backtesting (only 5-day forecast API, so weather features are neutral defaults for all historical data)
4. Weather adjustments clamped to ±15% — too conservative for extreme conditions
5. No park-specific wind sensitivity multipliers (e.g., Wrigley wind is 1.5x more impactful than average park)

**What research shows**:
- Wind at Wrigley can swing totals by **3-4 runs** (blowing out vs blowing in)
- Temperature effect: games at 90°F+ average **0.5-1 run more** than games <60°F
- These are largest weather signals but only matter for **totals** (which you don't model yet)

**Data sources for historical weather**: Retrosheet (game-time conditions), NOAA/NCDC (detailed meteorological), Open-Meteo (free historical weather API).

### 3.2 High-Priority Gaps

#### Gap 7: No Full-Game Run Line Model
- Only F5 run line exists.
- Full-game run line (typically -1.5) requires a run differential regression model.

#### Gap 8: No Model Diversity (XGBoost Only)
- Every model is XGBoost.
- Research shows optimal ensemble: XGBoost (40%) + LightGBM (30%) + Calibrated Logistic Regression (30%).
- Diversity between tree-based and linear models reduces variance. From Kevin Garnett's Beat the Streak model: "The final production model blends LightGBM (20%) and MLP (80%)... diversity reduces variance at the top of the ranking."

#### Gap 9: No Travel / Rest / Fatigue Features
- Zero travel, schedule, or fatigue features exist.
- **Quantified effects**:
  - West-to-East travel: circadian disadvantage ~0.3 runs
  - Day game after night game: measurable performance dip
  - 3+ consecutive games without off day: compounding bullpen fatigue
  - Series position (getaway day): different lineup construction

#### Gap 10: No Velocity Trend / Decline Detection
- `avg_fastball_velocity` is tracked as a rolling average, but there's no comparison to pitcher's own baseline to detect decline.
- A >1.5 mph drop from season average is a strong fatigue/injury predictor (+0.2 ERA).
- **Research**: Each pitch thrown in the preceding game increased ERA by 0.007 in the following game. Cumulative over 10 games: +0.022 ERA per average pitch.

#### Gap 11: CLV Not Primary Evaluation Metric
- CLV tracking code exists (`sync_closing_lines_from_snapshots` in odds_client.py) but isn't wired into model evaluation or decision making.
- **Research from Pinnacle**: "Bettors with positive CLV were almost universally profitable over time, regardless of short-term variance. Meanwhile, bettors with negative CLV were almost universally unprofitable."
- Your model should be evaluated primarily on CLV, not bankroll ROI.

### 3.3 Medium-Priority Gaps

#### Gap 12: No Poisson Run-Scoring Sub-Model
- Poisson regression naturally models run counts (non-negative integers) and gives probability distributions.
- Ideal for totals: predict expected runs per team, then compute P(over) and P(under) against posted total.
- Can also inform moneyline via simulation.

#### Gap 13: No Line Shopping / Multi-Book Integration
- Currently uses a single odds source (The Odds API + SBR scraping).
- Professional bettors use 5+ books for 2-5% ROI improvement.
- Shopping across books adds direct, measurable edge.

#### Gap 14: No Speed-to-Market / Opening Line Focus
- Model doesn't prioritize getting bets down when lines first open.
- Opening lines have more variance and inefficiency than lines shaped by sharp money.
- Timing of bet placement is a meaningful edge source.

#### Gap 15: No Player Props Capability
- Player props (strikeouts, hits, total bases, etc.) are increasingly the highest-edge market.
- Requires bottom-up player-level modeling (different architecture from team-level).
- Soft lines, lower limits, less sharp action = more inefficiency.

#### Gap 16: No Situational Features
- Missing: division games, series position, day/night splits, getaway day flags, holiday flags, standings/playoff implications, streak length.
- These have marginal individual impact but collectively meaningful.

#### Gap 17: Stacking Meta-Features Too Narrow
- Current meta-features are all home-centric: `home_team_f5_pythagorean_wp_30g`, `home_team_log5_30g`, `park_runs_factor`.
- Missing: away-team baselines, home-away differentials, any matchup-derived meta-features.
- Stacking consistently hurts performance (all 4 experiments) — likely because meta-features are too narrow to add value.

#### Gap 18: Umpire Features Underutilized
- Current features focus on `home_win_pct` and `total_runs_avg`.
- Missing: strike zone size metrics, K% impact, walk rate impact, O/U historical record.
- Research shows specific umpires can shift totals by 0.5+ runs:
  - Doug Eddings: Known for large strike zone, favors unders
  - Bill Miller: +1,100 extra strikes vs league average (2009-2016)
  - Alfonso Marquez: 8-2 home team record, +5.32 units on home dogs

---

## 4. Research: What Profitable Models Do

### 4.1 Features Used by Top Models

Based on analysis of academic papers, open-source repositories, and professional betting literature:

#### Tier 1: Highest Impact Features
| Feature Category | Specific Metrics | Why It Matters |
|-----------------|------------------|----------------|
| **Platoon Splits** | Lineup handedness composition vs starter hand, % LHH/RHH | Most persistent signal in baseball; +25-35 points wOBA |
| **Pitcher Quality (Modern)** | Stuff+, CSW%, SIERA | Outperforms traditional ERA/FIP/xFIP for prediction |
| **Bullpen Availability** | Pitch count L3/L5 days, consecutive appearance days, leverage usage | Modern starters avg <5.5 innings; 40% of game is relievers |
| **Weather (for Totals)** | Temperature, wind vector (speed × direction × park sensitivity) | Wind at Wrigley swings totals by 3-4 runs |

#### Tier 2: Significant Impact Features
| Feature Category | Specific Metrics | Why It Matters |
|-----------------|------------------|----------------|
| **Velocity Trends** | 3-start rolling vs season baseline | >1.5 mph drop = fatigue/injury signal |
| **Umpire Zone** | Strike zone size, K% impact, O/U record | Specific umpires shift totals 0.5+ runs |
| **Travel/Rest** | West→East flag, day-after-night, days off | Circadian disruption ~0.3 runs |
| **BvP (properly weighted)** | Batter wOBA vs specific pitcher at 50+ PA | Adds signal when sample size sufficient |

#### Tier 3: Marginal but Meaningful
| Feature Category | Specific Metrics | Why It Matters |
|-----------------|------------------|----------------|
| **Baserunning** | BsR (FanGraphs), sprint speed | 1-2 runs per season for elite runners |
| **Catcher Framing** | Framing runs (diminishing with ABS) | 2-4 runs in challenge-system parks |
| **Situational** | Division game, getaway day, streak, standings | Marginal individual, collectively meaningful |
| **Defensive Positioning** | OAA, FRV, DRS | Elite defense worth 1-3 wins per season |

### 4.2 Model Architectures That Work

#### Academic Research Findings (MDPI Applied Sciences 2025)
| Model | Accuracy | AUC-ROC | Best For |
|-------|----------|---------|----------|
| XGBoost | 89-93% | 0.97-0.98 | Classification (ML bets) |
| Logistic Regression | 89-93% | 0.97-0.98 | Calibrated probabilities |
| LightGBM | ~91% | ~0.97 | Speed + accuracy |
| Random Forest | 87-91% | 0.95-0.97 | Strong baseline |
| Neural Network (MLP) | 85-90% | 0.94-0.96 | Complex patterns (needs more data) |

**Key finding**: Logistic Regression and XGBoost perform nearly identically on AUC-ROC but LR has better calibrated probabilities out of the box. Ensembling diverse algorithms (tree + linear) consistently outperforms any single model.

#### Recommended Model Architecture Per Bet Type
| Bet Type | Recommended Model | Target Variable |
|----------|------------------|-----------------|
| Moneyline (F5 or Full) | XGBoost + LightGBM + LR ensemble | Win/Loss (binary classification) |
| Run Line | XGBoost Regression | Run differential (continuous) |
| Totals | Poisson Regression + weather adjustment | Expected runs (count) |
| Team Totals | Per-team Poisson model | Single team expected runs |
| Player Props | Bottom-up simulation | Player-specific outcomes |

#### Why Separate Models Per Bet Type
- Each bet type has different feature importance. Weather matters much more for totals than for sides.
- Different targets require different loss functions (classification vs regression vs count).
- Run-scoring models (Poisson) give you totals naturally but don't directly give win probability.
- Win probability models don't naturally give you total runs.

### 4.3 What Makes Models Actually Profitable

#### CLV: The Gold Standard
**Closing Line Value** measures whether you got a better price than the closing line.

Research from Pinnacle: "Bettors with positive CLV were almost universally profitable over time. Meanwhile, bettors with negative CLV were almost universally unprofitable."

| CLV Range | Assessment | Expected Long-Term Outcome |
|-----------|------------|---------------------------|
| +2% to +4% | Excellent | Sustained profitability |
| +1% to +2% | Good | Profitable with volume |
| 0% to +1% | Marginal | Break-even to slight profit |
| Negative | Poor | Long-term loss |

**Your model's job isn't just to predict winners — it's to find prices that will look better than the closing line.**

#### Line Shopping
- Shopping across 5+ books can add **2-5% to ROI**
- Target books with reduced juice (Pinnacle, Heritage)
- Target books with slower line movement for sharps

#### Speed to Market
- Opening lines have more variance before sharp money shapes them
- Get bets down early when model identifies value
- Track line movement to validate model (if line moves your way = confirmation)

#### Bet Selectivity
- Research consistently shows: **bet fewer games at higher confidence**
- Your 84% bet rate is far too high — target 10-30%
- From Forrest31's Baseball-Betting-Model: 59% accuracy overall, but 66%+ when filtering to >66% model probability

#### Bankroll Management
- Quarter Kelly (your current approach) is good
- Max 2-3% per game is standard
- **Never** exceed 5% regardless of perceived edge
- Kill-switch at 30% drawdown (your current approach) is reasonable

### 4.4 Notable Open-Source MLB Models

#### romanesquibel562/mlb-sports-betting-predictions
- **Features**: 30-day rolling Statcast metrics for pitchers and batters, team form indicators
- **Model**: XGBoost classifier with probability calibration
- **Performance**: Claims 64% accuracy
- **Notable**: Uses Gemini AI for game summaries

#### borsheng/Swing_Decision_Model
- **Approach**: Bottom-up pitch-level analysis
- **Method**: Buckets pitches by location (0.3 ft grid) + count + pitch type, calculates expected value of swing vs take
- **Application**: Player props, strikeout/walk prediction
- **Model**: XGBoost regressor with GroupKFold by batter

#### Forrest31/Baseball-Betting-Model
- **Features**: Pythagorean record (season and last 5), Log5 probability
- **Model**: XGBoost classifier
- **Performance**: 59% overall, 66%+ at high-confidence thresholds
- **Key insight**: Only betting high-confidence picks significantly improves profitability

#### pybaseball (jldbc/pybaseball)
- Not a model but the gold standard data acquisition library
- Wraps Statcast, FanGraphs, Baseball Reference
- Most serious MLB models use this for data ingestion

---

## 5. Recommended Target Architecture

### What a "Complete" System Looks Like

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION LAYER                            │
├─────────────────────────────────────────────────────────────────────────┤
│  Statcast (pitch-level)  │ FanGraphs (Stuff+, BsR, DRS, SIERA)        │
│  Baseball Savant         │ Retrosheet (historical, umpires, lineups)    │
│  Chadwick Register       │ Odds API (5+ books, F5 + FG + Totals)       │
│  Weather: OpenWeather    │ Weather: Open-Meteo or NOAA (historical)     │
│  SBR scraping            │ Schedule/Travel data (MLB API)               │
└─────────┬───────────────────────────────────────────┬───────────────────┘
          │                                           │
          ▼                                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       FEATURE ENGINEERING LAYER                         │
├──────────────────┬───────────────────────┬──────────────────────────────┤
│ CORE (existing)  │ MATCHUP (new)         │ CONTEXT (new)               │
│ • Team offense   │ • Platoon splits      │ • Weather (full: temp,      │
│   metrics (wRC+, │   (lineup handedness  │   wind vector, precip,      │
│   wOBA, ISO,     │   vs pitcher hand)    │   park-specific sensitivity)│
│   BABIP, K%, BB%)│ • BvP history         │ • Travel/rest/fatigue       │
│ • Starter quality│   (sample-weighted)   │   (W→E, day-after-night,    │
│   (xFIP, xERA,  │ • Stuff+ / CSW%       │   series position)          │
│   K%, BB%, GB%)  │ • Lineup vs pitcher   │ • Situational flags         │
│ • Bullpen fatigue│   composition match   │   (division, getaway,       │
│ • Defense (DRS,  │ • Velocity trends     │   streak, standings)        │
│   OAA, framing)  │   (decline detection) │ • Umpire zone metrics       │
│ • Baselines      │ • Pitch mix matchup   │   (zone size, K%/BB%        │
│   (Pythagorean,  │   (batter vs pitch    │   impact, O/U record)       │
│   Log5)          │   type effectiveness) │ • Historical weather        │
│ • Park factors   │                       │   (for training data)       │
│ • ABS adjustment │                       │ • Line movement features    │
└──────────┬───────┴───────────┬───────────┴──────────┬───────────────────┘
           │                   │                       │
           ▼                   ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    MODEL LAYER (separate model per market)              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  FULL GAME ML:  XGBoost + LightGBM + LR ensemble → win probability     │
│  FULL GAME RL:  XGBoost regression → run differential                   │
│  F5 ML:         XGBoost + LightGBM + LR ensemble → F5 win probability  │
│  F5 RL:         Direct cover + margin pricing (current architecture +)  │
│  GAME TOTALS:   Poisson regression + weather adjustment → expected runs │
│  F5 TOTALS:     Poisson regression (starter-focused) → F5 expected runs│
│  TEAM TOTALS:   Per-team Poisson model → single team expected runs      │
│                                                                         │
│  Each model: Calibration (Isotonic) → Ensemble blend                    │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│              ALL models feed into unified decision engine                │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         DECISION ENGINE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│ • De-vig across all books (Shin method for extreme lines)               │
│ • Edge calculation per market per book                                  │
│ • CLV prediction (will this line move our way?)                         │
│ • Quarter Kelly sizing with cross-market correlation handling           │
│ • 30% drawdown kill-switch                                              │
│ • Primary eval metric: CLV (not ROI)                                    │
│ • Bet selectivity target: 15-25% of games (not 84%)                    │
│ • Line shopping: Best odds across all available books                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Prioritized Roadmap

### Phase 1: Fix Foundation (Before Adding Anything)

These must be done first because all subsequent evaluation depends on them.

| # | Action | Why | Files Involved |
|---|--------|-----|----------------|
| 1 | **Wire CLV as primary evaluation metric** | Can't improve what you can't measure correctly | `src/engine/edge_calculator.py`, `src/pipeline/daily.py`, `src/clients/odds_client.py` |
| 2 | **Backtest only against real historical odds** | All synthetic-odds results are meaningless | `src/backtest/walk_forward.py`, historical odds DB |
| 3 | **Move strategy optimization to nested walk-forward** | Current in-sample optimization guarantees overfit | `src/backtest/walk_forward.py` |
| 4 | **Add early stopping to walk-forward models** | Severe overfitting risk at current feature-sample ratio | `src/backtest/walk_forward.py` |
| 5 | **Prune to 50-70 features** by stable importance | 224 features is way too many; most are noise | `src/model/data_builder.py`, `src/model/xgboost_trainer.py` |

### Phase 2: Add Highest-Impact Missing Signals

| # | Action | Expected Impact | Files Involved |
|---|--------|----------------|----------------|
| 6 | **Add platoon split features** | One of strongest, most reliable signals | New: `src/features/platoon.py`; Modify: `src/clients/lineup_client.py` (add handedness), `src/model/data_builder.py` |
| 7 | **Fix weather**: Parse `pop`, add historical weather (Open-Meteo/NOAA), improve wind modeling with park-specific sensitivity | Unlocks totals modeling | `src/clients/weather_client.py`, `src/features/adjustments/weather.py`, new: historical weather client |
| 8 | **Add Stuff+/CSW% pitcher metrics** | Better pitcher quality prediction | `src/features/pitching.py`, new data source (Baseball Savant Stuff+ endpoint or pybaseball) |
| 9 | **Add velocity decline detection** | Strong fatigue/injury predictor | `src/features/pitching.py` (compare 3-start rolling to season baseline) |
| 10 | **Add travel/rest/fatigue features** | Measurable 0.3+ run effect | New: `src/features/travel.py`; data from MLB schedule API |

### Phase 3: Expand Bet Types

| # | Action | Expected Impact | Files Involved |
|---|--------|----------------|----------------|
| 11 | **Build totals model** (Poisson regression for run scoring) | Highest-edge niche market | New: `src/model/totals_trainer.py`; Modify: `src/models/_base.py` (MarketType), `src/engine/edge_calculator.py`, `src/engine/settlement.py`, `src/clients/odds_client.py`, `src/pipeline/daily.py` |
| 12 | **Build F5 totals model** | Natural extension of current F5 focus | Same as above but F5-specific |
| 13 | **Build full-game ML model** | Largest volume market | New training script; Modify: MarketType, settlement, odds, pipeline |
| 14 | **Build full-game RL model** | Run differential regression | Same as above |
| 15 | **Build team totals model** | Offense-specific modeling | Per-team Poisson variant |

### Phase 4: Model Diversity & Sophistication

| # | Action | Expected Impact | Files Involved |
|---|--------|----------------|----------------|
| 16 | **Add LightGBM as second base learner** | Algorithm diversity reduces variance | New: `src/model/lightgbm_trainer.py`; Modify: stacking to ensemble both |
| 17 | **Add BvP features** (sample-size weighted) | Pitcher-vs-lineup specific signal | New: `src/features/bvp.py`; uses Statcast pitch-level data |
| 18 | **Enrich stacking meta-features** | Stacking currently hurts — needs richer context | `src/model/stacking.py` (add away-team baselines, differentials) |
| 19 | **Implement Shin de-vig method** for extreme lines | More accurate than proportional de-vig at tails | `src/clients/odds_client.py` |
| 20 | **Add line shopping across multiple books** | 2-5% ROI improvement | `src/clients/odds_client.py`, `src/pipeline/daily.py` |

---

## 7. Quick-Reference Comparison Table

| Dimension | Your System Now | What Profitable Models Do | Gap Severity |
|-----------|----------------|--------------------------|--------------|
| **Bet types** | F5 ML + F5 RL only | F5 ML/RL + Full Game ML/RL + Totals + F5 Totals + Team Totals | 🔴 Critical |
| **Platoon splits** | None | Full lineup handedness vs pitcher hand | 🔴 Critical |
| **BvP matchups** | None | Sample-size-weighted BvP history | 🟠 High |
| **Weather** | Minimal (mostly neutral defaults) | Temperature + wind vector + precip + historical data | 🟠 High |
| **Pitcher quality** | xFIP, xERA, K%, BB%, velocity | Stuff+, CSW%, SIERA, velocity trend detection | 🟠 High |
| **Totals modeling** | None | Poisson regression + weather adjustment | 🔴 Critical |
| **Model diversity** | XGBoost only | XGBoost + LightGBM + LR ensemble | 🟠 High |
| **Travel/rest** | None | West→East, day-after-night, series position | 🟡 Medium |
| **Umpire depth** | Basic (home win %, runs avg) | Zone size, K% impact, walk rate, O/U record | 🟡 Medium |
| **CLV tracking** | Exists but not primary | Primary evaluation metric | 🔴 Critical |
| **Line shopping** | Single source | 5+ books | 🟡 Medium |
| **Eval metric** | Bankroll ROI on synthetic odds | CLV against sharp closing lines | 🔴 Critical |
| **Feature count** | 224 (most are noise) | 50-70 curated, high-importance features | 🟠 High |
| **Bet selectivity** | 84% of games | 10-30% of games at high confidence | 🟠 High |
| **Expected ROI** | +24.4% (artifact of methodology) | 3-6% (realistic world-class) | ⚠️ Recalibrate |

---

## Appendix A: Research Sources

### Academic Papers
- "Application of Machine Learning Models for Baseball Outcome Prediction" — MDPI Applied Sciences 2025
- "Assessing win strength in MLB win prediction models" — arXiv 2025
- "Use of Machine Learning and Deep Learning to Predict MLB Outcomes" — MDPI 2021
- Dartmouth temperature/HR research

### Betting Strategy Literature
- Pinnacle CLV study (via XCLSV Media)
- VSiN MLB Bullpen Systems 2026
- Core Sports Betting — Umpire Tendency Analysis
- OddsJam — Weather Impact on Totals
- RotoGrinders — Wind Speed/Direction Study (2000-2023, 22,215 games)

### Open Source Repositories
- romanesquibel562/mlb-sports-betting-predictions (full pipeline)
- borsheng/Swing_Decision_Model (bottom-up pitch analysis)
- Forrest31/Baseball-Betting-Model (Pythagorean + Log5 + XGBoost)
- jldbc/pybaseball (data acquisition library)

### Sabermetric Research
- Tom Tango, "The Book" — BvP sample size requirements
- Baseball Savant Stuff+ methodology
- FanGraphs SIERA documentation

---

*Report generated by comprehensive codebase analysis (30+ source files read directly) and parallel research agents (4 codebase explorers, 2 external research agents). No code was modified.*
