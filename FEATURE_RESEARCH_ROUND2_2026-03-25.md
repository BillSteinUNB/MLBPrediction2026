# MLB Model Research: Feature Engineering Deep Dive (Round 2)

**Date**: March 25, 2026  
**Sources**: 5 parallel research agents + GitHub code analysis + 10 web sources + academic papers  
**Scope**: What features do successful MLB models actually use? What should we add next?

---

## Executive Summary

After analyzing 8+ open-source models, 6 academic papers, and extensive sabermetric research, three findings stand out:

1. **Fewer features wins.** Alex Zheng (60% accuracy) uses 11. Kevin Garnett's Beat the Streak model found 50 > 172. You currently have ~80 after pruning — still potentially too many.
2. **Pitcher vs specific team is noise.** Research unanimously says skip it. Sample sizes are too small (25-30 PA/season). Use overall pitcher quality + platoon splits instead.
3. **Classification may beat regression for betting.** The best models predict P(home_win) directly at 59-63%, rather than predicting runs and converting. Your run regression approach hits an R² ceiling of ~5-10%. But run models are valuable for totals/spread markets.

---

## What Top Open-Source Models Actually Use

### Forrest31/Baseball-Betting-Model (66% at High Confidence)

**Algorithm**: XGBoost binary classifier  
**Features**: ~20 (all differentials: home minus away)

| Feature | Description | SHAP Rank |
|---------|-------------|-----------|
| `ProbOfTmWin` | Log5 win probability (matchup-derived) | #2 |
| `Pythagdelta` | Pythagorean record differential | #4 |
| `RollingOppLosses` | Opponent losses last 5 games | #1 |
| `Opp_WinPct` | Opponent season win percentage | #3 |
| `Rolling5_Tm_Runs_For/Against` | 5-day rolling run production | Top 10 |
| `RollingProbofTmWin` | Rolling 5-game Log5 | Top 10 |
| `Tm_Game_Runs_For/Against` | Season cumulative runs | Top 10 |

**Key insight**: No individual pitcher stats. No Statcast. No weather. Just team quality + recent form + matchup probability. **And it hits 66% on high-confidence picks.**

### Alex Zheng (60% Accuracy, Positive ROI)

**Algorithm**: SVM with RBF kernel  
**Features**: Started with 40, pruned to **11**

| Feature | Description |
|---------|-------------|
| Team ERA differential | Home ERA - Away ERA |
| Runs scored differential | Recent runs scored difference |
| Runs allowed differential | Recent runs allowed difference |
| Win rate differential | Season win% difference |
| Recent form (7-day rolling) | Last 7 games performance |

**Key insight**: "Team ERA dominated as the single most predictive factor, far outweighing starting pitcher stats." Recency (7-day) mattered more than season-to-date.

### Academic Best (Li et al. 2022 — 65.75% with SVM)

**Features after selection**: R, H, RBI, SO, OBP, OPS, LOB, BB, ER, Win%

**Key insight**: Feature selection raised accuracy from 64.25% → 65.75%. Fewer, better features > more features.

---

## What Research Says About Specific Features You Asked About

### Pitcher vs Specific Team History: SKIP ❌

**Tom Tango's work + FanGraphs Splits Research**:  
> "Batter-pitcher matchup stats are mostly noise. How well a player performs against a particular team usually doesn't tell you much."

| Sample | PA per season vs specific team | Reliability |
|--------|-------------------------------|-------------|
| Pitcher vs Team | 25-30 PA | **Noise** |
| 3-year accumulation | 75-90 PA | **Barely meaningful** |
| Reliable threshold | 100+ PA | **Rarely reached** |

**Why it fails**: Rosters change 20-40% annually. The "team" a pitcher faced last year is a different lineup. Use overall pitcher quality metrics instead.

### Team vs Team Records: SKIP ❌

No peer-reviewed research supports team-vs-team persistence beyond what team quality metrics (Pythagorean, Log5) already capture. An 8-2 head-to-head record reflects schedule timing and pitching rotation alignment, not a genuine matchup advantage.

### Platoon Splits by Team: USE ✅

Unlike pitcher-vs-team, TEAM-LEVEL platoon splits have huge samples:
- Team vs LHP: 1,500-2,000 PA/season (very reliable)
- Team vs RHP: 3,500-4,000 PA/season (very reliable)
- Effect size: 50-100 OPS points between splits

**Features to add**:
- `team_wrc_plus_vs_LHP` and `team_wrc_plus_vs_RHP`
- `platoon_gap` = team performance vs starter's handedness minus opposite
- Already have `platoon_advantage_pct` — good foundation

### Timezone/Travel Effects: USE ✅

**PNAS Study (Song et al., 46,535 games, 20 seasons)**:

| Direction | Win% Impact | Runs Impact | HR Impact |
|-----------|-------------|-------------|-----------|
| **Eastward (2+ TZ)** | **-3.5%** | **+0.08-0.12/TZ** | **+0.17/game** |
| Westward | -1.0% (not sig) | minimal | minimal |

You already have `timezone_crossings_east` and `is_day_after_night_game` — these are the right features.

**Additional to consider**: `circadian_disadvantage_hours` = max(0, tz_diff - days_since_travel) to account for the 1-hour/day adjustment rate.

### Starter Rest / Workload: USE ✅

**Bradbury & Forman study (1,058 pitchers, 22 seasons)**:
- Rest days: Weak effect (-0.015 ERA/extra day)
- **Pitch count last start**: +0.007 ERA per pitch — this adds up!
- **Cumulative load (10-game)**: +0.022 ERA per pitch — strongest signal

**Features to add**:
- `starter_days_rest` (days since last start)
- `starter_last_start_pitch_count`
- `starter_cumulative_pitch_load_10` (rolling 10-start average pitch count)

### Bullpen Fatigue: USE ✅

**Most predictive window**: Last 3 days (not 5, not 7)

| Window | Predictive Power | Why |
|--------|-----------------|-----|
| **3 days** | **Highest** | Captures immediate availability state |
| 5 days | Moderate | Accumulation tracking |
| 7 days | Lower | Too diluted |

**Already have**: `bullpen_pitch_count_3d`, `bullpen_pitch_count_5d`. Good.

**Missing**: High-leverage innings weighting. Research shows 1 high-leverage inning ≈ 1.5-2.0 normal innings in fatigue impact.

---

## Advanced Statcast Features Worth Adding

### Tier 1: High-Value (backed by research)

| Feature | Why Better Than Current | Year-to-Year r | Data Source |
|---------|------------------------|----------------|-------------|
| **xwOBA** | More predictive than wOBA for future performance | 0.55+ | `statcast_batter_expected_stats()` |
| **Barrel rate** | More predictive than ISO for run scoring | 0.55 | `statcast_batter_exitvelo_barrels()` |
| **wOBA - xwOBA** | Predicts regression direction (luck indicator) | N/A | Derived |
| **Pitch run value/100** | Better than pitch mix entropy for arsenal quality | Moderate | Baseball Savant |
| **Whiff rate by pitch type** | Granular strikeout prediction | High | Statcast pitch-level |

### Tier 2: Moderate Value

| Feature | Value | Note |
|---------|-------|------|
| Catcher framing runs (FRV) | +/- 0.1 runs per framed strike | Requires 2,000 framing opportunities to stabilize |
| Team baserunning run value | 20-run spread between best/worst teams | r² = 0.31 with runs scored |
| OAA (Outs Above Average) | Most predictive defensive metric (r=0.27 y/y) | Better than DRS for range |

### Tier 3: Experimental

| Feature | Note |
|---------|------|
| Batted ball profile matchup (pitcher FB% × team FB tendency) | Limited production use |
| Sprint speed | Small marginal value beyond OBP/SLG |
| Recent form (last 7 games) | Hot hand is largely myth; r ≈ 0.05-0.10 |

---

## The Biggest Strategic Finding: Classification vs Regression

### Your Current Approach
Predict runs (regression) → Convert to win probability via distributions

### What Research Shows

| Approach | Best Accuracy | Best For |
|----------|--------------|----------|
| **Binary classification** (XGBoost classifier) | **59-63%** | **Moneyline betting** |
| Run regression (XGBoost regressor) | 55-58% (implied) | Totals/spread betting |
| Elo-based | 57-58% | Baseline comparison |
| Ensemble (LogR + SVM + XGB) | **62.83%** | Best overall |

**Key paper (Allen & Savala 2025, arXiv:2511.02815)**:
> "XGBoost has low agreement with other models but high standalone accuracy, making it ideal for ensembling." Triple ensemble achieved 62.83%.

### Recommendation: Build BOTH

**For moneyline**: Add a binary classifier alongside your run models
```
home_win = XGBClassifier(objective='binary:logistic').predict_proba(features)
```

**For totals/spreads**: Keep your run regression models (they're needed for over/under pricing)

**For final edge**: Compare the two. When classifier and regression-derived probabilities AGREE, you have a high-confidence pick (this is Forrest31's secret — 66% on filtered picks).

### Calibration > Accuracy

**From systematic review (arXiv:2410.21484)**:
> "Optimizing for calibration rather than accuracy leads to 69.86% higher average returns."

Use `CalibratedClassifierCV` with isotonic regression on any classification model. A well-calibrated 57% model is more profitable than a poorly-calibrated 60% model.

---

## R² Ceiling: What's Actually Achievable?

| Metric | Your Current | Realistic Target | Theoretical Ceiling |
|--------|-------------|-----------------|---------------------|
| **Run regression R²** | 1.7-4.3% | **5-8%** | ~10-12% |
| **Win classification accuracy** | N/A | **58-61%** | ~65% |
| **Log loss** | N/A | 0.67-0.68 | ~0.65 |

**Why the ceiling is low**: Baseball is the hardest major sport to predict. Low-scoring, high-variance, compressed talent distribution. Even the best models leave 90%+ of variance unexplained.

**Your R² of 4.3% (f5_away) is already within striking distance of the ceiling.** Getting to 8% would be excellent.

---

## Implementation Priority: What to Add Next

### Phase 4A: New Features (Highest Expected Impact)

| # | Feature | Expected Lift | Effort | Data Source |
|---|---------|--------------|--------|-------------|
| 1 | **xwOBA replacing wOBA** (or add both) | +0.5-1.0% R² | Medium | Statcast expected stats |
| 2 | **Barrel rate replacing ISO** (or add both) | +0.3-0.5% R² | Medium | Statcast exit velo barrels |
| 3 | **Starter days rest** | +0.1-0.3% R² | Low | MLB schedule API |
| 4 | **Starter last start pitch count** | +0.1-0.3% R² | Low | Statcast pitcher data |
| 5 | **wOBA - xwOBA** (luck regression indicator) | +0.2-0.5% R² | Low | Derived from 1 |
| 6 | **Team wRC+ vs LHP/RHP splits** | +0.2-0.4% R² | Medium | FanGraphs splits |

### Phase 4B: Classification Model (for Moneyline)

| # | Action | Expected Result |
|---|--------|----------------|
| 7 | **Add XGBClassifier for P(home_win)** | 58-61% accuracy |
| 8 | **Add LogisticRegression** | 57-59% accuracy |
| 9 | **Ensemble classifier + regressor agreement** | 63-66% on filtered picks |
| 10 | **Calibrate with isotonic regression** | Better probability reliability |

### Phase 4C: Market Features (when odds data is ready)

| # | Feature | Expected Lift |
|---|---------|--------------|
| 11 | **Market implied probability as feature** | +1-2% accuracy |
| 12 | **Line movement direction** | +3-5% ROI |
| 13 | **Opening vs closing line delta** | Edge identification |

---

## Features That Successful Models DON'T Use (Save Your Time)

| Feature | Why Skip | Source |
|---------|----------|--------|
| Pitcher vs specific team ERA | Sample too small (25-30 PA/season) | Tango, FanGraphs |
| Team vs team head-to-head record | Noise; roster turnover | Multiple studies |
| Batter vs pitcher history (<50 PA) | Pure noise | The Book (Tango) |
| Win/loss streaks | Regression to mean dominates | Multiple studies |
| Hot hand / momentum | r ≈ 0.05 after controlling for quality | Bock et al. 2023 |
| Neural networks / LSTM | No improvement over XGBoost for MLB | Allen & Savala 2025, Zhao 2025 |
| Individual batter vs pitch type vs specific pitcher | Way too sparse | Sample size math |

---

## Sources

**Academic Papers**:
- Allen & Savala 2025: "Assessing win strength in MLB win prediction models" (arXiv:2511.02815)
- Li et al. 2022: "Exploring and Selecting Features to Predict MLB Games" (Entropy 24, 288)
- Song et al. 2017: "Jet Lag in MLB" (PNAS)
- Bradbury & Forman 2012: "Pitch Counts and Days of Rest" (J Strength & Conditioning)
- Systematic Review of ML in Sports Betting 2024 (arXiv:2410.21484)

**Open-Source Models**:
- Forrest31/Baseball-Betting-Model (R + XGBoost, 66% filtered)
- romanesquibel562/mlb-sports-betting-predictions (Python + XGBoost calibrated)
- whrg/MLB_prediction (Python, 70% claimed)
- nyanp/mlb-player-digital-engagement (Kaggle 3rd place, LightGBM + NN)

**Practitioner Sources**:
- Alex Zheng: "How I Beat the Sportsbook in Baseball with ML" (Medium, 2025)
- Nick Faddis: "Predicting Win Probability for MLB Games" (Medium, 2024)
- Kevin Garnett: "Chasing $5.6M with ML" (Medium, 2026)
- FiveThirtyEight MLB Elo Methodology
- FanGraphs Splits Research Library
