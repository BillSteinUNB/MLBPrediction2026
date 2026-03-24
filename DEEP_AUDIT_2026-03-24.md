# Deep Audit: MLB F5 ML Betting System

**Date:** 2026-03-24  
**Method:** Full source code review of all critical files (walk_forward.py, edge_calculator.py, xgboost_trainer.py, stacking.py, market_recalibration.py, odds_client.py, data_builder.py, bankroll.py, test_antileak.py, margin_pricing.py, margin_trainer.py, calibration.py, settings.yaml, all config YAMLs), plus 4 parallel specialist agent investigations, plus external ML betting research. Zero code changes.

**Note:** An existing audit (AUDIT_REPORT_2026-03-24.md) already covers several issues. This report validates those and focuses on **new findings** and **deeper analysis** it missed.

---

## Part 1: The Fundamental Problem

The prior audit correctly identified C1 (synthetic odds circularity) and C2 (in-sample strategy optimization) as system-breaking. Both are fully confirmed. But there's a deeper structural question the prior audit danced around without fully answering:

### Can this model EVER be profitable, even with perfect evaluation methodology?

The model's top features are:

- **Log5** (team win probability derived from Pythagorean record) — ~4-8% importance
- **Starter K%**, **xERA**, **xFIP** — standard sabermetric stats
- **Lineup wOBA/wRC+** — publicly available batting metrics

Every one of these is publicly available information that sharp books (Pinnacle, Circa) have been pricing for decades. Research consensus from the University of Bath (2024), Wilkens (2021, Journal of Sports Analytics), and multiple industry sources is clear: **models using only publicly available sabermetric data rarely beat efficient closing lines.** Even extensive ML ensembles capped at ~70% accuracy in tennis; betting returns were "mainly negative over the longer term."

**Where genuine edge could come from:**

1. **Speed** — getting to market before lines move (the system doesn't do this)
2. **Niche market inefficiency** — F5 markets ARE less liquid/efficient than full-game (the system targets this — good)
3. **Non-public information** — lineup changes before announcement, injury intel (not systematically exploited)
4. **Novel signal combination** — combining information in ways the market doesn't (debatable whether XGBoost on standard stats achieves this)

The F5 market choice is the system's strongest strategic decision. F5 lines are less efficiently priced because fewer sharp bettors participate, and starter dominance makes the outcome more predictable. But even here, the achievable edge is 3-6%, not 24%.

---

## Part 2: Issues Confirmed from Prior Audit (with deeper code evidence)

Every line of the relevant code was read. These prior findings are **confirmed** with specific code evidence:

### C1: Synthetic Odds from Log5

- **Code:** `walk_forward.py:1024-1028` — `default_home_fair = np.clip(test_frame["home_team_log5_30g"], 0.02, 0.98)`
- **Verdict:** Confirmed. However, the code now has a `profitability_metrics_valid` guard (line 831-857) that sets `bet_side = "none"` when odds are synthetic. **The +24.4% ROI was likely from an earlier code version before this guard existed.** Current code won't produce ROI numbers with synthetic odds.

### C2: Strategy Parameters Optimized In-Sample

- **Code:** `market_strategy_sweep.py` loads training data and evaluates sampled params on same data
- **Verdict:** Confirmed.

### C3: Closing vs Opening Lines

- **Code:** `historical_odds_client.py` uses `MAX(fetched_at)` by default; `walk_forward.py` default is `"opening"`
- **Verdict:** Confirmed — but the walk-forward **defaults to opening** (`DEFAULT_HISTORICAL_ODDS_SNAPSHOT_SELECTION = "opening"` at line 75), which is actually correct. The mismatch exists between training pipeline (closing) and backtest (opening).

### H3: Hyperparameter Stuck at Corner

- **Code:** `xgboost_trainer.py:38-48` now has expanded search space including `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`
- **Verdict:** Partially fixed. The search space was expanded since the prior audit, but with `DEFAULT_SEARCH_ITERATIONS = 40` over ~384,000 combinations, coverage is 0.01%. Still insufficient.

### M1: No Embargo Gap

- **Code:** `walk_forward.py:199` — `train_end = test_start`
- **Verdict:** Confirmed. Zero gap.

---

## Part 3: New Issues Not in Prior Audit

### NEW-1: Walk-Forward Hyperparameters Don't Match Training Pipeline (HIGH)

**Location:** `walk_forward.py:83-89` vs `xgboost_trainer.py` search results

The walk-forward backtest hardcodes:

```python
DEFAULT_ESTIMATOR_KWARGS = {
    "max_depth": 3,
    "n_estimators": 120,
    "learning_rate": 0.05,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
}
```

But the training pipeline's RandomizedSearch consistently finds `max_depth=6, n_estimators=100, lr=0.01` (per all 4 experiments in EXPERIMENT_REVIEW).

**These are fundamentally different models:**

- Walk-forward: shallow (depth 3), fast learner (lr 0.05), more trees (120)
- Training search result: deep (depth 6), slow learner (lr 0.01), fewer trees (100)

**Impact:** Backtest performance reflects a model that would NEVER be deployed in production. Production would use the search-optimized params, which could perform completely differently. Every ROI/Brier number from the walk-forward is for the wrong model.

---

### NEW-2: No Early Stopping in XGBoost (HIGH)

**Location:** `xgboost_trainer.py:362-370`

```python
def _build_estimator(*, random_state: int) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        tree_method="hist",
        n_jobs=DEFAULT_XGBOOST_N_JOBS,
        verbosity=0,
    )
```

No `early_stopping_rounds`, no `eval_set`. The model trains for exactly `n_estimators` trees regardless of whether it started overfitting at tree 50. With 224 features and ~2,400 rows per walk-forward window, this is a **10:1 sample-to-feature ratio** — XGBoost will absolutely overfit without early stopping or aggressive regularization.

**Best practice:** `early_stopping_rounds=50` with a temporal validation set (last 20% of training window).

---

### NEW-3: Edge Bucketing Has Inverted Bet Sizing (MEDIUM)

**Location:** `walk_forward.py:1532-1542`

```python
def _edge_bucketed_units(edge: float) -> float:
    if resolved_edge >= 0.30:
        return 0.5   # SMALLEST bet for LARGEST edge
    if resolved_edge >= 0.20:
        return 2.0   # LARGEST bet
    if resolved_edge >= 0.15:
        return 1.5
    if resolved_edge >= 0.12:
        return 1.0
    return 0.5
```

The highest edge tier (30%+) gets the **smallest** bet (0.5u), while 20-30% edges get the **largest** (2.0u). The reasoning is presumably that very large edges are noise/data errors — which is correct intuition.

**But the proper solution isn't inverted sizing — it's a max edge filter.** Research shows edges above 10-15% in efficient markets are almost always data errors or model failures. The system should cap maximum bettable edge (via `max_edge_to_bet` parameter, which exists but defaults to `None`).

---

### NEW-4: No Class Imbalance Handling (MEDIUM)

**Location:** `xgboost_trainer.py:362-370` — no `scale_pos_weight`

Home teams win F5 ~52-53% of games. This slight imbalance isn't catastrophic, but the model makes no effort to handle it. No `scale_pos_weight` parameter, no `sample_weight` passed to fit. For a system trying to extract 3-5% edge, even a 1% systematic bias matters.

---

### NEW-5: Model Serialization Has No Version Safety (MEDIUM)

**Location:** All `joblib.dump`/`joblib.load` calls across `xgboost_trainer.py`, `stacking.py`, `calibration.py`, `daily.py`

Models are saved with `joblib.dump` but metadata records **no xgboost or sklearn version**. XGBoost pickles are notoriously version-sensitive — upgrading xgboost from 1.7 to 2.0 can silently produce different predictions from the same saved model. The daily pipeline loads these models without any version check.

**Fix:** Record `xgboost.__version__`, `sklearn.__version__` in metadata. Add loader assertion.

---

### NEW-6: Proportional De-Vig May Be Inaccurate for F5 Lines (LOW-MEDIUM)

**Location:** `odds_client.py:762-772`

```python
def devig_probabilities(home_odds: int, away_odds: int) -> tuple[float, float]:
    home_implied = american_to_implied(home_odds)
    away_implied = american_to_implied(away_odds)
    total_implied = home_implied + away_implied
    return home_implied / total_implied, away_implied / total_implied
```

Proportional de-vig assumes vig is distributed proportionally across outcomes. For typical MLB F5 lines (-150/+130), this is generally adequate. But for extreme lines (-300/+250), books often embed asymmetric vig (more on the favorite side). The Shin method or power method would be more accurate.

**Estimated impact:** 0.5-1.5% probability error on extreme lines, which could flip marginal edge decisions.

---

### NEW-7: The `shrink_factor=0.78` for F5 Estimation Is a Magic Number (LOW)

**Location:** `odds_client.py:713` — `build_estimated_f5_ml_snapshots`

When no F5 odds are available, the system estimates F5 moneylines from full-game odds using a hardcoded `shrink_factor=0.78`. This assumes F5 probabilities are 78% as extreme as full-game probabilities. No derivation or validation is documented. If this factor is wrong, estimated F5 odds are systematically biased.

---

### NEW-8: Stacking Meta-Features Have Systematic Home-Team Bias (MEDIUM)

**Location:** `stacking.py:41-45`

```python
DEFAULT_RAW_META_FEATURE_COLUMNS = (
    "home_team_f5_pythagorean_wp_30g",
    "home_team_log5_30g",
    "park_runs_factor",
)
```

All three meta-features are home-centric. No away-team baselines, no differential features. The LR meta-learner sees: `[xgb_prob, home_pyth, home_log5, park_factor]`. This creates a systematic bias where the stacking layer can only adjust based on home-team strength, not matchup quality.

The data confirms: stacking **hurts F5 ML in all 4 experiments** (Brier delta: -0.0003, -0.0002, -0.0002, -0.0008). The meta-learner is adding noise, not signal, because it lacks the away-team context needed to make meaningful corrections.

---

### NEW-9: Anti-Leak Tests Are Dangerously Narrow (HIGH)

**Location:** `tests/test_antileak.py` — only 78 lines, 2 tests

The anti-leak test suite checks only ONE thing: `as_of_timestamp < scheduled_start`. It does **NOT** test:

- Whether feature rolling windows use only prior games (e.g., that `wrc_plus_7g` for April 10 uses only games before April 10)
- Whether stacking OOF generation is leak-free across folds
- Whether historical odds features use only pre-game snapshots
- Whether the live inference pipeline produces identical features to training for the same inputs
- Whether `build_live_feature_frame` is restricted to prediction-time-available data

The `as_of_timestamp` check catches the most obvious leakage vector, but the feature engineering modules have many subtle places where look-ahead could occur (e.g., Marcel blending using full-season stats, defense module fetching all 30 teams including future-game data). These are **undefended gaps**.

---

### NEW-10: Autoresearch Optimizes a Noisy Metric (MEDIUM)

**Location:** `config/autoresearch_*.yaml` configs and `src/ops/autoresearch.py`

The autoresearch framework optimizes `bankroll_return_pct` as the primary objective. This is one of the noisiest possible metrics — heavily influenced by a few lucky/unlucky bets, edge bucket sizing, and random variance. A model that happens to bet large on a few winning underdogs looks spectacular on this metric.

Research says: **Brier score or CLV should be the primary optimization target.** Brier score measures probability accuracy (which is skill-based), while `bankroll_return_pct` conflates model skill with betting strategy and luck.

---

## Part 4: Comparison to Best Practices

| Dimension | Your System | Research Best Practice | Gap |
|---|---|---|---|
| **Primary eval metric** | `bankroll_return_pct` | CLV or Brier score | Critical |
| **Expected ROI** | +24.4% (historical, synthetic odds) | 3-6% (world-class) | 4-8x too high = artifact |
| **Bet selectivity** | 84% of games | 10-30% | Way too high |
| **Feature count** | 224 | 20-50 curated | 4-10x too many |
| **Sample:feature ratio** | ~10:1 per window | 50:1+ recommended | Severe overfitting risk |
| **XGB regularization** | `subsample=1.0` in walk-forward | 0.7-0.9 | Absent |
| **Early stopping** | None | 50-100 rounds | Absent |
| **Calibration validation** | Platt on small holdout | Isotonic on large sample, or skip | Platt hurts 7/8 times |
| **Walk-forward gap** | 0 days | 1-7 days embargo | Missing |
| **Strategy tuning** | In-sample sweep | Nested walk-forward | Leakage |
| **CLV tracking** | Exists but not wired to eval | Primary evaluation metric | Not integrated |
| **De-vig method** | Proportional only | Shin/Power for extreme lines | Adequate for most |
| **Model diversity** | XGBoost only | XGBoost + LightGBM + Poisson | Single model |
| **Walk-forward duration** | 3 months (Q2 2024) | 2+ full seasons minimum | Way too short |
| **Hyperparameter search** | 40 iter / 384K combos | 100+ iter or Bayesian (Optuna) | Under-explored |

---

## Part 5: What the System Does Well

Credit where it's due — the engineering quality IS high in many areas:

1. **Anti-leak timestamps** — Features use `as_of_timestamp = game_date - 1 day`. Feature modules filter `game_date < target_day`. `assert_training_data_is_leakage_free()` enforces `as_of < scheduled_start`.
2. **Temporal CV throughout** — `TimeSeriesSplit` used everywhere, never random k-fold.
3. **Proper OOF stacking** — `_WarmupAugmentedClassifier` + `_PartitionedTemporalCV` ensures temporal ordering is preserved in stacking cross-validation. Each fold trains only on prior data + warmup.
4. **De-vig math** — Proportional de-vig is correctly implemented for two-way markets.
5. **Kelly with safeguards** — Quarter Kelly, 5% max bet fraction, 30% drawdown kill-switch, correlated-bet deduplication.
6. **Feature symmetry** — Home/away features are consistently generated with matching prefixes.
7. **Label integrity** — Targets derived from official F5 scores, excluded from feature columns via `_feature_columns` filtering.
8. **Walk-forward data rebuild per window** — Each window gets fresh data with proper `scheduled_start_before` cutoff.
9. **`profitability_metrics_valid` guard** — Synthetic odds now correctly prevent ROI reporting (this appears to be a fix implemented after the prior audit identified C1).
10. **Edge calculation is correct** — `edge = model_probability - fair_probability` where fair_probability is properly de-vigged. Standard +EV computation.
11. **Margin pricing** — `margin_to_cover_probability` correctly uses Gaussian CDF with `residual_std` computed from training residuals only (no look-ahead).
12. **Market recalibration is safe** — Deterministic shrink toward market, never amplifies model conviction, no fitting to future data.

---

## Part 6: Prioritized Action Plan

### Tier 1 — Fix the Evaluation (Nothing else matters until this is done)

| # | Action | Why |
|---|---|---|
| 1 | **Backtest only with real historical odds** | All ROI numbers without real odds are meaningless |
| 2 | **Implement systematic CLV tracking** as primary eval metric | Only reliable predictor of future profitability |
| 3 | **Move strategy optimization to nested walk-forward** | In-sample strategy optimization guarantees overfit |
| 4 | **Align walk-forward hyperparams with training search** | Backtest currently tests a model that would never be deployed |
| 5 | **Run walk-forward across 2+ full seasons (2021-2025)** | 3 months is insufficient for statistical conclusions |

### Tier 2 — Fix the Model

| # | Action | Why |
|---|---|---|
| 6 | **Add early stopping** (`early_stopping_rounds=50`) | Without it, XGBoost overfits on 10:1 ratio |
| 7 | **Prune to 50-70 features** by stable importance | 224 features = noise dimensions dominating |
| 8 | **Drop stacking for F5 ML** or add away-team meta-features | Hurts in ALL 4 experiments |
| 9 | **Drop Platt calibration** permanently (use identity) | Hurts in 7/8 measurements |
| 10 | **Use Optuna for hyperparameter search** with 100+ trials | 40 random iterations over 384K combos has no power |

### Tier 3 — Strengthen Defenses

| # | Action | Why |
|---|---|---|
| 11 | **Expand anti-leak tests** to cover rolling windows and OOF | Current tests only check timestamp, not feature computation |
| 12 | **Add model serialization version checks** | xgboost version changes silently break saved models |
| 13 | **Set `max_edge_to_bet=0.15`** instead of inverted bucketing | Edges >15% in efficient markets are data errors |
| 14 | **Add 1-day embargo gap** between train and test windows | Same-day information could leak |
| 15 | **Add train/live feature parity test** | Ensure production features match training features |

---

## Bottom Line

The system has solid engineering foundations — proper temporal splits, anti-leak assertions, correct de-vig math, comprehensive feature engineering, walk-forward framework, and good risk management. The code quality is genuinely high.

**But the evaluation methodology has fundamental flaws that make current performance numbers unreliable.** The +24.4% ROI is almost certainly an artifact of either synthetic odds (competing against its own features) or in-sample strategy optimization — or both. The walk-forward also uses completely different hyperparameters than what the training pipeline would select, so even the Brier scores may not reflect production model behavior.

**The honest question:** Can this system be profitable with fixes? Possibly, but expectations must be radically recalibrated:

- **Realistic ROI target:** 3-6% (not 24%)
- **Realistic bet rate:** 15-25% of games (not 84%)
- **The only metric that matters:** Positive CLV against sharp book closing lines over 500+ bets

The F5 market focus is the system's strongest strategic advantage — it's less efficiently priced than full-game lines. But the model's features are all publicly available sabermetrics that sharp markets already price. Genuine edge will come from **speed** (betting before lines move), **lineup-specific information** (before official announcements), or **novel signal combinations** that the market doesn't replicate — not from running XGBoost on the same stats every handicapper uses.

**Before any model improvements: fix the evaluation.** Only then will you know if the model has genuine predictive power.

---

## Appendix A: Research Sources

- University of Bath 2024 (Walsh & Joshi) — Calibration > accuracy for profitability (+34.69% ROI vs -35.17%)
- Wilkens 2021, Journal of Sports Analytics — ML ensembles cap at ~70% accuracy, returns "mainly negative"
- Pinnacle CLV studies — Positive CLV is the single best predictor of long-term profitability
- Lopez de Prado, Advances in Financial Machine Learning — Walk-forward validation, embargo gaps, purging
- Machine Learning with Applications 2024 — XGBoost calibration, post-hoc calibration methods
- 25+ industry publications (SportBot AI, WagerProof, EdgeSlip, ExPrysm, Sports-AI.dev, BetCommand, FairOdds)

## Appendix B: CLV Performance Benchmarks

| CLV Range | Interpretation |
|---|---|
| +1% to +2% | On right track, expect long-term profit |
| +2% to +4% | Strong performance, professional level |
| +4%+ | Elite — expect sportsbook limits |
| Negative CLV | Strategy failing regardless of current bankroll |

## Appendix C: Kelly Criterion Recommendations

| Approach | Return | Max Drawdown | Sharpe |
|---|---|---|---|
| Full Kelly | +42% | -35% | 1.34 |
| Quarter Kelly | +38% | -12% | 2.1 |

Quarter Kelly sacrifices 10% returns for 65% drawdown reduction. The system correctly uses fractional Kelly. Professional recommendation: 25% Kelly for confident models with 500+ tracked bets, 10-20% Kelly for new models.
