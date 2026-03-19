# Draft: Autoresearch Adaptation for MLB Prediction Model

## User Decisions
- **Approach**: Hybrid (autoresearch loop for creative experiments + Optuna for HP sweeps)
- **Primary Metric**: CLV (Closing Line Value) — with weighted composites as secondary experiments
- **Agent Power**: OpenCode agent session (runs in terminal with program_mlb.md as skill)

---

## What Autoresearch IS (Core Mechanics)

### The Loop (from program.md)
```
LOOP FOREVER:
  1. Look at git state (current branch/commit)
  2. Modify train.py with an experimental idea
  3. git commit
  4. Run experiment: uv run train.py > run.log 2>&1
  5. Read results: grep val_bpb from run.log
  6. If crashed → attempt fix or log crash and move on
  7. Log to results.tsv (commit, val_bpb, memory, status, description)
  8. If improved → keep commit ("advance" branch)
  9. If equal/worse → git reset to previous best
  NEVER STOP until human interrupts
```

### Three-File Architecture
- **prepare.py** (FIXED): Data prep, tokenizer, evaluation function
- **train.py** (AGENT MODIFIES): Model, optimizer, training loop
- **program.md** (HUMAN WRITES): Research instructions, constraints, strategy

### Proven Results
- **Karpathy**: 700 experiments over 2 days → 11% efficiency gain, agents rediscovered RMSNorm and tied embeddings
- **Tobi Lutke (Shopify CEO)**: 19% improvement in query expansion model overnight
- **Community**: 43K stars, 6K forks, ports to every platform within weeks

---

## Hybrid Architecture: Autoresearch + Optuna for MLB

### Two Optimization Modes

**Mode 1: Creative Experiments (Autoresearch/OpenCode Loop)**
- OpenCode agent reads `program_mlb.md`
- Agent proposes STRUCTURAL changes: new features, different stacking, feature interactions, novel derived metrics
- Each experiment modifies `experiment.py` → runs backtest → measures CLV
- Git-based: keep improvements, revert failures
- **This is where the LLM's REASONING matters** — inventing things Optuna can't

**Mode 2: HP Sweeps (Optuna, runs without LLM)**
- After creative experiments find promising structural changes
- Optuna fine-tunes numerical hyperparameters: learning rates, window sizes, thresholds
- Bayesian optimization (TPE) is MORE EFFICIENT than LLM guessing for pure numbers
- Can run 50-100 trials overnight WITHOUT API costs
- **This is systematic, exhaustive, and cheap**

### The Overnight Schedule
```
6:00 PM — Human kicks off OpenCode session with program_mlb.md
6:00 PM - 12:00 AM — Mode 1: Creative experiments (~70 experiments)
  - Agent proposes feature engineering ideas, model architecture tweaks
  - Tests novel derived features (e.g., "xFIP × park factor × weather composite")
  - Tries different stacking configurations
  - Ablates features to find minimal effective set

12:00 AM — Human-set timer triggers Mode 2 (OR agent decides to switch)
12:00 AM - 6:00 AM — Mode 2: Optuna HP sweep (~100-200 trials)
  - Takes BEST structural configuration from Mode 1
  - Optimizes: XGBoost max_depth, learning_rate, n_estimators, subsample
  - Optimizes: edge threshold, Kelly fraction, calibration set size
  - Optimizes: Marcel regression weights per metric type

6:00 AM — Human wakes up to:
  - results.tsv with full experiment log (Mode 1)
  - Optuna study with best hyperparameters (Mode 2)
  - Git branch with accumulated improvements
  - CLV improvement trajectory chart
```

---

## File Architecture for the Hybrid System

```
MLBPrediction2026/
├── src/                          # Main prediction system (from plan)
├── autorefine/                   # The overnight refinement system
│   ├── program_mlb.md            # Agent instructions (human writes)
│   ├── evaluate.py               # FIXED — walk-forward backtest + CLV computation
│   ├── experiment.py             # AGENT MODIFIES — feature config, model config
│   ├── optuna_sweep.py           # Mode 2 — Optuna HP optimization (no LLM needed)
│   ├── results.tsv               # Experiment log (untracked by git)
│   ├── analysis.ipynb            # Morning review notebook
│   └── configs/
│       ├── baseline.yaml         # Current best configuration
│       └── search_space.yaml     # Optuna search space definition
└── data/                         # Cached Parquet + SQLite (shared)
```

### evaluate.py (FIXED — Agent Cannot Touch)
```python
# This runs the SAME walk-forward backtest every time
# Uses cached 2019-2025 data (no API calls, pure CPU)
# Time budget: ~5 minutes per experiment

def run_evaluation(config_path: str) -> dict:
    """
    Fixed evaluation harness.
    Loads config from experiment.py, runs walk-forward backtest,
    returns metrics including CLV.
    """
    config = load_config(config_path)
    
    # Walk-forward backtest (2022-2025, 6-month train, 1-month test)
    results = walk_forward_backtest(
        data=cached_parquet_data,
        features=config['features'],
        model_params=config['model_params'],
        calibration=config['calibration'],
        edge_threshold=config['edge_threshold'],
        kelly_fraction=config['kelly_fraction'],
    )
    
    # Compute CLV using cached historical odds
    clv = compute_clv(
        predictions=results['predictions'],
        bet_time_odds=results['bet_time_odds'],
        closing_odds=cached_closing_odds,
    )
    
    # Output format (matches autoresearch grep pattern)
    print(f"---")
    print(f"clv:              {clv:.6f}")
    print(f"brier:            {results['brier']:.6f}")
    print(f"roi:              {results['roi']:.4f}")
    print(f"ece:              {results['ece']:.6f}")
    print(f"total_bets:       {results['total_bets']}")
    print(f"win_rate:         {results['win_rate']:.4f}")
    print(f"runtime_seconds:  {results['runtime']:.1f}")
    
    return {'clv': clv, 'brier': results['brier'], 'roi': results['roi']}
```

### experiment.py (AGENT MODIFIES — The Playground)
```python
# This is the ONLY file the agent modifies
# It defines the complete model configuration

FEATURE_CONFIG = {
    'offense': {
        'enabled': True,
        'metrics': ['wRC+', 'wOBA', 'ISO', 'BABIP', 'K%', 'BB%'],
        'windows': [7, 14, 30, 60],
        'lineup_weighted': True,
    },
    'pitching': {
        'enabled': True,
        'metrics': ['xFIP', 'xERA', 'K%', 'BB%', 'GB%', 'velocity'],
        'windows': [7, 14, 30, 60],  # counted by starts
        'include_pitch_entropy': True,
    },
    'defense': {'enabled': True, 'abs_framing_retention': 0.75},
    'bullpen': {'enabled': True, 'pc_windows': [3, 5]},
    'park': {'enabled': True},
    'weather': {'enabled': True},
    'baselines': {
        'pythagorean_exp': 1.83,
        'use_f5_pythagorean': True,
        'log5_team_only': True,  # Never batter-pitcher (biased)
    },
}

MODEL_CONFIG = {
    'xgboost': {
        'max_depth': 6,
        'n_estimators': 300,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
    },
    'stacking': {
        'meta_features': ['pythagorean_wp', 'log5_prob', 'park_factor'],
        'lr_C': 1.0,
    },
    'calibration': {
        'method': 'isotonic',  # or 'platt'
        'cal_set_fraction': 0.10,
    },
}

DECISION_CONFIG = {
    'edge_threshold': 0.03,
    'kelly_fraction': 0.25,
    'max_bet_pct': 0.05,
}

MARCEL_CONFIG = {
    'offense_regression_games': 30,
    'pitching_regression_starts': 15,
    'defense_regression_games': 50,
}
```

### program_mlb.md (Human Writes — Agent Instructions)
```markdown
# MLB Prediction Model Overnight Refinement

## Context
You are optimizing an MLB F5 betting prediction model. The primary metric
is CLV (Closing Line Value) — higher is better. This measures whether the
model consistently identifies value before the market corrects.

## Setup
1. Create branch: git checkout -b autorefine/<date>
2. Read: evaluate.py (FIXED), experiment.py (YOU MODIFY), this file
3. Run baseline: python evaluate.py > run.log 2>&1
4. Initialize results.tsv with baseline

## What You CAN Modify
- experiment.py — feature config, model config, decision config, Marcel config
- Everything in experiment.py is fair game

## What You CANNOT Modify
- evaluate.py — fixed evaluation harness
- Any files in src/ — the production pipeline
- Do NOT install new packages

## The Goal
Maximize CLV. Secondary: minimize Brier score. Tertiary: maximize ROI.

## Experiment Strategies (try these categories)
1. FEATURE ABLATION: Remove one feature group, measure impact on CLV
2. WINDOW OPTIMIZATION: Try [10,20,40], [14,28,56], [7,21,42] etc.
3. NEW FEATURES: Add interaction terms (e.g., starter_xFIP × park_factor)
4. STACKING VARIANTS: Add/remove baselines from meta-learner
5. CALIBRATION: Try Platt vs Isotonic, vary cal_set_fraction
6. THRESHOLD TUNING: Edge 2%→5%, Kelly 0.15→0.35
7. MARCEL TUNING: Regression weights 10→50 games
8. SIMPLIFICATION: Remove features/complexity for same CLV (always keep)

## The Experiment Loop
LOOP FOREVER (same as Karpathy autoresearch):
1. Modify experiment.py with ONE idea
2. git commit
3. Run: python evaluate.py > run.log 2>&1
4. Read: grep "^clv:" run.log
5. If CLV improved → keep. If not → git reset.
6. Log to results.tsv
7. NEVER STOP. Run until interrupted.

## Handoff to Optuna (Mode 2)
When you've exhausted structural ideas (new features, architecture changes),
switch to Optuna for fine-tuning numerical hyperparameters:
  python optuna_sweep.py --trials 200 --timeout 21600
This runs Bayesian optimization on the numerical parameters in experiment.py
using the structural configuration you've established.
```

---

## CLV as Primary Metric — Implementation

### Why CLV is the RIGHT metric for this
- **CLV (Closing Line Value)**: Measures if you're consistently betting BEFORE the market moves in your direction
- Positive CLV = your model identifies value early → the market catches up → you had the better price
- This is the #1 predictor of long-term profitability in sports betting
- It's cheap to verify: one number comparison (like val_bpb)

### CLV Computation in evaluate.py
```python
def compute_clv(predictions, bet_time_odds, closing_odds):
    """
    CLV = average(closing_implied - bet_time_implied) for all bets placed.
    Positive CLV = beating the market. Higher = better.
    """
    clv_values = []
    for pred in predictions:
        if pred['bet_placed']:
            bet_implied = american_to_implied(pred['odds_at_bet'])
            close_implied = american_to_implied(closing_odds[pred['game_pk']])
            clv = close_implied - bet_implied
            clv_values.append(clv)
    return np.mean(clv_values) if clv_values else 0.0
```

### Weighted Composite as Secondary Experiment
```python
# The agent can also experiment with composites in experiment.py:
METRIC_CONFIG = {
    'primary': 'clv',  # What gets tracked in results.tsv
    'composite': {
        'clv_weight': 0.5,
        'brier_weight': 0.3,     # Lower brier = better calibration
        'roi_weight': 0.2,       # Higher ROI = more profitable
    },
    'use_composite': False,  # Agent can toggle this to True
}
```

---

## Research Findings Summary

### Real-World Autoresearch Results
- Karpathy: 11% efficiency gain over 700 experiments (2 days)
- Tobi Lutke: 19% improvement in query expansion model overnight
- Agents independently rediscovered RMSNorm and tied embeddings (8 years of human research)
- Community: 43K stars, 6K forks, ports to every platform in weeks

### Key Insight: Why This Works for MLB
The autoresearch pattern works when:
1. **Verification is cheap** → Walk-forward backtest: ~5 minutes on cached data ✅
2. **Metric is a single number** → CLV: one number, higher is better ✅
3. **Search space is rich** → 40+ features, 20+ hyperparameters, architectural choices ✅
4. **Agent can reason about domain** → LLM can read baseball stats documentation ✅
5. **Each experiment is independent** → No state between experiments ✅

### What Optuna Brings to the Hybrid
- TPE (Tree-Parzen Estimator) is ~3-5x more sample-efficient than random search for HP tuning
- Can run WITHOUT LLM (zero API cost) for pure numerical optimization
- Built-in pruning (early stopping of bad trials)
- Study persistence (can resume overnight runs)
- Visualization of parameter importance

---

## Implementation Timeline

### Phase 1 (After v1 MLB model is stable)
1. Create `autorefine/` directory
2. Build `evaluate.py` — wrap walk-forward backtest as fixed harness
3. Extract `experiment.py` — parameterize all model/feature choices
4. Write `program_mlb.md` — initial research directions
5. Cache historical odds data for CLV computation
6. Test: run 5 experiments manually to verify loop works

### Phase 2 (First Overnight Run)
1. Start OpenCode session with program_mlb.md
2. Let it run overnight (Mode 1: creative experiments)
3. Morning review: analyze results.tsv + git log
4. Identify which structural changes stuck

### Phase 3 (Add Optuna Integration)
1. Build `optuna_sweep.py` — Bayesian HP optimization
2. Define search space from experiment.py parameter ranges
3. Run combined overnight: Mode 1 → Mode 2 handoff
4. Compare: agent's creative finds vs Optuna's numerical optimization

### Phase 4 (Refinement Loop)
1. Update program_mlb.md based on what worked/didn't
2. Narrow search space based on accumulated results
3. Run weekly overnight sessions to keep model sharp
4. Track CLV trend over time — is the model improving week-over-week?
