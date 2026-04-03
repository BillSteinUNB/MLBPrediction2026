# Away Run Research 5-Stage Plan

## Intent

This document is the execution plan for implementing the accepted deep-research MLB modeling ideas without deleting or destabilizing the current working away run-count lane.

The repo will intentionally support **multiple viewpoints**:

- `Control lane`: current canonical mean-focused away run-count model
- `Distribution lane`: new probabilistic away run model
- `MCMC lane`: new sequencing/simulation away run model

The goal is not to immediately replace the control lane. The goal is to build a second and third viewpoint that can outperform it on betting-relevant metrics or provide useful disagreement signals.

## Research Assumptions Accepted For This Plan

This plan assumes the following research claims are directionally correct and worth implementing:

1. Single-game `R^2` is a weak north-star for betting; distribution quality matters more.
2. MLB run counts are overdispersed and zero-heavy; mean-only Poisson modeling is insufficient.
3. Sequencing matters enough that a Markov / MCMC simulation lane is worth building.
4. Shrinkage, calibration, and market priors are important.
5. Air density, catcher framing, and umpire effects are real but should be expressed in betting-relevant ways.

Important repo note:

- The repo now uses [docs/research/ResearchInformation.md](/C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/docs/research/ResearchInformation.md) as the research landing page. The unrelated Micro-SaaS material lives under `archive/research/ResearchInformation.micro_saas.md`.
- Future agents must **not** treat that file as the source of truth for this roadmap.
- Stage 1 and Stage 5 include specific cleanup work to fix this repo confusion.

## Current Control Baseline

The control baseline that must remain preserved during this roadmap is:

- Artifact: [full_game_away_runs_model_20260328T012328Z_cc10c0f6.metadata.json](/C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/data/models/2026-away-flat-xgbonly-forceddelta8-fast-120x3-cc10/full_game_away_runs_model_20260328T012328Z_cc10c0f6.metadata.json)
- Experiment family: `flat` selector, `xgb_only`, `3` folds, repaired `ParquetDefault.parquet`, forced delta retention
- Holdout metrics:
  - `R^2 = 0.0382468335`
  - `RMSE = 3.2941466123`
  - `Poisson deviance = 2.5095824478`

This remains the control until a research lane materially proves it is better for betting or meaningfully useful as a second opinion.

## Program-Level Rules

1. Do not modify or delete the control lane.
2. Keep apples-to-apples comparisons:
   - same parquet
   - same holdout year unless explicitly changed
   - same season range unless explicitly changed
   - same target when comparing away-run experiments
3. New research work must log artifacts separately from control runs.
4. Every stage must leave behind machine-readable reports, not just chat summaries.
5. Repo cleanup must improve navigation without breaking current commands.

## Research-To-Implementation Map

This section translates the accepted research claims into concrete repo workstreams.

| Research Claim | Repo Interpretation | Stage |
| --- | --- | --- |
| `R^2` is not enough | add distribution scoring and betting-facing evaluation | 2 |
| public-data MLB scores are overdispersed and zero-heavy | add Negative Binomial and zero-adjusted distribution logic | 2, 3 |
| sequencing matters | add a separate Markov / Monte Carlo lane | 4 |
| shrinkage matters | reuse current Marcel-style shrinkage, do not rewrite it first | 1, 3 |
| calibration matters | add run-distribution calibration diagnostics, not just classifier calibration | 2, 3 |
| market priors matter | add market-prior data and market-anchor features in a separate lane | 3, 5 |
| air density matters | reuse current weather implementation, do not rebuild it | 1, 3, 4 |
| catcher framing / umpire effects matter | reuse current coarse features first, then deepen the feature family later | 3, 4 |
| MCMC is worth pursuing | build it as a separate slow lane, not as a replacement for control | 4 |

## Existing Repo Components To Reuse

Future agents should reuse these existing components rather than re-implementing them.

- Canonical manual build/train workflow in [ORCHESTRATOR_HANDOFF.md](/C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/docs/ORCHESTRATOR_HANDOFF.md)
- Away run-count control trainer in [run_count_trainer.py](/C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/src/model/run_count_trainer.py)
- Training parquet and feature assembly in [data_builder.py](/C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/src/model/data_builder.py)
- Current score pricing utilities in [score_pricing.py](/C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/src/model/score_pricing.py)
- Market shrinkage logic in [market_recalibration.py](/C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/src/model/market_recalibration.py)
- Existing probability calibration utilities in [calibration.py](/C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/src/model/calibration.py)
- Existing Marcel-style shrinkage in [marcel_blend.py](/C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/src/features/marcel_blend.py)
- Existing weather / air-density logic in [weather.py](/C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/src/features/adjustments/weather.py)
- Existing framing logic in [defense.py](/C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/src/features/defense.py)
- Existing umpire historical features in [umpires.py](/C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/src/features/umpires.py)

Future agents should not spend time on these duplicate tasks in Stage 1:

- rebuilding weather logic
- rebuilding air-density logic
- rebuilding generic calibration utilities
- replacing the control trainer
- reorganizing all tests before the registry/reporting foundation exists

## What Improvement Means

Improvement must be tracked in three layers.

### Layer A: Mean / Trainer Diagnostics

These remain useful but are not the primary research gate:

- Holdout `R^2`
- Holdout `RMSE`
- Holdout `MAE`
- Holdout Poisson deviance
- Mean bias: `predicted_mean - actual_mean`

### Layer B: Distribution Quality

These are the primary research metrics for the new lanes:

- `CRPS` on away runs
- Discrete log score / negative log likelihood for away-run PMF
- Calibration error for:
  - `P(away_runs = 0)`
  - `P(away_runs >= 1)`
  - `P(away_runs >= 3)`
  - `P(away_runs >= 5)`
  - `P(away_runs >= 10)`
- Tail sharpness and tail reliability
- Coverage of central intervals:
  - `50%`
  - `80%`
  - `95%`

### Layer C: Betting-Relevant Performance

These determine whether a research lane matters in practice:

- Walk-forward ROI
- CLV where available
- Max drawdown
- Bet count and pick stability
- Edge-bucket performance
- Disagreement value versus control lane

## Canonical Output Structure To Build Toward

By the end of this roadmap, the repo should be navigable through these top-level anchors:

```text
docs/
  INDEX.md
  roadmaps/
  research/
  repo/
  runbooks/

data/
  models/
  reports/
    run_count/
      registry/
      distribution_eval/
      dual_view/
      mcmc/

src/
  model/
  ops/
  pipeline/

tests/
  model/
  ops/
  pipeline/
  features/
  integration/
```

Do not attempt the full cleanup in Stage 1. Stage 1 creates the map and the scaffolding. Stage 5 performs the reorganization.

---

## Stage 1: Foundation, Measurement, And Repo Navigation

### Objective

Create the documentation, registry, and reporting foundation needed to support all later stages. This stage must be executable by an AI agent with no ambiguity.

### Mandatory Scope

Stage 1 must do all of the following:

1. Create a clear docs entry structure so future agents stop guessing where things are.
2. Create a run-count research registry that scans existing away-run artifacts and writes machine-readable summaries.
3. Register the current control baseline and recent comparables.
4. Create a clean place for future distribution reports.
5. Write the MLB research assumptions into a repo-local file so future agents are not misled by the wrong root research file.

### Stage 1 Execution Order

The Stage 1 agent must execute work in this order:

1. Read [ORCHESTRATOR_HANDOFF.md](/C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/docs/ORCHESTRATOR_HANDOFF.md)
2. Read [docs/research/MODEL_COMPARISON_TRACKER.md](/C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/docs/research/MODEL_COMPARISON_TRACKER.md)
3. Inspect the existing top-level repo layout
4. Create the new docs and report directories
5. Implement `scripts/report_run_count_research_state.py`
6. Implement `tests/ops/test_run_count_research_state.py`
7. Run the registry script
8. Inspect the produced registry output and verify that the current control artifact is correct
9. Write the documentation files using the verified registry output
10. Run the required tests
11. Run fast validation
12. Stop and hand off

The Stage 1 agent must not change the execution order unless blocked by a discovered repo inconsistency.

### Non-Goals

Stage 1 must **not**:

- train new models
- modify the current trainer logic
- change feature generation
- change the daily pipeline
- move or delete existing temp folders yet
- rename existing experiments

### Files To Create

The Stage 1 agent must create exactly these files and directories:

```text
docs/
  INDEX.md
  research/
    MLB_RUN_MODEL_RESEARCH_ASSUMPTIONS.md
  repo/
    REPO_MAP.md
  runbooks/
    RUN_COUNT_RESEARCH_WORKFLOW.md

data/
  reports/
    run_count/
      README.md
      registry/

scripts/
  report_run_count_research_state.py

tests/
  ops/
    test_run_count_research_state.py
```

If `tests/ops/` does not exist, create it.

### Required Contents By File

#### `docs/INDEX.md`

Must contain:

- one-screen entrypoint to the repo
- where to find:
  - control modeling docs
  - research docs
  - repo map
  - runbooks
  - AutoResearch docs
- a "start here" path for:
  - new orchestrator
  - modeling agent
  - pipeline/debugging agent

#### `docs/research/MLB_RUN_MODEL_RESEARCH_ASSUMPTIONS.md`

Must contain:

- the accepted MLB research assumptions from this roadmap
- a note that `docs/research/ResearchInformation.md` is the current research landing page and the unrelated material is archived
- a concise rationale for:
  - distribution scoring
  - overdispersion / zero mass
  - market priors
  - sequencing / MCMC

#### `docs/repo/REPO_MAP.md`

Must contain:

- top-level folder map
- what each top-level folder is for
- "safe to ignore for most agents" list
- current high-signal files for away-run research:
  - `docs/ORCHESTRATOR_HANDOFF.md`
  - `docs/research/MODEL_COMPARISON_TRACKER.md`
  - `src/model/run_count_trainer.py`
  - `src/model/data_builder.py`
  - `src/pipeline/daily.py`
  - `src/model/score_pricing.py`

#### `docs/runbooks/RUN_COUNT_RESEARCH_WORKFLOW.md`

Must contain:

- canonical PowerShell build/train commands from [ORCHESTRATOR_HANDOFF.md](/C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/docs/ORCHESTRATOR_HANDOFF.md)
- explicit distinction between:
  - control lane
  - distribution lane
  - MCMC lane
- naming convention for future experiment folders

Recommended naming convention:

- Control mean lane: `2026-away-*`
- Distribution lane: `2026-away-dist-*`
- MCMC lane: `2026-away-mcmc-*`

#### `data/reports/run_count/README.md`

Must define the purpose of:

- `registry/`
- future `distribution_eval/`
- future `dual_view/`
- future `mcmc/`

#### `scripts/report_run_count_research_state.py`

This script is the key Stage 1 deliverable.

It must:

1. Scan `data/models/` recursively for `full_game_away_runs_model_*.metadata.json`
2. Parse metadata and emit:
   - `data/reports/run_count/registry/full_game_away_runs_registry.csv`
   - `data/reports/run_count/registry/full_game_away_runs_registry.json`
   - `data/reports/run_count/registry/current_control.json`
3. Support a `--models-root` argument defaulting to `data\models`
4. Support an `--output-dir` argument defaulting to `data\reports\run_count\registry`
5. Support a `--target-model` argument defaulting to `full_game_away_runs_model`

The script must extract at minimum these fields:

- `experiment_dir`
- `artifact_path`
- `model_name`
- `target_column`
- `model_version`
- `data_version_hash`
- `holdout_season`
- `cv_metric_name`
- `cv_best_score`
- `cv_best_rmse`
- `holdout_r2`
- `holdout_rmse`
- `holdout_mae`
- `holdout_poisson_deviance`
- `selected_feature_count`
- `feature_selection_mode` if present
- `blend_mode`
- `cv_aggregation_mode`
- `forced_delta_count`
- `forced_delta_feature_count`
- `weather_feature_count`
- `log5_feature_count`
- `plate_umpire_feature_count`
- `framing_feature_count`
- `delta_feature_count`
- `top_5_features`

The script must sort the registry output using this priority:

1. latest artifact modified time descending
2. `holdout_r2` descending
3. `holdout_rmse` ascending

The script must write `top_5_features` in JSON as an array of objects:

```json
[
  {"feature": "home_team_log5_30g", "importance": 0.0442},
  {"feature": "away_team_log5_30g", "importance": 0.0345}
]
```

The CSV version may serialize `top_5_features` as a compact JSON string.

The script must also determine the current control run by matching this exact preference order:

1. explicit experiment name contains `forceddelta8`
2. `blend_mode == xgb_only`
3. `feature_selection_mode == flat` if present
4. `holdout_season == 2025`
5. highest `holdout_r2`

If that logic fails, the script must still emit a registry and write `current_control.json` with a warning field explaining the fallback.

The script must write `current_control.json` with at minimum this schema:

```json
{
  "selected_artifact_path": "data/models/...metadata.json",
  "experiment_dir": "data/models/...",
  "selection_reason": "matched forceddelta8 + xgb_only + holdout 2025",
  "warning": null,
  "holdout_r2": 0.0382,
  "holdout_rmse": 3.2941,
  "holdout_poisson_deviance": 2.5096,
  "data_version_hash": "cc10c0f6",
  "blend_mode": "xgb_only",
  "feature_selection_mode": "flat"
}
```

#### `tests/ops/test_run_count_research_state.py`

Must validate:

- registry generation on synthetic metadata files
- feature-family counting logic
- control selection logic
- missing optional fields do not crash the script

### Implementation Notes For The Stage 1 Agent

The Stage 1 agent must prefer read-only parsing of existing metadata. It must not infer values that are not present unless they can be safely counted from `feature_columns`.

The Stage 1 agent should not load or rebuild parquet data unless needed for a validator. Metadata parsing is the source of truth for the registry.

Feature-family counts must use these rules:

- `weather_` prefix -> weather
- `_log5_` token -> log5
- `plate_umpire_` prefix -> umpire
- `adjusted_framing` token -> framing
- `_delta_7v30` token -> delta

If a count cannot be determined safely, the script must write `null` rather than guessing.

### Stage 1 Blocker Rules

If the Stage 1 agent hits one of these blockers, it must stop and report instead of wandering into unrelated work:

- metadata files are malformed and cannot be parsed
- control-artifact selection is ambiguous after the defined fallback logic
- `scripts\validate_modeling.py --profile fast` fails for unrelated pre-existing reasons
- the repo contains a second conflicting orchestrator handoff

If `validate_modeling.py --profile fast` fails because of a clearly unrelated existing failure, the Stage 1 agent must:

1. keep the newly created files intact
2. report the validator failure as a blocker
3. not attempt unrelated repairs in Stage 1

### Validation Commands For Stage 1

The Stage 1 agent must run:

```powershell
.\.venv\Scripts\python.exe `
  scripts\report_run_count_research_state.py `
  --models-root data\models `
  --output-dir data\reports\run_count\registry
```

```powershell
.\.venv\Scripts\python.exe `
  -m pytest `
  tests\ops\test_run_count_research_state.py
```

```powershell
.\.venv\Scripts\python.exe `
  scripts\validate_modeling.py `
  --profile fast
```

### Stage 1 Exit Criteria

Stage 1 is complete only if:

- all required files exist
- the registry files are generated successfully
- the current control baseline is correctly written to `current_control.json`
- tests pass
- fast modeling validation passes
- a future agent can open `docs/INDEX.md` and know where to start

### Stage 1 Handoff Format

The Stage 1 agent must report:

1. created files
2. registry row count
3. selected control artifact
4. any metadata inconsistencies found
5. whether the research landing page and archive split were documented

---

## Stage 2: Distribution Scoring And Baseline Probabilistic Evaluation

### Objective

Measure the current control lane as a run distribution, not just a point predictor.

### Why Stage 2 Exists

The repo currently trains mean-focused run-count models in [run_count_trainer.py](/C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/src/model/run_count_trainer.py), but downstream pricing in [score_pricing.py](/C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/src/model/score_pricing.py) already behaves as if a full distribution exists. Stage 2 closes that measurement gap before architecture changes are made.

### Files To Add

```text
src/model/run_distribution_metrics.py
scripts/evaluate_run_distribution.py
tests/model/test_run_distribution_metrics.py
```

### Required Implementation

Stage 2 must:

1. Load an existing away-run artifact and regenerate holdout predictions from parquet.
2. Build discrete away-run PMFs for at least these baseline families:
   - Poisson
   - Negative Binomial
   - zero-adjusted Negative Binomial
3. Score those PMFs on holdout with:
   - CRPS
   - discrete log score
   - zero calibration
   - tail calibration
   - interval coverage
4. Write machine-readable output to:
   - `data/reports/run_count/distribution_eval/`

### Required Output Files

For each evaluated artifact, write:

- `<model_version>.distribution_eval.json`
- `<model_version>.distribution_eval.csv`

### Required Comparison Logic

Stage 2 must compare:

- current control artifact
- at least one weaker comparable artifact

The purpose is to confirm that distribution metrics separate good and bad runs meaningfully.

### Success Criteria

Stage 2 is successful if:

- distribution evaluation runs on historical away-run artifacts
- CRPS and log score are computed without inventing new training
- the report clearly shows whether the control lane is better than weaker comparables on distribution metrics

### Stage 2 Gate

Do not begin Stage 3 until Stage 2 demonstrates that the new distribution metrics are stable enough to distinguish:

- the control artifact
- at least one weaker comparable artifact

If the metrics do not separate those runs meaningfully, fix the evaluation layer before building the new trainer.

---

## Stage 3: Distribution Lane V1

### Objective

Build a new experimental away-run lane that predicts more than the mean.

### Design Rule

Do **not** replace the current trainer. Create a parallel trainer.

### Files To Add

```text
src/model/run_distribution_trainer.py
scripts/train_run_distribution.py
tests/model/test_run_distribution_trainer.py
```

### Required Modeling Approach

Distribution Lane V1 must predict:

1. `mu`: expected away runs
2. `dispersion`: game-specific overdispersion or variance parameter
3. `p_zero_extra`: excess shutout probability or zero-mass adjustment

Recommended first implementation:

- keep the existing mean-model feature surface close to the control lane
- derive dispersion from a separate supervised head
- derive zero-mass from a separate classification head
- combine them into a zero-adjusted Negative Binomial PMF

This lane must stay compatible with the current repaired parquet and current manual workflow philosophy.

### Required Controls

The first Stage 3 experiments must keep fixed:

- `data/training/ParquetDefault.parquet`
- season window `2018-2025`
- holdout `2025`
- `3` folds for first comparable experiments
- `flat` selector for first comparable experiments

### Required Outputs

The trainer must emit metadata including:

- mean metrics
- distribution metrics
- zero calibration
- tail calibration
- selected features by head if heads differ

The Stage 3 trainer should also emit:

- fitted distribution family name
- dispersion summary statistics on holdout
- zero-mass summary statistics on holdout
- calibration bins for shutout probability
- calibration bins for tail probabilities

### Success Criteria

Stage 3 is promising if at least one distribution-lane run:

- improves CRPS or discrete log score versus the control lane
- does not catastrophically regress RMSE
- shows better zero/tail calibration than the control lane

Catastrophic regression means any of:

- RMSE worse by more than `2%`
- mean bias worse by more than `0.15` runs
- tail calibration clearly unstable

---

## Stage 4: MCMC / Markov Simulation Lane

### Objective

Build the sequencing-aware away-run lane that the research argues is the correct long-term direction.

### Design Rule

This lane is deliberately separate from the control lane and the Stage 3 distribution lane. It is allowed to be slower and more experimental.

### Stage 4 Design Intent

The MCMC lane is being built because the accepted research argues that sequencing is fundamentally meaningful. This lane is allowed to remain on the back burner if it is slower to mature, but it must still be implemented as a first-class experimental path.

### Files To Add

```text
src/model/mcmc_engine.py
src/model/mcmc_feature_builder.py
src/model/mcmc_pricing.py
scripts/run_mcmc_distribution.py
tests/model/test_mcmc_engine.py
tests/model/test_mcmc_pricing.py
```

### Required Architecture

Stage 4 must use a **24 base-out state engine** and Monte Carlo simulation.

Required design:

1. Represent inning state with:
   - outs: `0, 1, 2`
   - base occupancy: `000` through `111`
2. Parameterize plate appearance event probabilities for at least:
   - out
   - walk / HBP
   - single
   - double
   - triple
   - home run
3. Simulate full away offensive half-innings and aggregate to game-level away run distributions.
4. Run enough simulations per game to stabilize the PMF:
   - default target: `20,000`
   - must be configurable

Required first-version simplifying assumptions:

- one away offensive simulation path is sufficient for initial build
- starter-to-bullpen handoff may begin as a rule-based innings threshold
- event probabilities may begin from team / lineup / starter summaries before batter-level modeling is available
- park, weather, and umpire modifiers may be applied multiplicatively in V1

### Stage 4 Practical Rule

Do not block this stage on perfect batter-level data if it is unavailable in the current feature path.

Stage 4 may start with:

- lineup-order summaries
- team and lineup event-rate estimates
- starter and bullpen context
- park/weather/umpire modifiers

Then iterate toward finer batter-level resolution later.

### Required Validation

The MCMC engine must have deterministic tests for:

- state transitions
- inning termination
- run accounting
- probability normalization
- reproducibility with a fixed seed

### Required Outputs

The Stage 4 lane must write:

- away-run PMF
- expected away runs
- shutout probability
- tail probabilities
- simulation diagnostics

### Success Criteria

Stage 4 is promising if it beats either:

- the control lane on distribution metrics, or
- the Stage 3 distribution lane on distribution metrics

If it does not, it remains an exploratory lane and is not promoted.

---

## Stage 5: Dual-View Operation, Promotion Rules, And Repo Cleanup

### Objective

Make the repo usable, navigable, and capable of running multiple viewpoints side by side.

### Required Outcomes

1. The control lane stays intact.
2. The best research lane is available as a second opinion.
3. Repo navigation becomes clean enough that opening the repo does not feel like opening a wall of unrelated tests.

### Required Dual-View Behavior

The repo must support side-by-side comparison between:

- control mean lane
- best distribution lane
- best MCMC lane if available

At minimum, dual-view reporting must show:

- expected away runs from each lane
- shutout probability from each lane
- `P(away_runs >= 3)`
- `P(away_runs >= 5)`
- disagreement flags

Recommended disagreement flags:

- mean difference `>= 0.35` runs
- shutout probability gap `>= 0.05`
- tail probability gap `>= 0.04`

### Required Repo Cleanup Work

Stage 5 must:

1. Create `docs/INDEX.md` driven navigation if not already present.
2. Move or archive confusing non-MLB research material:
   - the research landing page should live under `docs/research/` and non-MLB material should stay under `archive/`
3. Reorganize tests into subfolders:
   - `tests/model/`
   - `tests/ops/`
   - `tests/pipeline/`
   - `tests/features/`
   - `tests/integration/`
4. Move temp backtest folders under a single archival or ignored location if they are not needed for active work.
5. Leave a clear "where to look first" path for future agents.

Recommended Stage 5 cleanup targets:

- `temp_backtest_*`
- `temp_walkforward_*`
- `temp_strategy_*`
- `tmp_pytest_*`
- stray top-level HTML scratch files that are no longer part of an active workflow

The cleanup must preserve any file still referenced by current scripts, tests, or docs.

### Promotion Rule

A research lane is promotable only if it satisfies all of:

1. better distribution metrics than the control lane
2. stable or acceptable mean diagnostics
3. no alarming calibration failures
4. at least neutral betting performance in walk-forward terms

If no research lane clears promotion, keep it as a secondary opinion only.

---

## Operating Model If Success Appears

If a research lane starts showing real value, the intended steady-state is:

- Control lane remains the stable benchmark
- Distribution lane becomes the probabilistic betting view
- MCMC lane becomes the sequencing-aware challenge model

Use cases:

- agreement between lanes -> higher conviction
- disagreement between lanes -> review trigger
- control stronger on mean, research stronger on tails -> use both in decision support

## Immediate Next Action

The next execution step after this document exists is:

- assign an agent to **Stage 1 only**
- do not let that agent jump ahead into modeling changes
- require the Stage 1 exit criteria before starting Stage 2

## Copy-Paste Stage 1 Agent Prompt

```text
Work only on Stage 1 from docs/roadmaps/AWAY_RUN_RESEARCH_5_STAGE_PLAN.md.

Your job is to build the documentation and registry foundation only.

Do not:
- train new models
- change the existing trainer
- change feature generation
- reorganize the whole repo
- start Stage 2

You must:
1. create the required docs and data/report directories
2. implement scripts/report_run_count_research_state.py
3. implement tests/ops/test_run_count_research_state.py
4. generate the away-run registry and current_control.json
5. run the required tests and fast validation
6. stop after Stage 1 and report blockers if any

Use the roadmap as the source of truth.
```

## Canonical PowerShell Commands To Preserve

Build parquet:

```powershell
.\.venv\Scripts\python.exe `
  scripts\build_parquet.py `
  --training-data data\training\ParquetDefault.parquet `
  --start 2018 `
  --end 2025 `
  --FeatureWorker 10
```

Train control-style run-count model:

```powershell
.\.venv\Scripts\python.exe `
  scripts\train_run_count.py `
  --profile flat-fast `
  --training-data data\training\ParquetDefault.parquet `
  --start 2018 `
  --end 2025 `
  --holdout 2025 `
  --XGBWork 4 `
  --OptunaWork 3 `
  --Iterations 120 `
  --Folds 3 `
  --feature-selection-mode flat `
  --blend-mode xgb_only `
  --cv-aggregation-mode mean
```

These commands remain canonical for the control lane even while the research lanes are added.
