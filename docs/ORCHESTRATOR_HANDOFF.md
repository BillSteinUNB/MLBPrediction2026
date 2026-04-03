# Orchestrator Handoff

Use this document to onboard a new AI chat that is acting as the project orchestrator for this repo.

## Project Summary

- Repo: `MLBPrediction2026`
- Primary goal: build a leak-safe MLB run/F5 prediction system that is strong enough to support betting workflows for the 2026 season.
- Current active work is concentrated on the **run-count modeling pipeline**, especially the single-model research path for `full_game_away_runs` / `final_away_score`.
- Core model family: **XGBoost + LightGBM blend**
- Current training objective: **Poisson-style count modeling**
- Main evaluation focus: **holdout season performance**, especially holdout `R²`, `RMSE`, and Poisson deviance.

## What The User Is Working On

The user is iterating on why holdout `R²` remains modest and unstable even after trainer fixes and feature work.

The important current framing is:

- trainer-side metric alignment was already improved
- bigger Optuna searches do not automatically improve holdout
- feature representation and feature selection are likely still the main bottlenecks
- the user often runs experiments manually and asks the agent to interpret the resulting artifacts

Do not assume the latest run number from this document. The run number may have advanced. Always inspect the latest artifacts directly.

## Where To Look First

### High-signal files

- `docs/research/MODEL_COMPARISON_TRACKER.md`
  - human-maintained history of major modeling runs
  - best place to understand what changed run-to-run

- `src/model/run_count_trainer.py`
  - Optuna search
  - CV setup
  - feature selection entry points
  - early stopping
  - blending
  - metadata emitted for each run

- `src/model/data_builder.py`
  - training/inference feature assembly
  - temporal windows
  - feature fill defaults
  - matchup and delta feature generation

- `src/model/single_model_profiles.py`
  - smoke / fast / full single-model experiment profiles
  - search iterations, folds, early stopping, search spaces

- `scripts/train_run_count.py`
  - canonical manual training entrypoint for run-count research from an existing parquet

- `scripts/build_parquet.py`
  - canonical manual parquet builder for run-count research

- `scripts/validate_modeling.py`
  - fast/full validation workflow for modeling-related files

### Common test files

- `tests/model/test_run_count_trainer.py`
- `tests/pipeline/test_data_builder.py`
- `tests/model/test_single_model_profiles.py`

## Where The Latest State Lives

Do not trust only chat history. Inspect the filesystem.

### Experiment outputs

- Directory: `data/models/`
- Find the newest experiment folder by modified time
- Inside each experiment folder, inspect:
  - `*.metadata.json` for holdout metrics, selected features, hyperparameters, and Optuna info
  - `run_count_training_run_*.json`
  - `*_model_training_run_*.json`
  - `optuna_studies.db`

### Training data

- Main parquet path typically used by the manual run-count scripts:
  - `data/training/training_data_2018_2025.parquet`

### Current narrative / run history

- `docs/research/MODEL_COMPARISON_TRACKER.md`

If the tracker and artifacts disagree, trust the artifact files first, then update the tracker if requested.

## Current Working Style

The user is on Windows and usually runs commands in **PowerShell**.

### Canonical manual commands

For the manual run-count workflow, there are exactly two canonical commands:

1. Build the parquet
2. Train from an existing parquet

Use these first before suggesting anything else.

### Canonical build command

```powershell
.\.venv\Scripts\python.exe `
  scripts\build_parquet.py `
  --training-data data\training\ParquetDefault.parquet `
  --start 2018 `
  --end 2025 `
  --FeatureWorker 10
```

What an agent should edit:
- Change `--training-data` only if the user explicitly wants a different parquet filename or path
- Change `--start` and `--end` only if the user explicitly wants a different season window
- Change `--FeatureWorker` only if the user wants a different feature-build worker count; valid range is `1` to `12`
- Add `--refresh-data` only if the user explicitly wants to refresh upstream raw sources
- Default manual rebuilds should usually omit `--refresh-data`
  - In the manual build script, no `--refresh-data` now means local-only weather lookups with no live weather backfill

What an agent should not casually change:
- Do not swap this to an archived wrapper
- Do not replace it with an inline Python command
- Do not change the year range unless the user explicitly asks for a different training window

### Canonical train command

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

What an agent should edit:
- Change `--training-data` only if the user wants to train from a different existing parquet
- Change `--profile` only if the user explicitly wants `smoke`, `fast`, `full`, `flat-fast`, or `flat-full`
- Change `--start` and `--end` only if the user wants to filter the existing parquet to a different season range before training
- Change `--holdout` only if the user wants a different holdout season
- Change `--XGBWork` only to adjust XGBoost worker count
- Change `--OptunaWork` only to adjust Optuna parallelism
- Change `--Iterations` and `--Folds` only when the user wants a different search budget or validation geometry
- Change `--feature-selection-mode` only when the user is explicitly comparing selector regimes
- Change `--blend-mode` only when the user is explicitly comparing XGBoost-only vs blend behavior
- Change `--cv-aggregation-mode` only when the user is explicitly comparing validation geometry
- Change `--forced-delta-count` only when the user is explicitly testing delta retention in the flat selector

What an agent should not casually change:
- Do not add rebuild behavior to the train command
- Do not switch to old scripts like `archive/scripts/manual_run_count/train_single_model.py` or `archive/scripts/manual_run_count/rebuild_and_train_single_model.py`
- Do not add extra trainer flags unless the user is explicitly asking for a modeling change

Preferred behavior for future agents:
- Start from one of the two canonical commands above
- Make the smallest possible edit to satisfy the user’s request
- Keep PowerShell backtick formatting when giving multi-line commands
- If the user only wants a fresh parquet, give only the build command
- If the user wants to train on an existing parquet, give only the train command

### Research-lane command source

For Stage 3 distribution, Stage 4 MCMC, and walk-forward evidence commands, use:

- `docs/runbooks/RUN_COUNT_RESEARCH_WORKFLOW.md`

Keep the same PowerShell backtick formatting there. Current manual worker settings to preserve on this machine are:

- `--FeatureWorker 10`
- `--XGBWork 4`
- `--OptunaWork 3`

Do not invent worker flags for scripts that do not expose them.

### Current best comparable manual-training baseline

As of the current repaired `ParquetDefault.parquet` workflow:

- manual training should be treated as **XGBoost-only** unless a new blend experiment proves otherwise
- `flat` selector currently beats `grouped` on the repaired parquet
- `recent_weighted` CV has not shown a holdout benefit yet
- more search budget alone has not improved the result on the current selector surface

The best current comparable manual baseline is:

```powershell
.\.venv\Scripts\python.exe `
  scripts\train_run_count.py `
  --profile flat-fast `
  --experiment 2026-away-flat-xgbonly-fast-120x3-cc10 `
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

### Delta-forcing experiment path

The trainer now supports a manual delta-retention experiment path through:

- `--forced-delta-count <N>`

Important constraints:

- currently supported only with `--feature-selection-mode flat`
- this reserves up to `N` top-ranked delta features inside the flat top-`80` set
- use this only for explicit delta-retention tests, not as the default baseline

Canonical first delta-forcing experiment:

```powershell
.\.venv\Scripts\python.exe `
  scripts\train_run_count.py `
  --profile flat-fast `
  --experiment 2026-away-flat-xgbonly-forceddelta12-fast-120x3-cc10 `
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
  --cv-aggregation-mode mean `
  --forced-delta-count 12
```

### Experiment stacking for sequential runs

When the user wants to run multiple manual experiments back-to-back, prefer a single PowerShell loop that:

- defines one shared common argument block
- defines a small list of per-run overrides like experiment name and forced-delta count
- runs each experiment sequentially
- stops immediately on the first failure

This is the preferred manual batching pattern for the canonical train script. Do not invent a new Python orchestrator for this unless the user explicitly asks for one.

Canonical example:

```powershell
$common = @(
  "scripts\train_run_count.py",
  "--profile", "flat-fast",
  "--training-data", "data\training\ParquetDefault.parquet",
  "--start", "2018",
  "--end", "2025",
  "--holdout", "2025",
  "--XGBWork", "4",
  "--OptunaWork", "3",
  "--Iterations", "120",
  "--Folds", "3",
  "--feature-selection-mode", "flat",
  "--blend-mode", "xgb_only",
  "--cv-aggregation-mode", "mean"
)

$runs = @(
  @{ Name = "2026-away-flat-xgbonly-forceddelta8-fast-120x3-cc10";  ForcedDelta = "8"  }
  @{ Name = "2026-away-flat-xgbonly-forceddelta10-fast-120x3-cc10"; ForcedDelta = "10" }
  @{ Name = "2026-away-flat-xgbonly-forceddelta14-fast-120x3-cc10"; ForcedDelta = "14" }
)

foreach ($run in $runs) {
  Write-Host ""
  Write-Host "Running $($run.Name) ..." -ForegroundColor Green

  & .\.venv\Scripts\python.exe `
    $common `
    --experiment $run.Name `
    --forced-delta-count $run.ForcedDelta

  if ($LASTEXITCODE -ne 0) {
    throw "Run failed: $($run.Name)"
  }
}
```

How future agents should use this:

- change only the values inside `$runs` for a small batch comparison
- keep the shared block identical across all runs unless the user explicitly wants a second axis changed
- keep batch comparisons apples-to-apples
- do not mix selector mode, CV mode, blend mode, season range, and search budget changes all in one batch unless the user explicitly wants a broad sweep

### Important command-format preference

When sending commands to the user, default to **vertically stacked PowerShell commands with backtick continuation**. Do **not** send single-line commands unless the user explicitly asks for a one-line version. The user's terminal/paste flow may auto-break long single-line commands, so the multiline backtick form is the canonical format for this repo handoff.

Example:

```powershell
.venv\Scripts\python.exe `
  scripts\train_single_model.py `
  --profile fast
```

Do not debate this preference. If the user asks for commands, send them in this stacked backtick format by default.

### Practical shell notes

- Prefer Windows-style paths in commands shown to the user
- Use `.venv\Scripts\python.exe` for Python commands when possible
- The user often runs long training jobs manually and then asks the agent to inspect outputs afterward

## Typical Modeling Workflow

1. Inspect `docs/research/MODEL_COMPARISON_TRACKER.md`
2. Inspect newest experiment artifact in `data/models/`
3. Compare holdout metrics against recent comparable runs
4. Decide whether the issue is:
   - search variance
   - trainer/CV behavior
   - feature generation
   - feature selection
   - data quality / dead features
5. If code changes are made, validate with:
   - `scripts/validate_modeling.py --profile fast`
   - then broader validation if needed

## Useful Commands

### Dry-run manual run-count profile

```powershell
.venv\Scripts\python.exe `
  scripts\train_run_count.py `
  --profile fast `
  --dry-run
```

### Train manual run-count model

```powershell
.venv\Scripts\python.exe `
  scripts\train_run_count.py `
  --profile fast `
  --training-data data\training\training_data_2018_2025.parquet
```

### Train manual run-count model with explicit experiment name

```powershell
.venv\Scripts\python.exe `
  scripts\train_run_count.py `
  --profile full `
  --experiment "custom-experiment-name"
```

### Build parquet only

```powershell
.venv\Scripts\python.exe `
  scripts\build_parquet.py `
  --training-data data\training\ParquetDefault.parquet `
  --start 2018 `
  --end 2025 `
  --FeatureWorker 10
```

### Fast validators

```powershell
.venv\Scripts\python.exe `
  scripts\validate_modeling.py `
  --profile fast
```

### Full validators

```powershell
.venv\Scripts\python.exe `
  scripts\validate_modeling.py `
  --profile full
```

## What A New Orchestrator Should Pay Attention To

### 1. Do not assume more compute is better

In this repo, bigger Optuna budgets and more folds have repeatedly failed to guarantee better holdout performance. More search only helps if the feature representation and CV surface are aligned.

### 2. Distinguish discovery from confirmation

The user has been using lower-fold runs to find promising regions, then larger runs to validate them. Treat these as different purposes:

- fewer folds + enough trials = search/discovery
- more folds = confirmation/stability check

### 3. Watch for feature-selection bottlenecks

A recurring issue is that engineered features may exist in `data_builder.py` but never survive the selector into the final model. Always inspect the emitted metadata:

- `feature_columns`
- `selected_features_by_bucket`
- `omitted_top_features_by_bucket`
- `feature_importance_rankings`
- `forced_delta_features`
- `feature_health_diagnostics`

Important recent lesson:

- the repaired parquet now has real weather, near-complete umpire coverage, and healthy delta columns
- even so, the default selectors still tended to choose **zero delta features**
- that means the current bottleneck is no longer missing data; it is the effective feature mix reaching the model

### 4. Check for dead or default-heavy features

When a run underperforms, verify whether supposedly useful features are:

- missing
- default-filled too often
- near-constant
- filtered out before training

### 5. Keep comparisons apples-to-apples

Compare runs only when the following are similar enough:

- target
- training parquet/version
- search space
- folds
- trial count
- feature-selection regime
- forced-delta setting
- objective / CV metric

### 6. Local-only parquet builds are now trustworthy

Recent fixes corrected two major manual-build issues:

- local-only weather lookups were previously being pointed at a temporary build DB instead of `data/mlb.db`
- umpire matching previously used UTC dates instead of home-team local baseball dates and under-matched same-day rematches

Current expectation for a healthy manual local-only build:

- weather factors should have real variation, not be flat/default
- umpire coverage should be roughly `16998 / 17006` rows on the current historical window
- source coverage is broad, but the artifact itself should still be checked if a run looks suspicious

## Guardrails For Future Agents

- Read before editing; match existing code style
- Do not treat the tracker as perfect ground truth
- Do not overwrite unrelated user changes
- If you change code, run validators before summarizing
- Be careful with git because the repo may already contain unrelated modified files
- Long-running experiments are usually run by the user, not by the agent

## How To Resume Quickly In A New Chat

Suggested first steps for a new agent:

1. Read this file
2. Read `docs/research/MODEL_COMPARISON_TRACKER.md`
3. Inspect `git status`
4. Inspect the newest folder under `data/models/`
5. Read the newest `*.metadata.json`
6. Summarize:
   - latest result
   - best known comparable result
   - likely current bottleneck
   - recommended next experiment or code change

## Paste-Ready Resume Prompt

```text
I am resuming work in the MLBPrediction2026 repo as the orchestration agent.

Start by reading:
1. docs/ORCHESTRATOR_HANDOFF.md
2. docs/research/MODEL_COMPARISON_TRACKER.md

Then inspect:
- git status
- the newest experiment directory under data/models/
- the newest *.metadata.json inside that directory

Your job:
- figure out the latest modeling state from artifacts, not assumptions
- compare the latest run against the best comparable prior runs
- identify the current bottleneck
- recommend the highest-leverage next step

Important user preferences:
- commands should be given for Windows PowerShell
- when formatting multi-line commands, use backticks for line continuation
- do not argue about that formatting preference
```
