# AutoResearch Handoff

Use this document to onboard a new AI chat that is specifically helping with the `AutoResearch/` system.

## Purpose

`AutoResearch/` is an overnight research orchestrator for the MLB run-count model.

Its job is to:

- run repeated constrained experiments
- log exactly what happened
- preserve session context across runs
- record suspicious failures and retest them
- generate morning-review artifacts

This is not the main modeling pipeline. It is the automation layer around it.

## Current Overnight Goal

The current narrow overnight objective is:

- hold the repaired manual setup fixed
- search the local forced-delta region
- avoid stale axes

### Frozen for the night

- parquet: `data/training/ParquetDefault.parquet`
- target: `final_away_score`
- model: `full_game_away_runs_model`
- selector: flat (`pearson`)
- blend: `xgb_only`
- CV aggregation: `mean`
- folds: `3`
- holdout season: `2025`
- no data/build code changes
- no 500x5 promotion

### Primary search axis

- `forced_delta_count`

### Secondary search axis

- `max_features`

## Core Files

### Orchestration

- `AutoResearch/launcher.py`
  - startup prompts
  - git checkpoint + branch creation
  - session loop
  - live terminal progress table

- `AutoResearch/agent.py`
  - planner orchestration
  - experiment DB management
  - notes, suspicious issue retests, summaries
  - planner fallback logic

- `AutoResearch/train.py`
  - constrained train-time config surface
  - applies overrides to the real run-count trainer
  - executes the actual training run

- `AutoResearch/program.md`
  - planner instructions for the current overnight mission

## Persistent State

- `AutoResearch/experiments.db`
  - sessions
  - experiments
  - notes
  - issue validations

- `AutoResearch/reports/nightly_log.md`
- `AutoResearch/reports/notes.jsonl`
- `AutoResearch/reports/session_history.jsonl`
- `AutoResearch/reports/sessions/`

## External Artifacts

Model artifacts may still be written outside `AutoResearch/`, typically under:

- `data/models/`

AutoResearch records the exact output directory and file names in its own reports so morning review can trace what it created.

## Current Design Constraints

- keep AutoResearch’s own logs/reports/state inside `AutoResearch/`
- branch workflow is part of startup
- startup commits and pushes current repo state to `main`
- startup then creates a branch like `AutoResearch-YYYY-MM-DD`
- suspicious failures are retested on the next loop
- morning-review prompt files are generated automatically

## Important Current Risks

### 1. LLM planner permission / timeout failures

`droid exec` may:

- time out
- fail to return valid JSON
- fail due to permissions

AutoResearch currently falls back to heuristic planning when this happens.

### 2. Wrapper drift in `AutoResearch/train.py`

The most fragile part of the system is the monkey-patching in `apply_training_overrides()`.

When the underlying trainer changes selector signatures, AutoResearch wrappers may break.

Past failures already included:

- missing `forced_delta_count` forwarding
- recursion through the patched flat selector
- brittle AGENT_CONFIG replacement behavior

If AutoResearch suddenly starts failing across all runs, inspect `train.py` first.

### 3. Adaptive holdout overfitting

The loop repeatedly compares against the same holdout season. Treat overnight winners as research candidates, not permanent truth.

## How To Debug Fast

If AutoResearch fails:

1. Inspect latest logs in:
   - `AutoResearch/logs/autoresearch/`
2. Inspect latest notes and nightly log:
   - `AutoResearch/reports/nightly_log.md`
   - `AutoResearch/reports/notes.jsonl`
3. Inspect recent rows in:
   - `AutoResearch/experiments.db`
4. Check whether failure is:
   - planner failure
   - training-wrapper failure
   - artifact parsing failure
   - session-state / retry logic failure

## Recommended First Files To Read

1. `AutoResearch/program.md`
2. `AutoResearch/train.py`
3. `AutoResearch/agent.py`
4. `AutoResearch/launcher.py`
5. latest `AutoResearch/reports/sessions/*.md`
6. latest `AutoResearch/reports/nightly_log.md`

## Validation Commands

### AutoResearch tests

```powershell
.\.venv\Scripts\python.exe `
  -m pytest `
  AutoResearch\tests\test_autoresearch_system.py
```

### Modeling validators

```powershell
.\.venv\Scripts\python.exe `
  scripts\validate_modeling.py `
  --profile fast `
  --skip-lint
```

## What A Future Agent Should Preserve

- do not casually widen the overnight scope
- do not switch parquet/holdout/target mid-session
- do not add unrelated repo changes into AutoResearch logic
- keep logs readable for morning diagnosis
- prefer explicit notes when a failure mode is discovered

## Morning Review Workflow

At the end of a session, AutoResearch should produce:

- session summary JSON
- session summary Markdown
- morning-review prompt file

These are the main handoff artifacts for the next AI chat or morning analysis.
