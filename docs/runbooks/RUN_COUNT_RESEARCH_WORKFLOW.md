# Run Count Research Workflow

This is the canonical manual workflow for run-count research in this repo.

## Canonical Build Command

```powershell
.\.venv\Scripts\python.exe `
  scripts\build_parquet.py `
  --training-data data\training\ParquetDefault.parquet `
  --start 2018 `
  --end 2025 `
  --FeatureWorker 10
```

## Canonical Control-Lane Train Command

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

## Lane Definitions

- Control lane:
  - Stable mean-focused away-run model.
  - Current verified control is `2026-away-flat-xgbonly-forceddelta8-fast-120x3-cc10`.
  - Preserve this lane while research lanes are added.

- Distribution lane:
  - Probabilistic lane for PMFs, calibration, CRPS, and log-score evaluation.
  - Current best distribution lane is `2026-away-dist-zanb-v1-controlmu-flat-3f-cc10`.
  - Stage 5 promotes this lane as the current second opinion, not as a control replacement.

- MCMC lane:
  - Sequencing-aware simulation lane using Markov or Monte Carlo logic.
  - Current exploratory lane is `2026-away-mcmc-markov-v1-controlmu-s1000-cc10`.
  - Keep it separate from control and distribution training paths.

## Naming Convention

- Control mean lane:
  - `2026-away-*`

- Distribution lane:
  - `2026-away-dist-*`

- MCMC lane:
  - `2026-away-mcmc-*`

## Verified Stage 1 Registry Outputs

- Full registry:
  - `data/reports/run_count/registry/full_game_away_runs_registry.json`

- Current control selection:
  - `data/reports/run_count/registry/current_control.json`

## Stage 5 Dual-View Command

```powershell
.\.venv\Scripts\python.exe `
  scripts\report_run_count_dual_view.py `
  --current-control data\reports\run_count\registry\current_control.json `
  --distribution-report-dir data\reports\run_count\distribution_eval `
  --mcmc-report-dir data\reports\run_count\mcmc `
  --output-dir data\reports\run_count\dual_view
```

## Current Stage 5 Outputs

- Dual-view markdown:
  - `data/reports/run_count/dual_view/current_dual_view.md`

- Dual-view JSON:
  - `data/reports/run_count/dual_view/current_dual_view.json`

- Dual-view CSV:
  - `data/reports/run_count/dual_view/current_dual_view.csv`

## Current Promotion State

- Control lane:
  - stays intact as the stable benchmark.

- Stage 3 distribution lane:
  - best current research lane.
  - promoted as the second-opinion lane.
  - not production-promotable yet because no walk-forward betting report is attached.

- Stage 4 MCMC lane:
  - remains exploratory.

## Practical Rules

- Use PowerShell backtick continuation for multiline commands.
- Keep comparisons apples-to-apples:
  - same parquet
  - same season window
  - same holdout season
  - same target
- Do not swap the workflow to archived wrappers or ad hoc inline scripts.
- Do not start Stage 2+ work from this runbook alone. Check the roadmap first.
