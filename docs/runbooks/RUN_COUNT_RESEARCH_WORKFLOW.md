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
  - Current best distribution lane is `2026-away-dist-zanb-v2-adv-marketfallback-cc10`.
  - Stage 5 promotes this lane as the current second opinion, not as a control replacement.
  - Market-prior features are isolated in lane metadata and currently run in fallback mode because matching historical away-run market data is unavailable in repo-local history.

- MCMC lane:
  - Sequencing-aware simulation lane using Markov or Monte Carlo logic.
  - Current exploratory lane is `2026-away-mcmc-markov-v2-adv-marketfallback-cc10`.
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

## Distribution-Lane Train Command

```powershell
.\.venv\Scripts\python.exe `
  scripts\train_run_distribution.py `
  --experiment 2026-away-dist-zanb-v2-adv-marketfallback-cc10 `
  --training-data data\training\ParquetDefault.parquet `
  --start 2018 `
  --end 2025 `
  --holdout 2025 `
  --Folds 3 `
  --feature-selection-mode flat `
  --forced-delta-count 0 `
  --XGBWork 4 `
  --current-control data\reports\run_count\registry\current_control.json `
  --distribution-report-dir data\reports\run_count\distribution_eval `
  --enable-market-priors `
  --historical-odds-db data\mlb.db `
  --research-lane-name distribution_market_priors_adv_v1
```

## MCMC-Lane Command

```powershell
.\.venv\Scripts\python.exe `
  scripts\run_mcmc_distribution.py `
  --experiment 2026-away-mcmc-markov-v2-adv-marketfallback-cc10 `
  --training-data data\training\ParquetDefault.parquet `
  --start 2018 `
  --end 2025 `
  --holdout 2025 `
  --current-control data\reports\run_count\registry\current_control.json `
  --stage3-report-json data\reports\run_count\distribution_eval\20260328T203522Z_cc10c0f6.distribution_eval.json `
  --distribution-report-dir data\reports\run_count\distribution_eval `
  --mcmc-report-dir data\reports\run_count\mcmc `
  --simulations 1000 `
  --starter-innings 5 `
  --seed 20260328 `
  --enable-market-priors `
  --historical-odds-db data\mlb.db `
  --research-lane-name mcmc_market_priors_adv_v1
```

## Walk-Forward Evidence Command

```powershell
.\.venv\Scripts\python.exe `
  scripts\evaluate_run_count_walk_forward.py `
  --training-data data\training\ParquetDefault.parquet `
  --stage3-metadata data\models\2026-away-dist-zanb-v2-adv-marketfallback-cc10\full_game_away_runs_distribution_model_20260328T203522Z_cc10c0f6.metadata.json `
  --mcmc-metadata data\models\2026-away-mcmc-markov-v2-adv-marketfallback-cc10\full_game_away_runs_mcmc_model_20260328T203600Z_cc10c0f6.metadata.json `
  --holdout 2025 `
  --output-dir data\reports\run_count\walk_forward `
  --enable-market-priors `
  --historical-odds-db data\mlb.db
```

## Stage 5 Dual-View Command

```powershell
.\.venv\Scripts\python.exe `
  scripts\report_run_count_dual_view.py `
  --current-control data\reports\run_count\registry\current_control.json `
  --distribution-report-dir data\reports\run_count\distribution_eval `
  --stage3-report-json data\reports\run_count\distribution_eval\20260328T203522Z_cc10c0f6.distribution_eval.json `
  --stage3-vs-control-json data\reports\run_count\distribution_eval\20260328T203522Z_cc10c0f6.vs_control.json `
  --mcmc-report-dir data\reports\run_count\mcmc `
  --mcmc-report-json data\reports\run_count\mcmc\20260328T203600Z_cc10c0f6.mcmc_eval.json `
  --mcmc-vs-control-json data\reports\run_count\mcmc\20260328T203600Z_cc10c0f6.vs_control.json `
  --mcmc-vs-stage3-json data\reports\run_count\mcmc\20260328T203600Z_cc10c0f6.vs_stage3.json `
  --stage3-walk-forward-report-json data\reports\run_count\walk_forward\20260328T203522Z_cc10c0f6.stage3_walk_forward.json `
  --mcmc-walk-forward-report-json data\reports\run_count\walk_forward\20260328T203600Z_cc10c0f6.mcmc_walk_forward.json `
  --output-dir data\reports\run_count\dual_view
```

## Current Stage 5 Outputs

- Dual-view markdown:
  - `data/reports/run_count/dual_view/current_dual_view.md`

- Dual-view JSON:
  - `data/reports/run_count/dual_view/current_dual_view.json`

- Dual-view CSV:
  - `data/reports/run_count/dual_view/current_dual_view.csv`

- Walk-forward reports:
  - `data/reports/run_count/walk_forward/20260328T203522Z_cc10c0f6.stage3_walk_forward.json`
  - `data/reports/run_count/walk_forward/20260328T203600Z_cc10c0f6.mcmc_walk_forward.json`

## Current Promotion State

- Control lane:
  - stays intact as the stable benchmark.

- Stage 3 distribution lane:
  - best current research lane.
  - promoted as the second-opinion lane.
  - not production-promotable because betting evidence is explicitly unavailable in repo-local history.

- Stage 4 MCMC lane:
  - remains exploratory.

## Practical Rules

- Use PowerShell backtick continuation for multiline commands.
- Keep comparisons apples-to-apples:
  - same parquet
  - same season window
  - same holdout season
  - same target
- Keep market-prior work isolated to research lanes and inspect `research_feature_metadata` before comparing runs.
- Read the walk-forward JSON before making any promotion claim.
- Do not swap the workflow to archived wrappers or ad hoc inline scripts.
- Do not start Stage 2+ work from this runbook alone. Check the roadmap first.

## Current Partial-History Override

Use this season window while the historical odds backfill is still incomplete:

- exclude `2020`
- use `2021-2025` for historical-odds-dependent reruns
- keep the same manual worker counts as the control lane
- Stage 4 MCMC currently has no worker flag; do not add one

### Stage 3 Distribution Command For Current Archive

```powershell
.\.venv\Scripts\python.exe `
  scripts\train_run_distribution.py `
  --training-data data\training\ParquetDefault.parquet `
  --start 2021 `
  --end 2025 `
  --holdout 2025 `
  --Folds 3 `
  --feature-selection-mode flat `
  --forced-delta-count 0 `
  --XGBWork 4 `
  --current-control data\reports\run_count\registry\current_control.json `
  --distribution-report-dir data\reports\run_count\distribution_eval `
  --enable-market-priors `
  --historical-odds-db OddsScraper\data\mlb_odds.db `
  --research-lane-name distribution_market_priors_adv_v1
```

### Stage 4 MCMC Command For Current Archive

```powershell
.\.venv\Scripts\python.exe `
  scripts\run_mcmc_distribution.py `
  --training-data data\training\ParquetDefault.parquet `
  --start 2021 `
  --end 2025 `
  --holdout 2025 `
  --current-control data\reports\run_count\registry\current_control.json `
  --distribution-report-dir data\reports\run_count\distribution_eval `
  --mcmc-report-dir data\reports\run_count\mcmc `
  --simulations 1000 `
  --starter-innings 5 `
  --seed 20260328 `
  --enable-market-priors `
  --historical-odds-db OddsScraper\data\mlb_odds.db `
  --research-lane-name mcmc_market_priors_adv_v1
```

### Walk-Forward Command For Current Archive

```powershell
.\.venv\Scripts\python.exe `
  scripts\evaluate_run_count_walk_forward.py `
  --training-data data\training\ParquetDefault.parquet `
  --stage3-metadata <stage3_metadata.json> `
  --mcmc-metadata <mcmc_metadata.json> `
  --holdout 2025 `
  --output-dir data\reports\run_count\walk_forward `
  --enable-market-priors `
  --historical-odds-db OddsScraper\data\mlb_odds.db
```
