# Repo Map

This map is for fast orientation and is the authoritative paths document for the repo.

If a long-lived directory moves, update this file in the same change.

## Top-Level Folders

- `AutoResearch/`
  - Separate automation system and its own docs, logs, tests, and launcher flow.

- `archive/`
  - Archived material not needed for the active away-run modeling workflows.
  - Stage 5 root-temp cleanup lives under `archive/repo_root_scratch/2026-03-28_stage5_cleanup/`.
  - Standard archive buckets are `archive/logs/`, `archive/experiments/`, `archive/subprojects/`, and `archive/repo_root_scratch/`.

- `config/`
  - Project configuration files.

- `dashboard/`
  - Dashboard application assets and related code.

- `data/`
  - Training data, model artifacts, and generated reports.

- `docs/`
  - Human navigation docs, handoffs, roadmap, repo map, and runbooks.

- `OddsScraper/`
  - Separate scraping subsystem.

- `scripts/`
  - Canonical command entrypoints, including parquet build, model train, validation, and reporting scripts.

- `src/`
  - Main Python source tree.

- `tests/`
  - Repo tests.
  - Canonical test buckets now live under `tests/model/`, `tests/ops/`, `tests/pipeline/`, `tests/features/`, and `tests/integration/`.
  - Dashboard suites now live under `tests/integration/dashboard/` and `tests/integration/dashboard_e2e/`.

## `src/` High-Level Map

- `src/model/`
  - Training, scoring, profiles, calibration, and model-side utilities.

- `src/ops/`
  - Reporting and operational support code.

- `src/pipeline/`
  - Daily pipeline orchestration and related production workflow code.

- `src/features/`
  - Feature generation families, including weather, defense, umpire, and shrinkage logic.

- `src/clients/`, `src/engine/`, `src/notifications/`, `src/dashboard/`, `src/models/`, `src/backtest/`
  - Supporting systems outside the Stage 1 away-run registry deliverable.

## High-Signal Files For Away-Run Research

- `docs/ORCHESTRATOR_HANDOFF.md`
- `docs/research/MODEL_COMPARISON_TRACKER.md`
- `data/reports/run_count/dual_view/current_dual_view.md`
- `data/reports/run_count/dual_view/current_dual_view.json`
- `data/reports/run_count/walk_forward/`
- `src/model/run_count_trainer.py`
- `src/model/run_distribution_trainer.py`
- `src/model/mcmc_feature_builder.py`
- `src/model/data_builder.py`
- `src/pipeline/daily.py`
- `src/model/score_pricing.py`

## Safe To Ignore For Most Agents

- `temp_backtest_*`
- `temp_strategy_*`
- `temp_walkforward_*`
- `.pytest*`
- `.ruff_cache/`
- `.uv-cache/`
- `.uv-python/`
- `nul`
- `.sisyphus/`
- `.factory/`
- `.meteostat-cache/`
- `archive/repo_root_scratch/`
- `archive/repo_root_scratch/2026-03-28_stage5_cleanup/`
- `archive/research/ResearchInformation.micro_saas.md`

## Research Navigation

- `docs/research/ResearchInformation.md`
  - acts as the research landing page for MLB modeling docs and the Stage 5 dual-view outputs.
  - the old unrelated Micro-SaaS material was archived to `archive/research/ResearchInformation.micro_saas.md`.

- `docs/repo/CONTRIBUTING_REPO_HYGIENE.md`
  - defines root, docs, archive, and data hygiene rules for keeping the repo readable on GitHub.

- `docs/handoffs/`
  - holds dated handoff material that used to live at the repo root.

- `archive/logs/`
  - holds preserved historical log bundles that were removed from the active top-level tree.

## Current Reporting Anchors

- Registry outputs:
  - `data/reports/run_count/registry/full_game_away_runs_registry.json`
  - `data/reports/run_count/registry/full_game_away_runs_registry.csv`
  - `data/reports/run_count/registry/current_control.json`

- Stage 5 dual view:
  - `data/reports/run_count/dual_view/current_dual_view.json`
  - `data/reports/run_count/dual_view/current_dual_view.csv`
  - `data/reports/run_count/dual_view/current_dual_view.md`

- Walk-forward research reports:
  - `data/reports/run_count/walk_forward/*.stage3_walk_forward.json`
  - `data/reports/run_count/walk_forward/*.mcmc_walk_forward.json`

- Current verified control:
  - `data/models/2026-away-flat-xgbonly-forceddelta8-fast-120x3-cc10/full_game_away_runs_model_20260328T012328Z_cc10c0f6.metadata.json`

- Current verified second opinion:
  - `data/models/2026-away-dist-zanb-v2-adv-marketfallback-cc10/full_game_away_runs_distribution_model_20260328T203522Z_cc10c0f6.metadata.json`

- Current exploratory MCMC lane:
  - `data/models/2026-away-mcmc-markov-v2-adv-marketfallback-cc10/full_game_away_runs_mcmc_model_20260328T203600Z_cc10c0f6.metadata.json`
