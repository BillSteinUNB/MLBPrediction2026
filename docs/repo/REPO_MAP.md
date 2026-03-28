# Repo Map

This map is for fast orientation. It is not a cleanup plan.

## Top-Level Folders

- `AutoResearch/`
  - Separate automation system and its own docs, logs, tests, and launcher flow.

- `archive/`
  - Archived material not needed for the active Stage 1 away-run registry work.

- `config/`
  - Project configuration files.

- `dashboard/`
  - Dashboard application assets and related code.

- `data/`
  - Training data, model artifacts, and generated reports.

- `docs/`
  - Human navigation docs, handoffs, roadmap, repo map, and runbooks.

- `logs/`
  - Runtime logs.

- `OddsScraper/`
  - Separate scraping subsystem.

- `scripts/`
  - Canonical command entrypoints, including parquet build, model train, validation, and reporting scripts.

- `src/`
  - Main Python source tree.

- `tests/`
  - Repo tests.
  - Current away-run research tests live primarily under `tests/model/` and `tests/ops/`.
  - Legacy broad-suite tests still exist at the top level of `tests/` and are safe to leave in place during Stage 5.

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
- `MODEL_COMPARISON_TRACKER.md`
- `data/reports/run_count/dual_view/current_dual_view.md`
- `data/reports/run_count/dual_view/current_dual_view.json`
- `src/model/run_count_trainer.py`
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
- `logs/`
- `nul`
- `archive/repo_root_scratch/`
- `archive/research/ResearchInformation.micro_saas.md`

## Repo Root Clarifier

- `ResearchInformation.md`
  - now acts as a root-level pointer to the MLB research docs and the Stage 5 dual-view outputs.
  - the old unrelated Micro-SaaS material was archived to `archive/research/ResearchInformation.micro_saas.md`.

## Current Reporting Anchors

- Registry outputs:
  - `data/reports/run_count/registry/full_game_away_runs_registry.json`
  - `data/reports/run_count/registry/full_game_away_runs_registry.csv`
  - `data/reports/run_count/registry/current_control.json`

- Stage 5 dual view:
  - `data/reports/run_count/dual_view/current_dual_view.json`
  - `data/reports/run_count/dual_view/current_dual_view.csv`
  - `data/reports/run_count/dual_view/current_dual_view.md`

- Current verified control:
  - `data/models/2026-away-flat-xgbonly-forceddelta8-fast-120x3-cc10/full_game_away_runs_model_20260328T012328Z_cc10c0f6.metadata.json`

- Current verified second opinion:
  - `data/models/2026-away-dist-zanb-v1-controlmu-flat-3f-cc10/full_game_away_runs_distribution_model_20260328T190153Z_cc10c0f6.metadata.json`

- Current exploratory MCMC lane:
  - `data/models/2026-away-mcmc-markov-v1-controlmu-s1000-cc10/full_game_away_runs_mcmc_model_20260328T192349Z_cc10c0f6.metadata.json`
