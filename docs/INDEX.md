# Docs Index

Start here when the work touches MLB modeling, research lanes, pipeline debugging, or repo navigation.

## Start Here Paths

- New orchestrator:
  - Read `docs/ORCHESTRATOR_HANDOFF.md`
  - Then read `docs/research/MODEL_COMPARISON_TRACKER.md`
  - Then read `data/reports/run_count/dual_view/current_dual_view.md`
  - Then inspect `data/reports/run_count/registry/current_control.json`
  - Then inspect `data/reports/run_count/walk_forward/`

- Modeling agent:
  - Read `docs/research/MLB_RUN_MODEL_RESEARCH_ASSUMPTIONS.md`
  - Then read `docs/runbooks/RUN_COUNT_RESEARCH_WORKFLOW.md`
  - Then read `data/reports/run_count/dual_view/current_dual_view.md`
  - Then inspect `src/model/run_distribution_trainer.py`, `src/model/mcmc_feature_builder.py`, and `src/model/data_builder.py`

- Pipeline or debugging agent:
  - Read `docs/repo/REPO_MAP.md`
  - Then inspect `src/pipeline/daily.py`
  - Then inspect `scripts/validate_modeling.py`

## Core Modeling Docs

- Control modeling handoff: `docs/ORCHESTRATOR_HANDOFF.md`
- Run history and comparable results: `docs/research/MODEL_COMPARISON_TRACKER.md`
- Current dual-view summary: `data/reports/run_count/dual_view/current_dual_view.md`
- Current dual-view JSON: `data/reports/run_count/dual_view/current_dual_view.json`
- Current walk-forward reports: `data/reports/run_count/walk_forward/`
- Current control registry output: `data/reports/run_count/registry/current_control.json`
- Full away-run registry: `data/reports/run_count/registry/full_game_away_runs_registry.json`

## Research Docs

- Stage roadmap: `docs/roadmaps/AWAY_RUN_RESEARCH_5_STAGE_PLAN.md`
- MLB research assumptions: `docs/research/MLB_RUN_MODEL_RESEARCH_ASSUMPTIONS.md`

## Repo Navigation

- Repo map: `docs/repo/REPO_MAP.md`
- Repo hygiene: `docs/repo/CONTRIBUTING_REPO_HYGIENE.md`
- Runbooks: `docs/runbooks/RUN_COUNT_RESEARCH_WORKFLOW.md`
- Research index: `docs/research/ResearchInformation.md`
- Handoffs: `docs/handoffs/`
- Archived root temp folders: `archive/repo_root_scratch/2026-03-28_stage5_cleanup/`

## Housekeeping

- `docs/` is curated for active guidance.
- `archive/` is frozen for preserved history.
- The repo root should contain only long-lived entrypoints and core project files.
- `docs/repo/REPO_MAP.md` is the authoritative paths document and should be updated with every directory move.

## AutoResearch Docs

- Overview: `AutoResearch/README.md`
- Handoff: `AutoResearch/AUTORESEARCH_HANDOFF.md`
- Program notes: `AutoResearch/program.md`

## Verified Stage 5 State

- Away-run metadata artifacts indexed: `39`
- Selected control artifact:
  - `data/models/2026-away-flat-xgbonly-forceddelta8-fast-120x3-cc10/full_game_away_runs_model_20260328T012328Z_cc10c0f6.metadata.json`
- Best current research lane:
  - `data/models/2026-away-dist-zanb-v2-adv-marketfallback-cc10/full_game_away_runs_distribution_model_20260328T203522Z_cc10c0f6.metadata.json`
- Exploratory MCMC lane:
  - `data/models/2026-away-mcmc-markov-v2-adv-marketfallback-cc10/full_game_away_runs_mcmc_model_20260328T203600Z_cc10c0f6.metadata.json`
- Stage 3 status:
  - promoted as the current second-opinion lane
  - has walk-forward reports, but is not production-promotable because repo-local betting evidence is unavailable
- Stage 4 status:
  - remains exploratory
