# Docs Index

Start here from the repo root when the work touches MLB modeling, research lanes, or pipeline debugging.

## Start Here Paths

- New orchestrator:
  - Read `docs/ORCHESTRATOR_HANDOFF.md`
  - Then read `MODEL_COMPARISON_TRACKER.md`
  - Then read `data/reports/run_count/dual_view/current_dual_view.md`
  - Then inspect `data/reports/run_count/registry/current_control.json`

- Modeling agent:
  - Read `docs/research/MLB_RUN_MODEL_RESEARCH_ASSUMPTIONS.md`
  - Then read `docs/runbooks/RUN_COUNT_RESEARCH_WORKFLOW.md`
  - Then read `data/reports/run_count/dual_view/current_dual_view.md`
  - Then inspect `src/model/run_count_trainer.py` and `src/model/data_builder.py`

- Pipeline or debugging agent:
  - Read `docs/repo/REPO_MAP.md`
  - Then inspect `src/pipeline/daily.py`
  - Then inspect `scripts/validate_modeling.py`

## Core Modeling Docs

- Control modeling handoff: `docs/ORCHESTRATOR_HANDOFF.md`
- Run history and comparable results: `MODEL_COMPARISON_TRACKER.md`
- Current dual-view summary: `data/reports/run_count/dual_view/current_dual_view.md`
- Current dual-view JSON: `data/reports/run_count/dual_view/current_dual_view.json`
- Current control registry output: `data/reports/run_count/registry/current_control.json`
- Full away-run registry: `data/reports/run_count/registry/full_game_away_runs_registry.json`

## Research Docs

- Stage roadmap: `docs/roadmaps/AWAY_RUN_RESEARCH_5_STAGE_PLAN.md`
- MLB research assumptions: `docs/research/MLB_RUN_MODEL_RESEARCH_ASSUMPTIONS.md`

## Repo Navigation

- Repo map: `docs/repo/REPO_MAP.md`
- Runbooks: `docs/runbooks/RUN_COUNT_RESEARCH_WORKFLOW.md`
- Root research pointer: `ResearchInformation.md`

## AutoResearch Docs

- Overview: `AutoResearch/README.md`
- Handoff: `AutoResearch/AUTORESEARCH_HANDOFF.md`
- Program notes: `AutoResearch/program.md`

## Verified Stage 5 State

- Away-run metadata artifacts indexed: `39`
- Selected control artifact:
  - `data/models/2026-away-flat-xgbonly-forceddelta8-fast-120x3-cc10/full_game_away_runs_model_20260328T012328Z_cc10c0f6.metadata.json`
- Best current research lane:
  - `data/models/2026-away-dist-zanb-v1-controlmu-flat-3f-cc10/full_game_away_runs_distribution_model_20260328T190153Z_cc10c0f6.metadata.json`
- Exploratory MCMC lane:
  - `data/models/2026-away-mcmc-markov-v1-controlmu-s1000-cc10/full_game_away_runs_mcmc_model_20260328T192349Z_cc10c0f6.metadata.json`
- Stage 3 status:
  - promoted as the current second-opinion lane
  - not production-promotable yet because no walk-forward betting report is attached
- Stage 4 status:
  - remains exploratory
