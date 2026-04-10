# Run Count Reports

This folder is the reporting anchor for away-run research outputs.

## Folder Purposes

- `registry/`
  - Machine-readable inventory of existing away-run model artifacts.
  - Stage 1 currently writes:
    - `full_game_away_runs_registry.csv`
    - `full_game_away_runs_registry.json`
    - `current_control.json`

- `distribution_eval/`
  - Stage 2 and Stage 3 distribution scoring outputs such as CRPS, log score, calibration, interval coverage, and research-lane feature metadata.

- `dual_view/`
  - Stage 5 side-by-side control vs research-lane comparisons.
  - Current files:
    - `current_dual_view.json`
    - `current_dual_view.csv`
    - `current_dual_view.md`

- `mcmc/`
  - Stage 4 sequencing and simulation outputs, including holdout PMF summaries and lane-comparison reports.

- `walk_forward/`
  - Machine-readable holdout walk-forward evidence for research lanes.
  - Current files include:
    - `*.stage3_walk_forward.json`
    - `*.stage3_walk_forward.csv`
    - `*.mcmc_walk_forward.json`
    - `*.mcmc_walk_forward.csv`

## Verified Stage 1 State

- Indexed away-run metadata artifacts: `39`
- Selected control artifact:
  - `data/models/2026-away-flat-xgbonly-forceddelta8-fast-120x3-cc10/full_game_away_runs_model_20260328T012328Z_cc10c0f6.metadata.json`
- Metadata inconsistencies currently flagged in the registry:
  - `6` AutoResearch artifacts where `forced_delta_feature_count` conflicts with `forced_delta_features`

## Current Research State

- Best distribution lane:
  - `data/models/2026-away-dist-zanb-v2-adv-marketfallback-cc10/full_game_away_runs_distribution_model_20260328T203522Z_cc10c0f6.metadata.json`
- Exploratory MCMC lane:
  - `data/models/2026-away-mcmc-markov-v2-adv-marketfallback-cc10/full_game_away_runs_mcmc_model_20260328T203600Z_cc10c0f6.metadata.json`
- Market-prior coverage blocker:
  - current repo-local historical odds do not contain matching away-run market rows for the training game ids, so market-prior features run in explicit fallback mode and record zero coverage in metadata and walk-forward reports
