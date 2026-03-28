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
  - Reserved for Stage 2 distribution scoring outputs such as CRPS, log score, calibration, and interval coverage.

- `dual_view/`
  - Stage 5 side-by-side control vs research-lane comparisons.
  - Current files:
    - `current_dual_view.json`
    - `current_dual_view.csv`
    - `current_dual_view.md`

- `mcmc/`
  - Reserved for Stage 4 sequencing and simulation outputs.

## Verified Stage 1 State

- Indexed away-run metadata artifacts: `39`
- Selected control artifact:
  - `data/models/2026-away-flat-xgbonly-forceddelta8-fast-120x3-cc10/full_game_away_runs_model_20260328T012328Z_cc10c0f6.metadata.json`
- Metadata inconsistencies currently flagged in the registry:
  - `6` AutoResearch artifacts where `forced_delta_feature_count` conflicts with `forced_delta_features`
