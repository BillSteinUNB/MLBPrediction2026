# TODO

This file tracks the current MLB model iteration backlog after the first full `2024` holdout run.

## Next Up

1. Wire Retrosheet historical lineups and plate-umpire data into feature generation.
2. Use Chadwick register mappings to bridge Retrosheet IDs into MLBAM/FanGraphs joins.
3. Compare the new `2024` metrics against the current baseline:
   - `f5_ml` base Brier: `0.244783`
   - `f5_ml` base ROC AUC: `0.585442`
   - `f5_rl` base Brier: `0.215245`
   - `f5_rl` base ROC AUC: `0.586068`
4. Check whether formerly constant offense and defensive-efficiency columns now vary in the rebuilt parquet.

## High Priority

- Fix real historical offense features.
  Status: implemented
  Notes: team logs now use `Batting Stats_*` columns, and the parser no longer falls back to defaults.

- Keep dead constant features out of training.
  Status: implemented
  Notes: training now drops numeric columns with only one value.

- Add free-source ingestion for historical lineups/umpires and cross-source IDs.
  Status: implemented
  Notes: Retrosheet and Chadwick ingestion clients now cache public source data under `data/raw`.

- Re-evaluate stacking and calibration after the offense fix.
  Notes: stacking is nearly neutral; calibration hurt the `2024` holdout.

## Deferred But Important

- Historical weather features
  Reason deferred: current backtests use forecast fallbacks for past games, so weather is mostly neutral and not yet trustworthy.
  Later work:
  - add a real historical weather source
  - set `weather_data_missing` honestly for fallback/default weather
  - decide whether weather should be excluded until real history exists

- Framing features
  Reason deferred: current framing source is leaderboard-style and does not provide the team/date history needed by the feature builder.

- Historical lineup features
  Reason deferred: backtests currently default to empty lineups, so lineup features are not yet adding real information.

## Experiments To Run Later

- Start-year comparison:
  - `2018+`
  - `2021+`
  - `2023+`

- Pure XGBoost vs stacking vs calibration after the offense fix.

- Feature drift / inflection analysis by season and month for:
  - offense metrics
  - starter metrics
  - bullpen usage
  - response variables

## Nice To Have

- Add a dedicated "build dataset only" CLI for faster iteration.
- Add runtime timing logs by feature module.
- Add prediction export for holdout runs so game-by-game misses are easier to inspect.
