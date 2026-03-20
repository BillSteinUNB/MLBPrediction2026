## Workspace state notes

- During feature `t7-mlb-lineup-client`, the working tree contained unrelated local weather/test WIP.
- To keep the lineup commit clean, these repo-local stashes were created:
  - `worker-8eb1cd35 isolate unrelated weather WIP`
  - `worker-8eb1cd35 isolate unrelated test scaffolding`
  - `worker-8eb1cd35 isolate leftover weather test`
- Reapply with `git stash list` / `git stash pop` only if that unrelated weather work needs to be resumed.

- During feature `fix-t5-odds-api-commit-games`, additional repo-local stashes were created to isolate unrelated feature-engineering and local note WIP:
  - `worker-ea65a6a9 isolate unrelated feature-engineering WIP retry`
  - `worker-ea65a6a9 isolate reserved nul path`
  - `worker-ea65a6a9 isolate unrelated feature-engineering WIP`
- A stray untracked `nul` path required Git Bash removal (`rm -f`) after Windows git operations hit reserved-path handling; if `git status` ever shows `?? nul` again, inspect it from Bash under `/mnt/c/...` instead of standard Windows file APIs.

- During feature `fix-t8-weather-cache-ttl`, pre-existing local `src/model/data_builder.py` and `tests/test_data_builder.py` WIP surfaced while validating the weather cache fix.
- `tests/test_data_builder.py` broke full-suite pytest collection because `src.model.data_builder` is not part of the committed milestone yet.
- To run milestone validators cleanly without committing unrelated WIP, these repo-local stashes were created:
  - `worker-0cac70ef isolate unrelated data_builder test`
  - `worker-0cac70ef isolate unrelated data_builder WIP`

- During feature `fix-t7-lineup-date-scoping`, another repo-local stash was created to isolate unrelated training-data work before validation:
  - `worker-ba06611d isolate unrelated data_builder WIP`

- During feature `fix-t6-cache-invalidation`, another repo-local stash was created to isolate unrelated model/training work before validation:
  - `worker-4991f8d1 isolate unrelated model WIP`

- During feature `t9-offensive-features`, another repo-local stash was created to isolate unrelated engine, notifications, notes, and local sandbox WIP before validation/commit:
  - `worker-1259c3dc isolate unrelated engine and local WIP`

- During feature `t16-training-data-builder`, the worker built `data/training/training_data_2019_2025.parquet` plus raw leaderboard/cache artifacts to verify the historical dataset, then stashed those untracked data files to keep the working tree clean:
  - `worker-bceefdea isolate raw data artifacts`
- Pop that stash if a future worker wants the generated training parquet or cached raw-season snapshots without rebuilding.
- The training-data builder now defaults `team_logs_fetcher` to `fetch_team_game_logs()` and threads those dated team logs through the offense, defense, and bullpen feature modules when rebuilding historical datasets. The old schedule-only / season-snapshot-only fallback description is no longer accurate for the current builder path.
- The training-data builder backfills 2018 and skips shortened 2020 by default so the effective sample covers seven full regular seasons and lands near the mission's ~17k-row expectation.
- The training-data builder now also defaults `weather_fetcher` to `fetch_game_weather()` on the normal rebuild path, and the trainer / stacking / calibration / walk-forward entrypoints all pass that live weather fetcher through when they need to rebuild training data.
- `_fill_missing_feature_values()` no longer computes whole-dataset column means; residual missing feature values now use leakage-safe feature-family defaults (for example module baselines, neutral weather defaults, and prior-derived fallbacks) so earlier rows are not imputed from future seasons.
