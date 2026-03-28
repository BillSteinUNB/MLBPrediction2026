# MLB Run-Count Autoresearch Program

## Objective

Tonight's job is narrow: map the forced-delta region around the current best repaired-parquet manual setup and do not waste the night on stale search axes. The planner is an LLM research agent, not a fixed heuristic search script.

## Tonight's frozen baseline

Hold these fixed all night:

- parquet: `data/training/ParquetDefault.parquet`
- target: `final_away_score`
- model: `full_game_away_runs_model`
- selector family: `flat` (`SELECTOR_TYPE = "pearson"`)
- blend_mode: `xgb_only`
- cv_aggregation_mode: `mean`
- holdout season: `2025`
- folds: `3`
- no data/build code changes
- no 500x5 promotion

Current best known manual region:

- forced-delta count: `8`
- holdout R²: about `3.82%`

## Fast Experiment Loop

1. Read `experiments.db` and sort prior fast runs by:
   - highest `holdout_r2`
   - lowest `holdout_poisson_deviance`
   - lowest `holdout_rmse`
   - earliest successful run as the final tie-breaker
2. Form the next hypothesis with the LLM from the current leader plus recent failures:
   - primarily vary forced-delta retention around the current best manual region
   - keep `max_features = 80` during the first discovery block
   - only after the delta-count curve starts to stabilize, vary `max_features` locally
   - avoid repeating any already-tried fast config
3. Edit only the config block at the top of `train.py`.
4. Run `python train.py --mode fast`.
5. Log the full config snapshot, metrics, hypothesis, planner type, planner model, planner prompt/response logs, stdout, stderr, and artifact paths into `experiments.db`.
6. Never rerun the same fast config fingerprint unless every other proposal has already been exhausted.

## Metrics and ranking

- Rank runs by:
  1. `holdout_r2`
  2. `holdout_poisson_deviance`
  3. `holdout_rmse`
- Use CV only as a weak sanity check, not the main ranker.

## Research Bias

- Do not reopen stale axes tonight.
- Do not test grouped vs flat vs bucketed.
- Do not test learned blend vs xgb_only.
- Do not test 3 folds vs 5 folds.
- Do not rebuild data or switch parquet versions.
- Keep comparisons apples-to-apples by freezing all nonessential modeling choices.

## Search Space

- Stage 1 discovery:
  - forced_delta_count in `{4, 6, 8, 10, 12, 14, 16}`
  - `max_features = 80`
  - `Iterations = 120`
  - `Folds = 3`
- Stage 2 local refinement:
  - take top 2 forced-delta counts from Stage 1
  - test `max_features` in `{72, 80, 88}`
  - still `120x3`
- Stage 3 confirmation:
  - best 1 config at `300x3`
  - optional second-best config at `300x3`
- If a clean delta-family composition ablation is easy to express, do at most one small test late in the night. Otherwise skip it.

## Repo-Specific Note

The underlying trainer already strips most `60g` and `60s` candidate windows before selection, so those are effectively pre-ablated in the current pipeline. Spend overnight trials on the remaining active axes instead of burning time on duplicate no-op runs.

## Artifact-Aware Diagnosis

- Do not trust only top-line metrics if richer artifacts are available.
- When recent runs underperform or become unstable, inspect emitted metadata/artifacts for:
  - `feature_columns`
  - `selected_features_by_bucket`
  - `omitted_top_features_by_bucket`
  - `feature_importance_rankings`
- Prefer hypotheses that explain why useful engineered features may not be surviving selection.
- Treat the following as suspicious and worth recording for morning review:
  - useful feature families missing from the selected set
  - default-heavy or near-constant features dominating the winner
  - repeated omission of promising short-window or delta features
  - top-line regressions that reproduce on rerun

## Cutoff Logic

- Cutoff time defaults to `04:00` local time.
- Do not start another fast run unless at least 30 minutes remain before cutoff.
- Once the remaining time is below that threshold, stop the fast loop.
- Promotion tonight means `300x3`, not `500x5`.
- Exit cleanly once the full run finishes.

## Hypothesis Rules

- Change one major lever at a time around the current leader.
- Prefer forced-delta-count changes first.
- Only after the forced-delta region is mapped should `max_features` move locally.
- Keep the winner as the anchor and test neighbors, not random jumps.
- If the LLM output is invalid, fall back to a safe heuristic rather than skipping the night.

## Example Flow

1. Hypothesis: "Bucketed 80-feature baseline is stable; test whether flat Pearson top-80 recovers the old Run 3 behavior."
2. Edit `train.py`:
   - `MAX_FEATURES = 80`
   - `SELECTOR_TYPE = "pearson"`
   - `BUCKET_QUOTAS = [24, 28, 12, 16]`
   - `EXCLUDE_PATTERNS = []`
   - `FORCE_INCLUDE_PATTERNS = []`
3. Run: `python train.py --mode fast`
4. Log:
   - hypothesis text
   - config snapshot
   - `holdout_r2`
   - `holdout_poisson_deviance`
   - `holdout_rmse`
   - `cv_rmse`
   - duration
   - summary / model artifact paths

## Start Commands

- Foreground: `python launcher.py`
- POSIX background: `python launcher.py &`
- PowerShell background: `Start-Process python -ArgumentList 'launcher.py'`
