# MLB Run-Count Autoresearch Program

## Objective

Run fast overnight experiments on `full_game_away_runs_model`, compare them consistently, stop launching new fast runs at the cutoff, then promote the best fast configuration into one full training run before morning. The planner is an LLM research agent, not a fixed heuristic search script.

## Fast Experiment Loop

1. Read `experiments.db` and sort prior fast runs by:
   - highest `holdout_r2`
   - lowest `cv_rmse`
   - earliest successful run as the final tie-breaker
2. Form the next hypothesis with the LLM from the current leader plus recent failures:
   - infer whether Pearson top-N or bucketed selection is currently more promising
   - vary feature counts (`60`, `80`, `100`, `120`) around the best observed region
   - vary bucket quotas if bucketed selection is leading
   - use ablations such as weather exclusion or forced `7g` metrics when recent results suggest they may help
   - avoid repeating any already-tried fast config
3. Edit only the config block at the top of `train.py`.
4. Run `python train.py --mode fast`.
5. Log the full config snapshot, metrics, hypothesis, planner type, planner model, planner prompt/response logs, stdout, stderr, and artifact paths into `experiments.db`.
6. Never rerun the same fast config fingerprint unless every other proposal has already been exhausted.

## Metrics

- Optimize `cv_rmse` as the fast proxy metric.
- Treat `holdout_r2` as the ground-truth score for promotion decisions.
- Use `holdout_rmse` as a secondary sanity check.
- If available in emitted artifacts, also inspect Poisson deviance as an extra count-model sanity check.

## Research Bias

- Do not assume more Optuna trials or more folds automatically improve holdout performance.
- Bias hypotheses toward feature representation and feature-selection bottlenecks before assuming search budget is the main issue.
- Treat discovery and confirmation as different jobs:
  - `fast` runs are for discovery
  - `full` runs are for confirmation / stability
- Keep comparisons apples-to-apples whenever possible:
  - same target
  - similar training parquet/version
  - similar folds and trial counts
  - similar selector family
  - similar objective / CV metric

## Search Space

- Feature counts: `60`, `80`, `100`, `120`
- Selector types:
  - `pearson` = flat top-N
  - `bucketed` = bucket quotas
  - `ablation` = bucketed plus exclude / force-include patterns
- Bucket quota families:
  - `short_form`
  - `medium_form`
  - `delta`
  - `context`
- Primary ablations:
  - exclude `weather_*`
  - force include `*_7g`, `*_7s`, `*_delta_7v30g`, `*_delta_7v30s`

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
- Promote the best successful fast run into `python train.py --mode full`.
- Full mode uses the same feature config but upgrades search depth to `500` trials and `5` folds.
- Exit cleanly once the full run finishes.

## Hypothesis Rules

- Change one major lever at a time around the current leader.
- Keep the winner as the anchor and test neighbors, not random jumps.
- Prefer small deltas in feature count or quota mix after a promising run.
- If several runs regress together, fall back to the best known config and switch selector family.
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
   - `holdout_rmse`
   - `cv_rmse`
   - duration
   - summary / model artifact paths

## Start Commands

- Foreground: `python launcher.py`
- POSIX background: `python launcher.py &`
- PowerShell background: `Start-Process python -ArgumentList 'launcher.py'`
