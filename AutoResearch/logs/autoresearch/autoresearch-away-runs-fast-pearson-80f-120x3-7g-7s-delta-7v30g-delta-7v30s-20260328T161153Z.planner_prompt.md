Research instructions:
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

Current repo constraints:
- Only edit the AGENT_CONFIG block in train.py.
- Exploration mode for this session is `fast`.
- This session must use trials=120 and folds=3.
- max_features must be one of [72, 80, 88].
- selector_type must be one of ['pearson', 'bucketed', 'ablation'].
- bucket_quotas must contain 3 or 4 non-negative integers with total <= max_features.
- exclude_patterns and force_include_patterns must be short lists of feature name patterns.
- The trainer already strips most 60g/60s candidate windows before selection.
- Avoid repeating any tried non-full configuration fingerprint.
- Use artifact diagnostics when available to explain why runs were good or bad.
- Inspect selected/omitted features, fill health, drift, and Poisson deviance before proposing the next test.

Current session context:
{
  "artifact_reviews": [
    {
      "cv_rmse": 3.206906254965057,
      "diagnostics": {
        "cv_best_score": null,
        "cv_metric_name": null,
        "expected_delta_columns": null,
        "feature_column_count": 80,
        "holdout_metrics": {},
        "omitted_top_features_by_bucket": {},
        "selected_feature_counts": null,
        "selected_feature_drift": null,
        "selected_feature_fill_health": null,
        "selected_features_by_bucket": {},
        "top_feature_importance": []
      },
      "experiment_id": 4,
      "experiment_name": "autoresearch-away-runs-fast-pearson-80f-120x3-7g-7s-delta-7v30g-delta-7v30s-20260328T154503Z",
      "holdout_r2": 0.023376247825425578
    },
    {
      "cv_rmse": 3.2076445588268494,
      "diagnostics": {
        "cv_best_score": null,
        "cv_metric_name": null,
        "expected_delta_columns": null,
        "feature_column_count": 88,
        "holdout_metrics": {},
        "omitted_top_features_by_bucket": {},
        "selected_feature_counts": null,
        "selected_feature_drift": null,
        "selected_feature_fill_health": null,
        "selected_features_by_bucket": {},
        "top_feature_importance": []
      },
      "experiment_id": 5,
      "experiment_name": "autoresearch-away-runs-fast-pearson-88f-120x3-7g-7s-delta-7v30g-delta-7v30s-20260328T155229Z",
      "holdout_r2": 0.02375547682668977
    },
    {
      "cv_rmse": 3.207380749175654,
      "diagnostics": {
        "cv_best_score": null,
        "cv_metric_name": null,
        "expected_delta_columns": null,
        "feature_column_count": 72,
        "holdout_metrics": {},
        "omitted_top_features_by_bucket": {},
        "selected_feature_counts": null,
        "selected_feature_drift": null,
        "selected_feature_fill_health": null,
        "selected_features_by_bucket": {},
        "top_feature_importance": []
      },
      "experiment_id": 6,
      "experiment_name": "autoresearch-away-runs-fast-pearson-72f-120x3-7g-7s-delta-7v30g-delta-7v30s-20260328T160324Z",
      "holdout_r2": 0.023118057576344886
    }
  ],
  "best_run": {
    "cv_rmse": 3.2075190826533215,
    "experiment_name": "autoresearch-away-runs-fast-pearson-80f-120x3-7g-7s-delta-7v30g-delta-7v30s-20260328T151401Z",
    "holdout_r2": 0.024468603600688477,
    "mode": "fast"
  },
  "progress": {
    "completed_runs": 6,
    "remaining_minutes": null,
    "successful_runs": 6
  },
  "recent_notes": [
    {
      "body": "Forced include patterns ['*_7g', '*_7s', '*_delta_7v30g', '*_delta_7v30s']. Use this result to judge whether short-window features deserve more quota.",
      "created_at": "2026-03-28T16:02:12.541674+00:00",
      "experiment_id": 5,
      "id": 10,
      "importance": "medium",
      "metadata": {
        "force_include_patterns": [
          "*_7g",
          "*_7s",
          "*_delta_7v30g",
          "*_delta_7v30s"
        ]
      },
      "note_type": "feature_bias",
      "title": "Forced feature family test"
    },
    {
      "body": "Good because: tested forced feature family ['*_7g', '*_7s', '*_delta_7v30g', '*_delta_7v30s']. Bad because: no obvious negative artifact pattern was detected. Suspicious because: no reproducible suspicious artifact signal was detected.",
      "created_at": "2026-03-28T16:02:12.541674+00:00",
      "experiment_id": 5,
      "id": 11,
      "importance": "medium",
      "metadata": {
        "bad": [
          "no obvious negative artifact pattern was detected"
        ],
        "good": [
          "tested forced feature family ['*_7g', '*_7s', '*_delta_7v30g', '*_delta_7v30s']"
        ],
        "suspicious": [
          "no reproducible suspicious artifact signal was detected"
        ]
      },
      "note_type": "diagnostic_summary",
      "title": "Run diagnostic summary"
    },
    {
      "body": "Forced include patterns ['*_7g', '*_7s', '*_delta_7v30g', '*_delta_7v30s']. Use this result to judge whether short-window features deserve more quota.",
      "created_at": "2026-03-28T16:10:54.388992+00:00",
      "experiment_id": 6,
      "id": 12,
      "importance": "medium",
      "metadata": {
        "force_include_patterns": [
          "*_7g",
          "*_7s",
          "*_delta_7v30g",
          "*_delta_7v30s"
        ]
      },
      "note_type": "feature_bias",
      "title": "Forced feature family test"
    },
    {
      "body": "Good because: tested forced feature family ['*_7g', '*_7s', '*_delta_7v30g', '*_delta_7v30s']. Bad because: no obvious negative artifact pattern was detected. Suspicious because: no reproducible suspicious artifact signal was detected.",
      "created_at": "2026-03-28T16:10:54.388992+00:00",
      "experiment_id": 6,
      "id": 13,
      "importance": "medium",
      "metadata": {
        "bad": [
          "no obvious negative artifact pattern was detected"
        ],
        "good": [
          "tested forced feature family ['*_7g', '*_7s', '*_delta_7v30g', '*_delta_7v30s']"
        ],
        "suspicious": [
          "no reproducible suspicious artifact signal was detected"
        ]
      },
      "note_type": "diagnostic_summary",
      "title": "Run diagnostic summary"
    }
  ],
  "recent_session_runs": [
    {
      "config": {
        "blend_mode": "xgb_only",
        "bucket_quotas": [
          80,
          0,
          0,
          0
        ],
        "bucket_targets": {
          "context": 0,
          "delta": 0,
          "medium_form": 0,
          "short_form": 80
        },
        "cv_aggregation_mode": "mean",
        "early_stopping_rounds": 30,
        "exclude_patterns": [],
        "feature_selection_mode": "flat",
        "force_include_patterns": [
          "*_7g",
          "*_7s",
          "*_delta_7v30g",
          "*_delta_7v30s"
        ],
        "forced_delta_count": 8,
        "lightgbm_param_mode": "derived",
        "max_features": 80,
        "mode": "fast",
        "optuna_workers": 2,
        "search_iterations": 120,
        "selector_type": "pearson",
        "time_series_splits": 3,
        "xgboost_n_jobs": 3
      },
      "cv_rmse": 3.2075190826533215,
      "diagnostics": {
        "cv_best_score": null,
        "cv_metric_name": null,
        "expected_delta_columns": null,
        "feature_column_count": 80,
        "holdout_metrics": {},
        "omitted_top_features_by_bucket": {},
        "selected_feature_counts": null,
        "selected_feature_drift": null,
        "selected_feature_fill_health": null,
        "selected_features_by_bucket": {},
        "top_feature_importance": []
      },
      "error_message": null,
      "experiment_name": "autoresearch-away-runs-fast-pearson-80f-120x3-7g-7s-delta-7v30g-delta-7v30s-20260328T151401Z",
      "holdout_r2": 0.024468603600688477,
      "holdout_rmse": 3.317658937407337,
      "hypothesis": "Baseline tonight's best known manual region: flat, 80 features, forced_delta_count=8.",
      "id": 1,
      "planner_model": "droid:custom:GLM-5.1-(Z.AI-Coding)-4",
      "planner_type": "heuristic_fallback",
      "status": "succeeded"
    },
    {
      "config": {
        "blend_mode": "xgb_only",
        "bucket_quotas": [
          80,
          0,
          0,
          0
        ],
        "bucket_targets": {
          "context": 0,
          "delta": 0,
          "medium_form": 0,
          "short_form": 80
        },
        "cv_aggregation_mode": "mean",
        "early_stopping_rounds": 30,
        "exclude_patterns": [],
        "feature_selection_mode": "flat",
        "force_include_patterns": [
          "*_7g",
          "*_7s",
          "*_delta_7v30g",
          "*_delta_7v30s"
        ],
        "forced_delta_count": 10,
        "lightgbm_param_mode": "derived",
        "max_features": 80,
        "mode": "fast",
        "optuna_workers": 2,
        "search_iterations": 120,
        "selector_type": "pearson",
        "time_series_splits": 3,
        "xgboost_n_jobs": 3
      },
      "cv_rmse": 3.2082564413546786,
      "diagnostics": {
        "cv_best_score": null,
        "cv_metric_name": null,
        "expected_delta_columns": null,
        "feature_column_count": 80,
        "holdout_metrics": {},
        "omitted_top_features_by_bucket": {},
        "selected_feature_counts": null,
        "selected_feature_drift": null,
        "selected_feature_fill_health": null,
        "selected_features_by_bucket": {},
        "top_feature_importance": []
      },
      "error_message": null,
      "experiment_name": "autoresearch-away-runs-fast-pearson-80f-120x3-7g-7s-delta-7v30g-delta-7v30s-20260328T152444Z",
      "holdout_r2": 0.023311497764217037,
      "holdout_rmse": 3.3196259396260483,
      "hypothesis": "Map the forced-delta curve upward from the anchor at 8 by testing forced_delta_count=10 with all other levers frozen, to see if adding two more delta features improves holdout R\u00b2 beyond the current 2.45%.",
      "id": 2,
      "planner_model": "droid:custom:GLM-5.1-(Z.AI-Coding)-4",
      "planner_type": "llm",
      "status": "succeeded"
    },
    {
      "config": {
        "blend_mode": "xgb_only",
        "bucket_quotas": [
          80,
          0,
          0,
          0
        ],
        "bucket_targets": {
          "context": 0,
          "delta": 0,
          "medium_form": 0,
          "short_form": 80
        },
        "cv_aggregation_mode": "mean",
        "early_stopping_rounds": 30,
        "exclude_patterns": [],
        "feature_selection_mode": "flat",
        "force_include_patterns": [
          "*_7g",
          "*_7s",
          "*_delta_7v30g",
          "*_delta_7v30s"
        ],
        "forced_delta_count": 6,
        "lightgbm_param_mode": "derived",
        "max_features": 80,
        "mode": "fast",
        "optuna_workers": 2,
        "search_iterations": 120,
        "selector_type": "pearson",
        "time_series_splits": 3,
        "xgboost_n_jobs": 3
      },
      "cv_rmse": 3.2054652162442405,
      "diagnostics": {
        "cv_best_score": null,
        "cv_metric_name": null,
        "expected_delta_columns": null,
        "feature_column_count": 80,
        "holdout_metrics": {},
        "omitted_top_features_by_bucket": {},
        "selected_feature_counts": null,
        "selected_feature_drift": null,
        "selected_feature_fill_health": null,
        "selected_features_by_bucket": {},
        "top_feature_importance": []
      },
      "error_message": null,
      "experiment_name": "autoresearch-away-runs-fast-pearson-80f-120x3-7g-7s-delta-7v30g-delta-7v30s-20260328T153254Z",
      "holdout_r2": 0.02396094142558014,
      "holdout_rmse": 3.3185220725645483,
      "hypothesis": "Test forced_delta_count=6 to map the downward side of the forced-delta curve around the current anchor at 8, since moving up to 10 slightly degraded holdout R\u00b2 (2.45% \u2192 2.33%), suggesting the peak may lie at or below 8.",
      "id": 3,
      "planner_model": "droid:custom:GLM-5.1-(Z.AI-Coding)-4",
      "planner_type": "llm",
      "status": "succeeded"
    },
    {
      "config": {
        "blend_mode": "xgb_only",
        "bucket_quotas": [
          80,
          0,
          0,
          0
        ],
        "bucket_targets": {
          "context": 0,
          "delta": 0,
          "medium_form": 0,
          "short_form": 80
        },
        "cv_aggregation_mode": "mean",
        "early_stopping_rounds": 30,
        "exclude_patterns": [],
        "feature_selection_mode": "flat",
        "force_include_patterns": [
          "*_7g",
          "*_7s",
          "*_delta_7v30g",
          "*_delta_7v30s"
        ],
        "forced_delta_count": 12,
        "lightgbm_param_mode": "derived",
        "max_features": 80,
        "mode": "fast",
        "optuna_workers": 2,
        "search_iterations": 120,
        "selector_type": "pearson",
        "time_series_splits": 3,
        "xgboost_n_jobs": 3
      },
      "cv_rmse": 3.206906254965057,
      "diagnostics": {
        "cv_best_score": null,
        "cv_metric_name": null,
        "expected_delta_columns": null,
        "feature_column_count": 80,
        "holdout_metrics": {},
        "omitted_top_features_by_bucket": {},
        "selected_feature_counts": null,
        "selected_feature_drift": null,
        "selected_feature_fill_health": null,
        "selected_features_by_bucket": {},
        "top_feature_importance": []
      },
      "error_message": null,
      "experiment_name": "autoresearch-away-runs-fast-pearson-80f-120x3-7g-7s-delta-7v30g-delta-7v30s-20260328T154503Z",
      "holdout_r2": 0.023376247825425578,
      "holdout_rmse": 3.3195158996568592,
      "hypothesis": "Continue Stage 1 forced-delta mapping at forced_delta_count=12 to confirm the downward trend observed from 8\u219210 (R\u00b2 dropped from 2.45% to 2.33%), establishing whether the peak is firmly at 8 before moving to Stage 2 local refinement.",
      "id": 4,
      "planner_model": "droid:custom:GLM-5.1-(Z.AI-Coding)-4",
      "planner_type": "llm",
      "status": "succeeded"
    },
    {
      "config": {
        "blend_mode": "xgb_only",
        "bucket_quotas": [
          88,
          0,
          0,
          0
        ],
        "bucket_targets": {
          "context": 0,
          "delta": 0,
          "medium_form": 0,
          "short_form": 88
        },
        "cv_aggregation_mode": "mean",
        "early_stopping_rounds": 30,
        "exclude_patterns": [],
        "feature_selection_mode": "flat",
        "force_include_patterns": [
          "*_7g",
          "*_7s",
          "*_delta_7v30g",
          "*_delta_7v30s"
        ],
        "forced_delta_count": 8,
        "lightgbm_param_mode": "derived",
        "max_features": 88,
        "mode": "fast",
        "optuna_workers": 2,
        "search_iterations": 120,
        "selector_type": "pearson",
        "time_series_splits": 3,
        "xgboost_n_jobs": 3
      },
      "cv_rmse": 3.2076445588268494,
      "diagnostics": {
        "cv_best_score": null,
        "cv_metric_name": null,
        "expected_delta_columns": null,
        "feature_column_count": 88,
        "holdout_metrics": {},
        "omitted_top_features_by_bucket": {},
        "selected_feature_counts": null,
        "selected_feature_drift": null,
        "selected_feature_fill_health": null,
        "selected_features_by_bucket": {},
        "top_feature_importance": []
      },
      "error_message": null,
      "experiment_name": "autoresearch-away-runs-fast-pearson-88f-120x3-7g-7s-delta-7v30g-delta-7v30s-20260328T155229Z",
      "holdout_r2": 0.02375547682668977,
      "holdout_rmse": 3.3188713428735106,
      "hypothesis": "The forced-delta curve has stabilized with a clear peak at 8 (R\u00b2=2.45% vs 2.40% at 6 and 2.33% at 10/12); move to Stage 2 local refinement by testing max_features=88 with the winning forced_delta_count=8 to see if expanding the feature budget captures additional signal just below the top-80 threshold.",
      "id": 5,
      "planner_model": "droid:custom:GLM-5.1-(Z.AI-Coding)-4",
      "planner_type": "llm",
      "status": "succeeded"
    },
    {
      "config": {
        "blend_mode": "xgb_only",
        "bucket_quotas": [
          72,
          0,
          0,
          0
        ],
        "bucket_targets": {
          "context": 0,
          "delta": 0,
          "medium_form": 0,
          "short_form": 72
        },
        "cv_aggregation_mode": "mean",
        "early_stopping_rounds": 30,
        "exclude_patterns": [],
        "feature_selection_mode": "flat",
        "force_include_patterns": [
          "*_7g",
          "*_7s",
          "*_delta_7v30g",
          "*_delta_7v30s"
        ],
        "forced_delta_count": 8,
        "lightgbm_param_mode": "derived",
        "max_features": 72,
        "mode": "fast",
        "optuna_workers": 2,
        "search_iterations": 120,
        "selector_type": "pearson",
        "time_series_splits": 3,
        "xgboost_n_jobs": 3
      },
      "cv_rmse": 3.207380749175654,
      "diagnostics": {
        "cv_best_score": null,
        "cv_metric_name": null,
        "expected_delta_columns": null,
        "feature_column_count": 72,
        "holdout_metrics": {},
        "omitted_top_features_by_bucket": {},
        "selected_feature_counts": null,
        "selected_feature_drift": null,
        "selected_feature_fill_health": null,
        "selected_features_by_bucket": {},
        "top_feature_importance": []
      },
      "error_message": null,
      "experiment_name": "autoresearch-away-runs-fast-pearson-72f-120x3-7g-7s-delta-7v30g-delta-7v30s-20260328T160324Z",
      "holdout_r2": 0.023118057576344886,
      "holdout_rmse": 3.3199546612560993,
      "hypothesis": "Stage 1 established forced_delta_count=8 as the peak; Stage 2 showed max_features=88 degraded R\u00b2 (2.45%\u21922.38%), so test max_features=72 with the winning delta=8 to determine whether the optimum lies below 80 and whether tighter feature budgets reduce noise from weak marginal features.",
      "id": 6,
      "planner_model": "droid:custom:GLM-5.1-(Z.AI-Coding)-4",
      "planner_type": "llm",
      "status": "succeeded"
    }
  ],
  "session": {
    "duration_hours": 0,
    "ended_at": null,
    "exploration_mode": "fast",
    "id": 1,
    "run_full_at_end": false,
    "started_at": "2026-03-28T15:08:06.274862+00:00",
    "status": "running",
    "stop_at": null,
    "until_interrupted": true
  }
}

Experiment history digest:
{
  "recent_runs": [
    {
      "config": {
        "blend_mode": "xgb_only",
        "bucket_quotas": [
          80,
          0,
          0,
          0
        ],
        "bucket_targets": {
          "context": 0,
          "delta": 0,
          "medium_form": 0,
          "short_form": 80
        },
        "cv_aggregation_mode": "mean",
        "early_stopping_rounds": 30,
        "exclude_patterns": [],
        "feature_selection_mode": "flat",
        "force_include_patterns": [
          "*_7g",
          "*_7s",
          "*_delta_7v30g",
          "*_delta_7v30s"
        ],
        "forced_delta_count": 8,
        "lightgbm_param_mode": "derived",
        "max_features": 80,
        "mode": "fast",
        "optuna_workers": 2,
        "search_iterations": 120,
        "selector_type": "pearson",
        "time_series_splits": 3,
        "xgboost_n_jobs": 3
      },
      "cv_rmse": 3.2075190826533215,
      "diagnostics": {
        "cv_best_score": null,
        "cv_metric_name": null,
        "expected_delta_columns": null,
        "feature_column_count": 80,
        "holdout_metrics": {},
        "omitted_top_features_by_bucket": {},
        "selected_feature_counts": null,
        "selected_feature_drift": null,
        "selected_feature_fill_health": null,
        "selected_features_by_bucket": {},
        "top_feature_importance": []
      },
      "error_message": null,
      "experiment_name": "autoresearch-away-runs-fast-pearson-80f-120x3-7g-7s-delta-7v30g-delta-7v30s-20260328T151401Z",
      "holdout_r2": 0.024468603600688477,
      "holdout_rmse": 3.317658937407337,
      "hypothesis": "Baseline tonight's best known manual region: flat, 80 features, forced_delta_count=8.",
      "id": 1,
      "planner_model": "droid:custom:GLM-5.1-(Z.AI-Coding)-4",
      "planner_type": "heuristic_fallback",
      "status": "succeeded"
    },
    {
      "config": {
        "blend_mode": "xgb_only",
        "bucket_quotas": [
          80,
          0,
          0,
          0
        ],
        "bucket_targets": {
          "context": 0,
          "delta": 0,
          "medium_form": 0,
          "short_form": 80
        },
        "cv_aggregation_mode": "mean",
        "early_stopping_rounds": 30,
        "exclude_patterns": [],
        "feature_selection_mode": "flat",
        "force_include_patterns": [
          "*_7g",
          "*_7s",
          "*_delta_7v30g",
          "*_delta_7v30s"
        ],
        "forced_delta_count": 10,
        "lightgbm_param_mode": "derived",
        "max_features": 80,
        "mode": "fast",
        "optuna_workers": 2,
        "search_iterations": 120,
        "selector_type": "pearson",
        "time_series_splits": 3,
        "xgboost_n_jobs": 3
      },
      "cv_rmse": 3.2082564413546786,
      "diagnostics": {
        "cv_best_score": null,
        "cv_metric_name": null,
        "expected_delta_columns": null,
        "feature_column_count": 80,
        "holdout_metrics": {},
        "omitted_top_features_by_bucket": {},
        "selected_feature_counts": null,
        "selected_feature_drift": null,
        "selected_feature_fill_health": null,
        "selected_features_by_bucket": {},
        "top_feature_importance": []
      },
      "error_message": null,
      "experiment_name": "autoresearch-away-runs-fast-pearson-80f-120x3-7g-7s-delta-7v30g-delta-7v30s-20260328T152444Z",
      "holdout_r2": 0.023311497764217037,
      "holdout_rmse": 3.3196259396260483,
      "hypothesis": "Map the forced-delta curve upward from the anchor at 8 by testing forced_delta_count=10 with all other levers frozen, to see if adding two more delta features improves holdout R\u00b2 beyond the current 2.45%.",
      "id": 2,
      "planner_model": "droid:custom:GLM-5.1-(Z.AI-Coding)-4",
      "planner_type": "llm",
      "status": "succeeded"
    },
    {
      "config": {
        "blend_mode": "xgb_only",
        "bucket_quotas": [
          80,
          0,
          0,
          0
        ],
        "bucket_targets": {
          "context": 0,
          "delta": 0,
          "medium_form": 0,
          "short_form": 80
        },
        "cv_aggregation_mode": "mean",
        "early_stopping_rounds": 30,
        "exclude_patterns": [],
        "feature_selection_mode": "flat",
        "force_include_patterns": [
          "*_7g",
          "*_7s",
          "*_delta_7v30g",
          "*_delta_7v30s"
        ],
        "forced_delta_count": 6,
        "lightgbm_param_mode": "derived",
        "max_features": 80,
        "mode": "fast",
        "optuna_workers": 2,
        "search_iterations": 120,
        "selector_type": "pearson",
        "time_series_splits": 3,
        "xgboost_n_jobs": 3
      },
      "cv_rmse": 3.2054652162442405,
      "diagnostics": {
        "cv_best_score": null,
        "cv_metric_name": null,
        "expected_delta_columns": null,
        "feature_column_count": 80,
        "holdout_metrics": {},
        "omitted_top_features_by_bucket": {},
        "selected_feature_counts": null,
        "selected_feature_drift": null,
        "selected_feature_fill_health": null,
        "selected_features_by_bucket": {},
        "top_feature_importance": []
      },
      "error_message": null,
      "experiment_name": "autoresearch-away-runs-fast-pearson-80f-120x3-7g-7s-delta-7v30g-delta-7v30s-20260328T153254Z",
      "holdout_r2": 0.02396094142558014,
      "holdout_rmse": 3.3185220725645483,
      "hypothesis": "Test forced_delta_count=6 to map the downward side of the forced-delta curve around the current anchor at 8, since moving up to 10 slightly degraded holdout R\u00b2 (2.45% \u2192 2.33%), suggesting the peak may lie at or below 8.",
      "id": 3,
      "planner_model": "droid:custom:GLM-5.1-(Z.AI-Coding)-4",
      "planner_type": "llm",
      "status": "succeeded"
    },
    {
      "config": {
        "blend_mode": "xgb_only",
        "bucket_quotas": [
          80,
          0,
          0,
          0
        ],
        "bucket_targets": {
          "context": 0,
          "delta": 0,
          "medium_form": 0,
          "short_form": 80
        },
        "cv_aggregation_mode": "mean",
        "early_stopping_rounds": 30,
        "exclude_patterns": [],
        "feature_selection_mode": "flat",
        "force_include_patterns": [
          "*_7g",
          "*_7s",
          "*_delta_7v30g",
          "*_delta_7v30s"
        ],
        "forced_delta_count": 12,
        "lightgbm_param_mode": "derived",
        "max_features": 80,
        "mode": "fast",
        "optuna_workers": 2,
        "search_iterations": 120,
        "selector_type": "pearson",
        "time_series_splits": 3,
        "xgboost_n_jobs": 3
      },
      "cv_rmse": 3.206906254965057,
      "diagnostics": {
        "cv_best_score": null,
        "cv_metric_name": null,
        "expected_delta_columns": null,
        "feature_column_count": 80,
        "holdout_metrics": {},
        "omitted_top_features_by_bucket": {},
        "selected_feature_counts": null,
        "selected_feature_drift": null,
        "selected_feature_fill_health": null,
        "selected_features_by_bucket": {},
        "top_feature_importance": []
      },
      "error_message": null,
      "experiment_name": "autoresearch-away-runs-fast-pearson-80f-120x3-7g-7s-delta-7v30g-delta-7v30s-20260328T154503Z",
      "holdout_r2": 0.023376247825425578,
      "holdout_rmse": 3.3195158996568592,
      "hypothesis": "Continue Stage 1 forced-delta mapping at forced_delta_count=12 to confirm the downward trend observed from 8\u219210 (R\u00b2 dropped from 2.45% to 2.33%), establishing whether the peak is firmly at 8 before moving to Stage 2 local refinement.",
      "id": 4,
      "planner_model": "droid:custom:GLM-5.1-(Z.AI-Coding)-4",
      "planner_type": "llm",
      "status": "succeeded"
    },
    {
      "config": {
        "blend_mode": "xgb_only",
        "bucket_quotas": [
          88,
          0,
          0,
          0
        ],
        "bucket_targets": {
          "context": 0,
          "delta": 0,
          "medium_form": 0,
          "short_form": 88
        },
        "cv_aggregation_mode": "mean",
        "early_stopping_rounds": 30,
        "exclude_patterns": [],
        "feature_selection_mode": "flat",
        "force_include_patterns": [
          "*_7g",
          "*_7s",
          "*_delta_7v30g",
          "*_delta_7v30s"
        ],
        "forced_delta_count": 8,
        "lightgbm_param_mode": "derived",
        "max_features": 88,
        "mode": "fast",
        "optuna_workers": 2,
        "search_iterations": 120,
        "selector_type": "pearson",
        "time_series_splits": 3,
        "xgboost_n_jobs": 3
      },
      "cv_rmse": 3.2076445588268494,
      "diagnostics": {
        "cv_best_score": null,
        "cv_metric_name": null,
        "expected_delta_columns": null,
        "feature_column_count": 88,
        "holdout_metrics": {},
        "omitted_top_features_by_bucket": {},
        "selected_feature_counts": null,
        "selected_feature_drift": null,
        "selected_feature_fill_health": null,
        "selected_features_by_bucket": {},
        "top_feature_importance": []
      },
      "error_message": null,
      "experiment_name": "autoresearch-away-runs-fast-pearson-88f-120x3-7g-7s-delta-7v30g-delta-7v30s-20260328T155229Z",
      "holdout_r2": 0.02375547682668977,
      "holdout_rmse": 3.3188713428735106,
      "hypothesis": "The forced-delta curve has stabilized with a clear peak at 8 (R\u00b2=2.45% vs 2.40% at 6 and 2.33% at 10/12); move to Stage 2 local refinement by testing max_features=88 with the winning forced_delta_count=8 to see if expanding the feature budget captures additional signal just below the top-80 threshold.",
      "id": 5,
      "planner_model": "droid:custom:GLM-5.1-(Z.AI-Coding)-4",
      "planner_type": "llm",
      "status": "succeeded"
    },
    {
      "config": {
        "blend_mode": "xgb_only",
        "bucket_quotas": [
          72,
          0,
          0,
          0
        ],
        "bucket_targets": {
          "context": 0,
          "delta": 0,
          "medium_form": 0,
          "short_form": 72
        },
        "cv_aggregation_mode": "mean",
        "early_stopping_rounds": 30,
        "exclude_patterns": [],
        "feature_selection_mode": "flat",
        "force_include_patterns": [
          "*_7g",
          "*_7s",
          "*_delta_7v30g",
          "*_delta_7v30s"
        ],
        "forced_delta_count": 8,
        "lightgbm_param_mode": "derived",
        "max_features": 72,
        "mode": "fast",
        "optuna_workers": 2,
        "search_iterations": 120,
        "selector_type": "pearson",
        "time_series_splits": 3,
        "xgboost_n_jobs": 3
      },
      "cv_rmse": 3.207380749175654,
      "diagnostics": {
        "cv_best_score": null,
        "cv_metric_name": null,
        "expected_delta_columns": null,
        "feature_column_count": 72,
        "holdout_metrics": {},
        "omitted_top_features_by_bucket": {},
        "selected_feature_counts": null,
        "selected_feature_drift": null,
        "selected_feature_fill_health": null,
        "selected_features_by_bucket": {},
        "top_feature_importance": []
      },
      "error_message": null,
      "experiment_name": "autoresearch-away-runs-fast-pearson-72f-120x3-7g-7s-delta-7v30g-delta-7v30s-20260328T160324Z",
      "holdout_r2": 0.023118057576344886,
      "holdout_rmse": 3.3199546612560993,
      "hypothesis": "Stage 1 established forced_delta_count=8 as the peak; Stage 2 showed max_features=88 degraded R\u00b2 (2.45%\u21922.38%), so test max_features=72 with the winning delta=8 to determine whether the optimum lies below 80 and whether tighter feature budgets reduce noise from weak marginal features.",
      "id": 6,
      "planner_model": "droid:custom:GLM-5.1-(Z.AI-Coding)-4",
      "planner_type": "llm",
      "status": "succeeded"
    }
  ],
  "top_runs": [
    {
      "config": {
        "blend_mode": "xgb_only",
        "bucket_quotas": [
          80,
          0,
          0,
          0
        ],
        "bucket_targets": {
          "context": 0,
          "delta": 0,
          "medium_form": 0,
          "short_form": 80
        },
        "cv_aggregation_mode": "mean",
        "early_stopping_rounds": 30,
        "exclude_patterns": [],
        "feature_selection_mode": "flat",
        "force_include_patterns": [
          "*_7g",
          "*_7s",
          "*_delta_7v30g",
          "*_delta_7v30s"
        ],
        "forced_delta_count": 8,
        "lightgbm_param_mode": "derived",
        "max_features": 80,
        "mode": "fast",
        "optuna_workers": 2,
        "search_iterations": 120,
        "selector_type": "pearson",
        "time_series_splits": 3,
        "xgboost_n_jobs": 3
      },
      "cv_rmse": 3.2075190826533215,
      "diagnostics": {
        "cv_best_score": null,
        "cv_metric_name": null,
        "expected_delta_columns": null,
        "feature_column_count": 80,
        "holdout_metrics": {},
        "omitted_top_features_by_bucket": {},
        "selected_feature_counts": null,
        "selected_feature_drift": null,
        "selected_feature_fill_health": null,
        "selected_features_by_bucket": {},
        "top_feature_importance": []
      },
      "error_message": null,
      "experiment_name": "autoresearch-away-runs-fast-pearson-80f-120x3-7g-7s-delta-7v30g-delta-7v30s-20260328T151401Z",
      "holdout_r2": 0.024468603600688477,
      "holdout_rmse": 3.317658937407337,
      "hypothesis": "Baseline tonight's best known manual region: flat, 80 features, forced_delta_count=8.",
      "id": 1,
      "planner_model": "droid:custom:GLM-5.1-(Z.AI-Coding)-4",
      "planner_type": "heuristic_fallback",
      "status": "succeeded"
    },
    {
      "config": {
        "blend_mode": "xgb_only",
        "bucket_quotas": [
          80,
          0,
          0,
          0
        ],
        "bucket_targets": {
          "context": 0,
          "delta": 0,
          "medium_form": 0,
          "short_form": 80
        },
        "cv_aggregation_mode": "mean",
        "early_stopping_rounds": 30,
        "exclude_patterns": [],
        "feature_selection_mode": "flat",
        "force_include_patterns": [
          "*_7g",
          "*_7s",
          "*_delta_7v30g",
          "*_delta_7v30s"
        ],
        "forced_delta_count": 6,
        "lightgbm_param_mode": "derived",
        "max_features": 80,
        "mode": "fast",
        "optuna_workers": 2,
        "search_iterations": 120,
        "selector_type": "pearson",
        "time_series_splits": 3,
        "xgboost_n_jobs": 3
      },
      "cv_rmse": 3.2054652162442405,
      "diagnostics": {
        "cv_best_score": null,
        "cv_metric_name": null,
        "expected_delta_columns": null,
        "feature_column_count": 80,
        "holdout_metrics": {},
        "omitted_top_features_by_bucket": {},
        "selected_feature_counts": null,
        "selected_feature_drift": null,
        "selected_feature_fill_health": null,
        "selected_features_by_bucket": {},
        "top_feature_importance": []
      },
      "error_message": null,
      "experiment_name": "autoresearch-away-runs-fast-pearson-80f-120x3-7g-7s-delta-7v30g-delta-7v30s-20260328T153254Z",
      "holdout_r2": 0.02396094142558014,
      "holdout_rmse": 3.3185220725645483,
      "hypothesis": "Test forced_delta_count=6 to map the downward side of the forced-delta curve around the current anchor at 8, since moving up to 10 slightly degraded holdout R\u00b2 (2.45% \u2192 2.33%), suggesting the peak may lie at or below 8.",
      "id": 3,
      "planner_model": "droid:custom:GLM-5.1-(Z.AI-Coding)-4",
      "planner_type": "llm",
      "status": "succeeded"
    },
    {
      "config": {
        "blend_mode": "xgb_only",
        "bucket_quotas": [
          88,
          0,
          0,
          0
        ],
        "bucket_targets": {
          "context": 0,
          "delta": 0,
          "medium_form": 0,
          "short_form": 88
        },
        "cv_aggregation_mode": "mean",
        "early_stopping_rounds": 30,
        "exclude_patterns": [],
        "feature_selection_mode": "flat",
        "force_include_patterns": [
          "*_7g",
          "*_7s",
          "*_delta_7v30g",
          "*_delta_7v30s"
        ],
        "forced_delta_count": 8,
        "lightgbm_param_mode": "derived",
        "max_features": 88,
        "mode": "fast",
        "optuna_workers": 2,
        "search_iterations": 120,
        "selector_type": "pearson",
        "time_series_splits": 3,
        "xgboost_n_jobs": 3
      },
      "cv_rmse": 3.2076445588268494,
      "diagnostics": {
        "cv_best_score": null,
        "cv_metric_name": null,
        "expected_delta_columns": null,
        "feature_column_count": 88,
        "holdout_metrics": {},
        "omitted_top_features_by_bucket": {},
        "selected_feature_counts": null,
        "selected_feature_drift": null,
        "selected_feature_fill_health": null,
        "selected_features_by_bucket": {},
        "top_feature_importance": []
      },
      "error_message": null,
      "experiment_name": "autoresearch-away-runs-fast-pearson-88f-120x3-7g-7s-delta-7v30g-delta-7v30s-20260328T155229Z",
      "holdout_r2": 0.02375547682668977,
      "holdout_rmse": 3.3188713428735106,
      "hypothesis": "The forced-delta curve has stabilized with a clear peak at 8 (R\u00b2=2.45% vs 2.40% at 6 and 2.33% at 10/12); move to Stage 2 local refinement by testing max_features=88 with the winning forced_delta_count=8 to see if expanding the feature budget captures additional signal just below the top-80 threshold.",
      "id": 5,
      "planner_model": "droid:custom:GLM-5.1-(Z.AI-Coding)-4",
      "planner_type": "llm",
      "status": "succeeded"
    },
    {
      "config": {
        "blend_mode": "xgb_only",
        "bucket_quotas": [
          80,
          0,
          0,
          0
        ],
        "bucket_targets": {
          "context": 0,
          "delta": 0,
          "medium_form": 0,
          "short_form": 80
        },
        "cv_aggregation_mode": "mean",
        "early_stopping_rounds": 30,
        "exclude_patterns": [],
        "feature_selection_mode": "flat",
        "force_include_patterns": [
          "*_7g",
          "*_7s",
          "*_delta_7v30g",
          "*_delta_7v30s"
        ],
        "forced_delta_count": 12,
        "lightgbm_param_mode": "derived",
        "max_features": 80,
        "mode": "fast",
        "optuna_workers": 2,
        "search_iterations": 120,
        "selector_type": "pearson",
        "time_series_splits": 3,
        "xgboost_n_jobs": 3
      },
      "cv_rmse": 3.206906254965057,
      "diagnostics": {
        "cv_best_score": null,
        "cv_metric_name": null,
        "expected_delta_columns": null,
        "feature_column_count": 80,
        "holdout_metrics": {},
        "omitted_top_features_by_bucket": {},
        "selected_feature_counts": null,
        "selected_feature_drift": null,
        "selected_feature_fill_health": null,
        "selected_features_by_bucket": {},
        "top_feature_importance": []
      },
      "error_message": null,
      "experiment_name": "autoresearch-away-runs-fast-pearson-80f-120x3-7g-7s-delta-7v30g-delta-7v30s-20260328T154503Z",
      "holdout_r2": 0.023376247825425578,
      "holdout_rmse": 3.3195158996568592,
      "hypothesis": "Continue Stage 1 forced-delta mapping at forced_delta_count=12 to confirm the downward trend observed from 8\u219210 (R\u00b2 dropped from 2.45% to 2.33%), establishing whether the peak is firmly at 8 before moving to Stage 2 local refinement.",
      "id": 4,
      "planner_model": "droid:custom:GLM-5.1-(Z.AI-Coding)-4",
      "planner_type": "llm",
      "status": "succeeded"
    },
    {
      "config": {
        "blend_mode": "xgb_only",
        "bucket_quotas": [
          80,
          0,
          0,
          0
        ],
        "bucket_targets": {
          "context": 0,
          "delta": 0,
          "medium_form": 0,
          "short_form": 80
        },
        "cv_aggregation_mode": "mean",
        "early_stopping_rounds": 30,
        "exclude_patterns": [],
        "feature_selection_mode": "flat",
        "force_include_patterns": [
          "*_7g",
          "*_7s",
          "*_delta_7v30g",
          "*_delta_7v30s"
        ],
        "forced_delta_count": 10,
        "lightgbm_param_mode": "derived",
        "max_features": 80,
        "mode": "fast",
        "optuna_workers": 2,
        "search_iterations": 120,
        "selector_type": "pearson",
        "time_series_splits": 3,
        "xgboost_n_jobs": 3
      },
      "cv_rmse": 3.2082564413546786,
      "diagnostics": {
        "cv_best_score": null,
        "cv_metric_name": null,
        "expected_delta_columns": null,
        "feature_column_count": 80,
        "holdout_metrics": {},
        "omitted_top_features_by_bucket": {},
        "selected_feature_counts": null,
        "selected_feature_drift": null,
        "selected_feature_fill_health": null,
        "selected_features_by_bucket": {},
        "top_feature_importance": []
      },
      "error_message": null,
      "experiment_name": "autoresearch-away-runs-fast-pearson-80f-120x3-7g-7s-delta-7v30g-delta-7v30s-20260328T152444Z",
      "holdout_r2": 0.023311497764217037,
      "holdout_rmse": 3.3196259396260483,
      "hypothesis": "Map the forced-delta curve upward from the anchor at 8 by testing forced_delta_count=10 with all other levers frozen, to see if adding two more delta features improves holdout R\u00b2 beyond the current 2.45%.",
      "id": 2,
      "planner_model": "droid:custom:GLM-5.1-(Z.AI-Coding)-4",
      "planner_type": "llm",
      "status": "succeeded"
    }
  ],
  "tried_config_fingerprints": [
    "9cb4c2e1683abde728c577c2bb5f2b00db99c8626582cce31ba76cb0b8bc0d98",
    "21a1eb2de6c223dd84318d881ab93c4d5b442749632910ade363ee56c4d480e4",
    "19ad8d6f2443a154571d7ea0e4b7743dd3ab857ccc38e9fee45ca9e57533b291",
    "0af760957d4216126b90728af89d25ac9d9c5e9ccfd8b845760df75953950876",
    "49c43683b1096cea6638d95c17034155677edc28f5fb813fa872a31f72fcd346",
    "b69799e729a271af80c7d49c88ce00bb96914ff6516eb341620b7c14ecefac2b"
  ]
}

Return JSON with this exact schema:
{
  "hypothesis": "one concise sentence",
  "reasoning": "brief explanation of why this experiment should improve accuracy",
  "config": {
    "max_features": 80,
    "selector_type": "pearson",
    "bucket_quotas": [80, 0, 0, 0],
    "exclude_patterns": [],
    "force_include_patterns": ["*_7g", "*_7s", "*_delta_7v30g", "*_delta_7v30s"],
    "forced_delta_count": 8,
    "trials": 120,
    "folds": 3
  }
}