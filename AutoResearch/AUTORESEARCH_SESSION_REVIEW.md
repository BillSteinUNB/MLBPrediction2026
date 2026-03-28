# AutoResearch Session Review Guide

Use this document when an overnight AutoResearch session has stopped and you want another orchestrator to review what happened without re-running anything.

## Start Here

Read these in order:

1. `AutoResearch/reports/sessions/session_<id>_<timestamp>.md`
2. `AutoResearch/reports/sessions/session_<id>_<timestamp>.json`
3. `AutoResearch/reports/sessions/session_<id>_<timestamp>_morning_review_prompt.md`

Those three files are the session-end package. They are written by `write_session_summary(...)` when the launcher finishes a session, whether the stop was normal, graceful-after-run, or interrupted.

## What Gets Written At Session End

When a session finishes, AutoResearch does all of this:

- writes a session summary JSON into `AutoResearch/reports/sessions/`
- writes a human-readable session summary Markdown into `AutoResearch/reports/sessions/`
- writes a ready-to-paste morning review prompt into `AutoResearch/reports/sessions/`
- appends a one-line session record into `AutoResearch/reports/session_history.jsonl`
- stores the final session status plus summary paths in `AutoResearch/experiments.db`

The stopped session is still valid review material even if the launcher was interrupted by the user.

## Session Status Meanings

These are the important session-level statuses:

- `completed`: the session ended normally without interruption
- `stopped_after_run`: the user pressed `Esc`, the current run was allowed to finish, and the session then stopped before starting another loop
- `interrupted`: the launcher was stopped immediately, typically with `Ctrl+C`
- `completed` plus a promoted full run: the overnight loop ended and then a full promotion run was executed
- `stopped_after_run_then_full`: graceful stop after the active run, followed by the configured full promotion run
- `interrupted_then_full`: hard interrupt of the loop, followed by the configured full promotion run

For review purposes:

- `stopped_after_run` is a clean partial night
- `interrupted` is a less clean stop and may leave a final in-progress run without a completed payload

## Run Kind Meanings

Experiment rows can have these run kinds:

- `fast`: normal exploration run
- `validation-fix`: a replay/diagnostic run triggered by a genuinely broken prior run
- `validation-improve`: a focused follow-up run on a clearly promising fast result
- `full`: the promoted long run at the end of the session

Current expected policy:

- the first three healthy exploration loops are `fast`
- `validation-fix` only exists to investigate broken runs
- `validation-improve` only exists to test whether a clearly better fast result holds up under more rigor

## Primary Review Files

### `AutoResearch/reports/sessions/`

This is the most important folder for morning review.

- `session_*.md`: condensed human summary
- `session_*.json`: machine-readable summary with leaderboard, notes, and artifact references
- `session_*_morning_review_prompt.md`: direct prompt to hand to another orchestrator/model

If you only give one file to another orchestrator, give it this guide first and then the newest `session_*.md` or `session_*_morning_review_prompt.md`.

### `AutoResearch/reports/session_history.jsonl`

One JSON line per completed session summary. Good for quickly comparing nights.

Useful fields:

- `session_id`
- `started_at`
- `ended_at`
- `status`
- `experiment_count`
- `best_experiment_name`
- `best_holdout_r2`
- `best_cv_rmse`

### `AutoResearch/reports/nightly_log.md`

Human-readable event log for the night. This is the best narrative view of what happened in order.

Look here for:

- session start
- planner self-check
- fast run completions
- validation completions or skips
- full promotion completion
- session completion

### `AutoResearch/reports/notes.jsonl`

Structured per-run notes. These are generated from experiment outcomes and diagnostics.

Common note types:

- `new_best`
- `feature_bias`
- `diagnostic_summary`
- `failure`
- `warning`
- `suspicious_issue`

Important interpretation:

- `new_best` means session-relative leader, not necessarily globally good
- `diagnostic_summary` is the easiest per-run explanation of what looked good, bad, or suspicious
- `failure` and `suspicious_issue` deserve follow-up before trusting the run

### `AutoResearch/reports/debug_trace.jsonl`

Structured debug trace. Use this when the terminal output was unclear or the launcher crashed.

Common event types:

- `planner_self_check_passed`
- `planner_failure`
- `fast_experiment_completed`
- `validation_improvement_completed`
- `issue_validation_completed`
- `experiment_diagnostics`
- `session_completed`
- `launcher_interrupted`

This file is the best source for the true failure path when something breaks at runtime.

## Raw Run Logs

### `AutoResearch/logs/autoresearch/`

One set of raw logs per experiment name.

Common files:

- `*.planner_prompt.md`: exact planning prompt sent to the planner
- `*.planner_response.txt`: raw planner response
- `*.stdout.log`: captured train-process stdout
- `*.stderr.log`: captured train-process stderr

Use these when you need to answer:

- what exactly the planner was asked
- what the model actually returned
- whether the trainer printed warnings or crashed

## Database Of Record

### `AutoResearch/experiments.db`

This SQLite database is the authoritative structured record for the session.

Important tables:

- `sessions`: one row per AutoResearch session
- `experiments`: one row per fast, validation, or full run
- `notes`: structured note records tied to sessions/experiments
- `issue_validations`: records of fix-validation attempts and outcomes

Useful experiment fields:

- `mode`
- `status`
- `experiment_name`
- `config_json`
- `holdout_r2`
- `cv_rmse`
- `output_dir`
- `summary_path`
- `stdout_path`
- `stderr_path`
- `planner_type`
- `planner_prompt_path`
- `planner_response_path`
- `parent_experiment_id`

## Model Artifact Locations

### `data/models/autoresearch-away-runs-*`

Each experiment writes to its own output directory.

Typical files inside:

- `run_count_training_run_*.json`: run-level summary payload
- `full_game_away_runs_model_training_run_*.json`: model-level training summary
- `full_game_away_runs_model_*.metadata.json`: saved model metadata
- `full_game_away_runs_model_*.joblib`: serialized model
- `optuna_studies.db`: Optuna study database for that experiment

These directories are the source of truth for the actual trained artifact and training summaries.

## How To Judge The Night

Use this order:

1. Find the newest `session_*.md` and `session_*.json`.
2. Identify the session status.
3. Read the best exploration run and leaderboard.
4. Check `notes.jsonl` for `failure`, `warning`, or `suspicious_issue`.
5. If something looks broken, inspect `debug_trace.jsonl`.
6. If a specific run needs deeper inspection, open its `output_dir` and matching files in `AutoResearch/logs/autoresearch/`.

## What "Good" And "Broken" Mean

Treat a run as healthy if:

- it has a completed experiment row
- `status` is `succeeded`
- holdout and CV metrics are present
- model artifacts were written

Treat a run as broken if any of these happen:

- non-zero process exit
- payload parse failure
- missing or unreadable artifacts
- impossible or missing metrics
- interrupted run with no completed result payload

Poor metrics do not mean the run is broken. They only mean the configuration was weak.

## Metric Reading Order

For the overnight away-runs search, review metrics in this order:

1. holdout `R2`
2. holdout Poisson deviance
3. holdout RMSE
4. CV RMSE / CV metric as supporting evidence

Use "clearly better than nearby runs" as the threshold for follow-up, not formal statistical significance.

## Recommended Instruction For Another Orchestrator

Use something like:

`Read AutoResearch/AUTORESEARCH_SESSION_REVIEW.md first. Then read the newest files in AutoResearch/reports/sessions/: the session markdown summary, the session JSON summary, and the morning review prompt. If the session looks broken or suspicious, use AutoResearch/reports/debug_trace.jsonl and AutoResearch/logs/autoresearch/ for the failure path.`
