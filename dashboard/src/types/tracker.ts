/**
 * TypeScript interfaces for the away-run research tracker.
 * These types define the structure of JSON artifacts from the AutoResearch pipeline.
 */

/**
 * WorkflowConfig - Configuration for a model training/research workflow.
 */
export interface WorkflowConfig {
  training_data_path: string;
  start_year: number;
  end_year: number;
  holdout_season: number;
  folds: number;
  feature_selection_mode: string;
  forced_delta_count: number;
  xgb_workers: number;
  enable_market_priors: boolean;
  historical_odds_db: string;
  historical_market_book: string | null;
  mu_delta_mode: string;
  stage3_experiment: string;
  stage4_experiment: string;
  stage3_research_lane_name: string;
  stage4_research_lane_name: string;
}

/**
 * ArtifactsPaths - File paths to generated evaluation reports.
 */
export interface ArtifactsPaths {
  stage3_report_path: string;
  stage3_vs_control_path: string;
  stage4_report_path: string;
  stage4_vs_control_path: string;
  stage4_vs_stage3_path: string;
  stage3_walk_forward_path: string;
  stage4_walk_forward_path: string;
  dual_view_path: string;
}

/**
 * DeltaBlock - Performance delta metrics comparing two models.
 * All fields are nullable because some comparisons may not be available.
 */
export interface DeltaBlock {
  mean_crps: number | null;
  mean_negative_log_score: number | null;
  rmse: number | null;
  delta_roi: number | null;
  delta_net_units: number | null;
}

/**
 * StageSummary - Performance metrics for a single stage (Stage 3 or Stage 4).
 * Includes accuracy metrics, calibration data, and profitability estimates.
 */
export interface StageSummary {
  model_version: string;
  research_lane_name: string;
  rmse: number;
  mae: number;
  mean_crps: number;
  mean_negative_log_score: number;
  market_data_coverage_pct: number;
  source_origins: string[];
  source_db_paths: string[];
  roi: number | null;
  net_units: number | null;
  bet_count: number;
  market_anchor_coverage: number;
  beats_control_on_crps: boolean;
  beats_control_on_negative_log_score: boolean;
  catastrophic_regression: boolean;
  delta_vs_control: DeltaBlock;
  delta_vs_prior_lane: DeltaBlock | null;
}

/**
 * WorkflowSummary - High-level summary of a workflow run outcome.
 */
export interface WorkflowSummary {
  headline_result: string;
  what_changed: string;
  best_research_lane_key: string;
  best_research_lane_label: string;
  promoted_second_opinion_lane_key: string | null;
  production_promotable_lane_key: string | null;
  is_profitable: boolean | null;
  is_more_accurate_than_control: boolean;
  is_more_stable_than_control: boolean;
  main_reason_failed: string;
  next_action_hint: string;
  stage3: StageSummary;
  stage4: StageSummary;
}

/**
 * BenchmarkCompare - Performance deltas between current run and active benchmark.
 */
export interface BenchmarkCompare {
  available: boolean;
  benchmark_label: string;
  benchmark_run_id: string;
  stage3: DeltaBlock;
  stage4: DeltaBlock;
  notes?: string[];
}

/**
 * LatestRun - The most recent completed workflow run, with full metadata and results.
 */
export interface LatestRun {
  run_type: string;
  run_id: string;
  tracked_at: string; // ISO 8601 timestamp
  run_label: string;
  hypothesis: string;
  benchmark_status: string; // e.g. "candidate", "benchmark"
  benchmark_label: string;
  workflow_config: WorkflowConfig;
  artifacts: ArtifactsPaths;
  summary: WorkflowSummary;
  benchmark_compare: BenchmarkCompare;
}

/**
 * BenchmarkFile - Metadata and summary for the active benchmark run.
 * Subset of LatestRun containing only the benchmark's summary and artifacts.
 */
export interface BenchmarkFile {
  benchmark_label: string;
  benchmark_run_id: string;
  tracked_at: string;
  summary: WorkflowSummary;
  artifacts: ArtifactsPaths;
}

/**
 * RunHistoryRow - Flattened history entry for a single workflow run.
 * Used in run_history_index.json for tabular view of all runs.
 */
export interface RunHistoryRow {
  run_id: string;
  tracked_at: string;
  run_label: string;
  benchmark_status: string;
  benchmark_label: string;
  headline_result: string;
  best_research_lane_key: string;
  production_promotable_lane_key: string | null;
  stage3_mean_crps: number;
  stage3_mean_negative_log_score: number;
  stage3_rmse: number;
  stage3_roi: number | null;
  stage4_mean_crps: number;
  stage4_mean_negative_log_score: number;
  stage4_rmse: number;
  stage4_roi: number | null;
  is_profitable: boolean | null;
  is_more_accurate_than_control: boolean;
}

/**
 * RunHistoryIndex - Index of all workflow runs for ledger and filtering.
 */
export interface RunHistoryIndex {
  generated_at: string;
  row_count: number;
  runs: RunHistoryRow[];
}

/**
 * RecommendedView - Definition of a recommended dashboard view.
 */
export interface RecommendedView {
  view_key: string;
  title: string;
  source: string;
  purpose: string;
}

/**
 * EntitySchema - Description of an entity type (e.g. workflow_run, lane_summary).
 */
export interface EntitySchema {
  primary_key?: string;
  child_path?: string;
  important_fields: string[];
}

/**
 * Semantics - Business logic rules for interpreting tracker data.
 */
export interface Semantics {
  profitability: string;
  accuracy: string;
  benchmarking: string;
  promotion: string;
}

/**
 * FrontendConnection - Manifest describing data sources and views for the frontend.
 * This is the contract between the backend tracker and the React dashboard.
 */
export interface FrontendConnection {
  generated_at: string;
  app_key: string;
  primary_data_sources: {
    latest_run: string;
    benchmark: string;
    run_history_index: string;
    run_history_jsonl: string;
    run_history_csv: string;
    current_dual_view: string;
  };
  recommended_default_views: RecommendedView[];
  entities: {
    workflow_run: EntitySchema;
    lane_summary: EntitySchema;
  };
  semantics: Semantics;
}
