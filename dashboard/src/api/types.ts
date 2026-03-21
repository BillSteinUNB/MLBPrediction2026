/**
 * TypeScript interfaces matching backend Pydantic schemas
 */

/**
 * Single feature importance entry
 */
export interface FeatureImportanceItem {
  feature: string
  importance: number
}

/**
 * Single bin in reliability diagram
 */
export interface BinItem {
  bin_index: number
  predicted_mean: number
  true_fraction: number
  count: number
}

/**
 * Summary-level run data for list views
 */
export interface RunSummary {
  // Run identifiers
  experiment_name: string
  summary_path: string
  run_kind: string
  model_name: string
  target_column: string
  model_version: string
  variant: string
  run_timestamp: string

  // Holdout / metadata
  holdout_season: number
  feature_column_count?: number | null

  // Performance metrics (all optional for training runs)
  accuracy?: number | null
  log_loss?: number | null
  roc_auc?: number | null
  brier?: number | null
  ece?: number | null
  reliability_gap?: number | null

  // Delta metrics (improvement vs previous)
  delta_vs_prev_roc_auc?: number | null
  delta_vs_prev_log_loss?: number | null
  delta_vs_prev_brier?: number | null
  delta_vs_prev_accuracy?: number | null

  // Comparison deltas
  comparison_brier_delta?: number | null
  comparison_log_loss_delta?: number | null
  comparison_roc_auc_delta?: number | null
  comparison_accuracy_delta?: number | null

  // Best flags
  is_best_accuracy: boolean
  is_best_log_loss: boolean
  is_best_roc_auc: boolean
  is_best_brier: boolean
}

/**
 * Extended run data with rich metadata for detail views
 */
export interface RunDetail extends RunSummary {
  // Rich metadata
  feature_importance?: FeatureImportanceItem[] | null
  best_params?: Record<string, unknown> | null
  reliability_diagram?: BinItem[] | null
  quality_gates?: Record<string, unknown> | null
  meta_feature_columns?: string[] | null
  calibration_method?: string | null
  train_row_count?: number | null
  holdout_row_count?: number | null
  stacking_metrics?: Record<string, unknown> | null
}

/**
 * Lane grouping runs by model/variant
 */
export interface Lane {
  lane_id: string
  model_name: string
  variant: string
  best_run?: RunSummary | null
  latest_run?: RunSummary | null
}

/**
 * Promotion record for a run
 */
export interface Promotion {
  promotion_id: string
  run_id: string
  from_stage: string
  to_stage: string
  promoted_timestamp: string
  metadata?: Record<string, unknown> | null
}

/**
 * Request to promote a run to next stage
 */
export interface PromotionRequest {
  run_id: string
  target_stage: string
  reason?: string | null
}

/**
 * Comparison between two runs
 */
export interface CompareResult {
  run_a_id: string
  run_b_id: string
  run_a?: RunSummary | null
  run_b?: RunSummary | null
  metric_deltas: Record<string, number | null>
  winner?: string | null  // "a", "b", or "tie"
}

/**
 * Health check response
 */
export interface HealthResponse {
  status: string
  version: string
  database_ok: boolean
  message?: string | null
}

/**
 * Dashboard overview response
 */
export interface OverviewResponse {
  total_runs: number
  active_lanes: number
  best_run?: RunSummary | null
  latest_run?: RunSummary | null
  recent_runs: RunSummary[]
}
