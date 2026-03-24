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

export interface SlatePrediction {
  game_pk: number
  model_version: string
  f5_ml_home_prob: number
  f5_ml_away_prob: number
  f5_rl_home_prob: number
  f5_rl_away_prob: number
  projected_f5_home_runs?: number | null
  projected_f5_away_runs?: number | null
  projected_f5_total_runs?: number | null
  projected_f5_home_margin?: number | null
  predicted_at: string
}

export interface SlateDecision {
  game_pk: number
  market_type: string
  side: string
  source_model?: string | null
  source_model_version?: string | null
  book_name?: string | null
  model_probability: number
  fair_probability: number
  edge_pct: number
  ev: number
  is_positive_ev: boolean
  kelly_stake: number
  odds_at_bet?: number | null
  line_at_bet?: number | null
  result?: string | null
  settled_at?: string | null
  profit_loss?: number | null
}

export interface SlateInputStatus {
  home_lineup_available: boolean
  home_lineup_confirmed: boolean
  home_lineup_source?: string | null
  away_lineup_available: boolean
  away_lineup_confirmed: boolean
  away_lineup_source?: string | null
  odds_available: boolean
  odds_books: string[]
  f5_odds_estimated: boolean
  f5_odds_sources: string[]
  full_game_odds_available: boolean
  full_game_odds_books: string[]
  full_game_home_ml?: number | null
  full_game_home_ml_book?: string | null
  full_game_away_ml?: number | null
  full_game_away_ml_book?: string | null
  full_game_home_spread?: number | null
  full_game_home_spread_odds?: number | null
  full_game_home_spread_book?: string | null
  full_game_away_spread?: number | null
  full_game_away_spread_odds?: number | null
  full_game_away_spread_book?: string | null
  weather_available: boolean
}

export interface SlateGame {
  game_pk: number
  matchup: string
  status: string
  prediction?: SlatePrediction | null
  selected_decision?: SlateDecision | null
  forced_decision?: SlateDecision | null
  no_pick_reason?: string | null
  error_message?: string | null
  notified: boolean
  paper_fallback: boolean
  input_status?: SlateInputStatus | null
}

export interface SlateResponse {
  run_id: string
  pipeline_date: string
  mode: string
  dry_run: boolean
  model_version: string
  pick_count: number
  no_pick_count: number
  error_count: number
  notification_type: string
  games: SlateGame[]
}

export interface LiveSeasonSummaryResponse {
  season: number
  tracked_games: number
  settled_games: number
  picks: number
  graded_picks: number
  wins: number
  losses: number
  pushes: number
  no_picks: number
  errors: number
  paper_fallback_picks: number
  flat_profit_units: number
  flat_roi?: number | null
  play_of_day_count: number
  play_of_day_graded_picks: number
  play_of_day_wins: number
  play_of_day_losses: number
  play_of_day_pushes: number
  play_of_day_profit_units: number
  play_of_day_roi?: number | null
  forced_picks: number
  forced_graded_picks: number
  forced_wins: number
  forced_losses: number
  forced_pushes: number
  forced_profit_units: number
  forced_roi?: number | null
  f5_ml_accuracy?: number | null
  f5_ml_brier?: number | null
  f5_ml_log_loss?: number | null
  f5_rl_accuracy?: number | null
  f5_rl_brier?: number | null
  f5_rl_log_loss?: number | null
  latest_capture_at?: string | null
}

export interface LiveSeasonGameResponse {
  season: number
  pipeline_date: string
  game_pk: number
  matchup: string
  run_id: string
  captured_at: string
  model_version?: string | null
  status: string
  paper_fallback: boolean
  f5_ml_home_prob?: number | null
  f5_ml_away_prob?: number | null
  f5_rl_home_prob?: number | null
  f5_rl_away_prob?: number | null
  selected_market_type?: string | null
  selected_side?: string | null
  source_model?: string | null
  source_model_version?: string | null
  is_play_of_day: boolean
  play_of_day_score?: number | null
  book_name?: string | null
  odds_at_bet?: number | null
  line_at_bet?: number | null
  fair_probability?: number | null
  model_probability?: number | null
  edge_pct?: number | null
  ev?: number | null
  kelly_stake?: number | null
  forced_market_type?: string | null
  forced_side?: string | null
  forced_source_model?: string | null
  forced_source_model_version?: string | null
  forced_book_name?: string | null
  forced_odds_at_bet?: number | null
  forced_line_at_bet?: number | null
  forced_fair_probability?: number | null
  forced_model_probability?: number | null
  forced_edge_pct?: number | null
  forced_ev?: number | null
  forced_kelly_stake?: number | null
  no_pick_reason?: string | null
  error_message?: string | null
  actual_status?: string | null
  actual_f5_home_score?: number | null
  actual_f5_away_score?: number | null
  settled_result?: string | null
  flat_profit_loss?: number | null
  forced_settled_result?: string | null
  forced_flat_profit_loss?: number | null
  settled_at?: string | null
}
