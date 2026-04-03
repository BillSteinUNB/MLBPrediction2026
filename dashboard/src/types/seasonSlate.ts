export interface SeasonSlateDecision {
  game_pk: number;
  market_type: string;
  side: string;
  source_model: string | null;
  source_model_version: string | null;
  book_name: string | null;
  model_probability: number;
  fair_probability: number;
  edge_pct: number;
  ev: number;
  is_positive_ev: boolean;
  kelly_stake: number;
  odds_at_bet: number | null;
  line_at_bet: number | null;
}

export interface SeasonSlateInputStatus {
  full_game_home_ml?: number | null;
  full_game_home_ml_book?: string | null;
  full_game_away_ml?: number | null;
  full_game_away_ml_book?: string | null;
  bet365_full_game_home_ml?: number | null;
  bet365_full_game_away_ml?: number | null;
  full_game_home_spread?: number | null;
  full_game_home_spread_odds?: number | null;
  full_game_home_spread_book?: string | null;
  full_game_away_spread?: number | null;
  full_game_away_spread_odds?: number | null;
  full_game_away_spread_book?: string | null;
  bet365_full_game_home_spread?: number | null;
  bet365_full_game_home_spread_odds?: number | null;
  bet365_full_game_away_spread?: number | null;
  bet365_full_game_away_spread_odds?: number | null;
  full_game_total?: number | null;
  full_game_total_over_odds?: number | null;
  full_game_total_under_odds?: number | null;
  full_game_total_book?: string | null;
  bet365_full_game_total?: number | null;
  bet365_full_game_total_over_odds?: number | null;
  bet365_full_game_total_under_odds?: number | null;
}

export interface SeasonSlatePrediction {
  game_pk: number;
  model_version: string;
  full_game_ml_home_prob: number | null;
  full_game_ml_away_prob: number | null;
  full_game_rl_home_prob: number | null;
  full_game_rl_away_prob: number | null;
  full_game_total_over_prob: number | null;
  full_game_total_under_prob: number | null;
  f5_ml_home_prob: number;
  f5_ml_away_prob: number;
  f5_rl_home_prob: number;
  f5_rl_away_prob: number;
  f5_total_over_prob: number | null;
  f5_total_under_prob: number | null;
  projected_full_game_total_runs: number | null;
  predicted_at: string;
}

export interface SeasonSlateGame {
  game_pk: number;
  matchup: string;
  status: string;
  game_status: string | null;
  is_completed: boolean;
  prediction: SeasonSlatePrediction | null;
  selected_decision: SeasonSlateDecision | null;
  forced_decision: SeasonSlateDecision | null;
  no_pick_reason: string | null;
  error_message: string | null;
  paper_fallback: boolean;
  narrative: string | null;
  candidate_decisions: SeasonSlateDecision[];
  input_status?: SeasonSlateInputStatus | null;
}

export interface SeasonSlateData {
  run_id: string;
  pipeline_date: string;
  mode: string;
  dry_run: boolean;
  model_version: string;
  release_name?: string;
  release_version?: string;
  model_display_name?: string;
  strategy_name?: string;
  strategy_version?: string;
  technical_model_version?: string;
  research_baseline_label?: string;
  policy_summary?: string;
  pick_count: number;
  no_pick_count: number;
  error_count: number;
  notification_type: string;
  games: SeasonSlateGame[];
}
