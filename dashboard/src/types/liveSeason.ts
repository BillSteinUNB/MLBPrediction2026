export interface LiveSeasonSummary {
  season: number;
  tracked_games: number;
  settled_games: number;
  picks: number;
  graded_picks: number;
  wins: number;
  losses: number;
  pushes: number;
  no_picks: number;
  errors: number;
  paper_fallback_picks: number;
  flat_profit_units: number;
  flat_roi: number | null;
  official_units_risked: number | null;
  official_profit_units: number | null;
  official_roi: number | null;
  latest_capture_at: string | null;
}

export interface LiveSeasonGame {
  id: number | null;
  season: number;
  pipeline_date: string;
  game_pk: number;
  matchup: string;
  bet_key: string | null;
  captured_at: string;
  tracking_source?: string;
  model_version: string | null;
  status: string;
  paper_fallback: boolean;
  selected_market_type: string | null;
  selected_side: string | null;
  source_model: string | null;
  source_model_version: string | null;
  book_name: string | null;
  odds_at_bet: number | null;
  line_at_bet: number | null;
  fair_probability: number | null;
  model_probability: number | null;
  edge_pct: number | null;
  ev: number | null;
  kelly_stake: number | null;
  bet_units: number | null;
  actual_status: string | null;
  actual_f5_home_score: number | null;
  actual_f5_away_score: number | null;
  actual_final_home_score: number | null;
  actual_final_away_score: number | null;
  settled_result: string | null;
  flat_profit_loss: number | null;
  no_pick_reason: string | null;
  narrative: string | null;
}

export interface LiveSeasonCapture {
  pipeline_date: string;
  season: number;
  captured: boolean;
  already_captured: boolean;
  settled_rows: number;
  tracked_games: number;
  run_id: string | null;
  pick_count: number | null;
  notification_type: string | null;
}

export interface LiveSeasonDashboardData {
  season: number;
  pipeline_date: string;
  release_name: string;
  release_version: string;
  model_display_name: string;
  strategy_name: string;
  strategy_version: string;
  technical_model_version: string;
  research_baseline_label: string;
  policy_summary: string;
  summary: LiveSeasonSummary;
  manual_summary: LiveSeasonSummary;
  forced_summary: LiveSeasonSummary;
  today_games: LiveSeasonGame[];
  historical_games: LiveSeasonGame[];
  manual_today_games: LiveSeasonGame[];
  manual_historical_games: LiveSeasonGame[];
  forced_today_games: LiveSeasonGame[];
  forced_historical_games: LiveSeasonGame[];
  capture: LiveSeasonCapture | null;
}
