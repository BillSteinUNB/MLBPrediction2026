/**
 * TypeScript interfaces for Picks (PICS) data contracts.
 * These types define the structure of JSON artifacts from the betting engine,
 * including daily picks and play-of-the-day selections.
 */

/**
 * MarketView - Odds and probability information for a single betting market.
 * Nullable fields indicate optional calculations or missing market data.
 */
export interface MarketView {
  /** ML/RL model probability for home team */
  ml_home_prob: number | null;
  /** ML/RL model probability for away team */
  ml_away_prob: number | null;
  /** Run-line model probability for home team */
  rl_home_prob: number | null;
  /** Run-line model probability for away team */
  rl_away_prob: number | null;
  /** Home team's American odds (e.g., -110, +150) */
  home_odds: number | null;
  /** Away team's American odds */
  away_odds: number | null;
  /** Implied probability derived from home odds (0.0-1.0) */
  home_implied_prob: number | null;
  /** Implied probability derived from away odds (0.0-1.0) */
  away_implied_prob: number | null;
  /** De-vigged edge % for home side (e.g., 0.03 = 3%) */
  home_edge_pct: number | null;
  /** De-vigged edge % for away side */
  away_edge_pct: number | null;
  /** Expected value for home bet (probability * odds) */
  home_ev: number | null;
  /** Expected value for away bet */
  away_ev: number | null;
  /** Signal strength/confidence for home side (0.0-1.0) */
  home_signal: number | null;
  /** Signal strength/confidence for away side (0.0-1.0) */
  away_signal: number | null;
}

/**
 * GamePick - Complete prediction and decision data for a single game.
 * Includes projections, market views, and the selected pick (if any).
 */
export interface GamePick {
  /** MLB official game ID (e.g., 748328) */
  game_pk: number;
  /** Game date in ISO 8601 format (YYYY-MM-DD) */
  game_date: string;
  /** Home team abbreviation (e.g., "NYY") */
  home_team: string;
  /** Away team abbreviation (e.g., "BOS") */
  away_team: string;
  /** Game status (e.g., "Scheduled", "In Progress", "Final") */
  game_status: string;
  /** Projected home team runs in first 5 innings */
  projected_home_runs: number | null;
  /** Projected away team runs in first 5 innings */
  projected_away_runs: number | null;
  /** Projected total runs in first 5 innings */
  projected_total_runs: number | null;
  /** Probability home team covers the run line */
  cover_home_prob: number | null;
  /** Probability away team covers the run line */
  cover_away_prob: number | null;
  /** ML and RL market data for home/away sides */
  market_view: MarketView;
  /** Betting market selected (e.g., "f5_ml", "f5_rl") */
  selected_market_type: string | null;
  /** Selected side: "home" or "away" */
  selected_side: string | null;
  /** American odds on selected pick */
  selected_odds: number | null;
  /** De-vigged edge % on selected pick */
  selected_edge_pct: number | null;
  /** Quarter-Kelly stake recommended (as % of bankroll) */
  selected_kelly_stake: number | null;
}

/**
 * PlayOfTheDayData - Best single pick from the day's game slate.
 * Includes game details, market context, and sizing recommendation.
 */
export interface PlayOfTheDayData {
  /** ISO 8601 date when play was generated (YYYY-MM-DD) */
  pick_date: string;
  /** MLB official game ID */
  game_pk: number;
  /** Game date in ISO 8601 format */
  game_date: string;
  /** Home team abbreviation */
  home_team: string;
  /** Away team abbreviation */
  away_team: string;
  /** Betting market type (e.g., "f5_ml", "f5_rl") */
  market_type: string;
  /** Selected side: "home" or "away" */
  side: string;
  /** American odds on the pick */
  odds: number;
  /** Model's probability estimate (0.0-1.0) */
  model_probability: number;
  /** Implied probability from odds (0.0-1.0) */
  implied_probability: number;
  /** De-vigged edge % (e.g., 0.05 = 5%) */
  edge_pct: number;
  /** Suggested Quarter-Kelly stake size as % of bankroll */
  kelly_stake_pct: number;
  /** Explanation/narrative for the pick */
  narrative: string | null;
}

/**
 * DailyPicsData - Container for all picks from a single day's game slate.
 * Includes metadata, all games, and optional play-of-the-day highlight.
 */
export interface DailyPicsData {
  /** ISO 8601 date when picks were generated (YYYY-MM-DD) */
  generated_date: string;
  /** Model version that produced these picks (e.g., "v2.3.1") */
  model_version: string;
  /** Total number of games in the slate */
  total_games: number;
  /** Number of games with qualifying picks */
  picks_count: number;
  /** All game predictions and picks */
  games: GamePick[];
  /** Best single pick of the day (if available) */
  play_of_the_day: PlayOfTheDayData | null;
}
