"""Pydantic v2 schemas for MLB prediction dashboard."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class RunSummary(BaseModel):
    """Summary-level run data for list views."""

    model_config = ConfigDict(from_attributes=True)

    # Run identifiers
    experiment_name: str
    summary_path: str
    run_kind: str
    model_name: str
    target_column: str
    model_version: str
    variant: str
    run_timestamp: str

    # Holdout / metadata
    holdout_season: int
    feature_column_count: Optional[int] = None

    # Performance metrics (all optional for training runs that may lack some metrics)
    accuracy: Optional[float] = None
    log_loss: Optional[float] = None
    roc_auc: Optional[float] = None
    brier: Optional[float] = None
    ece: Optional[float] = None
    reliability_gap: Optional[float] = None

    # Delta metrics (improvement vs previous)
    delta_vs_prev_roc_auc: Optional[float] = None
    delta_vs_prev_log_loss: Optional[float] = None
    delta_vs_prev_brier: Optional[float] = None
    delta_vs_prev_accuracy: Optional[float] = None

    # Comparison deltas
    comparison_brier_delta: Optional[float] = None
    comparison_log_loss_delta: Optional[float] = None
    comparison_roc_auc_delta: Optional[float] = None
    comparison_accuracy_delta: Optional[float] = None

    # Best flags
    is_best_accuracy: bool = False
    is_best_log_loss: bool = False
    is_best_roc_auc: bool = False
    is_best_brier: bool = False


class FeatureImportanceItem(BaseModel):
    """Single feature importance entry."""

    model_config = ConfigDict(from_attributes=True)

    feature: str
    importance: float


class BinItem(BaseModel):
    """Single bin in reliability diagram."""

    model_config = ConfigDict(from_attributes=True)

    bin_index: int
    predicted_mean: float
    true_fraction: float
    count: int


class RunDetail(RunSummary):
    """Extended run data with rich metadata for detail views."""

    # Rich metadata
    feature_importance: Optional[List[FeatureImportanceItem]] = None
    best_params: Optional[Dict[str, Any]] = None
    reliability_diagram: Optional[List[BinItem]] = None
    quality_gates: Optional[Dict[str, Any]] = None
    meta_feature_columns: Optional[List[str]] = None
    calibration_method: Optional[str] = None
    train_row_count: Optional[int] = None
    holdout_row_count: Optional[int] = None
    stacking_metrics: Optional[Dict[str, Any]] = None


class Lane(BaseModel):
    """Lane grouping runs by model/variant."""

    model_config = ConfigDict(from_attributes=True)

    lane_id: str
    model_name: str
    variant: str
    best_run: Optional[RunSummary] = None
    latest_run: Optional[RunSummary] = None


class Promotion(BaseModel):
    """Promotion record for a run."""

    model_config = ConfigDict(from_attributes=True)

    promotion_id: str
    run_id: str
    from_stage: str
    to_stage: str
    promoted_timestamp: str
    metadata: Optional[Dict[str, Any]] = None


class PromotionRequest(BaseModel):
    """Request to promote a run to next stage."""

    model_config = ConfigDict(from_attributes=True)

    run_id: str
    target_stage: str
    reason: Optional[str] = None


class CompareResult(BaseModel):
    """Comparison between two runs."""

    model_config = ConfigDict(from_attributes=True)

    run_a_id: str
    run_b_id: str
    run_a: Optional[RunSummary] = None
    run_b: Optional[RunSummary] = None
    metric_deltas: Dict[str, Optional[float]] = {}
    winner: Optional[str] = None  # "a", "b", or "tie"


class HealthResponse(BaseModel):
    """Health check response."""

    model_config = ConfigDict(from_attributes=True)

    status: str
    version: str
    database_ok: bool
    message: Optional[str] = None


class OverviewResponse(BaseModel):
    """Dashboard overview response."""

    model_config = ConfigDict(from_attributes=True)

    total_runs: int
    active_lanes: int
    best_run: Optional[RunSummary] = None
    latest_run: Optional[RunSummary] = None
    recent_runs: List[RunSummary] = []


class SlatePrediction(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    game_pk: int
    model_version: str
    full_game_ml_home_prob: Optional[float] = None
    full_game_ml_away_prob: Optional[float] = None
    full_game_rl_home_prob: Optional[float] = None
    full_game_rl_away_prob: Optional[float] = None
    full_game_total_over_prob: Optional[float] = None
    full_game_total_under_prob: Optional[float] = None
    f5_ml_home_prob: float
    f5_ml_away_prob: float
    f5_rl_home_prob: float
    f5_rl_away_prob: float
    f5_total_over_prob: Optional[float] = None
    f5_total_under_prob: Optional[float] = None
    projected_full_game_home_runs: Optional[float] = None
    projected_full_game_away_runs: Optional[float] = None
    projected_full_game_total_runs: Optional[float] = None
    projected_full_game_home_margin: Optional[float] = None
    projected_f5_home_runs: Optional[float] = None
    projected_f5_away_runs: Optional[float] = None
    projected_f5_total_runs: Optional[float] = None
    projected_f5_home_margin: Optional[float] = None
    predicted_at: str


class SlateDecision(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    game_pk: int
    market_type: str
    side: str
    source_model: Optional[str] = None
    source_model_version: Optional[str] = None
    book_name: Optional[str] = None
    model_probability: float
    fair_probability: float
    edge_pct: float
    ev: float
    is_positive_ev: bool
    kelly_stake: float
    odds_at_bet: Optional[int] = None
    line_at_bet: Optional[float] = None
    result: Optional[str] = None
    settled_at: Optional[str] = None
    profit_loss: Optional[float] = None


class SlateInputStatus(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    home_lineup_available: bool = False
    home_lineup_confirmed: bool = False
    home_lineup_source: Optional[str] = None
    away_lineup_available: bool = False
    away_lineup_confirmed: bool = False
    away_lineup_source: Optional[str] = None
    odds_available: bool = False
    odds_books: List[str] = []
    f5_odds_estimated: bool = False
    f5_odds_sources: List[str] = []
    bet365_f5_ml_home_odds: Optional[int] = None
    bet365_f5_ml_away_odds: Optional[int] = None
    consensus_f5_ml_home_odds: Optional[int] = None
    consensus_f5_ml_away_odds: Optional[int] = None
    bet365_f5_rl_home_point: Optional[float] = None
    bet365_f5_rl_home_odds: Optional[int] = None
    bet365_f5_rl_away_point: Optional[float] = None
    bet365_f5_rl_away_odds: Optional[int] = None
    consensus_f5_rl_home_point: Optional[float] = None
    consensus_f5_rl_home_odds: Optional[int] = None
    consensus_f5_rl_away_point: Optional[float] = None
    consensus_f5_rl_away_odds: Optional[int] = None
    full_game_odds_available: bool = False
    full_game_odds_books: List[str] = []
    full_game_home_ml: Optional[int] = None
    full_game_home_ml_book: Optional[str] = None
    full_game_away_ml: Optional[int] = None
    full_game_away_ml_book: Optional[str] = None
    bet365_full_game_home_ml: Optional[int] = None
    bet365_full_game_away_ml: Optional[int] = None
    consensus_full_game_home_ml: Optional[int] = None
    consensus_full_game_away_ml: Optional[int] = None
    full_game_home_spread: Optional[float] = None
    full_game_home_spread_odds: Optional[int] = None
    full_game_home_spread_book: Optional[str] = None
    full_game_away_spread: Optional[float] = None
    full_game_away_spread_odds: Optional[int] = None
    full_game_away_spread_book: Optional[str] = None
    bet365_full_game_home_spread: Optional[float] = None
    bet365_full_game_home_spread_odds: Optional[int] = None
    bet365_full_game_away_spread: Optional[float] = None
    bet365_full_game_away_spread_odds: Optional[int] = None
    consensus_full_game_home_spread: Optional[float] = None
    consensus_full_game_home_spread_odds: Optional[int] = None
    consensus_full_game_away_spread: Optional[float] = None
    consensus_full_game_away_spread_odds: Optional[int] = None
    weather_available: bool = False


class SlateGame(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    game_pk: int
    matchup: str
    status: str
    game_status: Optional[str] = None
    is_completed: bool = False
    prediction: Optional[SlatePrediction] = None
    selected_decision: Optional[SlateDecision] = None
    forced_decision: Optional[SlateDecision] = None
    no_pick_reason: Optional[str] = None
    error_message: Optional[str] = None
    notified: bool = False
    paper_fallback: bool = False
    input_status: Optional[SlateInputStatus] = None
    narrative: Optional[str] = None


class SlateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    run_id: str
    pipeline_date: str
    mode: str
    dry_run: bool
    model_version: str
    pick_count: int
    no_pick_count: int
    error_count: int
    notification_type: str
    games: List[SlateGame] = []


class LiveSeasonSummaryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    season: int
    tracked_games: int
    settled_games: int
    picks: int
    graded_picks: int
    wins: int
    losses: int
    pushes: int
    no_picks: int
    errors: int
    paper_fallback_picks: int
    flat_profit_units: float
    flat_roi: Optional[float] = None
    play_of_day_count: int
    play_of_day_graded_picks: int
    play_of_day_wins: int
    play_of_day_losses: int
    play_of_day_pushes: int
    play_of_day_profit_units: float
    play_of_day_roi: Optional[float] = None
    forced_picks: int
    forced_graded_picks: int
    forced_wins: int
    forced_losses: int
    forced_pushes: int
    forced_profit_units: float
    forced_roi: Optional[float] = None
    f5_ml_accuracy: Optional[float] = None
    f5_ml_brier: Optional[float] = None
    f5_ml_log_loss: Optional[float] = None
    f5_rl_accuracy: Optional[float] = None
    f5_rl_brier: Optional[float] = None
    f5_rl_log_loss: Optional[float] = None
    latest_capture_at: Optional[str] = None


class LiveSeasonGameResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    season: int
    pipeline_date: str
    game_pk: int
    matchup: str
    run_id: str
    captured_at: str
    model_version: Optional[str] = None
    status: str
    paper_fallback: bool
    f5_ml_home_prob: Optional[float] = None
    f5_ml_away_prob: Optional[float] = None
    f5_rl_home_prob: Optional[float] = None
    f5_rl_away_prob: Optional[float] = None
    selected_market_type: Optional[str] = None
    selected_side: Optional[str] = None
    source_model: Optional[str] = None
    source_model_version: Optional[str] = None
    is_play_of_day: bool = False
    play_of_day_score: Optional[float] = None
    book_name: Optional[str] = None
    odds_at_bet: Optional[int] = None
    line_at_bet: Optional[float] = None
    fair_probability: Optional[float] = None
    model_probability: Optional[float] = None
    edge_pct: Optional[float] = None
    ev: Optional[float] = None
    kelly_stake: Optional[float] = None
    forced_market_type: Optional[str] = None
    forced_side: Optional[str] = None
    forced_source_model: Optional[str] = None
    forced_source_model_version: Optional[str] = None
    forced_book_name: Optional[str] = None
    forced_odds_at_bet: Optional[int] = None
    forced_line_at_bet: Optional[float] = None
    forced_fair_probability: Optional[float] = None
    forced_model_probability: Optional[float] = None
    forced_edge_pct: Optional[float] = None
    forced_ev: Optional[float] = None
    forced_kelly_stake: Optional[float] = None
    no_pick_reason: Optional[str] = None
    error_message: Optional[str] = None
    actual_status: Optional[str] = None
    actual_f5_home_score: Optional[int] = None
    actual_f5_away_score: Optional[int] = None
    settled_result: Optional[str] = None
    flat_profit_loss: Optional[float] = None
    forced_settled_result: Optional[str] = None
    forced_flat_profit_loss: Optional[float] = None
    settled_at: Optional[str] = None
    narrative: Optional[str] = None
