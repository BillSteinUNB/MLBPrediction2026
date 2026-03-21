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
