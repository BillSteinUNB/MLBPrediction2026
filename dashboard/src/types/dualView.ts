/**
 * TypeScript interfaces for the dual-view comparison report.
 * These types define side-by-side control vs. research lane comparisons.
 */

/**
 * CalibrationBin - A single bin in a calibration histogram.
 */
export interface CalibrationBin {
  bin_index: number;
  count: number;
  mean_predicted_probability: number;
  empirical_rate: number;
  absolute_error: number;
}

/**
 * CalibrationEvent - Calibration metrics for a single probability event (p_0, p_ge_1, etc.).
 */
export interface CalibrationEvent {
  event: string;
  mean_predicted_probability: number;
  predicted_probability_std: number;
  empirical_rate: number;
  absolute_error: number;
  brier_score: number;
  bin_count: number;
  bins: CalibrationBin[];
}

/**
 * ZeroCalibration - Calibration for zero-count vs. one-or-more-count events.
 */
export interface ZeroCalibration {
  p_0: CalibrationEvent;
  p_ge_1: CalibrationEvent;
}

/**
 * TailCalibration - Calibration for tail events (p_ge_3, p_ge_5, p_ge_10).
 */
export interface TailCalibration {
  p_ge_3: CalibrationEvent;
  p_ge_5: CalibrationEvent;
  p_ge_10: CalibrationEvent;
}

/**
 * IntervalCoverage - Empirical vs. nominal coverage for prediction intervals.
 */
export interface IntervalCoverage {
  central_50: {
    nominal_coverage: number;
    empirical_coverage: number;
    coverage_error: number;
    mean_width: number;
  };
  central_80: {
    nominal_coverage: number;
    empirical_coverage: number;
    coverage_error: number;
    mean_width: number;
  };
  central_95: {
    nominal_coverage: number;
    empirical_coverage: number;
    coverage_error: number;
    mean_width: number;
  };
}

/**
 * PredictionSummary - Point estimates for predicted run distributions.
 */
export interface PredictionSummary {
  mean_predicted_runs: number;
  mean_predicted_p_0: number;
  mean_predicted_p_ge_3: number;
  mean_predicted_p_ge_5: number;
  mean_predicted_p_ge_10: number;
}

/**
 * DistributionMetrics - Full calibration and prediction metrics for a model.
 */
export interface DistributionMetrics {
  mean_crps: number;
  mean_log_score: number;
  mean_negative_log_score: number;
  zero_calibration: ZeroCalibration;
  tail_calibration: TailCalibration;
  interval_coverage: IntervalCoverage;
  prediction_summary: PredictionSummary;
}

/**
 * MeanMetrics - Traditional mean-based accuracy metrics.
 */
export interface MeanMetrics {
  mae: number;
  rmse: number;
  poisson_deviance: number;
  r2: number;
  actual_mean: number;
  predicted_mean: number;
  naive_mean_prediction: number;
  naive_mae: number;
  naive_rmse: number;
  naive_poisson_deviance: number;
  mae_improvement_vs_naive_pct: number;
  rmse_improvement_vs_naive_pct: number;
  poisson_deviance_improvement_vs_naive_pct: number;
}

/**
 * PromotionState - Promotion eligibility and reasoning for a research lane.
 */
export interface PromotionState {
  lane_key: string;
  lane_label: string;
  second_opinion_promoted: boolean;
  production_promotable: boolean;
  lane_status: string;
  summary_reason: string;
  checks: Record<string, unknown>;
}

/**
 * ComparisonToControl - Detailed metrics comparing a research lane to control.
 */
export interface ComparisonToControl {
  stage3_mean_metrics: MeanMetrics;
  control_mean_metrics: MeanMetrics;
  stage3_distribution_metrics: DistributionMetrics;
  control_distribution_metrics: DistributionMetrics;
}

/**
 * LaneSummary - Complete summary for a single lane (control or research).
 * Includes metadata, accuracy metrics, distribution calibration, and promotion state.
 */
export interface LaneSummary {
  lane_key: string;
  lane_label: string;
  lane_kind: string; // e.g. "control", "distribution", "mcmc"
  lane_status: string; // e.g. "control", "exploratory"
  artifact_path: string;
  expected_away_runs: number;
  shutout_probability: number;
  p_away_runs_ge_3: number;
  p_away_runs_ge_5: number;
  mean_metrics: MeanMetrics;
  distribution_metrics: DistributionMetrics;
  promotion_state: PromotionState;
  comparison_to_control?: ComparisonToControl; // Only present for non-control lanes
}

/**
 * SourcePaths - File paths to the reports used to generate this dual view.
 */
export interface SourcePaths {
  current_control_path: string;
  stage3_report_path: string;
  stage3_vs_control_path: string;
  mcmc_report_path: string;
  mcmc_vs_control_path: string;
  mcmc_vs_stage3_path: string;
  stage3_walk_forward_report_path: string;
  mcmc_walk_forward_report_path: string;
}

/**
 * DualView - Side-by-side comparison of control and research lanes.
 * This is the highest-level report summarizing stage 3 and stage 4 results.
 */
export interface DualView {
  generated_at: string;
  source_paths: SourcePaths;
  lane_summaries: LaneSummary[];
}
