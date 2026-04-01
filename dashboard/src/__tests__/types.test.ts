/**
 * Type-level smoke tests — verify key interfaces are structurally sound
 * by creating sample objects and accessing critical fields.
 */
import { describe, it, expect } from 'vitest';
import type { LatestRun, RunHistoryIndex } from '../types';

describe('Type smoke tests', () => {
  it('LatestRun object allows field access on all key paths', () => {
    const run: LatestRun = {
      run_type: 'research',
      run_id: 'run-type-test',
      tracked_at: '2026-03-30T00:00:00Z',
      run_label: 'Type Test',
      hypothesis: 'Verify types compile.',
      benchmark_status: 'candidate',
      benchmark_label: 'baseline-v1',
      workflow_config: {
        training_data_path: '/data/train.parquet',
        start_year: 2018,
        end_year: 2025,
        holdout_season: 2025,
        folds: 5,
        feature_selection_mode: 'auto',
        forced_delta_count: 0,
        xgb_workers: 4,
        enable_market_priors: false,
        historical_odds_db: '/data/odds.db',
        historical_market_book: null,
        mu_delta_mode: 'fitted',
        stage3_experiment: 'exp-s3',
        stage4_experiment: 'exp-s4',
        stage3_research_lane_name: 'distribution',
        stage4_research_lane_name: 'mcmc',
      },
      artifacts: {
        stage3_report_path: '/r/s3.json',
        stage3_vs_control_path: '/r/s3c.json',
        stage4_report_path: '/r/s4.json',
        stage4_vs_control_path: '/r/s4c.json',
        stage4_vs_stage3_path: '/r/s4s3.json',
        stage3_walk_forward_path: '/r/s3wf.json',
        stage4_walk_forward_path: '/r/s4wf.json',
        dual_view_path: '/r/dv.json',
      },
      summary: {
        headline_result: 'Works',
        what_changed: 'Nothing',
        best_research_lane_key: 'stage4',
        best_research_lane_label: 'Stage 4',
        promoted_second_opinion_lane_key: null,
        production_promotable_lane_key: null,
        is_profitable: null,
        is_more_accurate_than_control: false,
        is_more_stable_than_control: false,
        main_reason_failed: 'Not enough data',
        next_action_hint: 'Gather more data',
        stage3: {
          model_version: 'v1', research_lane_name: 'dist',
          rmse: 1.0, mae: 0.8, mean_crps: 0.5, mean_negative_log_score: 0.7,
          market_data_coverage_pct: 90, source_origins: ['src'], source_db_paths: ['/db'],
          roi: null, net_units: null, bet_count: 0, market_anchor_coverage: 80,
          beats_control_on_crps: false, beats_control_on_negative_log_score: false,
          catastrophic_regression: false,
          delta_vs_control: {
            mean_crps: null, mean_negative_log_score: null,
            rmse: null, delta_roi: null, delta_net_units: null,
          },
          delta_vs_prior_lane: null,
        },
        stage4: {
          model_version: 'v1', research_lane_name: 'mcmc',
          rmse: 0.9, mae: 0.7, mean_crps: 0.45, mean_negative_log_score: 0.65,
          market_data_coverage_pct: 92, source_origins: ['src'], source_db_paths: ['/db'],
          roi: 0.05, net_units: 2.0, bet_count: 10, market_anchor_coverage: 85,
          beats_control_on_crps: true, beats_control_on_negative_log_score: true,
          catastrophic_regression: false,
          delta_vs_control: {
            mean_crps: -0.05, mean_negative_log_score: -0.05,
            rmse: -0.1, delta_roi: 0.05, delta_net_units: 2.0,
          },
          delta_vs_prior_lane: null,
        },
      },
      benchmark_compare: {
        available: false,
        benchmark_label: '',
        benchmark_run_id: '',
        stage3: {
          mean_crps: null, mean_negative_log_score: null,
          rmse: null, delta_roi: null, delta_net_units: null,
        },
        stage4: {
          mean_crps: null, mean_negative_log_score: null,
          rmse: null, delta_roi: null, delta_net_units: null,
        },
      },
    };

    // Verify deep field access works at runtime
    expect(run.run_id).toBe('run-type-test');
    expect(run.summary.stage4.mean_crps).toBe(0.45);
    expect(run.summary.production_promotable_lane_key).toBeNull();
    expect(run.benchmark_compare.available).toBe(false);
    expect(run.workflow_config.folds).toBe(5);
  });

  it('RunHistoryIndex object supports row iteration and field access', () => {
    const index: RunHistoryIndex = {
      generated_at: '2026-03-30T00:00:00Z',
      row_count: 2,
      runs: [
        {
          run_id: 'r1', tracked_at: '2026-03-29T00:00:00Z',
          run_label: 'Run A', benchmark_status: 'benchmark',
          benchmark_label: 'baseline-v1', headline_result: 'Good',
          best_research_lane_key: 's4', production_promotable_lane_key: 's4',
          stage3_mean_crps: 0.50, stage3_mean_negative_log_score: 0.80,
          stage3_rmse: 1.3, stage3_roi: 0.10,
          stage4_mean_crps: 0.45, stage4_mean_negative_log_score: 0.75,
          stage4_rmse: 1.2, stage4_roi: 0.15,
          is_profitable: true, is_more_accurate_than_control: true,
        },
        {
          run_id: 'r2', tracked_at: '2026-03-30T00:00:00Z',
          run_label: 'Run B', benchmark_status: 'candidate',
          benchmark_label: 'baseline-v1', headline_result: 'Worse',
          best_research_lane_key: 's3', production_promotable_lane_key: null,
          stage3_mean_crps: 0.55, stage3_mean_negative_log_score: 0.85,
          stage3_rmse: 1.4, stage3_roi: null,
          stage4_mean_crps: 0.52, stage4_mean_negative_log_score: 0.82,
          stage4_rmse: 1.35, stage4_roi: null,
          is_profitable: false, is_more_accurate_than_control: false,
        },
      ],
    };

    expect(index.row_count).toBe(2);
    expect(index.runs).toHaveLength(2);
    expect(index.runs[0].run_label).toBe('Run A');
    expect(index.runs[1].production_promotable_lane_key).toBeNull();
    expect(index.runs[0].stage4_roi).toBe(0.15);
    expect(index.runs[1].is_profitable).toBe(false);
  });
});
