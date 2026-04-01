/**
 * Smoke tests — verify every page renders without crashing.
 *
 * Strategy: mock global.fetch so each hook (useLatestRun, useRunHistory,
 * useBenchmark, useDualView) receives minimal but structurally valid JSON.
 * Then render each page inside a BrowserRouter and assert basic content.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';

import LatestRunPage from '../pages/LatestRunPage';
import StageCardsPage from '../pages/StageCardsPage';
import BenchmarkComparePage from '../pages/BenchmarkComparePage';
import RunLedgerPage from '../pages/RunLedgerPage';
import BestRunsPage from '../pages/BestRunsPage';
import PromotionSummaryPage from '../pages/PromotionSummaryPage';

import type { LatestRun, RunHistoryIndex, BenchmarkFile } from '../types';
import type { DualView } from '../types/dualView';

/* ── Shared mock fragments ─────────────────────────────────────────── */

const DELTA_BLOCK = {
  mean_crps: -0.002,
  mean_negative_log_score: -0.003,
  rmse: -0.01,
  delta_roi: 0.05,
  delta_net_units: 1.2,
};

const STAGE_SUMMARY = {
  model_version: 'v1.0.0',
  research_lane_name: 'test-lane',
  rmse: 1.23,
  mae: 0.98,
  mean_crps: 0.456,
  mean_negative_log_score: 0.789,
  market_data_coverage_pct: 95.0,
  source_origins: ['statcast'],
  source_db_paths: ['/data/test.db'],
  roi: 0.12,
  net_units: 3.5,
  bet_count: 42,
  market_anchor_coverage: 88.0,
  beats_control_on_crps: true,
  beats_control_on_negative_log_score: true,
  catastrophic_regression: false,
  delta_vs_control: DELTA_BLOCK,
  delta_vs_prior_lane: null,
};

const MOCK_LATEST_RUN: LatestRun = {
  run_type: 'research',
  run_id: 'run-001',
  tracked_at: '2026-03-30T12:00:00Z',
  run_label: 'Smoke Test Run',
  hypothesis: 'Testing that pages render without crashing.',
  benchmark_status: 'candidate',
  benchmark_label: 'baseline-v1',
  workflow_config: {
    training_data_path: '/data/training.parquet',
    start_year: 2018,
    end_year: 2025,
    holdout_season: 2025,
    folds: 5,
    feature_selection_mode: 'auto',
    forced_delta_count: 0,
    xgb_workers: 4,
    enable_market_priors: true,
    historical_odds_db: '/data/odds.db',
    historical_market_book: null,
    mu_delta_mode: 'fitted',
    stage3_experiment: 'exp-s3',
    stage4_experiment: 'exp-s4',
    stage3_research_lane_name: 'distribution',
    stage4_research_lane_name: 'mcmc',
  },
  artifacts: {
    stage3_report_path: '/reports/s3.json',
    stage3_vs_control_path: '/reports/s3_vs_ctrl.json',
    stage4_report_path: '/reports/s4.json',
    stage4_vs_control_path: '/reports/s4_vs_ctrl.json',
    stage4_vs_stage3_path: '/reports/s4_vs_s3.json',
    stage3_walk_forward_path: '/reports/s3_wf.json',
    stage4_walk_forward_path: '/reports/s4_wf.json',
    dual_view_path: '/reports/dual_view.json',
  },
  summary: {
    headline_result: 'Stage 4 beats control on CRPS.',
    what_changed: 'Added market priors to the MCMC model.',
    best_research_lane_key: 'stage4_mcmc',
    best_research_lane_label: 'Stage 4 MCMC',
    promoted_second_opinion_lane_key: null,
    production_promotable_lane_key: 'stage4_mcmc',
    is_profitable: true,
    is_more_accurate_than_control: true,
    is_more_stable_than_control: true,
    main_reason_failed: '',
    next_action_hint: 'Promote stage4_mcmc to production.',
    stage3: { ...STAGE_SUMMARY, research_lane_name: 'distribution' },
    stage4: { ...STAGE_SUMMARY, research_lane_name: 'mcmc' },
  },
  benchmark_compare: {
    available: true,
    benchmark_label: 'baseline-v1',
    benchmark_run_id: 'run-000',
    stage3: DELTA_BLOCK,
    stage4: DELTA_BLOCK,
    notes: ['First comparison run.'],
  },
};

const MOCK_RUN_HISTORY: RunHistoryIndex = {
  generated_at: '2026-03-30T12:00:00Z',
  row_count: 1,
  runs: [
    {
      run_id: 'run-001',
      tracked_at: '2026-03-30T12:00:00Z',
      run_label: 'Smoke Test Run',
      benchmark_status: 'candidate',
      benchmark_label: 'baseline-v1',
      headline_result: 'Stage 4 beats control.',
      best_research_lane_key: 'stage4_mcmc',
      production_promotable_lane_key: 'stage4_mcmc',
      stage3_mean_crps: 0.456,
      stage3_mean_negative_log_score: 0.789,
      stage3_rmse: 1.23,
      stage3_roi: 0.12,
      stage4_mean_crps: 0.445,
      stage4_mean_negative_log_score: 0.770,
      stage4_rmse: 1.20,
      stage4_roi: 0.15,
      is_profitable: true,
      is_more_accurate_than_control: true,
    },
  ],
};

const MOCK_BENCHMARK: BenchmarkFile = {
  benchmark_label: 'baseline-v1',
  benchmark_run_id: 'run-000',
  tracked_at: '2026-03-25T08:00:00Z',
  summary: MOCK_LATEST_RUN.summary,
  artifacts: MOCK_LATEST_RUN.artifacts,
};

const CALIBRATION_EVENT = {
  event: 'p_0',
  mean_predicted_probability: 0.3,
  predicted_probability_std: 0.05,
  empirical_rate: 0.28,
  absolute_error: 0.02,
  brier_score: 0.18,
  bin_count: 10,
  bins: [],
};

const MOCK_DUAL_VIEW: DualView = {
  generated_at: '2026-03-30T12:00:00Z',
  source_paths: {
    current_control_path: '/reports/control.json',
    stage3_report_path: '/reports/s3.json',
    stage3_vs_control_path: '/reports/s3_vs_ctrl.json',
    mcmc_report_path: '/reports/mcmc.json',
    mcmc_vs_control_path: '/reports/mcmc_vs_ctrl.json',
    mcmc_vs_stage3_path: '/reports/mcmc_vs_s3.json',
    stage3_walk_forward_report_path: '/reports/s3_wf.json',
    mcmc_walk_forward_report_path: '/reports/mcmc_wf.json',
  },
  lane_summaries: [
    {
      lane_key: 'control',
      lane_label: 'Control',
      lane_kind: 'control',
      lane_status: 'control',
      artifact_path: '/artifacts/control',
      expected_away_runs: 4.2,
      shutout_probability: 0.08,
      p_away_runs_ge_3: 0.65,
      p_away_runs_ge_5: 0.30,
      mean_metrics: {
        mae: 1.0, rmse: 1.3, poisson_deviance: 0.5, r2: 0.15,
        actual_mean: 4.5, predicted_mean: 4.3, naive_mean_prediction: 4.5,
        naive_mae: 1.5, naive_rmse: 1.8, naive_poisson_deviance: 0.7,
        mae_improvement_vs_naive_pct: 33.3, rmse_improvement_vs_naive_pct: 27.8,
        poisson_deviance_improvement_vs_naive_pct: 28.6,
      },
      distribution_metrics: {
        mean_crps: 0.46, mean_log_score: -0.78, mean_negative_log_score: 0.78,
        zero_calibration: { p_0: CALIBRATION_EVENT, p_ge_1: CALIBRATION_EVENT },
        tail_calibration: {
          p_ge_3: CALIBRATION_EVENT, p_ge_5: CALIBRATION_EVENT, p_ge_10: CALIBRATION_EVENT,
        },
        interval_coverage: {
          central_50: { nominal_coverage: 0.5, empirical_coverage: 0.48, coverage_error: 0.02, mean_width: 3.0 },
          central_80: { nominal_coverage: 0.8, empirical_coverage: 0.78, coverage_error: 0.02, mean_width: 5.0 },
          central_95: { nominal_coverage: 0.95, empirical_coverage: 0.93, coverage_error: 0.02, mean_width: 7.0 },
        },
        prediction_summary: {
          mean_predicted_runs: 4.3, mean_predicted_p_0: 0.08,
          mean_predicted_p_ge_3: 0.65, mean_predicted_p_ge_5: 0.30, mean_predicted_p_ge_10: 0.02,
        },
      },
      promotion_state: {
        lane_key: 'control', lane_label: 'Control',
        second_opinion_promoted: false, production_promotable: false,
        lane_status: 'control', summary_reason: 'Control lane — not eligible.',
        checks: {},
      },
    },
    {
      lane_key: 'stage4_mcmc',
      lane_label: 'Stage 4 MCMC',
      lane_kind: 'mcmc',
      lane_status: 'promotable',
      artifact_path: '/artifacts/mcmc',
      expected_away_runs: 4.1,
      shutout_probability: 0.09,
      p_away_runs_ge_3: 0.63,
      p_away_runs_ge_5: 0.28,
      mean_metrics: {
        mae: 0.95, rmse: 1.2, poisson_deviance: 0.45, r2: 0.2,
        actual_mean: 4.5, predicted_mean: 4.1, naive_mean_prediction: 4.5,
        naive_mae: 1.5, naive_rmse: 1.8, naive_poisson_deviance: 0.7,
        mae_improvement_vs_naive_pct: 36.7, rmse_improvement_vs_naive_pct: 33.3,
        poisson_deviance_improvement_vs_naive_pct: 35.7,
      },
      distribution_metrics: {
        mean_crps: 0.44, mean_log_score: -0.75, mean_negative_log_score: 0.75,
        zero_calibration: { p_0: CALIBRATION_EVENT, p_ge_1: CALIBRATION_EVENT },
        tail_calibration: {
          p_ge_3: CALIBRATION_EVENT, p_ge_5: CALIBRATION_EVENT, p_ge_10: CALIBRATION_EVENT,
        },
        interval_coverage: {
          central_50: { nominal_coverage: 0.5, empirical_coverage: 0.49, coverage_error: 0.01, mean_width: 2.9 },
          central_80: { nominal_coverage: 0.8, empirical_coverage: 0.79, coverage_error: 0.01, mean_width: 4.8 },
          central_95: { nominal_coverage: 0.95, empirical_coverage: 0.94, coverage_error: 0.01, mean_width: 6.8 },
        },
        prediction_summary: {
          mean_predicted_runs: 4.1, mean_predicted_p_0: 0.09,
          mean_predicted_p_ge_3: 0.63, mean_predicted_p_ge_5: 0.28, mean_predicted_p_ge_10: 0.01,
        },
      },
      promotion_state: {
        lane_key: 'stage4_mcmc', lane_label: 'Stage 4 MCMC',
        second_opinion_promoted: false, production_promotable: true,
        lane_status: 'promotable', summary_reason: 'Beats control on CRPS and NLS.',
        checks: {},
      },
    },
  ],
};

/* ── Fetch mock router ─────────────────────────────────────────────── */

function mockFetchRouter(url: string | URL | Request): Promise<Response> {
  const urlStr = typeof url === 'string' ? url : url instanceof URL ? url.href : url.url;

  let body: unknown = null;

  if (urlStr.includes('latest_run.json')) body = MOCK_LATEST_RUN;
  else if (urlStr.includes('run_history_index.json')) body = MOCK_RUN_HISTORY;
  else if (urlStr.includes('benchmark.json')) body = MOCK_BENCHMARK;
  else if (urlStr.includes('current_dual_view.json')) body = MOCK_DUAL_VIEW;

  if (body !== null) {
    return Promise.resolve(
      new Response(JSON.stringify(body), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
  }

  return Promise.resolve(
    new Response(JSON.stringify({ error: 'Not found' }), { status: 404 }),
  );
}

/* ── Test setup ────────────────────────────────────────────────────── */

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn(mockFetchRouter));
});

/** Helper: render a page component inside BrowserRouter. */
function renderPage(Page: React.ComponentType) {
  return render(
    <BrowserRouter>
      <Page />
    </BrowserRouter>,
  );
}

/* ── Smoke tests ───────────────────────────────────────────────────── */

describe('Page smoke tests', () => {
  it('LatestRunPage renders run label', async () => {
    renderPage(LatestRunPage);
    await waitFor(() => {
      expect(screen.getByText('Smoke Test Run')).toBeDefined();
    });
  });

  it('StageCardsPage renders stage comparison heading', async () => {
    renderPage(StageCardsPage);
    await waitFor(() => {
      expect(screen.getByText('Stage Comparison')).toBeDefined();
    });
  });

  it('BenchmarkComparePage renders benchmark heading', async () => {
    renderPage(BenchmarkComparePage);
    await waitFor(() => {
      expect(screen.getByText('Benchmark Comparison')).toBeDefined();
    });
  });

  it('RunLedgerPage renders ledger heading', async () => {
    renderPage(RunLedgerPage);
    await waitFor(() => {
      expect(screen.getByText('Run Ledger')).toBeDefined();
    });
  });

  it('BestRunsPage renders best runs heading', async () => {
    renderPage(BestRunsPage);
    await waitFor(() => {
      expect(screen.getByText('Best Runs')).toBeDefined();
    });
  });

  it('PromotionSummaryPage renders promotion heading', async () => {
    renderPage(PromotionSummaryPage);
    await waitFor(() => {
      expect(screen.getByText('Promotion Summary')).toBeDefined();
    });
  });
});
