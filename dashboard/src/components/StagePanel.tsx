/**
 * StagePanel — Card displaying all metrics for one stage (Stage 3 or Stage 4).
 * Handles the ROI-when-no-bets semantic: shows "—" when bet_count is 0.
 */
import type { ReactNode } from 'react';
import type { StageSummary } from '../types';
import { GlassCard } from './GlassCard';
import { MetricCard } from './MetricCard';

interface StagePanelProps {
  /** Panel heading, e.g. "Stage 3 — Distribution" */
  title: string;
  /** Full stage summary data */
  stage: StageSummary;
  /** Optional icon for the card header */
  icon?: ReactNode;
}

export function StagePanel({ title, stage, icon }: StagePanelProps) {
  const hasBets = stage.bet_count > 0;

  return (
    <GlassCard title={title} icon={icon}>
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-3">
        <MetricCard
          label="CRPS"
          value={stage.mean_crps}
          delta={stage.delta_vs_control.mean_crps}
          deltaInverted
          precision={4}
        />
        <MetricCard
          label="Neg Log Score"
          value={stage.mean_negative_log_score}
          delta={stage.delta_vs_control.mean_negative_log_score}
          deltaInverted
          precision={4}
        />
        <MetricCard
          label="RMSE"
          value={stage.rmse}
          delta={stage.delta_vs_control.rmse}
          deltaInverted
          precision={4}
        />
        <MetricCard
          label="MAE"
          value={stage.mae}
          precision={4}
        />
        <MetricCard
          label="ROI"
          value={hasBets ? stage.roi : null}
          unit="%"
          delta={hasBets ? stage.delta_vs_control.delta_roi : null}
          precision={2}
        />
        <MetricCard
          label="Bet Count"
          value={stage.bet_count}
          precision={0}
        />
        <MetricCard
          label="Market Coverage"
          value={stage.market_data_coverage_pct}
          unit="%"
          precision={1}
        />
      </div>

      {stage.catastrophic_regression && (
        <div className="mt-4 rounded-lg border border-negative/30 bg-negative/5 px-4 py-2">
          <span className="text-xs font-bold uppercase tracking-widest text-negative">
            ⚠ Catastrophic Regression Detected
          </span>
        </div>
      )}
    </GlassCard>
  );
}
