/**
 * StageCardsPage — Side-by-side Stage 3 vs Stage 4 comparison.
 * Renders every StageSummary field: accuracy metrics, profitability,
 * source badges, delta blocks, and catastrophic-regression flag.
 */
import { motion } from 'motion/react';
import { useLatestRun } from '../hooks';
import {
  GlassCard,
  MetricCard,
  DeltaBadge,
  StatusBadge,
  LoadingState,
  ErrorState,
  EmptyState,
} from '../components';
import type { StageSummary, DeltaBlock } from '../types';

/** Renders a row of delta badges for a DeltaBlock (vs control or vs prior lane). */
function DeltaRow({ label, delta }: { label: string; delta: DeltaBlock | null }) {
  if (!delta) return null;
  return (
    <div className="space-y-2">
      <span className="text-[11px] font-bold uppercase tracking-widest text-ink-dim">
        {label}
      </span>
      <div className="flex flex-wrap gap-2">
        <span className="flex items-center gap-1 text-xs text-ink-dim">
          CRPS <DeltaBadge value={delta.mean_crps} inverted precision={4} />
        </span>
        <span className="flex items-center gap-1 text-xs text-ink-dim">
          NLS <DeltaBadge value={delta.mean_negative_log_score} inverted precision={4} />
        </span>
        <span className="flex items-center gap-1 text-xs text-ink-dim">
          RMSE <DeltaBadge value={delta.rmse} inverted precision={4} />
        </span>
        <span className="flex items-center gap-1 text-xs text-ink-dim">
          ROI <DeltaBadge value={delta.delta_roi} precision={2} />
        </span>
        <span className="flex items-center gap-1 text-xs text-ink-dim">
          Units <DeltaBadge value={delta.delta_net_units} precision={2} />
        </span>
      </div>
    </div>
  );
}

/** Full panel for one stage with all StageSummary fields. */
function FullStagePanel({ title, stage }: { title: string; stage: StageSummary }) {
  const hasBets = stage.bet_count > 0;

  return (
    <GlassCard title={title}>
      {/* Identity row */}
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <span className="rounded-lg bg-accent/10 px-3 py-1 text-xs font-bold text-accent">
          {stage.model_version}
        </span>
        <span className="text-xs text-ink-dim">{stage.research_lane_name}</span>
        {stage.catastrophic_regression && (
          <StatusBadge status="catastrophic_regression" />
        )}
      </div>

      {/* Accuracy metrics */}
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
        <MetricCard label="MAE" value={stage.mae} precision={4} />
        <MetricCard
          label="ROI"
          value={hasBets ? stage.roi : null}
          unit="%"
          delta={hasBets ? stage.delta_vs_control.delta_roi : null}
          precision={2}
        />
        <MetricCard
          label="Net Units"
          value={hasBets ? stage.net_units : null}
          delta={hasBets ? stage.delta_vs_control.delta_net_units : null}
          precision={2}
        />
        <MetricCard label="Bet Count" value={stage.bet_count} precision={0} />
        <MetricCard
          label="Market Coverage"
          value={stage.market_data_coverage_pct}
          unit="%"
          precision={1}
        />
        <MetricCard
          label="Anchor Coverage"
          value={stage.market_anchor_coverage}
          unit="%"
          precision={1}
        />
      </div>

      {/* Source origins */}
      <div className="mt-4 space-y-1">
        <span className="text-[11px] font-bold uppercase tracking-widest text-ink-dim">
          Sources
        </span>
        <div className="flex flex-wrap gap-1.5">
          {stage.source_origins.map((src) => (
            <span
              key={src}
              className="rounded-full bg-stroke/20 px-2.5 py-0.5 text-[11px] font-semibold text-ink-dim"
            >
              {src}
            </span>
          ))}
        </div>
      </div>

      {/* Delta blocks */}
      <div className="mt-4 space-y-3 border-t border-stroke/15 pt-4">
        <DeltaRow label="Δ vs Control" delta={stage.delta_vs_control} />
        <DeltaRow label="Δ vs Prior Lane" delta={stage.delta_vs_prior_lane} />
      </div>
    </GlassCard>
  );
}

export default function StageCardsPage() {
  const { data, loading, error, refetch } = useLatestRun();

  if (loading) return <LoadingState rows={4} />;
  if (error) return <ErrorState message={error} onRetry={refetch} />;
  if (!data?.summary?.stage3 || !data?.summary?.stage4) {
    return <EmptyState message="No stage data available — run the research pipeline first." />;
  }

  const { stage3, stage4 } = data.summary;

  return (
    <motion.div className="space-y-6" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
      <h1 className="font-heading text-2xl font-extrabold tracking-tight text-ink">
        Stage Comparison
      </h1>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <FullStagePanel title="Stage 3 — Distribution" stage={stage3} />
        <FullStagePanel title="Stage 4 — MCMC" stage={stage4} />
      </div>
    </motion.div>
  );
}
