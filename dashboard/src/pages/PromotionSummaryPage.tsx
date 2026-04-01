import { motion } from 'motion/react';
import { useLatestRun, useDualView } from '../hooks';
import {
  GlassCard,
  MetricCard,
  StatusBadge,
  LoadingState,
  ErrorState,
  EmptyState,
} from '../components';
import type { LaneSummary } from '../types';

/** Pass/fail indicator pill for quality gate checks. */
function GateCheck({ label, passed }: { label: string; passed: boolean }) {
  return (
    <div className="flex items-center gap-2">
      <span
        className={`inline-block h-2.5 w-2.5 rounded-full ${passed ? 'bg-positive' : 'bg-negative'}`}
      />
      <span className="text-sm text-ink-dim">{label}</span>
      <span className={`text-xs font-bold ${passed ? 'text-positive' : 'text-negative'}`}>
        {passed ? 'Pass' : 'Fail'}
      </span>
    </div>
  );
}

/** Sorted, non-control lane cards from the dual view. */
function ResearchLanes({ lanes }: { lanes: LaneSummary[] }) {
  const research = lanes
    .filter((l) => l.lane_kind !== 'control')
    .sort((a, b) => a.distribution_metrics.mean_crps - b.distribution_metrics.mean_crps);

  if (research.length === 0) {
    return <EmptyState message="No research lanes in dual view." />;
  }

  return (
    <div className="space-y-4">
      {research.map((lane, idx) => (
        <GlassCard key={lane.lane_key}>
          <div className="mb-4 flex flex-wrap items-center gap-3">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent/10 font-heading text-sm font-extrabold text-accent">
              #{idx + 1}
            </span>
            <h3 className="font-heading text-base font-extrabold text-ink">
              {lane.lane_label}
            </h3>
            <span className="rounded-md bg-well/80 px-2 py-0.5 text-[11px] font-bold uppercase tracking-widest text-ink-dim">
              {lane.lane_kind}
            </span>
            <StatusBadge status={lane.promotion_state.lane_status} />
          </div>

          <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
            <MetricCard
              label="CRPS"
              value={lane.distribution_metrics.mean_crps}
              deltaInverted
            />
            <MetricCard label="RMSE" value={lane.mean_metrics.rmse} deltaInverted />
            <MetricCard
              label="NLS"
              value={lane.distribution_metrics.mean_negative_log_score}
              deltaInverted
            />
            <MetricCard label="MAE" value={lane.mean_metrics.mae} deltaInverted />
          </div>

          {lane.promotion_state.summary_reason && (
            <p className="text-sm leading-relaxed text-ink-dim">
              {lane.promotion_state.summary_reason}
            </p>
          )}
        </GlassCard>
      ))}
    </div>
  );
}

export default function PromotionSummaryPage() {
  const { data: run, loading: runLoading, error: runError, refetch: retryRun } = useLatestRun();
  const {
    data: dualView,
    loading: dvLoading,
    error: dvError,
    refetch: retryDv,
  } = useDualView();

  /* ── Section 1: Promotion status from latest run ──────────── */
  const renderPromotion = () => {
    if (runLoading) return <LoadingState rows={4} />;
    if (runError) return <ErrorState message={runError} onRetry={retryRun} />;
    if (!run) return <EmptyState message="No run data — run the pipeline first." />;

    const s = run.summary;
    const promotable = s.production_promotable_lane_key != null;

    return (
      <div className="space-y-4">
        {/* Best + second opinion row */}
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="flex flex-col gap-2 rounded-2xl border border-stroke/15 bg-panel/40 p-5">
            <span className="text-[11px] font-bold uppercase tracking-widest text-ink-dim">
              Best Research Lane
            </span>
            <span className="font-heading text-lg font-extrabold text-ink">
              {s.best_research_lane_label}
            </span>
          </div>

          <div className="flex flex-col gap-2 rounded-2xl border border-stroke/15 bg-panel/40 p-5">
            <span className="text-[11px] font-bold uppercase tracking-widest text-ink-dim">
              Second Opinion Lane
            </span>
            <span className="font-heading text-lg font-extrabold text-ink">
              {s.promoted_second_opinion_lane_key ?? '—'}
            </span>
          </div>
        </div>

        {/* Production promotable */}
        <div className="flex flex-col gap-3 rounded-2xl border border-stroke/15 bg-panel/40 p-5">
          <span className="text-[11px] font-bold uppercase tracking-widest text-ink-dim">
            Production Promotable
          </span>
          <div className="flex items-center gap-3">
            <span className="font-heading text-lg font-extrabold text-ink">
              {promotable ? s.production_promotable_lane_key : 'Not promotable'}
            </span>
            <StatusBadge status={promotable ? 'promotable' : 'failed'} />
          </div>
        </div>

        {/* Quality gates */}
        <div className="rounded-2xl border border-stroke/15 bg-panel/40 p-5">
          <span className="mb-3 block text-[11px] font-bold uppercase tracking-widest text-ink-dim">
            Quality Gates
          </span>
          <div className="flex flex-wrap gap-x-8 gap-y-2">
            <GateCheck label="More stable than control" passed={s.is_more_stable_than_control} />
            <GateCheck
              label="More accurate than control"
              passed={s.is_more_accurate_than_control}
            />
          </div>
        </div>
      </div>
    );
  };

  /* ── Section 2: Research lanes from dual view ─────────────── */
  const renderLanes = () => {
    if (dvLoading) return <LoadingState rows={3} />;
    if (dvError) return <ErrorState message={dvError} onRetry={retryDv} />;
    if (!dualView?.lane_summaries?.length)
      return <EmptyState message="No lane data in dual view." />;

    return <ResearchLanes lanes={dualView.lane_summaries} />;
  };

  return (
    <motion.div className="mx-auto max-w-5xl space-y-8" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
      <header>
        <h1 className="font-heading text-3xl font-extrabold tracking-tight text-ink">
          Promotion Summary
        </h1>
        <p className="mt-1 text-sm text-ink-dim">
          Lane promotion status and quality gate results.
        </p>
      </header>

      {renderPromotion()}

      <section className="space-y-4">
        <h2 className="font-heading text-lg font-extrabold tracking-tight text-ink">
          Research Lanes
        </h2>
        {renderLanes()}
      </section>
    </motion.div>
  );
}
