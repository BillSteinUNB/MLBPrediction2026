import { motion } from 'motion/react';
import { useLatestRun, useBenchmark } from '../hooks';
import {
  GlassCard,
  MetricCard,
  DeltaBadge,
  LoadingState,
  ErrorState,
  EmptyState,
} from '../components';
import type { DeltaBlock, StageSummary } from '../types';

/** Render a side-by-side delta row for one stage. */
function StageDeltaGrid({
  label,
  delta,
  current,
}: {
  label: string;
  delta: DeltaBlock;
  current: StageSummary;
}) {
  return (
    <GlassCard title={label}>
      <div className="grid grid-cols-3 gap-3">
        <MetricCard
          label="RMSE"
          value={current.rmse}
          delta={delta.rmse}
          deltaInverted
        />
        <MetricCard
          label="CRPS"
          value={current.mean_crps}
          delta={delta.mean_crps}
          deltaInverted
        />
        <MetricCard
          label="NLS"
          value={current.mean_negative_log_score}
          delta={delta.mean_negative_log_score}
          deltaInverted
        />
      </div>
      {/* ROI / Net Units row when available */}
      {(delta.delta_roi != null || delta.delta_net_units != null) && (
        <div className="mt-3 grid grid-cols-2 gap-3">
          <MetricCard
            label="ROI Δ"
            value={delta.delta_roi}
            unit="%"
            precision={2}
          />
          <MetricCard
            label="Net Units Δ"
            value={delta.delta_net_units}
            precision={2}
          />
        </div>
      )}
    </GlassCard>
  );
}

export default function BenchmarkComparePage() {
  const { data: run, loading: runLoading, error: runError, refetch: runRefetch } = useLatestRun();
  const { data: bench, loading: benchLoading, error: benchError, refetch: benchRefetch } =
    useBenchmark();

  const loading = runLoading || benchLoading;
  const error = runError || benchError;
  const refetch = () => { runRefetch(); benchRefetch(); };

  if (loading) return <LoadingState rows={6} />;
  if (error) return <ErrorState message={error} onRetry={refetch} />;
  if (!run) return <EmptyState message="No research runs found — run the pipeline first." />;

  const bc = run.benchmark_compare;

  /* ── Benchmark not available ─────────────────────────────────── */
  if (!bc.available) {
    return (
      <div className="mx-auto max-w-5xl space-y-6">
        <h1 className="font-heading text-3xl font-extrabold tracking-tight text-ink">
          Benchmark Comparison
        </h1>
        <GlassCard title="No Benchmark Available" icon="🚫">
          <p className="text-sm leading-relaxed text-ink-dim">
            This run has no active benchmark to compare against.
          </p>
          {bc.notes && bc.notes.length > 0 && (
            <ul className="mt-4 list-inside list-disc space-y-1.5">
              {bc.notes.map((note) => (
                <li key={note} className="text-sm text-ink-dim">
                  {note}
                </li>
              ))}
            </ul>
          )}
        </GlassCard>
      </div>
    );
  }

  /* ── Benchmark available — side-by-side delta view ──────────── */
  const s3 = run.summary.stage3;
  const s4 = run.summary.stage4;

  return (
    <motion.div className="mx-auto max-w-5xl space-y-6" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
      {/* ── Header ──────────────────────────────────────────────── */}
      <header className="space-y-2">
        <h1 className="font-heading text-3xl font-extrabold tracking-tight text-ink">
          Benchmark Comparison
        </h1>
        <p className="text-sm leading-relaxed text-ink-dim">
          Current run{' '}
          <span className="font-bold text-ink">{run.run_label}</span>
          {' vs benchmark '}
          <span className="font-bold text-accent">{bc.benchmark_label}</span>
        </p>
      </header>

      {/* ── Summary badges ──────────────────────────────────────── */}
      <div className="flex flex-wrap gap-3">
        <div className="flex items-center gap-2 rounded-xl border border-stroke/15 bg-panel/40 px-4 py-2">
          <span className="text-[11px] font-bold uppercase tracking-widest text-ink-dim">
            Benchmark
          </span>
          <span className="font-heading text-sm font-extrabold text-accent">
            {bc.benchmark_label}
          </span>
        </div>
        {bench && (
          <div className="flex items-center gap-2 rounded-xl border border-stroke/15 bg-panel/40 px-4 py-2">
            <span className="text-[11px] font-bold uppercase tracking-widest text-ink-dim">
              Tracked
            </span>
            <span className="text-sm font-medium text-ink">
              {new Date(bench.tracked_at).toLocaleDateString('en-US', {
                month: 'short',
                day: 'numeric',
                year: 'numeric',
              })}
            </span>
          </div>
        )}
      </div>

      {/* ── Delta legend ────────────────────────────────────────── */}
      <div className="flex items-center gap-4 text-xs text-ink-dim">
        <span className="flex items-center gap-1.5">
          <DeltaBadge value={-0.01} inverted precision={2} />
          <span>= improvement (lower loss)</span>
        </span>
        <span className="flex items-center gap-1.5">
          <DeltaBadge value={0.01} inverted precision={2} />
          <span>= regression (higher loss)</span>
        </span>
      </div>

      {/* ── Stage 3 vs Benchmark ────────────────────────────────── */}
      <section className="space-y-3">
        <h2 className="font-heading text-lg font-extrabold tracking-tight text-ink">
          Deltas vs Benchmark
        </h2>
        <div className="grid gap-4 sm:grid-cols-2">
          <StageDeltaGrid
            label={`Stage 3 — ${s3.research_lane_name}`}
            delta={bc.stage3}
            current={s3}
          />
          <StageDeltaGrid
            label={`Stage 4 — ${s4.research_lane_name}`}
            delta={bc.stage4}
            current={s4}
          />
        </div>
      </section>

      {/* ── Notes (if present) ──────────────────────────────────── */}
      {bc.notes && bc.notes.length > 0 && (
        <GlassCard title="Comparison Notes" icon="📝">
          <ul className="list-inside list-disc space-y-1.5">
            {bc.notes.map((note) => (
              <li key={note} className="text-sm leading-relaxed text-ink-dim">
                {note}
              </li>
            ))}
          </ul>
        </GlassCard>
      )}
    </motion.div>
  );
}
