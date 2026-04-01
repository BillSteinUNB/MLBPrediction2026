import { motion } from 'motion/react';
import { useLatestRun } from '../hooks';
import {
  GlassCard,
  MetricCard,
  StatusBadge,
  LoadingState,
  ErrorState,
  EmptyState,
} from '../components';

/** Format ISO timestamp to a short readable date. */
function fmtDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

export default function LatestRunPage() {
  const { data: run, loading, error, refetch } = useLatestRun();

  if (loading) return <LoadingState rows={6} />;
  if (error) return <ErrorState message={error} onRetry={refetch} />;
  if (!run) return <EmptyState message="No research runs found — run the pipeline first." />;

  const s = run.summary;
  const s3 = s.stage3;
  const s4 = s.stage4;
  const promotable = s.production_promotable_lane_key != null;

  return (
    <motion.div className="mx-auto max-w-5xl space-y-6" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
      {/* ── Hero ────────────────────────────────────────────────── */}
      <header className="space-y-2">
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="font-heading text-3xl font-extrabold tracking-tight text-ink">
            {run.run_label}
          </h1>
          <span className="rounded-lg bg-accent/10 px-3 py-1 text-xs font-bold text-accent">
            {fmtDate(run.tracked_at)}
          </span>
          <StatusBadge status={run.benchmark_status} />
        </div>
        <p className="max-w-3xl text-sm leading-relaxed text-ink-dim">{run.hypothesis}</p>
      </header>

      {/* ── Headline result ─────────────────────────────────────── */}
      <GlassCard title="Headline Result" icon="📊">
        <p className="text-base font-medium leading-relaxed text-ink">
          {s.headline_result}
        </p>
      </GlassCard>

      {/* ── What changed ────────────────────────────────────────── */}
      <GlassCard title="What Changed" icon="🔬" className="border-accent/15">
        <p className="text-sm leading-relaxed text-ink-dim">{s.what_changed}</p>
      </GlassCard>

      {/* ── Best lane + promotability row ────────────────────────── */}
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="flex flex-col gap-3 rounded-2xl border border-stroke/15 bg-panel/40 p-5">
          <span className="text-[11px] font-bold uppercase tracking-widest text-ink-dim">
            Best Research Lane
          </span>
          <span className="font-heading text-lg font-extrabold text-ink">
            {s.best_research_lane_label}
          </span>
          <StatusBadge status={s.best_research_lane_key} />
        </div>

        <div className="flex flex-col gap-3 rounded-2xl border border-stroke/15 bg-panel/40 p-5">
          <span className="text-[11px] font-bold uppercase tracking-widest text-ink-dim">
            Production Promotability
          </span>
          <span className="font-heading text-lg font-extrabold text-ink">
            {promotable ? s.production_promotable_lane_key : 'No lane qualifies'}
          </span>
          <StatusBadge status={promotable ? 'promotable' : 'failed'} />
        </div>
      </div>

      {/* ── Main reason failed (hide if null/empty) ─────────────── */}
      {s.main_reason_failed && (
        <div className="rounded-2xl border border-caution/20 bg-caution/5 p-5">
          <span className="mb-2 block text-[11px] font-bold uppercase tracking-widest text-caution">
            Main Reason Failed
          </span>
          <p className="text-sm font-medium leading-relaxed text-ink">
            {s.main_reason_failed}
          </p>
        </div>
      )}

      {/* ── Next action hint ────────────────────────────────────── */}
      <div className="rounded-2xl border border-accent/20 bg-accent/5 p-5">
        <span className="mb-2 block text-[11px] font-bold uppercase tracking-widest text-accent">
          Next Action
        </span>
        <p className="text-sm font-medium leading-relaxed text-ink">
          {s.next_action_hint}
        </p>
      </div>

      {/* ── Stage comparison ────────────────────────────────────── */}
      <section className="space-y-3">
        <h2 className="font-heading text-lg font-extrabold tracking-tight text-ink">
          Stage Comparison
        </h2>
        <div className="grid gap-4 sm:grid-cols-2">
          {/* Stage 3 */}
          <GlassCard title={`Stage 3 — ${s3.research_lane_name}`}>
            <div className="grid grid-cols-3 gap-3">
              <MetricCard label="CRPS" value={s3.mean_crps} deltaInverted />
              <MetricCard label="NLS" value={s3.mean_negative_log_score} deltaInverted />
              <MetricCard label="RMSE" value={s3.rmse} deltaInverted />
            </div>
          </GlassCard>

          {/* Stage 4 */}
          <GlassCard title={`Stage 4 — ${s4.research_lane_name}`}>
            <div className="grid grid-cols-3 gap-3">
              <MetricCard
                label="CRPS"
                value={s4.mean_crps}
                delta={s4.delta_vs_prior_lane?.mean_crps ?? null}
                deltaInverted
              />
              <MetricCard
                label="NLS"
                value={s4.mean_negative_log_score}
                delta={s4.delta_vs_prior_lane?.mean_negative_log_score ?? null}
                deltaInverted
              />
              <MetricCard
                label="RMSE"
                value={s4.rmse}
                delta={s4.delta_vs_prior_lane?.rmse ?? null}
                deltaInverted
              />
            </div>
          </GlassCard>
        </div>
      </section>
    </motion.div>
  );
}
