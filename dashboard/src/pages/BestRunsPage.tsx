/**
 * BestRunsPage — Ranked list of best runs sorted by a user-selected metric.
 * Supports CRPS, NLS, RMSE (ascending = better) and ROI (descending = better).
 * ROI ranking excludes runs with bet_count === 0 per profitability semantics.
 */
import { useState, useMemo } from 'react';
import { motion } from 'motion/react';
import { useRunHistory } from '../hooks';
import { GlassCard, StatusBadge, LoadingState, ErrorState, EmptyState } from '../components';
import type { RunHistoryRow } from '../types';

type Metric = 'crps' | 'nls' | 'rmse' | 'roi';

interface MetricDef {
  label: string;
  short: string;
  accessor: (r: RunHistoryRow) => number | null;
  ascending: boolean;
  precision: number;
  unit?: string;
}

const METRICS: Record<Metric, MetricDef> = {
  crps: {
    label: 'Best CRPS',
    short: 'CRPS',
    accessor: (r) => r.stage4_mean_crps,
    ascending: true,
    precision: 4,
  },
  nls: {
    label: 'Best NLS',
    short: 'NLS',
    accessor: (r) => r.stage4_mean_negative_log_score,
    ascending: true,
    precision: 4,
  },
  rmse: {
    label: 'Best RMSE',
    short: 'RMSE',
    accessor: (r) => r.stage4_rmse,
    ascending: true,
    precision: 4,
  },
  roi: {
    label: 'Best ROI',
    short: 'ROI',
    accessor: (r) => r.stage4_roi,
    ascending: false,
    precision: 2,
    unit: '%',
  },
};

const METRIC_KEYS: Metric[] = ['crps', 'nls', 'rmse', 'roi'];

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
}

export default function BestRunsPage() {
  const { data: history, loading, error } = useRunHistory();
  const [active, setActive] = useState<Metric>('crps');
  const [benchmarkOnly, setBenchmarkOnly] = useState(false);
  const [promotableOnly, setPromotableOnly] = useState(false);

  const def = METRICS[active];

  const ranked = useMemo(() => {
    if (!history?.runs) return [];
    let rows = history.runs;

    if (benchmarkOnly) {
      rows = rows.filter((r) => r.benchmark_status === 'benchmark');
    }
    if (promotableOnly) {
      rows = rows.filter((r) => r.production_promotable_lane_key != null);
    }
    // ROI excludes runs with no bets (roi === null means bet_count was 0)
    if (active === 'roi') {
      rows = rows.filter((r) => r.stage4_roi != null);
    }

    return [...rows].sort((a, b) => {
      const va = def.accessor(a);
      const vb = def.accessor(b);
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      return def.ascending ? va - vb : vb - va;
    });
  }, [history, active, benchmarkOnly, promotableOnly, def]);

  if (loading) return <LoadingState rows={5} />;
  if (error) return <ErrorState message={error} />;
  if (!history?.runs.length) return <EmptyState message="No runs recorded yet." />;

  return (
    <motion.div className="space-y-6" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
      <h1 className="font-heading text-2xl font-extrabold tracking-tight text-ink">
        Best Runs
      </h1>

      {/* Metric selector */}
      <div className="flex flex-wrap items-center gap-2">
        {METRIC_KEYS.map((key) => (
          <button
            key={key}
            type="button"
            onClick={() => setActive(key)}
            className={`rounded-lg border px-4 py-2 text-xs font-bold uppercase tracking-widest transition-colors ${
              active === key
                ? 'border-accent/40 bg-accent/15 text-accent'
                : 'border-stroke/15 bg-panel/40 text-ink-dim hover:border-stroke/30 hover:text-ink'
            }`}
          >
            {METRICS[key].label}
          </button>
        ))}
      </div>

      {/* Filter toggles */}
      <div className="flex flex-wrap items-center gap-5">
        <label className="flex cursor-pointer items-center gap-2 text-xs text-ink-dim">
          <input
            type="checkbox"
            checked={benchmarkOnly}
            onChange={(e) => setBenchmarkOnly(e.target.checked)}
            className="accent-accent"
          />
          <span className="font-medium uppercase tracking-widest">Benchmark only</span>
        </label>
        <label className="flex cursor-pointer items-center gap-2 text-xs text-ink-dim">
          <input
            type="checkbox"
            checked={promotableOnly}
            onChange={(e) => setPromotableOnly(e.target.checked)}
            className="accent-accent"
          />
          <span className="font-medium uppercase tracking-widest">Promotable only</span>
        </label>
      </div>

      {/* Ranked list */}
      {ranked.length === 0 ? (
        <EmptyState message="No runs match the current filters." />
      ) : (
        <GlassCard>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr className="border-b border-stroke/30">
                  <th className="px-4 py-3 text-left text-[11px] font-bold uppercase tracking-widest text-ink-dim">
                    #
                  </th>
                  <th className="px-4 py-3 text-left text-[11px] font-bold uppercase tracking-widest text-ink-dim">
                    Run
                  </th>
                  <th className="px-4 py-3 text-right text-[11px] font-bold uppercase tracking-widest text-accent">
                    {def.short}
                  </th>
                  <th className="px-4 py-3 text-left text-[11px] font-bold uppercase tracking-widest text-ink-dim">
                    Status
                  </th>
                  <th className="px-4 py-3 text-right text-[11px] font-bold uppercase tracking-widest text-ink-dim">
                    Tracked
                  </th>
                </tr>
              </thead>
              <tbody>
                {ranked.map((row, i) => {
                  const val = def.accessor(row);
                  return (
                    <tr
                      key={row.run_id}
                      className="border-b border-stroke/10 transition-colors hover:bg-well/60"
                    >
                      <td className="px-4 py-3 font-heading text-lg font-extrabold text-accent/70">
                        {i + 1}
                      </td>
                      <td className="max-w-[280px] truncate px-4 py-3 font-medium text-ink">
                        {row.run_label}
                      </td>
                      <td className="px-4 py-3 text-right font-heading font-extrabold text-ink">
                        {val != null ? val.toFixed(def.precision) : '—'}
                        {val != null && def.unit ? (
                          <span className="ml-0.5 text-xs font-medium text-ink-dim">
                            {def.unit}
                          </span>
                        ) : null}
                      </td>
                      <td className="px-4 py-3">
                        <StatusBadge status={row.benchmark_status} />
                      </td>
                      <td className="px-4 py-3 text-right text-xs text-ink-dim">
                        {formatDate(row.tracked_at)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </GlassCard>
      )}
    </motion.div>
  );
}
