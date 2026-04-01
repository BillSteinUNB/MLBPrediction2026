/**
 * RunLedgerPage — Sortable, filterable ledger of all tracked pipeline runs.
 * Displays run metrics in a SortableTable with label search and benchmark filter.
 */
import { useState, useMemo } from 'react';
import { motion } from 'motion/react';
import { useRunHistory } from '../hooks';
import {
  SortableTable,
  GlassCard,
  LoadingState,
  ErrorState,
  EmptyState,
  StatusBadge,
} from '../components';
import type { Column } from '../components';
import type { RunHistoryRow } from '../types';

/** Format ISO timestamp to readable short date + time */
function fmtDate(iso: string): string {
  return new Date(iso).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

/** Format metric to 4 decimal places */
function fmtMetric(v: number): string {
  return v.toFixed(4);
}

/** Format ROI — null renders as dash, never 0 */
function fmtRoi(v: number | null): string {
  if (v == null) return '\u2014';
  return `${(v * 100).toFixed(1)}%`;
}

/** Semantic color class for ROI values */
function roiColor(v: number | null): string {
  if (v == null) return 'text-ink-dim';
  if (v > 0) return 'text-positive';
  if (v < 0) return 'text-negative';
  return 'text-ink-dim';
}

const COLUMNS: Column<RunHistoryRow>[] = [
  {
    key: 'tracked_at',
    header: 'Date',
    render: (r) => <span className="whitespace-nowrap">{fmtDate(r.tracked_at)}</span>,
    sortValue: (r) => r.tracked_at,
  },
  {
    key: 'run_label',
    header: 'Run Label',
    render: (r) => (
      <span className="font-medium text-accent" title={r.run_id}>
        {r.run_label}
      </span>
    ),
    sortValue: (r) => r.run_label,
  },
  {
    key: 'benchmark_status',
    header: 'Status',
    render: (r) => <StatusBadge status={r.benchmark_status} />,
    sortValue: (r) => r.benchmark_status,
    align: 'center',
  },
  {
    key: 's3_crps',
    header: 'S3 CRPS',
    render: (r) => fmtMetric(r.stage3_mean_crps),
    sortValue: (r) => r.stage3_mean_crps,
    align: 'right',
  },
  {
    key: 's3_nls',
    header: 'S3 NLS',
    render: (r) => fmtMetric(r.stage3_mean_negative_log_score),
    sortValue: (r) => r.stage3_mean_negative_log_score,
    align: 'right',
  },
  {
    key: 's3_rmse',
    header: 'S3 RMSE',
    render: (r) => fmtMetric(r.stage3_rmse),
    sortValue: (r) => r.stage3_rmse,
    align: 'right',
  },
  {
    key: 's4_crps',
    header: 'S4 CRPS',
    render: (r) => fmtMetric(r.stage4_mean_crps),
    sortValue: (r) => r.stage4_mean_crps,
    align: 'right',
  },
  {
    key: 's4_nls',
    header: 'S4 NLS',
    render: (r) => fmtMetric(r.stage4_mean_negative_log_score),
    sortValue: (r) => r.stage4_mean_negative_log_score,
    align: 'right',
  },
  {
    key: 's4_rmse',
    header: 'S4 RMSE',
    render: (r) => fmtMetric(r.stage4_rmse),
    sortValue: (r) => r.stage4_rmse,
    align: 'right',
  },
  {
    key: 's3_roi',
    header: 'S3 ROI',
    render: (r) => <span className={roiColor(r.stage3_roi)}>{fmtRoi(r.stage3_roi)}</span>,
    sortValue: (r) => r.stage3_roi,
    align: 'right',
  },
  {
    key: 's4_roi',
    header: 'S4 ROI',
    render: (r) => <span className={roiColor(r.stage4_roi)}>{fmtRoi(r.stage4_roi)}</span>,
    sortValue: (r) => r.stage4_roi,
    align: 'right',
  },
];

export default function RunLedgerPage() {
  const { data, loading, error, refetch } = useRunHistory();
  const [labelFilter, setLabelFilter] = useState('');
  const [benchmarkOnly, setBenchmarkOnly] = useState(false);

  const filtered = useMemo(() => {
    if (!data) return [];
    let rows = data.runs;
    if (labelFilter) {
      const q = labelFilter.toLowerCase();
      rows = rows.filter((r) => r.run_label.toLowerCase().includes(q));
    }
    if (benchmarkOnly) {
      rows = rows.filter((r) => r.benchmark_status === 'benchmark');
    }
    return rows;
  }, [data, labelFilter, benchmarkOnly]);

  if (loading) return <LoadingState rows={6} />;
  if (error) return <ErrorState message={error} onRetry={refetch} />;
  if (!data || data.runs.length === 0) return <EmptyState message="No runs tracked yet." />;

  return (
    <motion.div className="space-y-6" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
      <h1 className="font-heading text-3xl font-extrabold tracking-tight text-ink">
        Run Ledger
      </h1>

      <div className="flex flex-wrap items-center gap-4">
        <input
          type="text"
          value={labelFilter}
          onChange={(e) => setLabelFilter(e.target.value)}
          placeholder="Filter by run label\u2026"
          className="rounded-lg border border-stroke/30 bg-well/60 px-4 py-2 text-sm text-ink placeholder:text-ink-dim/50 focus:border-accent/50 focus:outline-none focus:ring-1 focus:ring-accent/30"
        />
        <label className="flex items-center gap-2 text-sm text-ink-dim">
          <input
            type="checkbox"
            checked={benchmarkOnly}
            onChange={(e) => setBenchmarkOnly(e.target.checked)}
            className="h-4 w-4 rounded border-stroke/30 bg-well/60 accent-accent"
          />
          Benchmark runs only
        </label>
        <span className="ml-auto text-xs text-ink-dim">
          {filtered.length} of {data.runs.length} runs
        </span>
      </div>

      <GlassCard>
        {filtered.length === 0 ? (
          <EmptyState message="No runs match current filters." />
        ) : (
          <SortableTable columns={COLUMNS} data={filtered} rowKey={(r) => r.run_id} />
        )}
      </GlassCard>
    </motion.div>
  );
}
