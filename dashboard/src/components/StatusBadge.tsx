/**
 * StatusBadge — Pill badge for run/lane states.
 * Color-coded: Benchmark (blue), Promotable (green), Failed/Catastrophic (red),
 * Candidate (amber), Exploratory (dim).
 */

const STATUS_STYLES: Record<string, { label: string; classes: string }> = {
  benchmark: {
    label: 'Benchmark',
    classes: 'bg-accent/15 text-accent border-accent/25',
  },
  promotable: {
    label: 'Promotable',
    classes: 'bg-positive/15 text-positive border-positive/25',
  },
  failed: {
    label: 'Failed',
    classes: 'bg-negative/15 text-negative border-negative/25',
  },
  catastrophic_regression: {
    label: 'Catastrophic Regression',
    classes: 'bg-negative/15 text-negative border-negative/25',
  },
  candidate: {
    label: 'Candidate',
    classes: 'bg-caution/15 text-caution border-caution/25',
  },
  exploratory: {
    label: 'Exploratory',
    classes: 'bg-ink-dim/15 text-ink-dim border-ink-dim/25',
  },
};

interface StatusBadgeProps {
  /** Status key — maps to predefined styles, falls back to neutral */
  status: string;
}

export function StatusBadge({ status }: StatusBadgeProps) {
  const config = STATUS_STYLES[status] ?? {
    label: status,
    classes: 'bg-stroke/20 text-ink-dim border-stroke/30',
  };

  return (
    <span
      className={`inline-flex items-center rounded-full border px-3 py-1 text-[11px] font-bold uppercase tracking-widest ${config.classes}`}
    >
      {config.label}
    </span>
  );
}
