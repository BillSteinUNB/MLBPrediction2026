/**
 * MetricCard — Displays a single metric with label, value, optional delta, and optional unit.
 * Handles null values by showing "—" (never "0").
 */
import type { ReactNode } from 'react';
import { DeltaBadge } from './DeltaBadge';

interface MetricCardProps {
  /** Uppercase label shown above the value */
  label: string;
  /** Numeric or string value; null renders as "—" */
  value: number | string | null;
  /** Unit suffix displayed after the value (e.g. "%") */
  unit?: string;
  /** Delta value shown as a badge next to the value */
  delta?: number | null;
  /** When true, negative delta is good (for loss metrics like CRPS) */
  deltaInverted?: boolean;
  /** Optional icon shown before the label */
  icon?: ReactNode;
  /** Decimal places for numeric values (default 3) */
  precision?: number;
}

export function MetricCard({
  label,
  value,
  unit,
  delta,
  deltaInverted = false,
  icon,
  precision = 3,
}: MetricCardProps) {
  const formatted =
    value == null
      ? '—'
      : typeof value === 'number'
        ? value.toFixed(precision)
        : value;

  return (
    <div className="flex flex-col gap-1.5 rounded-xl bg-well/80 px-4 py-3">
      <div className="flex items-center gap-2">
        {icon && <span className="text-ink-dim">{icon}</span>}
        <span className="text-[11px] font-bold uppercase tracking-widest text-ink-dim">
          {label}
        </span>
      </div>
      <div className="flex items-baseline gap-2">
        <span className="font-heading text-2xl font-extrabold text-ink">
          {formatted}
        </span>
        {unit && value != null && (
          <span className="text-xs font-medium text-ink-dim">{unit}</span>
        )}
        {delta != null && (
          <DeltaBadge value={delta} inverted={deltaInverted} precision={precision} />
        )}
      </div>
    </div>
  );
}
