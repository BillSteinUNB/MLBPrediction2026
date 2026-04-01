/**
 * DeltaBadge — Shows a delta value with color coding.
 * Positive = green for gain metrics, negative = green for loss metrics (inverted).
 * Returns null when value is null (hidden).
 */

interface DeltaBadgeProps {
  /** The delta value to display */
  value: number | null;
  /** When true, negative delta is good (for loss metrics like CRPS, NLS, RMSE) */
  inverted?: boolean;
  /** Decimal places (default 3) */
  precision?: number;
}

export function DeltaBadge({
  value,
  inverted = false,
  precision = 3,
}: DeltaBadgeProps) {
  if (value == null) return null;

  const isGood = inverted ? value < 0 : value > 0;
  const isBad = inverted ? value > 0 : value < 0;
  const sign = value > 0 ? '+' : '';

  const colorClass = isGood
    ? 'text-positive bg-positive/10'
    : isBad
      ? 'text-negative bg-negative/10'
      : 'text-ink-dim bg-stroke/20';

  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-bold ${colorClass}`}
    >
      {sign}{value.toFixed(precision)}
    </span>
  );
}
