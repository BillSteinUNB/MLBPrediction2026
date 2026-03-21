import React from "react";

export interface DeltaIndicatorProps {
  /** Raw delta value (e.g. 0.02 means +2% change) */
  value: number | null;
  /**
   * Direction rule: when true, positive delta = improvement (green).
   * When false (lower-is-better metrics like log_loss, brier), negative delta = improvement (green).
   */
  higherIsBetter: boolean;
  /** Optional label rendered before the badge */
  label?: string;
}

/* ---- colour tokens (matches layout.css palette) ---- */
const GREEN_BG = "#e6f6ea";
const GREEN_FG = "#0f7a3a";
const RED_BG = "#fdecea";
const RED_FG = "#9f1f1f";
const NEUTRAL_BG = "#e6e9ee";
const NEUTRAL_FG = "#6b7280";

const containerStyle: React.CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  gap: 8,
  fontFamily: "Arial, Helvetica, sans-serif",
  color: "#0b1220",
};

const labelStyle: React.CSSProperties = {
  fontSize: 12,
  color: "#495464",
  margin: 0,
};

const badgeBase: React.CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  gap: 4,
  padding: "3px 8px",
  borderRadius: 12,
  fontSize: 13,
  fontWeight: 500,
  minWidth: 56,
  justifyContent: "center",
  whiteSpace: "nowrap",
};

/**
 * Determine whether a delta represents an improvement given the metric direction.
 * Returns 1 for improvement, -1 for regression, 0 for neutral/zero.
 */
function classifyDelta(
  value: number,
  higherIsBetter: boolean,
): 1 | -1 | 0 {
  if (value === 0) return 0;
  const isPositive = value > 0;
  // For higher-is-better metrics: positive delta = improvement
  // For lower-is-better metrics: negative delta = improvement
  const isImprovement = higherIsBetter ? isPositive : !isPositive;
  return isImprovement ? 1 : -1;
}

export const DeltaIndicator: React.FC<DeltaIndicatorProps> = ({
  value,
  higherIsBetter,
  label,
}) => {
  const isNull = value === null || Number.isNaN(value);

  const formatted = isNull
    ? "\u2014"
    : `${value >= 0 ? "+" : ""}${(value * 100).toFixed(2)}%`;

  const direction = isNull ? 0 : classifyDelta(value, higherIsBetter);

  let bg = NEUTRAL_BG;
  let fg = NEUTRAL_FG;
  if (direction === 1) {
    bg = GREEN_BG;
    fg = GREEN_FG;
  } else if (direction === -1) {
    bg = RED_BG;
    fg = RED_FG;
  }

  const badgeStyle: React.CSSProperties = {
    ...badgeBase,
    background: bg,
    color: fg,
  };

  const arrow = isNull ? "\u2014" : direction === 1 ? "\u2191" : direction === -1 ? "\u2193" : "\u2014";

  return (
    <span
      style={containerStyle}
      data-label={label ? `delta-${label}` : "delta-indicator"}
    >
      {label ? <span style={labelStyle}>{label}</span> : null}
      <span style={badgeStyle}>
        <span aria-hidden="true">{arrow}</span>
        <span>{formatted}</span>
      </span>
    </span>
  );
};

export default DeltaIndicator;
