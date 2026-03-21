import React from "react";

export interface WarningBadgeProps {
  /** Short label for the warning (e.g. "ECE", "Reliability Gap") */
  label: string;
  /** Current metric value */
  value: number | null;
  /** Threshold — badge renders only when value > threshold */
  threshold: number;
}

/* ---- colour tokens ---- */
const WARN_BG = "#fff4e5";
const WARN_FG = "#6a4a00";
const WARN_BORDER = "#f5deb3";
const ERR_BG = "#fdecea";
const ERR_FG = "#9f1f1f";
const ERR_BORDER = "#f5c6c6";

const baseStyle: React.CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  gap: 4,
  padding: "3px 10px",
  borderRadius: 16,
  fontSize: 12,
  fontWeight: 500,
  fontFamily: "Arial, Helvetica, sans-serif",
  whiteSpace: "nowrap",
};

/**
 * Shows a warning badge when a metric value exceeds a threshold.
 * If value > 2 × threshold, badge escalates to error severity.
 * Renders nothing when value is null or below threshold.
 */
export const WarningBadge: React.FC<WarningBadgeProps> = ({
  label,
  value,
  threshold,
}) => {
  if (value === null || value === undefined || Number.isNaN(value)) return null;
  if (value <= threshold) return null;

  const isError = value > threshold * 2;

  const style: React.CSSProperties = {
    ...baseStyle,
    background: isError ? ERR_BG : WARN_BG,
    color: isError ? ERR_FG : WARN_FG,
    border: `1px solid ${isError ? ERR_BORDER : WARN_BORDER}`,
  };

  return (
    <span style={style} data-warning={label}>
      {isError ? "\u26d4" : "\u26a0\ufe0f"} {label}: {value.toFixed(4)}
    </span>
  );
};

export default WarningBadge;
