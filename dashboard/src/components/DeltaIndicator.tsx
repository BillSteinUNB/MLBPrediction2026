import React from "react";

export interface DeltaIndicatorProps {
  delta: number | null;
  label?: string;
}

const containerStyle: React.CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  gap: 8,
  fontFamily: "Arial, sans-serif",
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
  gap: 6,
  padding: "4px 8px",
  borderRadius: 12,
  fontSize: 13,
  minWidth: 56,
  justifyContent: "center",
};

export const DeltaIndicator: React.FC<DeltaIndicatorProps> = ({ delta, label }) => {
  const formatted =
    delta === null || Object.is(delta, NaN)
      ? "—"
      : `${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(2)}%`;

  const badgeStyle: React.CSSProperties =
    delta === null
      ? { ...badgeBase, background: "#e6e9ee", color: "#6b7280" }
      : delta > 0
      ? { ...badgeBase, background: "#e6f6ea", color: "#0f7a3a" }
      : delta < 0
      ? { ...badgeBase, background: "#fdecea", color: "#9f1f1f" }
      : { ...badgeBase, background: "#eef2f6", color: "#6b7280" };

  const arrow =
    delta === null
      ? "—"
      : delta > 0
      ? "↑"
      : delta < 0
      ? "↓"
      : "—";

  return (
    <div style={containerStyle} aria-label={label ? `delta-${label}` : "delta-indicator"}>
      {label ? <p style={labelStyle}>{label}</p> : null}
      <div style={badgeStyle}>
        <span aria-hidden>{arrow}</span>
        <span>{formatted}</span>
      </div>
    </div>
  );
};

export default DeltaIndicator;
