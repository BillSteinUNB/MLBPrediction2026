import React from "react";
import { DeltaIndicator } from "./DeltaIndicator";

export interface MetricCardProps {
  /** Metric display name (e.g. "ROC AUC") */
  name: string;
  /** Raw metric value — will be formatted to 4 decimal places */
  value: number | null;
  /** Optional delta vs previous run */
  delta?: number | null;
  /**
   * Direction rule: true for accuracy/roc_auc (higher = better),
   * false for log_loss/brier/ece/reliability_gap (lower = better).
   */
  higherIsBetter?: boolean;
  /** Optional unit suffix (e.g. "%") */
  unit?: string;
}

/* ---- Style tokens (aligned to layout.css variables) ---- */
const containerStyle: React.CSSProperties = {
  minWidth: 180,
  background: "#ffffff",
  border: "1px solid #e5e4e7",
  borderRadius: 6,
  padding: 14,
  boxSizing: "border-box",
  fontFamily: "Arial, Helvetica, sans-serif",
  color: "#08060d",
  display: "grid",
  gridTemplateRows: "auto 1fr auto",
  gap: 6,
};

const nameStyle: React.CSSProperties = {
  fontSize: 12,
  fontWeight: 500,
  color: "#6b6375",
  margin: 0,
  textTransform: "uppercase",
  letterSpacing: "0.04em",
};

const valueRowStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "baseline",
  gap: 6,
};

const valueStyle: React.CSSProperties = {
  fontSize: 24,
  fontWeight: 600,
  color: "#08060d",
  margin: 0,
  fontFamily: "ui-monospace, Consolas, monospace",
};

const unitStyle: React.CSSProperties = {
  fontSize: 12,
  color: "#9ca3af",
  margin: 0,
};

/**
 * Format a numeric value to 4 decimal places.
 * Returns "—" for null/undefined/NaN values.
 */
function formatValue(val: number | null): string {
  if (val === null || val === undefined || Number.isNaN(val)) return "\u2014";
  return val.toFixed(4);
}

export const MetricCard: React.FC<MetricCardProps> = ({
  name,
  value,
  delta,
  higherIsBetter = true,
  unit,
}) => {
  return (
    <div style={containerStyle} data-metric={name}>
      <p style={nameStyle}>{name}</p>
      <div style={valueRowStyle}>
        <p style={valueStyle}>{formatValue(value)}</p>
        {unit ? <span style={unitStyle}>{unit}</span> : null}
      </div>
      {delta !== undefined && delta !== null ? (
        <DeltaIndicator value={delta} higherIsBetter={higherIsBetter} />
      ) : null}
    </div>
  );
};

export default MetricCard;
