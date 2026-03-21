import React from "react";

export interface MetricCardProps {
  label: string;
  value: number | null;
  unit?: string;
  bgColor?: string;
}

const containerStyle: React.CSSProperties = {
  width: 200,
  background: "#f5f5f5",
  border: "1px solid #dbe2ea",
  borderRadius: 6,
  padding: 12,
  boxSizing: "border-box",
  fontFamily: "Arial, sans-serif",
  color: "#0b1220",
  display: "grid",
  gridTemplateRows: "auto 1fr",
  gap: 8,
};

const labelStyle: React.CSSProperties = {
  fontSize: 12,
  color: "#495464",
  margin: 0,
};

const valueRowStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "baseline",
  gap: 6,
};

const valueStyle: React.CSSProperties = {
  fontSize: 24,
  fontWeight: 600,
  color: "#0b1220",
  margin: 0,
};

const unitStyle: React.CSSProperties = {
  fontSize: 12,
  color: "#6b7280",
  margin: 0,
};

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  unit,
  bgColor,
}) => {
  const appliedStyle: React.CSSProperties = {
    ...containerStyle,
    background: bgColor ?? containerStyle.background,
  };

  const displayValue = value === null ? "—" : value;

  return (
    <div style={appliedStyle} aria-label={`metric-card-${label}`}>
      <p style={labelStyle}>{label}</p>
      <div style={valueRowStyle}>
        <p style={valueStyle}>{displayValue}</p>
        {unit ? <p style={unitStyle}>{unit}</p> : null}
      </div>
    </div>
  );
};

export default MetricCard;
