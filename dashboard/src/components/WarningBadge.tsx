import React from "react";

export type BadgeType = "warning" | "info" | "error";

export interface WarningBadgeProps {
  type: BadgeType;
  text: string;
}

const baseStyle: React.CSSProperties = {
  display: "inline-block",
  padding: "4px 10px",
  borderRadius: 16,
  fontSize: 13,
  fontFamily: "Arial, sans-serif",
  color: "#0b1220",
  border: "1px solid #dbe2ea",
};

const typeStyles: Record<BadgeType, React.CSSProperties> = {
  warning: {
    background: "#fff4e5",
    color: "#6a4a00",
  },
  info: {
    background: "#e8f0ff",
    color: "#0b63d8",
  },
  error: {
    background: "#fdecea",
    color: "#9f1f1f",
  },
};

export const WarningBadge: React.FC<WarningBadgeProps> = ({ type, text }) => {
  const style: React.CSSProperties = { ...baseStyle, ...typeStyles[type] };
  return (
    <span style={style} role="status" aria-label={`badge-${type}`}>
      {text}
    </span>
  );
};

export default WarningBadge;
