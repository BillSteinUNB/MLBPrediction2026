import React from "react";
import type { Layout } from "plotly.js";
import { LazyPlot } from "./LazyPlot";

export interface FeatureImportanceEntry {
  /** Feature name */
  feature: string;
  /** Importance score (e.g. gain, SHAP mean absolute) */
  importance: number;
}

export interface FeatureImportanceChartProps {
  /** Feature importance entries — will be sorted descending internally */
  data: FeatureImportanceEntry[];
  /** Optional chart title */
  title?: string;
  /** Maximum number of features to show (default 15) */
  topN?: number;
  /** Chart height in pixels (auto-sized per bar if omitted) */
  height?: number;
}

const baseLayout: Partial<Layout> = {
  font: { family: "Arial, Helvetica, sans-serif", color: "#495464", size: 13 },
  paper_bgcolor: "transparent",
  plot_bgcolor: "#ffffff",
  hovermode: "closest",
  xaxis: {
    gridcolor: "#e5e4e7",
    linecolor: "#dbe2ea",
    tickfont: { size: 11, color: "#6b7280" },
    zeroline: false,
  },
  yaxis: {
    gridcolor: "#e5e4e7",
    linecolor: "#dbe2ea",
    tickfont: { size: 11, color: "#495464" },
    automargin: true,
  },
};

const containerStyle: React.CSSProperties = {
  width: "100%",
  background: "#ffffff",
  border: "1px solid #dbe2ea",
  borderRadius: 6,
  overflow: "hidden",
  boxSizing: "border-box",
};

export const FeatureImportanceChart: React.FC<FeatureImportanceChartProps> = ({
  data,
  title,
  topN = 15,
  height: heightProp,
}) => {
  // Sort descending, take topN, then reverse so highest is at top of horizontal bar
  const sorted = [...data]
    .sort((a, b) => b.importance - a.importance)
    .slice(0, topN)
    .reverse();

  const features = sorted.map((d) => d.feature);
  const values = sorted.map((d) => d.importance);

  const barHeight = 28;
  const height = heightProp ?? Math.max(220, sorted.length * barHeight + 80);

  const trace: Plotly.Data = {
    x: values,
    y: features,
    type: "bar",
    orientation: "h",
    marker: { color: "#0b63d8" },
    hovertemplate: "%{y}<br>Importance: %{x:.4f}<extra></extra>",
  };

  const layout: Partial<Layout> = {
    ...baseLayout,
    height,
    margin: { l: 12, r: 24, t: 36, b: 40 },
    title: title
      ? { text: title, font: { size: 14, color: "#0b1220" } }
      : undefined,
    xaxis: {
      ...baseLayout.xaxis,
      title: { text: "Importance", standoff: 8 },
    },
    yaxis: {
      ...baseLayout.yaxis,
      tickfont: { size: 11, color: "#495464" },
      automargin: true,
    },
  };

  return (
    <div style={containerStyle}>
      <LazyPlot
        data={[trace]}
        layout={layout}
        config={{ displayModeBar: false, responsive: true }}
        useResizeHandler
        style={{ width: "100%", height }}
        fallbackHeight={height}
      />
    </div>
  );
};

export default FeatureImportanceChart;
