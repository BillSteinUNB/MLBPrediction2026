import React from "react";
import type { Layout } from "plotly.js";
import { LazyPlot } from "./LazyPlot";

export interface MetricTrendPoint {
  /** ISO date string or label for the x-axis */
  date: string;
  value: number;
}

export interface MetricTrendChartProps {
  /** Array of { date, value } points */
  data: MetricTrendPoint[];
  /** Y-axis label (e.g. "Brier Score") */
  metricName: string;
  /** Optional chart title */
  title?: string;
  /** Chart height in pixels */
  height?: number;
}

/** Design-system–aligned Plotly layout defaults */
const baseLayout: Partial<Layout> = {
  font: { family: "Arial, Helvetica, sans-serif", color: "#495464", size: 13 },
  paper_bgcolor: "transparent",
  plot_bgcolor: "#ffffff",
  margin: { l: 56, r: 24, t: 36, b: 48 },
  hovermode: "x unified" as Layout["hovermode"],
  xaxis: {
    gridcolor: "#e5e4e7",
    linecolor: "#dbe2ea",
    tickfont: { size: 11, color: "#6b7280" },
  },
  yaxis: {
    gridcolor: "#e5e4e7",
    linecolor: "#dbe2ea",
    tickfont: { size: 11, color: "#6b7280" },
    zeroline: false,
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

export const MetricTrendChart: React.FC<MetricTrendChartProps> = ({
  data,
  metricName,
  title,
  height = 320,
}) => {
  const dates = data.map((d) => d.date);
  const values = data.map((d) => d.value);

  const trace: Plotly.Data = {
    x: dates,
    y: values,
    type: "scatter",
    mode: "lines+markers",
    name: metricName,
    line: { color: "#0b63d8", width: 2, shape: "spline" },
    marker: { size: 5, color: "#0b63d8" },
    hovertemplate: `%{x}<br>${metricName}: %{y:.4f}<extra></extra>`,
  };

  const layout: Partial<Layout> = {
    ...baseLayout,
    height,
    title: title
      ? { text: title, font: { size: 14, color: "#0b1220" } }
      : undefined,
    xaxis: { ...baseLayout.xaxis, title: { text: "Date", standoff: 8 } },
    yaxis: { ...baseLayout.yaxis, title: { text: metricName, standoff: 8 } },
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

export default MetricTrendChart;
