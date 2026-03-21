import React from "react";
import type { Layout } from "plotly.js";
import { LazyPlot } from "./LazyPlot";

export interface MetricComparisonEntry {
  /** Metric name (x-axis category label) */
  metric: string;
  /** Value from the baseline / first run */
  valueA: number;
  /** Value from the candidate / second run */
  valueB: number;
}

export interface MetricComparisonChartProps {
  /** Array of metrics to compare */
  data: MetricComparisonEntry[];
  /** Label for first run (e.g. "Baseline") */
  labelA: string;
  /** Label for second run (e.g. "Candidate") */
  labelB: string;
  /** Optional chart title */
  title?: string;
  /** Chart height in pixels */
  height?: number;
}

const baseLayout: Partial<Layout> = {
  font: { family: "Arial, Helvetica, sans-serif", color: "#495464", size: 13 },
  paper_bgcolor: "transparent",
  plot_bgcolor: "#ffffff",
  margin: { l: 56, r: 24, t: 36, b: 56 },
  hovermode: "closest",
  xaxis: {
    gridcolor: "#e5e4e7",
    linecolor: "#dbe2ea",
    tickfont: { size: 11, color: "#495464" },
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

/** Color palette: blue for run A, teal/green for run B */
const COLOR_A = "#0b63d8";
const COLOR_B = "#0f7a3a";

export const MetricComparisonChart: React.FC<MetricComparisonChartProps> = ({
  data,
  labelA,
  labelB,
  title,
  height = 340,
}) => {
  const metrics = data.map((d) => d.metric);
  const valuesA = data.map((d) => d.valueA);
  const valuesB = data.map((d) => d.valueB);

  const traceA: Plotly.Data = {
    x: metrics,
    y: valuesA,
    type: "bar",
    name: labelA,
    marker: { color: COLOR_A },
    hovertemplate: `${labelA}<br>%{x}: %{y:.4f}<extra></extra>`,
  };

  const traceB: Plotly.Data = {
    x: metrics,
    y: valuesB,
    type: "bar",
    name: labelB,
    marker: { color: COLOR_B },
    hovertemplate: `${labelB}<br>%{x}: %{y:.4f}<extra></extra>`,
  };

  const layout: Partial<Layout> = {
    ...baseLayout,
    height,
    title: title
      ? { text: title, font: { size: 14, color: "#0b1220" } }
      : undefined,
    barmode: "group",
    xaxis: {
      ...baseLayout.xaxis,
      title: { text: "Metric", standoff: 8 },
    },
    yaxis: {
      ...baseLayout.yaxis,
      title: { text: "Value", standoff: 8 },
    },
    showlegend: true,
    legend: { x: 0.02, y: 0.98, font: { size: 11 } },
    bargap: 0.2,
    bargroupgap: 0.06,
  };

  return (
    <div style={containerStyle}>
      <LazyPlot
        data={[traceA, traceB]}
        layout={layout}
        config={{ displayModeBar: false, responsive: true }}
        useResizeHandler
        style={{ width: "100%", height }}
        fallbackHeight={height}
      />
    </div>
  );
};

export default MetricComparisonChart;
