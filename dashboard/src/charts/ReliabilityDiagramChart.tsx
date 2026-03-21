import React from "react";
import type { Layout } from "plotly.js";
import { LazyPlot } from "./LazyPlot";

export interface CalibrationBin {
  /** Bin midpoint predicted probability (e.g. 0.15) */
  predicted: number;
  /** Observed frequency within this bin */
  observed: number;
  /** Number of samples in this bin (shown in hover) */
  count: number;
}

export interface ReliabilityDiagramChartProps {
  /** Calibration bins from model evaluation */
  bins: CalibrationBin[];
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
    tickfont: { size: 11, color: "#6b7280" },
    range: [0, 1],
  },
  yaxis: {
    gridcolor: "#e5e4e7",
    linecolor: "#dbe2ea",
    tickfont: { size: 11, color: "#6b7280" },
    range: [0, 1],
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

export const ReliabilityDiagramChart: React.FC<ReliabilityDiagramChartProps> = ({
  bins,
  title,
  height = 360,
}) => {
  const predicted = bins.map((b) => b.predicted);
  const observed = bins.map((b) => b.observed);
  const counts = bins.map((b) => b.count);

  /** Calibration bars — observed frequency per bin */
  const barTrace: Plotly.Data = {
    x: predicted,
    y: observed,
    type: "bar",
    name: "Observed",
    marker: { color: "#0b63d8", opacity: 0.75 },
    width: predicted.length > 1 ? (predicted[1] - predicted[0]) * 0.85 : 0.08,
    customdata: counts,
    hovertemplate:
      "Predicted: %{x:.2f}<br>Observed: %{y:.3f}<br>n = %{customdata}<extra></extra>",
  };

  /** Perfect-calibration diagonal reference line */
  const refLine: Plotly.Data = {
    x: [0, 1],
    y: [0, 1],
    type: "scatter",
    mode: "lines",
    name: "Perfect",
    line: { color: "#9ca3af", width: 1.5, dash: "dash" },
    hoverinfo: "skip",
  };

  const layout: Partial<Layout> = {
    ...baseLayout,
    height,
    title: title
      ? { text: title, font: { size: 14, color: "#0b1220" } }
      : undefined,
    xaxis: {
      ...baseLayout.xaxis,
      title: { text: "Predicted Probability", standoff: 8 },
    },
    yaxis: {
      ...baseLayout.yaxis,
      title: { text: "Observed Frequency", standoff: 8 },
    },
    showlegend: true,
    legend: { x: 0.02, y: 0.98, font: { size: 11 } },
    bargap: 0.05,
  };

  return (
    <div style={containerStyle}>
      <LazyPlot
        data={[barTrace, refLine]}
        layout={layout}
        config={{ displayModeBar: false, responsive: true }}
        useResizeHandler
        style={{ width: "100%", height }}
        fallbackHeight={height}
      />
    </div>
  );
};

export default ReliabilityDiagramChart;
