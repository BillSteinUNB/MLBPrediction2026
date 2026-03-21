import React, { Suspense } from "react";
import type { PlotParams } from "react-plotly.js";

/**
 * Lazy-loaded Plotly component.
 *
 * Uses React.lazy to code-split the heavy plotly.js-basic-dist bundle.
 * The factory pattern avoids pulling the full plotly.js distribution.
 */
const PlotComponent = React.lazy(
  () =>
    import("plotly.js-basic-dist").then((Plotly) =>
      import("react-plotly.js/factory").then((factory) => ({
        default: factory.default(Plotly.default ?? Plotly),
      }))
    ) as Promise<{ default: React.ComponentType<PlotParams> }>
);

const fallbackStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  minHeight: 260,
  color: "#6b7280",
  fontFamily: "Arial, sans-serif",
  fontSize: 14,
};

export interface LazyPlotProps extends PlotParams {
  /** Minimum height shown while the bundle loads */
  fallbackHeight?: number;
}

export const LazyPlot: React.FC<LazyPlotProps> = ({
  fallbackHeight = 260,
  ...plotProps
}) => (
  <Suspense
    fallback={
      <div style={{ ...fallbackStyle, minHeight: fallbackHeight }}>
        Loading chart…
      </div>
    }
  >
    <PlotComponent {...plotProps} />
  </Suspense>
);

export default LazyPlot;
