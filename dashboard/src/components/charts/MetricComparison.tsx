import React, { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-basic-dist';

export interface MetricComparisonRow {
  label: string;
  run1: number;
  run2: number;
}

export interface MetricComparisonProps {
  title: string;
  run1Label: string;
  run2Label: string;
  metrics: MetricComparisonRow[];
}

const containerStyle: React.CSSProperties = {
  width: 600,
  maxWidth: '100%',
  height: 400,
};

const MetricComparison: React.FC<MetricComparisonProps> = ({ title, run1Label, run2Label, metrics }) => {
  const plotRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!plotRef.current) return;

    const labels = metrics.map((m) => m.label);
    const run1Values = metrics.map((m) => m.run1);
    const run2Values = metrics.map((m) => m.run2);

    const trace1: Partial<Plotly.PlotData> = {
      x: labels,
      y: run1Values,
      name: run1Label,
      type: 'bar',
      marker: { color: '#1f77b4' },
    };

    const trace2: Partial<Plotly.PlotData> = {
      x: labels,
      y: run2Values,
      name: run2Label,
      type: 'bar',
      marker: { color: '#ff7f0e' },
    };

    const layout: Partial<Plotly.Layout> = {
      title: { text: title },
      barmode: 'group',
      xaxis: { title: { text: 'Metric' } },
      yaxis: { title: { text: 'Value' } },
      autosize: true,
      margin: { t: 40, l: 60, r: 20, b: 80 },
    };

    Plotly.newPlot(plotRef.current, [trace1, trace2], layout as any, { responsive: true });

    return () => {
      if (plotRef.current) {
        Plotly.purge(plotRef.current);
      }
    };
  }, [title, run1Label, run2Label, metrics]);

  return <div ref={plotRef} style={containerStyle} />;
};

export default MetricComparison;
