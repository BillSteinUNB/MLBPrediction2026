import React, { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-basic-dist';

export interface Feature {
  name: string;
  importance: number;
}

export interface FeatureImportanceProps {
  title: string;
  features: Feature[];
  limit?: number;
}

const containerStyle: React.CSSProperties = {
  width: 600,
  maxWidth: '100%',
  height: 400,
};

const FeatureImportance: React.FC<FeatureImportanceProps> = ({ title, features, limit = 10 }) => {
  const plotRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!plotRef.current) return;

    // Sort descending and take top N
    const sorted = [...features].sort((a, b) => b.importance - a.importance).slice(0, limit);

    const y = sorted.map((f) => f.name).reverse(); // reverse so highest is at top
    const x = sorted.map((f) => f.importance).reverse();

    const trace: Partial<Plotly.PlotData> = {
      x,
      y,
      type: 'bar',
      orientation: 'h',
      text: x.map((v) => v.toFixed(3)),
      textposition: 'auto',
      marker: { color: '#66c2ff' },
    };

    const layout: Partial<Plotly.Layout> = {
      title: { text: title },
      xaxis: { title: { text: 'Importance' } },
      autosize: true,
      margin: { t: 40, l: 150, r: 20, b: 60 },
    };

    Plotly.newPlot(plotRef.current, [trace], layout as any, { responsive: true });

    return () => {
      if (plotRef.current) {
        Plotly.purge(plotRef.current);
      }
    };
  }, [features, limit, title]);

  return <div ref={plotRef} style={containerStyle} />;
};

export default FeatureImportance;
