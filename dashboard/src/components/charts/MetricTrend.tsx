import React, { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-basic-dist';

export interface MetricPoint {
  timestamp: string;
  value: number;
}

export interface MetricTrendProps {
  title: string;
  data: MetricPoint[];
  yLabel: string;
}

const containerStyle: React.CSSProperties = {
  width: 600,
  maxWidth: '100%',
  height: 400,
};

const MetricTrend: React.FC<MetricTrendProps> = ({ title, data, yLabel }) => {
  const plotRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!plotRef.current) return;

    const x = data.map((d) => d.timestamp);
    const y = data.map((d) => d.value);

    const trace: Partial<Plotly.PlotData> = {
      x,
      y,
      type: 'scatter',
      mode: 'lines',
      line: { color: '#4da6ff' }, // light blue
      name: yLabel,
    };

    const layout: Partial<Plotly.Layout> = {
      title: { text: title },
      xaxis: { title: { text: 'Time' } },
      yaxis: { title: { text: yLabel } },
      autosize: true,
      margin: { t: 40, l: 60, r: 20, b: 60 },
    };

    Plotly.newPlot(plotRef.current, [trace], layout as any, { responsive: true });

    return () => {
      if (plotRef.current) {
        Plotly.purge(plotRef.current);
      }
    };
  }, [data, title, yLabel]);

  return <div ref={plotRef} style={containerStyle} />;
};

export default MetricTrend;
