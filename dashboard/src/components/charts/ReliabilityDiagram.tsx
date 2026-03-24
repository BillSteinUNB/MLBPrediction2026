import React, { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-basic-dist';

export interface CalibrationPoint {
  predicted: number;
  actual: number;
}

export interface ReliabilityDiagramProps {
  title: string;
  calibrationData: CalibrationPoint[];
}

const containerStyle: React.CSSProperties = {
  width: 600,
  maxWidth: '100%',
  height: 400,
};

const ReliabilityDiagram: React.FC<ReliabilityDiagramProps> = ({ title, calibrationData }) => {
  const plotRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!plotRef.current) return;

    const x = calibrationData.map((d) => d.predicted);
    const y = calibrationData.map((d) => d.actual);

    const scatter: Partial<Plotly.PlotData> = {
      x,
      y,
      mode: 'markers',
      type: 'scatter',
      marker: { color: '#1f77b4' },
      name: 'Calibration',
    };

    const diag: Partial<Plotly.PlotData> = {
      x: [0, 1],
      y: [0, 1],
      mode: 'lines',
      type: 'scatter',
      line: { dash: 'dash', color: 'gray' },
      hoverinfo: 'skip',
      showlegend: false,
    };

    const layout: Partial<Plotly.Layout> = {
      title: { text: title },
      xaxis: { title: { text: 'Predicted probability' }, range: [0, 1] },
      yaxis: { title: { text: 'Observed (actual) probability' }, range: [0, 1] },
      autosize: true,
      margin: { t: 40, l: 60, r: 20, b: 60 },
    };

    Plotly.newPlot(plotRef.current, [scatter, diag], layout as any, { responsive: true });

    return () => {
      if (plotRef.current) {
        Plotly.purge(plotRef.current);
      }
    };
  }, [calibrationData, title]);

  return <div ref={plotRef} style={containerStyle} />;
};

export default ReliabilityDiagram;
