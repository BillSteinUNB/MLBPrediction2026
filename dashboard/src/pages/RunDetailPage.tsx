import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { getRunDetail } from "../api";
import type { RunDetail } from "../api";
import { MetricCard } from "../components/MetricCard";
import { WarningBadge } from "../components/WarningBadge";
import { FeatureImportanceChart } from "../charts/FeatureImportanceChart";
import { ReliabilityDiagramChart } from "../charts/ReliabilityDiagramChart";

/* ============================================================================
 * METRIC DEFINITIONS
 * ========================================================================= */

interface MetricDef {
  key: keyof RunDetail;
  label: string;
  higherIsBetter: boolean;
  deltaKey?: keyof RunDetail;
}

const METRIC_DEFS: MetricDef[] = [
  { key: "roc_auc", label: "ROC AUC", higherIsBetter: true, deltaKey: "delta_vs_prev_roc_auc" },
  { key: "log_loss", label: "Log Loss", higherIsBetter: false, deltaKey: "delta_vs_prev_log_loss" },
  { key: "brier", label: "Brier Score", higherIsBetter: false, deltaKey: "delta_vs_prev_brier" },
  { key: "accuracy", label: "Accuracy", higherIsBetter: true, deltaKey: "delta_vs_prev_accuracy" },
  { key: "ece", label: "ECE", higherIsBetter: false },
  { key: "reliability_gap", label: "Reliability Gap", higherIsBetter: false },
];

/* ============================================================================
 * STYLES (aligned to layout.css / design tokens)
 * ========================================================================= */

const headerStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "flex-start",
  justifyContent: "space-between",
  gap: 16,
  flexWrap: "wrap",
  marginBottom: 20,
};

const backBtnStyle: React.CSSProperties = {
  background: "none",
  border: "1px solid #e5e4e7",
  borderRadius: 6,
  padding: "6px 14px",
  fontSize: 13,
  fontFamily: "Arial, Helvetica, sans-serif",
  color: "#6b6375",
  cursor: "pointer",
  transition: "background 0.15s, color 0.15s",
  whiteSpace: "nowrap",
};

const titleRowStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 10,
  flexWrap: "wrap",
};

const kindBadgeStyle: React.CSSProperties = {
  display: "inline-block",
  padding: "3px 10px",
  borderRadius: 10,
  background: "#eef2f6",
  fontSize: 12,
  fontWeight: 500,
  color: "#495464",
  fontFamily: "Arial, Helvetica, sans-serif",
  textTransform: "capitalize",
};

const metaRowStyle: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: 20,
  fontSize: 13,
  color: "#6b6375",
  fontFamily: "Arial, Helvetica, sans-serif",
  marginBottom: 20,
};

const metaLabelStyle: React.CSSProperties = {
  fontWeight: 600,
  color: "#08060d",
  marginRight: 4,
};

const warningRowStyle: React.CSSProperties = {
  display: "flex",
  gap: 8,
  flexWrap: "wrap",
  marginBottom: 20,
};

const metricsGridStyle: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
  gap: 14,
  marginBottom: 24,
};

const sectionStyle: React.CSSProperties = {
  marginBottom: 24,
};

const sectionTitleStyle: React.CSSProperties = {
  fontSize: 14,
  fontWeight: 600,
  color: "#08060d",
  fontFamily: "Arial, Helvetica, sans-serif",
  margin: "0 0 10px 0",
};

const preStyle: React.CSSProperties = {
  fontFamily: "ui-monospace, Consolas, monospace",
  fontSize: 13,
  lineHeight: "145%",
  background: "#f4f3ec",
  border: "1px solid #e5e4e7",
  borderRadius: 6,
  padding: 14,
  margin: 0,
  overflowX: "auto",
  color: "#08060d",
};

const centerStyle: React.CSSProperties = {
  display: "flex",
  justifyContent: "center",
  alignItems: "center",
  minHeight: 200,
  color: "#6b6375",
  fontSize: 14,
  fontFamily: "Arial, Helvetica, sans-serif",
};

const errorBoxStyle: React.CSSProperties = {
  background: "#fdecea",
  border: "1px solid #f5c6c6",
  borderRadius: 6,
  padding: 20,
  color: "#9f1f1f",
  fontSize: 14,
  fontFamily: "Arial, Helvetica, sans-serif",
  textAlign: "center",
};

/* ============================================================================
 * COMPONENT
 * ========================================================================= */

const RunDetailPage: React.FC = () => {
  const { summaryPath } = useParams<{ summaryPath: string }>();
  const navigate = useNavigate();

  const [run, setRun] = useState<RunDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const decodedPath = summaryPath ? decodeURIComponent(summaryPath) : "";

  useEffect(() => {
    if (!decodedPath) return;

    let cancelled = false;
    setLoading(true);
    setError(null);

    getRunDetail(encodeURIComponent(decodedPath))
      .then((data) => {
        if (!cancelled) setRun(data);
      })
      .catch((err: Error) => {
        if (!cancelled) {
          const is404 = err.message.includes("404");
          setError(is404 ? "Run not found." : err.message);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [decodedPath]);

  /* ---- Loading state ---- */
  if (loading) {
    return (
      <div className="page">
        <div style={centerStyle}>Loading run detail\u2026</div>
      </div>
    );
  }

  /* ---- Error / 404 state ---- */
  if (error || !run) {
    return (
      <div className="page">
        <div style={{ marginBottom: 16 }}>
          <button
            type="button"
            style={backBtnStyle}
            onClick={() => navigate(-1)}
            onMouseEnter={(e) => {
              (e.currentTarget.style.background = "#f7f7f8");
              (e.currentTarget.style.color = "#08060d");
            }}
            onMouseLeave={(e) => {
              (e.currentTarget.style.background = "none");
              (e.currentTarget.style.color = "#6b6375");
            }}
          >
            \u2190 Back
          </button>
        </div>
        <div style={errorBoxStyle}>
          {error ?? "Run not found."}
        </div>
      </div>
    );
  }

  /* ---- Derived data ---- */
  const isTraining = run.run_kind === "training";
  const isCalibration = run.run_kind === "calibration";
  const isStacking = run.run_kind === "stacking";

  const hasFeatureImportance =
    isTraining && run.feature_importance && run.feature_importance.length > 0;

  const hasReliabilityDiagram =
    isCalibration && run.reliability_diagram && run.reliability_diagram.length > 0;

  const hasStackingMetrics =
    isStacking && run.stacking_metrics && Object.keys(run.stacking_metrics).length > 0;

  const hasBestParams =
    run.best_params && Object.keys(run.best_params).length > 0;

  const hasQualityGates =
    run.quality_gates && Object.keys(run.quality_gates).length > 0;

  /* Map API BinItem → CalibrationBin for the chart */
  const calibrationBins =
    run.reliability_diagram?.map((bin) => ({
      predicted: bin.predicted_mean,
      observed: bin.true_fraction,
      count: bin.count,
    })) ?? [];

  /* ---- Render ---- */
  return (
    <div className="page">
      {/* Header */}
      <div style={headerStyle}>
        <div>
          <div style={{ marginBottom: 8 }}>
            <button
              type="button"
              style={backBtnStyle}
              onClick={() => navigate(-1)}
              onMouseEnter={(e) => {
                (e.currentTarget.style.background = "#f7f7f8");
                (e.currentTarget.style.color = "#08060d");
              }}
              onMouseLeave={(e) => {
                (e.currentTarget.style.background = "none");
                (e.currentTarget.style.color = "#6b6375");
              }}
            >
              \u2190 Back
            </button>
          </div>
          <div style={titleRowStyle}>
            <h2 style={{ margin: 0 }}>{run.experiment_name}</h2>
            <span style={kindBadgeStyle}>{run.run_kind}</span>
          </div>
        </div>
      </div>

      {/* Meta row */}
      <div style={metaRowStyle}>
        <span>
          <span style={metaLabelStyle}>Model:</span>
          {run.model_name}
        </span>
        <span>
          <span style={metaLabelStyle}>Variant:</span>
          {run.variant}
        </span>
        <span>
          <span style={metaLabelStyle}>Version:</span>
          {run.model_version}
        </span>
        <span>
          <span style={metaLabelStyle}>Target:</span>
          {run.target_column}
        </span>
        <span>
          <span style={metaLabelStyle}>Holdout:</span>
          {run.holdout_season}
        </span>
        <span>
          <span style={metaLabelStyle}>Timestamp:</span>
          {run.run_timestamp}
        </span>
        {run.train_row_count != null && (
          <span>
            <span style={metaLabelStyle}>Train rows:</span>
            {run.train_row_count.toLocaleString()}
          </span>
        )}
        {run.holdout_row_count != null && (
          <span>
            <span style={metaLabelStyle}>Holdout rows:</span>
            {run.holdout_row_count.toLocaleString()}
          </span>
        )}
        {run.feature_column_count != null && (
          <span>
            <span style={metaLabelStyle}>Features:</span>
            {run.feature_column_count}
          </span>
        )}
        {run.calibration_method != null && (
          <span>
            <span style={metaLabelStyle}>Calibration:</span>
            {run.calibration_method}
          </span>
        )}
      </div>

      {/* Warning badges for high ECE / reliability_gap */}
      {(run.ece != null || run.reliability_gap != null) && (
        <div style={warningRowStyle}>
          <WarningBadge label="ECE" value={run.ece ?? null} threshold={0.05} />
          <WarningBadge
            label="Reliability Gap"
            value={run.reliability_gap ?? null}
            threshold={0.05}
          />
        </div>
      )}

      {/* Metric cards */}
      <div style={metricsGridStyle}>
        {METRIC_DEFS.map((def) => {
          const val = run[def.key] as number | null | undefined;
          const delta = def.deltaKey
            ? (run[def.deltaKey] as number | null | undefined) ?? null
            : null;

          return (
            <MetricCard
              key={def.key}
              name={def.label}
              value={val ?? null}
              delta={delta}
              higherIsBetter={def.higherIsBetter}
            />
          );
        })}
      </div>

      {/* Feature Importance Chart (training runs) */}
      {hasFeatureImportance && (
        <div style={sectionStyle}>
          <p style={sectionTitleStyle}>Feature Importance</p>
          <FeatureImportanceChart
            data={run.feature_importance!}
            title="Top Feature Importances"
          />
        </div>
      )}

      {/* Reliability Diagram (calibration runs) */}
      {hasReliabilityDiagram && (
        <div style={sectionStyle}>
          <p style={sectionTitleStyle}>Reliability Diagram</p>
          <ReliabilityDiagramChart
            bins={calibrationBins}
            title="Calibration Reliability"
          />
        </div>
      )}

      {/* Stacking metrics (stacking runs) */}
      {hasStackingMetrics && (
        <div style={sectionStyle}>
          <p style={sectionTitleStyle}>Stacking Metrics</p>
          <pre style={preStyle}>
            {JSON.stringify(run.stacking_metrics, null, 2)}
          </pre>
        </div>
      )}

      {/* Best params */}
      {hasBestParams && (
        <div style={sectionStyle}>
          <p style={sectionTitleStyle}>Best Parameters</p>
          <pre style={preStyle}>
            {JSON.stringify(run.best_params, null, 2)}
          </pre>
        </div>
      )}

      {/* Quality gates */}
      {hasQualityGates && (
        <div style={sectionStyle}>
          <p style={sectionTitleStyle}>Quality Gates</p>
          <QualityGatesTable gates={run.quality_gates!} />
        </div>
      )}
    </div>
  );
};

/* ============================================================================
 * QUALITY GATES TABLE
 * ========================================================================= */

interface QualityGatesTableProps {
  gates: Record<string, unknown>;
}

const gateTableStyle: React.CSSProperties = {
  width: "100%",
  borderCollapse: "collapse",
  fontFamily: "Arial, Helvetica, sans-serif",
  fontSize: 13,
};

const gateThStyle: React.CSSProperties = {
  textAlign: "left",
  padding: "8px 10px",
  borderBottom: "2px solid #e5e4e7",
  fontSize: 11,
  fontWeight: 600,
  color: "#6b6375",
  textTransform: "uppercase",
  letterSpacing: "0.04em",
};

const gateTdStyle: React.CSSProperties = {
  padding: "7px 10px",
  borderBottom: "1px solid #e5e4e7",
  color: "#08060d",
};

const passStyle: React.CSSProperties = {
  display: "inline-block",
  padding: "2px 8px",
  borderRadius: 10,
  background: "#e6f6ea",
  color: "#0f7a3a",
  fontSize: 12,
  fontWeight: 500,
};

const failStyle: React.CSSProperties = {
  display: "inline-block",
  padding: "2px 8px",
  borderRadius: 10,
  background: "#fdecea",
  color: "#9f1f1f",
  fontSize: 12,
  fontWeight: 500,
};

const QualityGatesTable: React.FC<QualityGatesTableProps> = ({ gates }) => {
  const entries = Object.entries(gates);

  if (entries.length === 0) return null;

  /* If gate values are booleans, render as pass/fail table */
  const isBooleanGates = entries.every(
    ([, v]) => typeof v === "boolean"
  );

  if (isBooleanGates) {
    return (
      <table style={gateTableStyle}>
        <thead>
          <tr>
            <th style={gateThStyle}>Gate</th>
            <th style={{ ...gateThStyle, textAlign: "center" }}>Status</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([key, val]) => (
            <tr key={key}>
              <td style={gateTdStyle}>{key}</td>
              <td style={{ ...gateTdStyle, textAlign: "center" }}>
                <span style={val ? passStyle : failStyle}>
                  {val ? "Pass" : "Fail"}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  }

  /* Fallback: render as key/value JSON-like */
  return (
    <pre style={preStyle}>{JSON.stringify(gates, null, 2)}</pre>
  );
};

export default RunDetailPage;
