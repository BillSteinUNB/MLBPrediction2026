import React from "react";
import { useNavigate } from "react-router-dom";
import type { RunSummary } from "../api/types";
import { DeltaIndicator } from "./DeltaIndicator";
import { TooltipLabel } from "./TooltipLabel";

export interface RunTableProps {
  /** Array of run summaries to display */
  runs: RunSummary[];
  /** Optional title above the table */
  title?: string;
  /** Hide the delta columns (useful for single-run lists with no comparison context) */
  hideDelta?: boolean;
}

/* ---- metric column definitions ---- */
interface MetricCol {
  key: keyof RunSummary;
  label: string;
  higherIsBetter: boolean;
  deltaKey?: keyof RunSummary;
}

const METRIC_COLS: MetricCol[] = [
  {
    key: "roc_auc",
    label: "ROC AUC",
    higherIsBetter: true,
    deltaKey: "delta_vs_prev_roc_auc",
  },
  {
    key: "log_loss",
    label: "Log Loss",
    higherIsBetter: false,
    deltaKey: "delta_vs_prev_log_loss",
  },
  {
    key: "brier",
    label: "Brier",
    higherIsBetter: false,
    deltaKey: "delta_vs_prev_brier",
  },
  {
    key: "accuracy",
    label: "Accuracy",
    higherIsBetter: true,
    deltaKey: "delta_vs_prev_accuracy",
  },
];

/* ---- styles ---- */
const tableStyle: React.CSSProperties = {
  width: "100%",
  borderCollapse: "collapse",
  fontFamily: "Arial, Helvetica, sans-serif",
  fontSize: 13,
  color: "var(--text-h)",
};

const thStyle: React.CSSProperties = {
  textAlign: "left",
  padding: "8px 10px",
  borderBottom: "2px solid var(--border)",
  fontSize: 11,
  fontWeight: 600,
  color: "var(--text)",
  textTransform: "uppercase",
  letterSpacing: "0.04em",
  whiteSpace: "nowrap",
};

const tdStyle: React.CSSProperties = {
  padding: "7px 10px",
  borderBottom: "1px solid var(--border)",
  verticalAlign: "middle",
  whiteSpace: "nowrap",
};

const rowHoverBg = "var(--bg-hover)";

const titleStyle: React.CSSProperties = {
  fontSize: 14,
  fontWeight: 600,
  color: "var(--text-h)",
  margin: "0 0 8px 0",
};

const emptyStyle: React.CSSProperties = {
  padding: 20,
  textAlign: "center",
  color: "var(--muted)",
  fontSize: 13,
};

const metricCellStyle: React.CSSProperties = {
  fontFamily: "ui-monospace, Consolas, monospace",
  fontSize: 13,
};

const deltaCellStyle: React.CSSProperties = {
  textAlign: "center",
};

/**
 * Format a metric value to 4 decimal places, or "—" for null.
 */
function fmt(val: number | null | undefined): string {
  if (val === null || val === undefined || Number.isNaN(val)) return "\u2014";
  return val.toFixed(4);
}

export const RunTable: React.FC<RunTableProps> = ({
  runs,
  title,
  hideDelta = false,
}) => {
  const navigate = useNavigate();

  const handleRowClick = (summaryPath: string) => {
    navigate(`/runs/${encodeURIComponent(summaryPath)}`);
  };

  if (runs.length === 0) {
    return (
      <div>
        {title ? <p style={titleStyle}>{title}</p> : null}
        <p style={emptyStyle}>No runs to display.</p>
      </div>
    );
  }

  return (
    <div>
      {title ? <p style={titleStyle}>{title}</p> : null}
      <div style={{ overflowX: "auto" }}>
        <table style={tableStyle}>
          <thead>
            <tr>
              <th style={thStyle}>
                <TooltipLabel label="Experiment" />
              </th>
              <th style={thStyle}>
                <TooltipLabel label="Variant" />
              </th>
              {METRIC_COLS.map((col) => (
                <React.Fragment key={col.key}>
                  <th style={{ ...thStyle, textAlign: "right" }}>
                    <TooltipLabel label={col.label} />
                  </th>
                  {!hideDelta && col.deltaKey ? (
                    <th style={{ ...thStyle, textAlign: "center" }}>
                      <TooltipLabel label="Δ" helpText="Change versus the previous run in the same lane. Positive is good for ROC AUC and Accuracy. Negative is good for Log Loss and Brier." />
                    </th>
                  ) : null}
                </React.Fragment>
              ))}
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => (
              <RunRow
                key={run.summary_path}
                run={run}
                hideDelta={hideDelta}
                onClick={() => handleRowClick(run.summary_path)}
              />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

/* ---- Row sub-component ---- */

interface RunRowProps {
  run: RunSummary;
  hideDelta: boolean;
  onClick: () => void;
}

const RunRow: React.FC<RunRowProps> = ({ run, hideDelta, onClick }) => {
  const [hovered, setHovered] = React.useState(false);

  const rowStyle: React.CSSProperties = {
    cursor: "pointer",
    background: hovered ? rowHoverBg : "transparent",
    transition: "background 0.12s",
  };

  return (
    <tr
      style={rowStyle}
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <td style={tdStyle}>{run.experiment_name}</td>
      <td style={tdStyle}>
        <span
          style={{
            display: "inline-block",
            padding: "2px 8px",
            borderRadius: 10,
            background: "var(--surface-2)",
            fontSize: 12,
            color: "var(--text)",
          }}
        >
          {run.variant}
        </span>
      </td>
      {METRIC_COLS.map((col) => {
        const val = run[col.key] as number | null | undefined;
        const deltaVal = col.deltaKey
          ? (run[col.deltaKey] as number | null | undefined)
          : null;

        return (
          <React.Fragment key={col.key}>
            <td style={{ ...tdStyle, ...metricCellStyle, textAlign: "right" }}>
              {fmt(val)}
            </td>
            {!hideDelta && col.deltaKey ? (
              <td style={{ ...tdStyle, ...deltaCellStyle }}>
                {deltaVal !== null && deltaVal !== undefined ? (
                  <DeltaIndicator
                    value={deltaVal}
                    higherIsBetter={col.higherIsBetter}
                  />
                ) : (
                  <span style={{ color: "var(--muted)" }}>{"\u2014"}</span>
                )}
              </td>
            ) : null}
          </React.Fragment>
        );
      })}
    </tr>
  );
};

export default RunTable;
