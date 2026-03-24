import React from "react";
import { listRuns, compareRuns } from "../api";
import type { RunSummary, CompareResult } from "../api";
import { MetricCard } from "../components";
import { MetricComparisonChart } from "../charts";
import type { MetricComparisonEntry } from "../charts";

/* ---- Metric definitions ---- */
interface MetricDef {
  key: string;
  label: string;
  higherIsBetter: boolean;
}

const METRICS: MetricDef[] = [
  { key: "roc_auc", label: "ROC AUC", higherIsBetter: true },
  { key: "log_loss", label: "Log Loss", higherIsBetter: false },
  { key: "brier", label: "Brier", higherIsBetter: false },
  { key: "accuracy", label: "Accuracy", higherIsBetter: true },
];

/* ---- Style tokens (aligned to layout.css / existing components) ---- */

const selectorBarStyle: React.CSSProperties = {
  display: "flex",
  gap: 16,
  alignItems: "flex-end",
  flexWrap: "wrap",
  marginBottom: 20,
};

const selectorGroupStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 4,
  flex: "1 1 240px",
  minWidth: 200,
};

const selectorLabelStyle: React.CSSProperties = {
  fontSize: 11,
  fontWeight: 600,
  color: "#6b6375",
  textTransform: "uppercase",
  letterSpacing: "0.04em",
  margin: 0,
};

const selectStyle: React.CSSProperties = {
  fontFamily: "Arial, Helvetica, sans-serif",
  fontSize: 13,
  color: "#08060d",
  padding: "8px 10px",
  border: "1px solid #e5e4e7",
  borderRadius: 6,
  background: "#ffffff",
  cursor: "pointer",
  outline: "none",
  width: "100%",
  boxSizing: "border-box",
};

const compareButtonStyle: React.CSSProperties = {
  fontFamily: "Arial, Helvetica, sans-serif",
  fontSize: 13,
  fontWeight: 600,
  color: "#ffffff",
  background: "#08060d",
  border: "1px solid #08060d",
  borderRadius: 6,
  padding: "8px 20px",
  cursor: "pointer",
  whiteSpace: "nowrap",
  alignSelf: "flex-end",
  transition: "opacity 0.15s",
};

const compareButtonDisabledStyle: React.CSSProperties = {
  ...compareButtonStyle,
  opacity: 0.4,
  cursor: "not-allowed",
};

const sectionHeadingStyle: React.CSSProperties = {
  fontSize: 14,
  fontWeight: 600,
  color: "#08060d",
  margin: "0 0 10px 0",
};

const metricsGridStyle: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
  gap: 12,
};

const sideBySideStyle: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "1fr 1fr",
  gap: 20,
};

const columnStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 12,
};

const columnHeaderStyle: React.CSSProperties = {
  fontSize: 12,
  fontWeight: 600,
  color: "#6b6375",
  textTransform: "uppercase",
  letterSpacing: "0.04em",
  padding: "6px 0",
  borderBottom: "2px solid #e5e4e7",
  margin: 0,
};

const laneWarningStyle: React.CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  gap: 6,
  padding: "6px 14px",
  borderRadius: 16,
  fontSize: 12,
  fontWeight: 500,
  fontFamily: "Arial, Helvetica, sans-serif",
  background: "#fff4e5",
  color: "#6a4a00",
  border: "1px solid #f5deb3",
  marginBottom: 16,
};

const winnerBadgeStyle: React.CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  gap: 6,
  padding: "4px 12px",
  borderRadius: 16,
  fontSize: 12,
  fontWeight: 600,
  fontFamily: "Arial, Helvetica, sans-serif",
  background: "#e6f6ea",
  color: "#0f7a3a",
  border: "1px solid #c3e6cb",
};

const tieBadgeStyle: React.CSSProperties = {
  ...winnerBadgeStyle,
  background: "#e6e9ee",
  color: "#6b7280",
  border: "1px solid #d1d5db",
};

const emptyStateStyle: React.CSSProperties = {
  textAlign: "center",
  color: "#9ca3af",
  fontSize: 13,
  padding: "40px 20px",
};

const errorStyle: React.CSSProperties = {
  padding: "10px 14px",
  borderRadius: 6,
  background: "#fdecea",
  color: "#9f1f1f",
  fontSize: 13,
  border: "1px solid #f5c6c6",
  marginBottom: 16,
};

const loadingStyle: React.CSSProperties = {
  color: "#9ca3af",
  fontSize: 13,
  padding: "20px 0",
};

const deltaSummaryRowStyle: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: 10,
  marginBottom: 16,
};

const deltaChipBaseStyle: React.CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  gap: 4,
  padding: "3px 10px",
  borderRadius: 12,
  fontSize: 12,
  fontWeight: 500,
  fontFamily: "ui-monospace, Consolas, monospace",
  whiteSpace: "nowrap",
};

/* ---- Helpers ---- */

function formatRunLabel(run: RunSummary): string {
  const ts = run.run_timestamp.slice(0, 16).replace("T", " ");
  return `${run.experiment_name} · ${run.variant} (${ts})`;
}

function getLaneId(run: RunSummary): string {
  return `${run.model_name}::${run.variant}`;
}

function getMetricVal(run: RunSummary | null | undefined, key: string): number | null {
  if (!run) return null;
  const val = (run as unknown as Record<string, unknown>)[key];
  if (val === null || val === undefined || typeof val !== "number" || Number.isNaN(val))
    return null;
  return val;
}

/* ---- Component ---- */

const ComparePage: React.FC = () => {
  const [runs, setRuns] = React.useState<RunSummary[]>([]);
  const [loadingRuns, setLoadingRuns] = React.useState(true);
  const [runsError, setRunsError] = React.useState<string | null>(null);

  const [runAId, setRunAId] = React.useState("");
  const [runBId, setRunBId] = React.useState("");

  const [result, setResult] = React.useState<CompareResult | null>(null);
  const [comparing, setComparing] = React.useState(false);
  const [compareError, setCompareError] = React.useState<string | null>(null);

  /* Fetch runs list on mount */
  React.useEffect(() => {
    let cancelled = false;
    setLoadingRuns(true);
    listRuns(0, 200)
      .then((data) => {
        if (!cancelled) {
          setRuns(data);
          setLoadingRuns(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setRunsError(err instanceof Error ? err.message : String(err));
          setLoadingRuns(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  /* Compare handler */
  const handleCompare = React.useCallback(() => {
    if (!runAId || !runBId) return;
    setComparing(true);
    setCompareError(null);
    setResult(null);
    compareRuns(runAId, runBId)
      .then((data) => {
        setResult(data);
        setComparing(false);
      })
      .catch((err) => {
        setCompareError(err instanceof Error ? err.message : String(err));
        setComparing(false);
      });
  }, [runAId, runBId]);

  const canCompare = runAId !== "" && runBId !== "" && runAId !== runBId && !comparing;

  /* Derived: lane mismatch */
  const isCrossLane =
    result?.run_a && result?.run_b
      ? getLaneId(result.run_a) !== getLaneId(result.run_b)
      : false;

  /* Derived: chart data */
  const chartData: MetricComparisonEntry[] = result
    ? METRICS.filter(
        (m) =>
          getMetricVal(result.run_a, m.key) !== null &&
          getMetricVal(result.run_b, m.key) !== null
      ).map((m) => ({
        metric: m.label,
        valueA: getMetricVal(result.run_a, m.key) ?? 0,
        valueB: getMetricVal(result.run_b, m.key) ?? 0,
      }))
    : [];

  /* Winner label */
  const winnerLabel = React.useMemo(() => {
    if (!result?.winner) return null;
    if (result.winner === "a" && result.run_a)
      return `${result.run_a.experiment_name} · ${result.run_a.variant}`;
    if (result.winner === "b" && result.run_b)
      return `${result.run_b.experiment_name} · ${result.run_b.variant}`;
    if (result.winner === "tie") return "Tie";
    return result.winner;
  }, [result]);

  return (
    <div className="page">
      <h2>Compare Runs</h2>

      {/* ---- Run selectors ---- */}
      <div style={selectorBarStyle}>
        <div style={selectorGroupStyle}>
          <label style={selectorLabelStyle} htmlFor="compare-run-a">
            Run A
          </label>
          <select
            id="compare-run-a"
            style={selectStyle}
            value={runAId}
            onChange={(e) => setRunAId(e.target.value)}
            disabled={loadingRuns}
          >
            <option value="">
              {loadingRuns ? "Loading runs…" : "Select run A"}
            </option>
            {runs.map((r) => (
              <option key={r.summary_path} value={r.summary_path}>
                {formatRunLabel(r)}
              </option>
            ))}
          </select>
        </div>

        <div style={selectorGroupStyle}>
          <label style={selectorLabelStyle} htmlFor="compare-run-b">
            Run B
          </label>
          <select
            id="compare-run-b"
            style={selectStyle}
            value={runBId}
            onChange={(e) => setRunBId(e.target.value)}
            disabled={loadingRuns}
          >
            <option value="">
              {loadingRuns ? "Loading runs…" : "Select run B"}
            </option>
            {runs.map((r) => (
              <option key={r.summary_path} value={r.summary_path}>
                {formatRunLabel(r)}
              </option>
            ))}
          </select>
        </div>

        <button
          type="button"
          style={canCompare ? compareButtonStyle : compareButtonDisabledStyle}
          disabled={!canCompare}
          onClick={handleCompare}
        >
          {comparing ? "Comparing…" : "Compare"}
        </button>
      </div>

      {/* ---- Errors ---- */}
      {runsError ? (
        <div style={errorStyle}>Failed to load runs: {runsError}</div>
      ) : null}
      {compareError ? (
        <div style={errorStyle}>Comparison failed: {compareError}</div>
      ) : null}

      {/* ---- Loading state ---- */}
      {comparing ? <p style={loadingStyle}>Running comparison…</p> : null}

      {/* ---- Empty state ---- */}
      {!result && !comparing && !compareError ? (
        <p style={emptyStateStyle}>
          Select two runs above and click Compare to see a side-by-side analysis.
        </p>
      ) : null}

      {/* ---- Results ---- */}
      {result ? (
        <>
          {/* Lane warning */}
          {isCrossLane ? (
            <div style={laneWarningStyle}>
              <span aria-hidden="true">{"\u26a0\ufe0f"}</span>
              <span>
                These runs are from <strong>different lanes</strong> (
                {result.run_a?.model_name}/{result.run_a?.variant} vs{" "}
                {result.run_b?.model_name}/{result.run_b?.variant}). Comparison
                may not be meaningful.
              </span>
            </div>
          ) : null}

          {/* Winner */}
          {winnerLabel ? (
            <div style={{ marginBottom: 16 }}>
              {result.winner === "tie" ? (
                <span style={tieBadgeStyle}>
                  <span aria-hidden="true">{"\u2014"}</span> Tie — no clear winner
                </span>
              ) : (
                <span style={winnerBadgeStyle}>
                  <span aria-hidden="true">{"\u2705"}</span> Winner:{" "}
                  {winnerLabel}
                </span>
              )}
            </div>
          ) : null}

          {/* Delta summary chips */}
          {Object.keys(result.metric_deltas).length > 0 ? (
            <>
              <p style={sectionHeadingStyle}>Metric Deltas (B − A)</p>
              <div style={deltaSummaryRowStyle}>
                {METRICS.map((m) => {
                  const delta = result.metric_deltas[m.key];
                  if (delta === null || delta === undefined) return null;
                  const improved = m.higherIsBetter ? delta > 0 : delta < 0;
                  const regressed = m.higherIsBetter ? delta < 0 : delta > 0;
                  const bg = improved
                    ? "#e6f6ea"
                    : regressed
                    ? "#fdecea"
                    : "#e6e9ee";
                  const fg = improved
                    ? "#0f7a3a"
                    : regressed
                    ? "#9f1f1f"
                    : "#6b7280";
                  const arrow = improved ? "\u2191" : regressed ? "\u2193" : "\u2014";
                  return (
                    <span
                      key={m.key}
                      style={{ ...deltaChipBaseStyle, background: bg, color: fg }}
                    >
                      {m.label}: {arrow}{" "}
                      {delta >= 0 ? "+" : ""}
                      {(delta * 100).toFixed(2)}%
                    </span>
                  );
                })}
              </div>
            </>
          ) : null}

          {/* Side-by-side metric cards */}
          <p style={sectionHeadingStyle}>Side-by-Side Metrics</p>
          <div style={sideBySideStyle}>
            {/* Column A */}
            <div style={columnStyle}>
              <p style={columnHeaderStyle}>
                {result.run_a
                  ? `${result.run_a.experiment_name} · ${result.run_a.variant}`
                  : "Run A"}
              </p>
              <div style={metricsGridStyle}>
                {METRICS.map((m) => (
                  <MetricCard
                    key={m.key}
                    name={m.label}
                    value={getMetricVal(result.run_a, m.key)}
                    higherIsBetter={m.higherIsBetter}
                  />
                ))}
              </div>
            </div>
            {/* Column B */}
            <div style={columnStyle}>
              <p style={columnHeaderStyle}>
                {result.run_b
                  ? `${result.run_b.experiment_name} · ${result.run_b.variant}`
                  : "Run B"}
              </p>
              <div style={metricsGridStyle}>
                {METRICS.map((m) => {
                  const delta = result.metric_deltas[m.key] ?? null;
                  return (
                    <MetricCard
                      key={m.key}
                      name={m.label}
                      value={getMetricVal(result.run_b, m.key)}
                      delta={delta}
                      higherIsBetter={m.higherIsBetter}
                    />
                  );
                })}
              </div>
            </div>
          </div>

          {/* Chart */}
          {chartData.length > 0 ? (
            <div style={{ marginTop: 24 }}>
              <p style={sectionHeadingStyle}>Metric Comparison Chart</p>
              <MetricComparisonChart
                data={chartData}
                labelA={
                  result.run_a
                    ? `${result.run_a.experiment_name} · ${result.run_a.variant}`
                    : "Run A"
                }
                labelB={
                  result.run_b
                    ? `${result.run_b.experiment_name} · ${result.run_b.variant}`
                    : "Run B"
                }
                title="Metric Comparison"
                height={360}
              />
            </div>
          ) : null}
        </>
      ) : null}
    </div>
  );
};

export default ComparePage;
