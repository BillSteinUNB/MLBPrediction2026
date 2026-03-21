import React, { useEffect, useState, useCallback, useMemo } from "react";
import { listLanes, getLaneRuns } from "../api";
import type { Lane, RunSummary } from "../api";
import { MetricCard } from "../components/MetricCard";
import { RunTable } from "../components/RunTable";
import { MetricTrendChart } from "../charts/MetricTrendChart";
import type { MetricTrendPoint } from "../charts/MetricTrendChart";

/* ====================================================================
   Metric definitions for the selector & trend chart
   ==================================================================== */

interface MetricDef {
  key: keyof RunSummary;
  label: string;
  higherIsBetter: boolean;
}

const METRICS: MetricDef[] = [
  { key: "roc_auc", label: "ROC AUC", higherIsBetter: true },
  { key: "log_loss", label: "Log Loss", higherIsBetter: false },
  { key: "brier", label: "Brier", higherIsBetter: false },
  { key: "accuracy", label: "Accuracy", higherIsBetter: true },
  { key: "ece", label: "ECE", higherIsBetter: false },
];

/* ====================================================================
   Inline styles — aligned to layout.css / component design tokens
   ==================================================================== */

const pageStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 24,
};

const headerRow: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  flexWrap: "wrap",
  gap: 12,
};

const seasonGroupStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 10,
};

const seasonHeadingStyle: React.CSSProperties = {
  fontSize: 15,
  fontWeight: 600,
  color: "#08060d",
  margin: 0,
  padding: "6px 0",
  borderBottom: "1px solid #e5e4e7",
  letterSpacing: "-0.1px",
};

const laneCardStyle: React.CSSProperties = {
  background: "#ffffff",
  border: "1px solid #e5e4e7",
  borderRadius: 6,
  overflow: "hidden",
  transition: "border-color 0.15s",
};

const laneCardExpandedStyle: React.CSSProperties = {
  ...laneCardStyle,
  borderColor: "#c4c2c7",
};

const laneHeaderStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 12,
  padding: "12px 14px",
  cursor: "pointer",
  userSelect: "none",
};

const laneHeaderHoverStyle: React.CSSProperties = {
  ...laneHeaderStyle,
  background: "#f7f7f8",
};

const laneNameStyle: React.CSSProperties = {
  fontSize: 14,
  fontWeight: 600,
  color: "#08060d",
  margin: 0,
  flex: "0 0 auto",
};

const variantBadgeStyle: React.CSSProperties = {
  display: "inline-block",
  padding: "2px 8px",
  borderRadius: 10,
  background: "#eef2f6",
  fontSize: 12,
  color: "#495464",
  fontWeight: 500,
};

const laneMetaStyle: React.CSSProperties = {
  fontSize: 12,
  color: "#9ca3af",
  marginLeft: "auto",
  display: "flex",
  gap: 16,
  alignItems: "center",
  whiteSpace: "nowrap",
};

const chevronStyle: React.CSSProperties = {
  fontSize: 14,
  color: "#9ca3af",
  transition: "transform 0.2s",
  flex: "0 0 auto",
};

const expandedBodyStyle: React.CSSProperties = {
  padding: "0 14px 14px",
  display: "flex",
  flexDirection: "column",
  gap: 16,
};

const metricRowStyle: React.CSSProperties = {
  display: "flex",
  gap: 10,
  flexWrap: "wrap",
};

const selectorRow: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 10,
};

const selectStyle: React.CSSProperties = {
  fontFamily: "Arial, Helvetica, sans-serif",
  fontSize: 13,
  padding: "5px 10px",
  border: "1px solid #e5e4e7",
  borderRadius: 6,
  background: "#ffffff",
  color: "#08060d",
  cursor: "pointer",
  outline: "none",
};

const selectLabelStyle: React.CSSProperties = {
  fontSize: 12,
  fontWeight: 500,
  color: "#6b6375",
  textTransform: "uppercase",
  letterSpacing: "0.04em",
};

const centerStyle: React.CSSProperties = {
  display: "flex",
  justifyContent: "center",
  alignItems: "center",
  minHeight: 200,
  color: "#9ca3af",
  fontSize: 14,
  fontFamily: "Arial, Helvetica, sans-serif",
};

const errorBoxStyle: React.CSSProperties = {
  padding: "14px 16px",
  background: "#fdecea",
  border: "1px solid #f5c6c6",
  borderRadius: 6,
  color: "#9f1f1f",
  fontSize: 13,
};

const retryBtnStyle: React.CSSProperties = {
  marginTop: 8,
  padding: "5px 14px",
  fontSize: 12,
  fontWeight: 600,
  border: "1px solid #e5e4e7",
  borderRadius: 6,
  background: "#ffffff",
  color: "#08060d",
  cursor: "pointer",
  fontFamily: "Arial, Helvetica, sans-serif",
};

/* ====================================================================
   Helper: format metric value
   ==================================================================== */

function fmt(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "\u2014";
  return v.toFixed(4);
}

/* ====================================================================
   LaneCard — single expandable lane
   ==================================================================== */

interface LaneCardProps {
  lane: Lane;
  selectedMetric: MetricDef;
}

const LaneCard: React.FC<LaneCardProps> = ({ lane, selectedMetric }) => {
  const [expanded, setExpanded] = useState(false);
  const [hovered, setHovered] = useState(false);
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [runsLoading, setRunsLoading] = useState(false);
  const [runsError, setRunsError] = useState<string | null>(null);

  const toggle = useCallback(() => {
    setExpanded((prev) => !prev);
  }, []);

  // Fetch lane runs when expanded
  useEffect(() => {
    if (!expanded || runs !== null) return;

    let cancelled = false;
    setRunsLoading(true);
    setRunsError(null);

    getLaneRuns(lane.lane_id, 0, 100)
      .then((data) => {
        if (!cancelled) setRuns(data);
      })
      .catch((err) => {
        if (!cancelled) setRunsError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setRunsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [expanded, runs, lane.lane_id]);

  // Build trend data from runs
  const trendData: MetricTrendPoint[] = useMemo(() => {
    if (!runs) return [];
    return runs
      .filter((r) => {
        const val = r[selectedMetric.key];
        return val !== null && val !== undefined && !Number.isNaN(val as number);
      })
      .sort((a, b) => a.run_timestamp.localeCompare(b.run_timestamp))
      .map((r) => ({
        date: r.run_timestamp,
        value: r[selectedMetric.key] as number,
      }));
  }, [runs, selectedMetric.key]);

  const best = lane.best_run;
  const latest = lane.latest_run;

  return (
    <div style={expanded ? laneCardExpandedStyle : laneCardStyle}>
      {/* Clickable header */}
      <button
        type="button"
        style={hovered ? laneHeaderHoverStyle : laneHeaderStyle}
        onClick={toggle}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      >
        <span
          style={{
            ...chevronStyle,
            transform: expanded ? "rotate(90deg)" : "rotate(0deg)",
          }}
          aria-hidden="true"
        >
          ▸
        </span>

        <p style={laneNameStyle}>{lane.model_name}</p>
        <span style={variantBadgeStyle}>{lane.variant}</span>

        <span style={laneMetaStyle}>
          {best ? (
            <span title="Best run ROC AUC">
              Best: {fmt(best.roc_auc)}
            </span>
          ) : null}
          {latest ? (
            <span title="Latest run timestamp">
              Latest: {latest.run_timestamp.slice(0, 10)}
            </span>
          ) : null}
        </span>
      </button>

      {/* Expanded body */}
      {expanded && (
        <div style={expandedBodyStyle}>
          {/* Best-run metric cards */}
          {best && (
            <div style={metricRowStyle}>
              {METRICS.map((m) => (
                <MetricCard
                  key={m.key}
                  name={m.label}
                  value={(best[m.key] as number | null) ?? null}
                  higherIsBetter={m.higherIsBetter}
                />
              ))}
            </div>
          )}

          {/* Metric trend chart */}
          {runsLoading && (
            <div style={centerStyle}>Loading lane runs\u2026</div>
          )}
          {runsError && (
            <div style={errorBoxStyle}>
              Failed to load runs: {runsError}
              <br />
              <button
                type="button"
                style={retryBtnStyle}
                onClick={() => {
                  setRuns(null);
                  setRunsError(null);
                }}
              >
                Retry
              </button>
            </div>
          )}
          {runs && trendData.length > 0 && (
            <MetricTrendChart
              data={trendData}
              metricName={selectedMetric.label}
              title={`${selectedMetric.label} over time — ${lane.model_name} (${lane.variant})`}
              height={280}
            />
          )}
          {runs && trendData.length === 0 && !runsLoading && (
            <div style={{ ...centerStyle, minHeight: 60 }}>
              No {selectedMetric.label} data for this lane.
            </div>
          )}

          {/* Run table */}
          {runs && runs.length > 0 && (
            <RunTable runs={runs} title="All Runs in Lane" />
          )}
        </div>
      )}
    </div>
  );
};

/* ====================================================================
   LaneExplorerPage — top level
   ==================================================================== */

const LaneExplorerPage: React.FC = () => {
  const [lanes, setLanes] = useState<Lane[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedMetricKey, setSelectedMetricKey] = useState<string>("roc_auc");

  const selectedMetric = METRICS.find((m) => m.key === selectedMetricKey) ?? METRICS[0];

  const fetchLanes = useCallback(() => {
    setLoading(true);
    setError(null);
    listLanes()
      .then((data) => setLanes(data))
      .catch((err) =>
        setError(err instanceof Error ? err.message : String(err)),
      )
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetchLanes();
  }, [fetchLanes]);

  // Group lanes by holdout_season (from best_run or latest_run)
  const grouped: Map<number | string, Lane[]> = useMemo(() => {
    const map = new Map<number | string, Lane[]>();
    for (const lane of lanes) {
      const season =
        lane.best_run?.holdout_season ??
        lane.latest_run?.holdout_season ??
        "Unknown";
      const arr = map.get(season);
      if (arr) {
        arr.push(lane);
      } else {
        map.set(season, [lane]);
      }
    }
    // Sort keys descending (most recent season first)
    return new Map(
      [...map.entries()].sort((a, b) => {
        if (typeof a[0] === "number" && typeof b[0] === "number")
          return b[0] - a[0];
        if (typeof a[0] === "number") return -1;
        if (typeof b[0] === "number") return 1;
        return 0;
      }),
    );
  }, [lanes]);

  return (
    <div className="page" style={pageStyle}>
      <div style={headerRow}>
        <h2 style={{ margin: 0 }}>Lane Explorer</h2>

        <div style={selectorRow}>
          <span style={selectLabelStyle}>Trend metric</span>
          <select
            style={selectStyle}
            value={selectedMetricKey}
            onChange={(e) => setSelectedMetricKey(e.target.value)}
          >
            {METRICS.map((m) => (
              <option key={m.key} value={m.key}>
                {m.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Loading state */}
      {loading && <div style={centerStyle}>Loading lanes\u2026</div>}

      {/* Error state */}
      {error && !loading && (
        <div style={errorBoxStyle}>
          Failed to load lanes: {error}
          <br />
          <button type="button" style={retryBtnStyle} onClick={fetchLanes}>
            Retry
          </button>
        </div>
      )}

      {/* Empty state */}
      {!loading && !error && lanes.length === 0 && (
        <div style={centerStyle}>No lanes found.</div>
      )}

      {/* Lane list grouped by holdout season */}
      {!loading &&
        !error &&
        [...grouped.entries()].map(([season, seasonLanes]) => (
          <div key={String(season)} style={seasonGroupStyle}>
            <p style={seasonHeadingStyle}>
              Holdout Season: {String(season)}
              <span
                style={{
                  fontWeight: 400,
                  color: "#9ca3af",
                  marginLeft: 8,
                  fontSize: 12,
                }}
              >
                {seasonLanes.length} lane{seasonLanes.length !== 1 ? "s" : ""}
              </span>
            </p>

            {seasonLanes.map((lane) => (
              <LaneCard
                key={lane.lane_id}
                lane={lane}
                selectedMetric={selectedMetric}
              />
            ))}
          </div>
        ))}
    </div>
  );
};

export default LaneExplorerPage;
