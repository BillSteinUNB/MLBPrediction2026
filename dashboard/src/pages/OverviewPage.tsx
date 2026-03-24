import React, { useEffect, useState } from "react";
import { getOverview } from "../api";
import type { OverviewResponse, RunSummary } from "../api/types";
import { MetricCard, RunTable } from "../components";
import { TooltipLabel } from "../components/TooltipLabel";

/* ---- Style tokens aligned to layout.css variables ---- */
const sectionHeading: React.CSSProperties = {
  fontSize: 16,
  fontWeight: 600,
  color: "var(--text-h)",
  margin: "0 0 12px 0",
};

const cardGrid: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: "var(--gap, 20px)",
  marginBottom: 28,
};

const sectionBlock: React.CSSProperties = {
  marginBottom: 32,
};

const statusText: React.CSSProperties = {
  fontSize: 14,
  color: "var(--muted)",
  padding: 24,
  textAlign: "center",
};

const errorText: React.CSSProperties = {
  ...statusText,
  color: "#dc2626",
};

/**
 * Derive the top N runs with the largest positive delta_vs_prev_roc_auc.
 */
function topImprovements(runs: RunSummary[], n: number): RunSummary[] {
  return runs
    .filter(
      (r) =>
        r.delta_vs_prev_roc_auc !== null &&
        r.delta_vs_prev_roc_auc !== undefined &&
        r.delta_vs_prev_roc_auc > 0
    )
    .sort(
      (a, b) =>
        (b.delta_vs_prev_roc_auc ?? 0) - (a.delta_vs_prev_roc_auc ?? 0)
    )
    .slice(0, n);
}

/**
 * Derive the top N runs with the largest negative delta_vs_prev_roc_auc.
 */
function topRegressions(runs: RunSummary[], n: number): RunSummary[] {
  return runs
    .filter(
      (r) =>
        r.delta_vs_prev_roc_auc !== null &&
        r.delta_vs_prev_roc_auc !== undefined &&
        r.delta_vs_prev_roc_auc < 0
    )
    .sort(
      (a, b) =>
        (a.delta_vs_prev_roc_auc ?? 0) - (b.delta_vs_prev_roc_auc ?? 0)
    )
    .slice(0, n);
}

const OverviewPage: React.FC = () => {
  const [data, setData] = useState<OverviewResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    getOverview()
      .then((res) => {
        if (!cancelled) {
          setData(res);
          setError(null);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load data");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  /* ---- Loading state ---- */
  if (loading) {
    return (
      <div className="page">
        <h2 style={sectionHeading}>Overview</h2>
        <p style={statusText}>Loading overview...</p>
      </div>
    );
  }

  /* ---- Error state ---- */
  if (error) {
    return (
      <div className="page">
        <h2 style={sectionHeading}>Overview</h2>
        <p style={errorText}>Failed to load data</p>
      </div>
    );
  }

  /* ---- Empty state ---- */
  if (!data || (data.total_runs === 0 && data.recent_runs.length === 0)) {
    return (
      <div className="page">
        <h2 style={sectionHeading}>Overview</h2>
        <p style={statusText}>No experiment data found</p>
      </div>
    );
  }

  /* ---- Derived lists ---- */
  const improvements = topImprovements(data.recent_runs, 5);
  const regressions = topRegressions(data.recent_runs, 5);

  return (
    <div className="page">
      <h2 style={{ ...sectionHeading, fontSize: 20, marginBottom: 20 }}>
        <TooltipLabel label="Overview" as="span" />
      </h2>

      {/* ---- Summary metric cards ---- */}
      <div style={cardGrid}>
        <MetricCard
          name="Total Runs"
          value={data.total_runs}
          unit=""
          higherIsBetter
        />
        <MetricCard
          name="Active Lanes"
          value={data.active_lanes}
          unit=""
          higherIsBetter
        />
        <MetricCard
          name="Best ROC AUC"
          value={data.best_run?.roc_auc ?? null}
          delta={data.best_run?.delta_vs_prev_roc_auc}
          higherIsBetter
        />
        <MetricCard
          name="Latest Experiment"
          value={data.latest_run?.roc_auc ?? null}
          delta={data.latest_run?.delta_vs_prev_roc_auc}
          higherIsBetter
        />
      </div>

      {/* ---- Best Run Per Lane ---- */}
      <div style={sectionBlock}>
        <RunTable title="Best Run Per Lane" runs={data.recent_runs} />
      </div>

      {/* ---- Biggest Improvements ---- */}
      {improvements.length > 0 && (
        <div style={sectionBlock}>
          <RunTable title="Biggest Improvements" runs={improvements} />
        </div>
      )}

      {/* ---- Biggest Regressions ---- */}
      {regressions.length > 0 && (
        <div style={sectionBlock}>
          <RunTable title="Biggest Regressions" runs={regressions} />
        </div>
      )}
    </div>
  );
};

export default OverviewPage;
