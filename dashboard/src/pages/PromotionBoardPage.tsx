import React, { useEffect, useState, useCallback } from "react";
import { listPromotions, promoteRun, listRuns } from "../api";
import type { Promotion, RunSummary } from "../api";

const STAGES = ["development", "staging", "candidate", "production"] as const;

interface FeedbackState {
  type: "success" | "error";
  message: string;
}

const PromotionBoardPage: React.FC = () => {
  const [promotions, setPromotions] = useState<Promotion[]>([]);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Modal state
  const [showModal, setShowModal] = useState(false);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [targetStage, setTargetStage] = useState<string>(STAGES[1]);
  const [reason, setReason] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [feedback, setFeedback] = useState<FeedbackState | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [promoData, runData] = await Promise.all([
        listPromotions(0, 100),
        listRuns(0, 200),
      ]);
      setPromotions(promoData);
      setRuns(runData);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load data");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Auto-dismiss feedback after 4s
  useEffect(() => {
    if (!feedback) return;
    const timer = setTimeout(() => setFeedback(null), 4000);
    return () => clearTimeout(timer);
  }, [feedback]);

  const openModal = () => {
    setSelectedRunId("");
    setTargetStage(STAGES[1]);
    setReason("");
    setFeedback(null);
    setShowModal(true);
  };

  const closeModal = () => {
    if (!submitting) setShowModal(false);
  };

  const runExists = (runId: string): boolean =>
    runs.some(
      (r) => r.summary_path === runId || r.experiment_name === runId
    );

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedRunId) {
      setFeedback({ type: "error", message: "Please select a run." });
      return;
    }
    if (!runExists(selectedRunId)) {
      setFeedback({ type: "error", message: "Selected run not found." });
      return;
    }
    if (!targetStage) {
      setFeedback({ type: "error", message: "Please select a target stage." });
      return;
    }

    setSubmitting(true);
    setFeedback(null);
    try {
      await promoteRun({
        run_id: selectedRunId,
        target_stage: targetStage,
        reason: reason.trim() || undefined,
      });
      setFeedback({ type: "success", message: "Run promoted successfully." });
      setShowModal(false);
      await fetchData();
    } catch (err) {
      setFeedback({
        type: "error",
        message: err instanceof Error ? err.message : "Promotion failed.",
      });
    } finally {
      setSubmitting(false);
    }
  };

  const formatTimestamp = (ts: string): string => {
    try {
      const d = new Date(ts);
      return d.toLocaleString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return ts;
    }
  };

  const runLabel = (r: RunSummary): string => {
    const parts = [r.model_name, r.variant, r.run_kind].filter(Boolean);
    return `${r.experiment_name} — ${parts.join(" / ")}`;
  };

  return (
    <div className="page">
      <div style={styles.header}>
        <h2 style={{ margin: 0 }}>Promotion Board</h2>
        <button
          type="button"
          onClick={openModal}
          style={styles.promoteBtn}
          disabled={loading}
        >
          + Promote Run
        </button>
      </div>

      {/* Feedback banner */}
      {feedback && (
        <div
          style={{
            ...styles.feedback,
            borderColor:
              feedback.type === "success"
                ? "var(--border)"
                : "#e5534b",
            background:
              feedback.type === "success" ? "#f0fdf4" : "#fef2f2",
            color:
              feedback.type === "success" ? "#166534" : "#991b1b",
          }}
        >
          {feedback.message}
        </div>
      )}

      {/* Loading / Error */}
      {loading && <p style={styles.muted}>Loading promotions…</p>}
      {error && <p style={styles.errorText}>{error}</p>}

      {/* Promotions table */}
      {!loading && !error && (
        <>
          {promotions.length === 0 ? (
            <p style={styles.muted}>
              No promotions recorded yet. Promote a run to get started.
            </p>
          ) : (
            <div style={styles.tableWrap}>
              <table style={styles.table}>
                <thead>
                  <tr>
                    <th style={styles.th}>Run ID</th>
                    <th style={styles.th}>From</th>
                    <th style={styles.th}>To</th>
                    <th style={styles.th}>Promoted</th>
                    <th style={styles.th}>Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {promotions.map((p) => (
                    <tr key={p.promotion_id} style={styles.row}>
                      <td style={styles.td}>
                        <code style={styles.code}>
                          {p.run_id.length > 32
                            ? `${p.run_id.slice(0, 30)}…`
                            : p.run_id}
                        </code>
                      </td>
                      <td style={styles.td}>
                        <span style={styles.stageBadge}>{p.from_stage}</span>
                      </td>
                      <td style={styles.td}>
                        <span
                          style={{
                            ...styles.stageBadge,
                            background:
                              p.to_stage === "production"
                                ? "#dcfce7"
                                : p.to_stage === "candidate"
                                ? "#fef9c3"
                                : "var(--code-bg)",
                            color:
                              p.to_stage === "production"
                                ? "#166534"
                                : p.to_stage === "candidate"
                                ? "#854d0e"
                                : "var(--text-h)",
                          }}
                        >
                          {p.to_stage}
                        </span>
                      </td>
                      <td style={styles.td}>
                        {formatTimestamp(p.promoted_timestamp)}
                      </td>
                      <td style={styles.td}>
                        {p.metadata?.reason
                          ? String(p.metadata.reason)
                          : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}

      {/* Modal overlay */}
      {showModal && (
        <div
          style={styles.overlay}
          role="dialog"
          aria-modal="true"
          onClick={closeModal}
          onKeyDown={(e) => { if (e.key === "Escape") closeModal(); }}
        >
          <div
            style={styles.modal}
            role="document"
            onClick={(e) => e.stopPropagation()}
            onKeyDown={(e) => e.stopPropagation()}
          >
            <div style={styles.modalHeader}>
              <h3 style={styles.modalTitle}>Promote Run</h3>
              <button
                type="button"
                onClick={closeModal}
                style={styles.closeBtn}
                disabled={submitting}
                aria-label="Close"
              >
                ×
              </button>
            </div>

            <form onSubmit={handleSubmit} style={styles.form}>
              {/* Run selector */}
              <label style={styles.label}>
                Run
                <select
                  value={selectedRunId}
                  onChange={(e) => setSelectedRunId(e.target.value)}
                  style={styles.select}
                  required
                  disabled={submitting}
                >
                  <option value="">Select a run…</option>
                  {runs.map((r) => (
                    <option key={r.summary_path} value={r.summary_path}>
                      {runLabel(r)}
                    </option>
                  ))}
                </select>
              </label>

              {/* Target stage */}
              <label style={styles.label}>
                Target Stage
                <select
                  value={targetStage}
                  onChange={(e) => setTargetStage(e.target.value)}
                  style={styles.select}
                  required
                  disabled={submitting}
                >
                  {STAGES.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </label>

              {/* Reason */}
              <label style={styles.label}>
                Reason
                <textarea
                  value={reason}
                  onChange={(e) => setReason(e.target.value)}
                  style={styles.textarea}
                  rows={3}
                  placeholder="Why is this run being promoted?"
                  disabled={submitting}
                />
              </label>

              {/* Modal-scoped feedback */}
              {feedback && showModal && (
                <div
                  style={{
                    ...styles.feedback,
                    borderColor:
                      feedback.type === "error" ? "#e5534b" : "var(--border)",
                    background:
                      feedback.type === "error" ? "#fef2f2" : "#f0fdf4",
                    color:
                      feedback.type === "error" ? "#991b1b" : "#166534",
                    marginTop: 0,
                  }}
                >
                  {feedback.message}
                </div>
              )}

              <div style={styles.modalActions}>
                <button
                  type="button"
                  onClick={closeModal}
                  style={styles.cancelBtn}
                  disabled={submitting}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  style={styles.submitBtn}
                  disabled={submitting || !selectedRunId}
                >
                  {submitting ? "Promoting…" : "Promote"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

/* -------------------------------------------------------------------------- */
/*  Inline styles — all reference CSS custom properties from index.css        */
/* -------------------------------------------------------------------------- */

const styles: Record<string, React.CSSProperties> = {
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 20,
    gap: 12,
    flexWrap: "wrap",
  },

  promoteBtn: {
    padding: "8px 16px",
    fontSize: 14,
    fontWeight: 600,
    background: "var(--text-h)",
    color: "#fff",
    border: "none",
    borderRadius: "var(--radius, 6px)",
    cursor: "pointer",
    whiteSpace: "nowrap",
  },

  feedback: {
    padding: "10px 14px",
    borderRadius: "var(--radius, 6px)",
    border: "1px solid",
    fontSize: 14,
    marginBottom: 16,
    marginTop: 0,
  },

  muted: {
    color: "var(--muted)",
    fontSize: 14,
  },

  errorText: {
    color: "#991b1b",
    fontSize: 14,
  },

  tableWrap: {
    overflowX: "auto",
    marginTop: 4,
  },

  table: {
    width: "100%",
    borderCollapse: "collapse",
    fontSize: 14,
  },

  th: {
    textAlign: "left",
    padding: "10px 12px",
    borderBottom: "2px solid var(--border)",
    color: "var(--text-h)",
    fontWeight: 600,
    fontSize: 13,
    whiteSpace: "nowrap",
  },

  td: {
    padding: "10px 12px",
    borderBottom: "1px solid var(--border)",
    verticalAlign: "top",
  },

  row: {
    transition: "background 0.1s",
  },

  code: {
    fontFamily: "var(--mono, ui-monospace, Consolas, monospace)",
    fontSize: 13,
    padding: "2px 6px",
    background: "var(--code-bg)",
    borderRadius: 4,
    color: "var(--text-h)",
  },

  stageBadge: {
    display: "inline-block",
    padding: "3px 10px",
    borderRadius: 4,
    fontSize: 12,
    fontWeight: 600,
    background: "var(--code-bg)",
    color: "var(--text-h)",
    textTransform: "capitalize",
  },

  /* Modal */
  overlay: {
    position: "fixed",
    inset: 0,
    background: "rgba(8, 6, 13, 0.35)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    zIndex: 1000,
    padding: 16,
  },

  modal: {
    background: "var(--bg, #fff)",
    border: "1px solid var(--border)",
    borderRadius: 8,
    width: "100%",
    maxWidth: 480,
    boxShadow: "0 8px 30px rgba(0,0,0,0.12)",
  },

  modalHeader: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "16px 20px 0",
  },

  modalTitle: {
    margin: 0,
    fontSize: 18,
    fontWeight: 600,
    color: "var(--text-h)",
  },

  closeBtn: {
    background: "none",
    border: "none",
    fontSize: 22,
    color: "var(--muted)",
    cursor: "pointer",
    padding: "0 4px",
    lineHeight: 1,
  },

  form: {
    padding: "16px 20px 20px",
    display: "flex",
    flexDirection: "column",
    gap: 14,
  },

  label: {
    display: "flex",
    flexDirection: "column",
    gap: 4,
    fontSize: 13,
    fontWeight: 600,
    color: "var(--text-h)",
  },

  select: {
    padding: "8px 10px",
    fontSize: 14,
    border: "1px solid var(--border)",
    borderRadius: "var(--radius, 6px)",
    background: "var(--bg, #fff)",
    color: "var(--text)",
    fontFamily: "inherit",
    fontWeight: 400,
  },

  textarea: {
    padding: "8px 10px",
    fontSize: 14,
    border: "1px solid var(--border)",
    borderRadius: "var(--radius, 6px)",
    background: "var(--bg, #fff)",
    color: "var(--text)",
    fontFamily: "inherit",
    resize: "vertical",
    fontWeight: 400,
  },

  modalActions: {
    display: "flex",
    justifyContent: "flex-end",
    gap: 8,
    marginTop: 4,
  },

  cancelBtn: {
    padding: "8px 16px",
    fontSize: 14,
    background: "transparent",
    border: "1px solid var(--border)",
    borderRadius: "var(--radius, 6px)",
    color: "var(--text)",
    cursor: "pointer",
  },

  submitBtn: {
    padding: "8px 18px",
    fontSize: 14,
    fontWeight: 600,
    background: "var(--text-h)",
    color: "#fff",
    border: "none",
    borderRadius: "var(--radius, 6px)",
    cursor: "pointer",
  },
};

export default PromotionBoardPage;
