import React, { useEffect, useMemo, useState } from "react";
import { getLiveSeasonGames, getLiveSeasonSummary } from "../api";
import type { LiveSeasonGameResponse, LiveSeasonSummaryResponse } from "../api";

const pageStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 20,
};

const topRowStyle: React.CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  gap: 12,
  flexWrap: "wrap",
};

const controlsStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 10,
  flexWrap: "wrap",
};

const inputStyle: React.CSSProperties = {
  fontFamily: "Arial, Helvetica, sans-serif",
  fontSize: 14,
  padding: "8px 10px",
  border: "1px solid var(--border)",
  borderRadius: 8,
  background: "var(--bg-panel)",
  color: "var(--text-h)",
};

const buttonStyle: React.CSSProperties = {
  fontFamily: "Arial, Helvetica, sans-serif",
  fontSize: 13,
  fontWeight: 600,
  padding: "8px 14px",
  border: "1px solid var(--border)",
  borderRadius: 8,
  background: "var(--surface-3)",
  color: "var(--text-h)",
  cursor: "pointer",
};

const summaryGridStyle: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
  gap: 12,
};

const summaryCardStyle: React.CSSProperties = {
  border: "1px solid var(--border)",
  borderRadius: 12,
  background: "var(--bg-panel)",
  padding: 14,
};

const tableStyle: React.CSSProperties = {
  width: "100%",
  borderCollapse: "collapse",
  fontSize: 13,
};

const thStyle: React.CSSProperties = {
  textAlign: "left",
  padding: "10px 12px",
  borderBottom: "1px solid var(--border)",
  color: "var(--text)",
  fontSize: 12,
  textTransform: "uppercase",
};

const tdStyle: React.CSSProperties = {
  padding: "10px 12px",
  borderBottom: "1px solid var(--border)",
  color: "var(--text-h)",
  verticalAlign: "top",
};

const panelStyle: React.CSSProperties = {
  border: "1px solid var(--border)",
  borderRadius: 14,
  background: "var(--bg-panel)",
  padding: 16,
};

function fmtPct(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function fmtOdds(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value > 0 ? `+${value}` : `${value}`;
}

function fmtUnits(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}u`;
}

function sourceModelLabel(value: string | null | undefined): string {
  if (!value) return "—";
  if (value === "legacy_f5_ml") return "Legacy F5 ML";
  if (value === "legacy_f5_ml_equiv") return "Legacy F5 ML Equivalent";
  if (value === "legacy_f5_rl") return "Legacy F5 RL";
  if (value === "rlv2_direct") return "RL V2 Direct";
  if (value === "rlv2_margin") return "RL V2 Margin";
  if (value === "rlv2_blend") return "RL V2 Blend";
  return value;
}

function statusBadge(label: string, tone: "neutral" | "good" | "warn" | "bad"): React.ReactElement {
  const palette = {
    neutral: { background: "var(--surface-2)", color: "var(--text-h)", border: "var(--border)" },
    good: { background: "var(--good-bg)", color: "var(--good-fg)", border: "#1f4f38" },
    warn: { background: "var(--warn-bg)", color: "var(--warn-fg)", border: "#614625" },
    bad: { background: "var(--bad-bg)", color: "var(--bad-fg)", border: "#693039" },
  }[tone];

  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 4,
        padding: "3px 8px",
        borderRadius: 999,
        border: `1px solid ${palette.border}`,
        background: palette.background,
        color: palette.color,
        fontSize: 12,
        fontWeight: 600,
      }}
    >
      {label}
    </span>
  );
}

function StrategyCell({
  marketType,
  side,
  odds,
  edgePct,
  result,
  profit,
}: {
  marketType?: string | null;
  side?: string | null;
  odds?: number | null;
  edgePct?: number | null;
  result?: string | null;
  profit?: number | null;
}) {
  if (!marketType || !side) {
    return <span style={{ color: "var(--text)" }}>—</span>;
  }
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div>{`${marketType} ${side} ${fmtOdds(odds)}`}</div>
      <div style={{ color: "var(--text)" }}>{`edge ${fmtPct(edgePct)}`}</div>
      <div style={{ color: "var(--text)" }}>{`${result ?? "PENDING"} • ${fmtUnits(profit)}`}</div>
    </div>
  );
}

void StrategyCell;

function StrategyCellWithModel({
  marketType,
  side,
  odds,
  edgePct,
  sourceModel,
  result,
  profit,
}: {
  marketType?: string | null;
  side?: string | null;
  odds?: number | null;
  edgePct?: number | null;
  sourceModel?: string | null;
  result?: string | null;
  profit?: number | null;
}) {
  if (!marketType || !side) {
    return <span style={{ color: "var(--text)" }}>—</span>;
  }
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div>{`${marketType} ${side} ${fmtOdds(odds)}`}</div>
      <div style={{ color: "var(--text)" }}>{`edge ${fmtPct(edgePct)}`}</div>
      <div style={{ color: "var(--text)" }}>{sourceModelLabel(sourceModel)}</div>
      <div style={{ color: "var(--text)" }}>{`${result ?? "PENDING"} • ${fmtUnits(profit)}`}</div>
    </div>
  );
}

const LiveSeasonPage: React.FC = () => {
  const [pipelineDate, setPipelineDate] = useState("");
  const [summary, setSummary] = useState<LiveSeasonSummaryResponse | null>(null);
  const [games, setGames] = useState<LiveSeasonGameResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function loadData(dateFilter?: string) {
    setLoading(true);
    setError(null);
    try {
      const [summaryResponse, gamesResponse] = await Promise.all([
        getLiveSeasonSummary(2026),
        getLiveSeasonGames(2026, dateFilter || undefined),
      ]);
      setSummary(summaryResponse);
      setGames(gamesResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load live season tracking");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadData();
  }, []);

  const filteredGames = useMemo(() => games, [games]);

  return (
    <section style={pageStyle}>
      <div style={topRowStyle}>
        <div>
          <h1 style={{ margin: 0 }}>Live Season 2026</h1>
          <p style={{ color: "var(--text)" }}>
            Track play of the day, all value plays, and forced picks separately.
          </p>
        </div>
        <div style={controlsStyle}>
          <input
            type="date"
            value={pipelineDate}
            onChange={(event) => setPipelineDate(event.target.value)}
            style={inputStyle}
          />
          <button type="button" style={buttonStyle} onClick={() => void loadData(pipelineDate)}>
            Load Date
          </button>
          <button type="button" style={buttonStyle} onClick={() => { setPipelineDate(""); void loadData(); }}>
            Full Season
          </button>
        </div>
      </div>

      <div style={panelStyle}>
        Pending means the game has been captured but not settled yet. Outcomes update only after the game is final and you run the settlement step, so opening-week rows will stay pending until then.
      </div>

      {error ? <div style={panelStyle}>{error}</div> : null}

      {summary ? (
        <div style={summaryGridStyle}>
          <div style={summaryCardStyle}>
            <div style={{ color: "var(--text)", fontSize: 12, textTransform: "uppercase" }}>Value Plays</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: "var(--text-h)" }}>{summary.picks}</div>
            <div style={{ color: "var(--text)" }}>{`ROI ${fmtPct(summary.flat_roi)} • ${fmtUnits(summary.flat_profit_units)}`}</div>
          </div>
          <div style={summaryCardStyle}>
            <div style={{ color: "var(--text)", fontSize: 12, textTransform: "uppercase" }}>Play of the Day</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: "var(--text-h)" }}>{summary.play_of_day_count}</div>
            <div style={{ color: "var(--text)" }}>{`ROI ${fmtPct(summary.play_of_day_roi)} • ${fmtUnits(summary.play_of_day_profit_units)}`}</div>
          </div>
          <div style={summaryCardStyle}>
            <div style={{ color: "var(--text)", fontSize: 12, textTransform: "uppercase" }}>Forced Picks</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: "var(--text-h)" }}>{summary.forced_picks}</div>
            <div style={{ color: "var(--text)" }}>{`ROI ${fmtPct(summary.forced_roi)} • ${fmtUnits(summary.forced_profit_units)}`}</div>
          </div>
          <div style={summaryCardStyle}>
            <div style={{ color: "var(--text)", fontSize: 12, textTransform: "uppercase" }}>Predictive Read</div>
            <div style={{ color: "var(--text-h)" }}>{`F5 ML acc ${fmtPct(summary.f5_ml_accuracy)}`}</div>
            <div style={{ color: "var(--text-h)" }}>{`F5 RL acc ${fmtPct(summary.f5_rl_accuracy)}`}</div>
          </div>
        </div>
      ) : null}

      <div style={panelStyle}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
          <h2 style={{ margin: 0 }}>Tracked Games</h2>
          <div style={{ color: "var(--text)" }}>
            {loading ? "Loading..." : `${filteredGames.length} games`}
          </div>
        </div>
        <div style={{ overflowX: "auto" }}>
          <table style={tableStyle}>
            <thead>
              <tr>
                <th style={thStyle}>Date</th>
                <th style={thStyle}>Matchup</th>
                <th style={thStyle}>Flags</th>
                <th style={thStyle}>Value Play</th>
                <th style={thStyle}>Forced Pick</th>
                <th style={thStyle}>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {filteredGames.map((game) => (
                <tr key={`${game.pipeline_date}-${game.game_pk}`}>
                  <td style={tdStyle}>{game.pipeline_date}</td>
                  <td style={tdStyle}>{game.matchup}</td>
                  <td style={tdStyle}>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                      {game.is_play_of_day ? statusBadge("Play of Day", "good") : null}
                      {game.status === "pick" ? statusBadge("Value", "good") : null}
                      {game.paper_fallback ? statusBadge("Paper", "warn") : null}
                    </div>
                  </td>
                  <td style={tdStyle}>
                    <StrategyCellWithModel
                      marketType={game.selected_market_type}
                      side={game.selected_side}
                      odds={game.odds_at_bet}
                      edgePct={game.edge_pct}
                      sourceModel={game.source_model}
                      result={game.settled_result}
                      profit={game.flat_profit_loss}
                    />
                  </td>
                  <td style={tdStyle}>
                    <StrategyCellWithModel
                      marketType={game.forced_market_type}
                      side={game.forced_side}
                      odds={game.forced_odds_at_bet}
                      edgePct={game.forced_edge_pct}
                      sourceModel={game.forced_source_model}
                      result={game.forced_settled_result}
                      profit={game.forced_flat_profit_loss}
                    />
                  </td>
                  <td style={tdStyle}>
                    <div style={{ color: "var(--text-h)" }}>
                      {game.actual_f5_home_score !== null && game.actual_f5_away_score !== null
                        ? `${game.actual_f5_away_score}-${game.actual_f5_home_score} F5`
                        : game.actual_status ?? "Pending"}
                    </div>
                    <div style={{ color: "var(--text)" }}>
                      {game.no_pick_reason ?? game.error_message ?? "—"}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
};

export default LiveSeasonPage;
