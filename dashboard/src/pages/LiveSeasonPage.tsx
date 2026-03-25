import React, { useCallback, useEffect, useMemo, useState } from "react";
import { getLiveSeasonGames, getLiveSeasonSummary } from "../api";
import type { LiveSeasonGameResponse, LiveSeasonSummaryResponse } from "../api";
import { useBetTracker } from "../hooks/useBetTracker";
import type { TrackedBet, SplitSummary } from "../hooks/useBetTracker";

/* ------------------------------------------------------------------ */
/*  Design tokens (inline CSSProperties, referencing CSS vars)        */
/* ------------------------------------------------------------------ */

const pageStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 24,
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
  fontFamily: "var(--sans)",
  fontSize: 14,
  padding: "8px 12px",
  border: "1px solid var(--border)",
  borderRadius: 8,
  background: "var(--bg-panel)",
  color: "var(--text-h)",
};

const buttonStyle: React.CSSProperties = {
  fontFamily: "var(--sans)",
  fontSize: 13,
  fontWeight: 600,
  padding: "8px 16px",
  border: "1px solid var(--border)",
  borderRadius: 8,
  background: "var(--surface-3)",
  color: "var(--text-h)",
  cursor: "pointer",
  transition: "background 0.15s, border-color 0.15s",
};

const buttonAccentStyle: React.CSSProperties = {
  ...buttonStyle,
  background: "linear-gradient(135deg, rgba(124,199,255,0.15), rgba(124,199,255,0.06))",
  borderColor: "rgba(124,199,255,0.35)",
  color: "var(--accent)",
};

const panelStyle: React.CSSProperties = {
  border: "1px solid var(--border)",
  borderRadius: 14,
  background: "var(--bg-panel)",
  padding: 16,
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
  fontSize: 11,
  fontWeight: 600,
  textTransform: "uppercase",
  letterSpacing: "0.5px",
};

const tdStyle: React.CSSProperties = {
  padding: "10px 12px",
  borderBottom: "1px solid var(--border)",
  color: "var(--text-h)",
  verticalAlign: "top",
};

/* ------------------------------------------------------------------ */
/*  Summary card gradient configs                                     */
/* ------------------------------------------------------------------ */

interface CardTheme {
  gradient: string;
  borderAccent: string;
  iconBg: string;
}

const cardThemes: Record<string, CardTheme> = {
  default: {
    gradient: "var(--bg-panel)",
    borderAccent: "var(--border)",
    iconBg: "var(--surface-3)",
  },
  potd: {
    gradient: "linear-gradient(135deg, rgba(127,224,168,0.08) 0%, rgba(127,224,168,0.02) 100%)",
    borderAccent: "rgba(127,224,168,0.25)",
    iconBg: "rgba(127,224,168,0.12)",
  },
};

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */

function fmtPct(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "\u2014";
  return `${(value * 100).toFixed(1)}%`;
}

function fmtOdds(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "\u2014";
  return value > 0 ? `+${value}` : `${value}`;
}

function fmtUnits(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "\u2014";
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}u`;
}

function sourceModelLabel(value: string | null | undefined): string {
  if (!value) return "\u2014";
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
        fontSize: 11,
        fontWeight: 600,
        letterSpacing: "0.3px",
      }}
    >
      {label}
    </span>
  );
}

function resultBadge(result: string | null | undefined): React.ReactElement {
  if (!result) return statusBadge("PENDING", "neutral");
  if (result === "WIN") return statusBadge("WIN", "good");
  if (result === "LOSS") return statusBadge("LOSS", "bad");
  if (result === "PUSH") return statusBadge("PUSH", "warn");
  return statusBadge(result, "neutral");
}

function makeGameKey(game: LiveSeasonGameResponse): string {
  return `${game.pipeline_date}-${game.game_pk}`;
}

function gameToTrackedBet(game: LiveSeasonGameResponse): TrackedBet {
  return {
    gameKey: makeGameKey(game),
    pipeline_date: game.pipeline_date,
    game_pk: game.game_pk,
    matchup: game.matchup,
    market_type: game.selected_market_type ?? "",
    side: game.selected_side ?? "",
    odds: game.odds_at_bet ?? null,
    edge_pct: game.edge_pct ?? null,
    is_play_of_day: game.is_play_of_day,
    trackedAt: new Date().toISOString(),
  };
}

/* ------------------------------------------------------------------ */
/*  Sub-components                                                    */
/* ------------------------------------------------------------------ */

function TrackerCheckbox({
  checked,
  onChange,
}: {
  checked: boolean;
  onChange: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onChange}
      aria-label={checked ? "Remove from tracked bets" : "Add to tracked bets"}
      style={{
        width: 22,
        height: 22,
        borderRadius: 6,
        border: checked
          ? "2px solid var(--good-fg)"
          : "2px solid var(--border)",
        background: checked
          ? "var(--good-bg)"
          : "transparent",
        cursor: "pointer",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 0,
        transition: "all 0.15s ease",
        flexShrink: 0,
      }}
    >
      {checked ? (
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true" role="img">
          <title>Checked</title>
          <path
            d="M2 6.5L4.5 9L10 3"
            stroke="var(--good-fg)"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      ) : null}
    </button>
  );
}

function SummaryCard({
  label,
  count,
  detail,
  theme,
}: {
  label: string;
  count: string | number;
  detail: string;
  theme: CardTheme;
  iconShape?: string;
}) {
  return (
    <div
      style={{
        border: `1px solid ${theme.borderAccent}`,
        borderRadius: 12,
        background: theme.gradient,
        padding: 16,
        display: "flex",
        flexDirection: "column",
        gap: 6,
      }}
    >
      <div
        style={{
          color: "var(--text)",
          fontSize: 11,
          fontWeight: 600,
          textTransform: "uppercase",
          letterSpacing: "0.6px",
        }}
      >
        {label}
      </div>
      <div style={{ fontSize: 28, fontWeight: 700, color: "var(--text-h)", lineHeight: 1.1 }}>
        {count}
      </div>
      <div style={{ color: "var(--muted)", fontSize: 12, marginTop: 2 }}>{detail}</div>
    </div>
  );
}

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
    return <span style={{ color: "var(--text)" }}>{"\u2014"}</span>;
  }
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
      <div style={{ fontWeight: 600 }}>{`${marketType} ${side} ${fmtOdds(odds)}`}</div>
      <div style={{ color: "var(--muted)", fontSize: 12 }}>{`edge ${fmtPct(edgePct)}`}</div>
      <div style={{ color: "var(--muted)", fontSize: 12 }}>{sourceModelLabel(sourceModel)}</div>
      <div style={{ fontSize: 12 }}>
        {resultBadge(result)}
        <span style={{ color: "var(--muted)", marginLeft: 6 }}>{fmtUnits(profit)}</span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  POTD Spotlight Card                                               */
/* ------------------------------------------------------------------ */

function PotdCard({
  game,
  tracked,
  onToggle,
}: {
  game: LiveSeasonGameResponse;
  tracked: boolean;
  onToggle: () => void;
}) {
  return (
    <div
      style={{
        border: "1px solid rgba(127,224,168,0.3)",
        borderRadius: 14,
        background: "linear-gradient(135deg, rgba(127,224,168,0.06) 0%, var(--bg-panel) 60%)",
        padding: 20,
        display: "flex",
        flexDirection: "column",
        gap: 14,
        position: "relative",
        overflow: "hidden",
      }}
    >
      {/* decorative accent line at top */}
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          height: 3,
          background: "linear-gradient(90deg, var(--good-fg), transparent)",
          borderRadius: "14px 14px 0 0",
        }}
      />

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
            {statusBadge("PLAY OF THE DAY", "good")}
            {game.paper_fallback ? statusBadge("Paper", "warn") : null}
          </div>
          <div style={{ fontSize: 20, fontWeight: 700, color: "var(--text-h)", lineHeight: 1.2 }}>
            {game.matchup}
          </div>
          <div style={{ color: "var(--muted)", fontSize: 12, marginTop: 4 }}>
            {game.pipeline_date}
            {game.play_of_day_score != null ? ` \u00B7 Score: ${game.play_of_day_score.toFixed(2)}` : ""}
          </div>
        </div>
        <TrackerCheckbox checked={tracked} onChange={onToggle} />
      </div>

      {/* Strategy details */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
          gap: 12,
          background: "rgba(0,0,0,0.15)",
          borderRadius: 10,
          padding: 14,
        }}
      >
        <div>
          <div style={{ color: "var(--muted)", fontSize: 11, fontWeight: 600, textTransform: "uppercase", marginBottom: 4 }}>Market</div>
          <div style={{ color: "var(--text-h)", fontWeight: 600 }}>
            {game.selected_market_type ?? "\u2014"} {game.selected_side ?? ""}
          </div>
        </div>
        <div>
          <div style={{ color: "var(--muted)", fontSize: 11, fontWeight: 600, textTransform: "uppercase", marginBottom: 4 }}>Odds</div>
          <div style={{ color: "var(--text-h)", fontWeight: 600 }}>{fmtOdds(game.odds_at_bet)}</div>
        </div>
        <div>
          <div style={{ color: "var(--muted)", fontSize: 11, fontWeight: 600, textTransform: "uppercase", marginBottom: 4 }}>Edge</div>
          <div style={{ color: "var(--good-fg)", fontWeight: 600 }}>{fmtPct(game.edge_pct)}</div>
        </div>
        <div>
          <div style={{ color: "var(--muted)", fontSize: 11, fontWeight: 600, textTransform: "uppercase", marginBottom: 4 }}>Model Prob</div>
          <div style={{ color: "var(--text-h)", fontWeight: 600 }}>{fmtPct(game.model_probability)}</div>
        </div>
        <div>
          <div style={{ color: "var(--muted)", fontSize: 11, fontWeight: 600, textTransform: "uppercase", marginBottom: 4 }}>Result</div>
          <div>{resultBadge(game.settled_result)}</div>
        </div>
        <div>
          <div style={{ color: "var(--muted)", fontSize: 11, fontWeight: 600, textTransform: "uppercase", marginBottom: 4 }}>P/L</div>
          <div style={{ color: "var(--text-h)", fontWeight: 600 }}>{fmtUnits(game.flat_profit_loss)}</div>
        </div>
      </div>

      {/* Narrative */}
      {game.narrative ? (
        <p style={{ color: "var(--muted)", fontSize: 13, lineHeight: 1.5, fontStyle: "italic", margin: 0 }}>
          {game.narrative}
        </p>
      ) : null}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Value Bet Card                                                    */
/* ------------------------------------------------------------------ */

function ValueBetCard({
  game,
  tracked,
  onToggle,
}: {
  game: LiveSeasonGameResponse;
  tracked: boolean;
  onToggle: () => void;
}) {
  return (
    <div
      style={{
        border: tracked ? "1px solid var(--good-fg)" : "1px solid var(--border)",
        borderRadius: 12,
        background: "var(--bg-panel)",
        padding: 14,
        display: "flex",
        flexDirection: "column",
        gap: 10,
        transition: "border-color 0.15s",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 8 }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 15, fontWeight: 700, color: "var(--text-h)", lineHeight: 1.2 }}>
            {game.matchup}
          </div>
          <div style={{ color: "var(--muted)", fontSize: 12, marginTop: 2 }}>{game.pipeline_date}</div>
        </div>
        <TrackerCheckbox checked={tracked} onChange={onToggle} />
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 8,
          fontSize: 12,
        }}
      >
        <div>
          <span style={{ color: "var(--muted)" }}>Pick: </span>
          <span style={{ color: "var(--text-h)", fontWeight: 600 }}>
            {game.selected_market_type ?? "\u2014"} {game.selected_side ?? ""}
          </span>
        </div>
        <div>
          <span style={{ color: "var(--muted)" }}>Odds: </span>
          <span style={{ color: "var(--text-h)", fontWeight: 600 }}>{fmtOdds(game.odds_at_bet)}</span>
        </div>
        <div>
          <span style={{ color: "var(--muted)" }}>Edge: </span>
          <span style={{ color: "var(--good-fg)", fontWeight: 600 }}>{fmtPct(game.edge_pct)}</span>
        </div>
        <div>
          {resultBadge(game.settled_result)}
          <span style={{ color: "var(--muted)", marginLeft: 6 }}>{fmtUnits(game.flat_profit_loss)}</span>
        </div>
      </div>

      {game.narrative ? (
        <p style={{ color: "var(--muted)", fontSize: 12, lineHeight: 1.4, fontStyle: "italic", margin: 0 }}>
          {game.narrative}
        </p>
      ) : null}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Tracked Bets Panel                                                */
/* ------------------------------------------------------------------ */

function TrackedBetsPanel({
  games,
  trackedBets,
  onRemove,
  onClearAll,
}: {
  games: LiveSeasonGameResponse[];
  trackedBets: Map<string, TrackedBet>;
  onRemove: (gameKey: string) => void;
  onClearAll: () => void;
}) {
  const [expanded, setExpanded] = useState(false);

  if (trackedBets.size === 0) return null;

  // Match tracked bets to game data for detail display
  const gameMap = new Map<string, LiveSeasonGameResponse>();
  for (const g of games) {
    gameMap.set(makeGameKey(g), g);
  }

  return (
    <div
      style={{
        border: "1px solid var(--border)",
        borderRadius: 14,
        background: "var(--bg-panel)",
        overflow: "hidden",
      }}
    >
      {/* Header */}
      <button
        type="button"
        onClick={() => setExpanded((p) => !p)}
        style={{
          width: "100%",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "14px 18px",
          background: "none",
          border: "none",
          cursor: "pointer",
          color: "var(--text-h)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ fontSize: 14, fontWeight: 600 }}>Manage Tracked Bets</span>
          <span
            style={{
              fontSize: 12,
              fontWeight: 600,
              background: "var(--surface-3)",
              padding: "2px 8px",
              borderRadius: 999,
              color: "var(--muted)",
            }}
          >
            {trackedBets.size}
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <svg
            width="14"
            height="14"
            viewBox="0 0 14 14"
            fill="none"
            aria-hidden="true"
            role="img"
            style={{
              transform: expanded ? "rotate(180deg)" : "rotate(0deg)",
              transition: "transform 0.2s ease",
            }}
          >
            <title>Toggle</title>
            <path d="M3 5L7 9L11 5" stroke="var(--muted)" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
        </div>
      </button>

      {/* Expanded content */}
      {expanded ? (
        <div style={{ padding: "0 18px 16px" }}>
          <div style={{ overflowX: "auto" }}>
            <table style={{ ...tableStyle, fontSize: 12 }}>
              <thead>
                <tr>
                  <th style={thStyle}>Date</th>
                  <th style={thStyle}>Matchup</th>
                  <th style={thStyle}>Pick</th>
                  <th style={thStyle}>Odds</th>
                  <th style={thStyle}>Result</th>
                  <th style={thStyle}>P/L</th>
                  <th style={{ ...thStyle, width: 40, textAlign: "center" }}></th>
                </tr>
              </thead>
              <tbody>
                {Array.from(trackedBets.values()).map((bet) => {
                  const game = gameMap.get(bet.gameKey);
                  const result = game?.settled_result ?? null;
                  const profit = game?.flat_profit_loss ?? null;
                  return (
                    <tr key={bet.gameKey}>
                      <td style={{ ...tdStyle, fontSize: 12 }}>{bet.pipeline_date}</td>
                      <td style={{ ...tdStyle, fontSize: 12, fontWeight: 600 }}>{bet.matchup}</td>
                      <td style={{ ...tdStyle, fontSize: 12 }}>
                        {bet.market_type} {bet.side}
                      </td>
                      <td style={{ ...tdStyle, fontSize: 12 }}>{fmtOdds(bet.odds)}</td>
                      <td style={{ ...tdStyle, fontSize: 12 }}>{resultBadge(result)}</td>
                      <td style={{ ...tdStyle, fontSize: 12 }}>{fmtUnits(profit)}</td>
                      <td style={{ ...tdStyle, fontSize: 12, textAlign: "center" }}>
                        <button
                          type="button"
                          onClick={() => onRemove(bet.gameKey)}
                          style={{
                            background: "none",
                            border: "none",
                            color: "var(--bad-fg)",
                            cursor: "pointer",
                            fontSize: 14,
                            padding: "2px 6px",
                            borderRadius: 4,
                            lineHeight: 1,
                          }}
                          aria-label="Remove tracked bet"
                        >
                          <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true" role="img">
                            <title>Remove</title>
                            <path d="M2 2L10 10M10 2L2 10" stroke="var(--bad-fg)" strokeWidth="1.5" strokeLinecap="round" />
                          </svg>
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div style={{ marginTop: 12, display: "flex", justifyContent: "flex-end" }}>
            <button
              type="button"
              onClick={onClearAll}
              style={{
                ...buttonStyle,
                fontSize: 12,
                padding: "6px 12px",
                color: "var(--bad-fg)",
                borderColor: "rgba(255,138,138,0.25)",
                background: "rgba(255,138,138,0.06)",
              }}
            >
              Clear All Tracked
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Section Header                                                    */
/* ------------------------------------------------------------------ */

function SectionHeader({
  title,
  count,
  rightContent,
}: {
  title: string;
  count?: number;
  rightContent?: React.ReactNode;
}) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <h2 style={{ margin: 0, fontSize: 18 }}>{title}</h2>
        {count !== undefined ? (
          <span
            style={{
              fontSize: 12,
              fontWeight: 600,
              background: "var(--surface-3)",
              padding: "2px 8px",
              borderRadius: 999,
              color: "var(--muted)",
            }}
          >
            {count}
          </span>
        ) : null}
      </div>
      {rightContent ?? null}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Main Page                                                         */
/* ------------------------------------------------------------------ */

const LiveSeasonPage: React.FC = () => {
  const [pipelineDate, setPipelineDate] = useState("");
  const [summary, setSummary] = useState<LiveSeasonSummaryResponse | null>(null);
  const [games, setGames] = useState<LiveSeasonGameResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const tracker = useBetTracker();

  const loadData = useCallback(async (dateFilter?: string) => {
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
  }, []);

  useEffect(() => {
    void loadData();
  }, [loadData]);

  // Compute tracked bets split summary using actual game data
  const trackerSplit: SplitSummary = useMemo(() => tracker.computeSplitSummary(games), [tracker, games]);

  // Game segments — POTD: show only the latest one (most recent date)
  const latestPotd = useMemo(() => {
    const sorted = [...games.filter((g) => g.is_play_of_day)].sort(
      (a, b) => b.pipeline_date.localeCompare(a.pipeline_date),
    );
    return sorted.length > 0 ? sorted[0] : null;
  }, [games]);
  const valueBets = useMemo(
    () => games.filter((g) => g.status === "pick" && !g.is_play_of_day),
    [games],
  );

  const handleToggle = (game: LiveSeasonGameResponse) => {
    tracker.toggleBet(gameToTrackedBet(game));
  };

  return (
    <section style={pageStyle}>
      {/* ---- Header ---- */}
      <div style={topRowStyle}>
        <div>
          <h1 style={{ margin: 0 }}>Live Season 2026</h1>
          <p style={{ color: "var(--muted)", fontSize: 14, marginTop: 4 }}>
            Track play of the day, value plays, and forced picks. Toggle bets you want to follow.
          </p>
        </div>
        <div style={controlsStyle}>
          <input
            type="date"
            value={pipelineDate}
            onChange={(event) => setPipelineDate(event.target.value)}
            style={inputStyle}
          />
          <button type="button" style={buttonAccentStyle} onClick={() => void loadData(pipelineDate)}>
            Load Date
          </button>
          <button
            type="button"
            style={buttonStyle}
            onClick={() => {
              setPipelineDate("");
              void loadData();
            }}
          >
            Full Season
          </button>
        </div>
      </div>

      {/* ---- Info Banner ---- */}
      <div
        style={{
          ...panelStyle,
          display: "flex",
          alignItems: "center",
          gap: 10,
          fontSize: 13,
          color: "var(--muted)",
          background: "linear-gradient(90deg, rgba(243,191,108,0.06), transparent 60%)",
          borderColor: "rgba(243,191,108,0.15)",
        }}
      >
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true" role="img" style={{ flexShrink: 0 }}>
          <title>Info</title>
          <circle cx="8" cy="8" r="7" stroke="var(--warn-fg)" strokeWidth="1.5" />
          <path d="M8 5V8.5" stroke="var(--warn-fg)" strokeWidth="1.5" strokeLinecap="round" />
          <circle cx="8" cy="11" r="0.75" fill="var(--warn-fg)" />
        </svg>
        <span>
          <strong style={{ color: "var(--warn-fg)" }}>Pending</strong> means the game has been captured but not
          settled yet. Outcomes update after the game is final and you run the settlement step.
        </span>
      </div>

      {/* ---- Error ---- */}
      {error ? (
        <div
          style={{
            ...panelStyle,
            borderColor: "rgba(255,138,138,0.25)",
            background: "rgba(255,138,138,0.06)",
            color: "var(--bad-fg)",
          }}
        >
          {error}
        </div>
      ) : null}

      {/* ---- Scorecard: 2x2 grid ---- */}
      {summary ? (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 14,
          }}
        >
          {/* Row 1: POTD */}
          <SummaryCard
            label="Model POTD"
            count={`${summary.play_of_day_wins}W-${summary.play_of_day_losses}L-${summary.play_of_day_pushes}P`}
            detail={`${summary.play_of_day_count} picks \u00B7 ${fmtUnits(summary.play_of_day_profit_units)}`}
            theme={cardThemes.potd}
          />
          <SummaryCard
            label="My POTD"
            count={trackerSplit.potd.totalTracked > 0
              ? `${trackerSplit.potd.wins}W-${trackerSplit.potd.losses}L-${trackerSplit.potd.pushes}P`
              : "\u2014"}
            detail={trackerSplit.potd.totalTracked > 0
              ? `${trackerSplit.potd.totalTracked} picks \u00B7 ${fmtUnits(trackerSplit.potd.totalUnits)}`
              : "No POTD tracked yet"}
            theme={cardThemes.potd}
          />
          {/* Row 2: Value Plays */}
          <SummaryCard
            label="Model Value Plays"
            count={`${summary.wins}W-${summary.losses}L-${summary.pushes}P`}
            detail={`${summary.picks} plays \u00B7 ${fmtUnits(summary.flat_profit_units)}`}
            theme={cardThemes.default}
          />
          <SummaryCard
            label="My Value Plays"
            count={trackerSplit.value.totalTracked > 0
              ? `${trackerSplit.value.wins}W-${trackerSplit.value.losses}L-${trackerSplit.value.pushes}P`
              : "\u2014"}
            detail={trackerSplit.value.totalTracked > 0
              ? `${trackerSplit.value.totalTracked} picks \u00B7 ${fmtUnits(trackerSplit.value.totalUnits)}`
              : "No value plays tracked yet"}
            theme={cardThemes.default}
          />
        </div>
      ) : null}

      {/* ---- Manage Tracked Bets (collapsed by default) ---- */}
      <TrackedBetsPanel
        games={games}
        trackedBets={tracker.trackedBets}
        onRemove={tracker.removeTracked}
        onClearAll={tracker.clearAll}
      />

      {/* ---- POTD Spotlight (single latest) ---- */}
      {latestPotd ? (
        <div>
          <SectionHeader title="Play of the Day" />
          <PotdCard
            game={latestPotd}
            tracked={tracker.isTracked(makeGameKey(latestPotd))}
            onToggle={() => handleToggle(latestPotd)}
          />
        </div>
      ) : null}

      {/* ---- Value Bets Grid ---- */}
      {valueBets.length > 0 ? (
        <div>
          <SectionHeader title="Value Bets" count={valueBets.length} />
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
              gap: 14,
            }}
          >
            {valueBets.map((game) => (
              <ValueBetCard
                key={makeGameKey(game)}
                game={game}
                tracked={tracker.isTracked(makeGameKey(game))}
                onToggle={() => handleToggle(game)}
              />
            ))}
          </div>
        </div>
      ) : null}

      {/* ---- All Games Table ---- */}
      <div style={panelStyle}>
        <SectionHeader
          title="All Games"
          count={games.length}
          rightContent={
            loading ? (
              <span style={{ color: "var(--muted)", fontSize: 13 }}>Loading...</span>
            ) : null
          }
        />
        <div style={{ overflowX: "auto" }}>
          <table style={tableStyle}>
            <thead>
              <tr>
                <th style={{ ...thStyle, width: 40, textAlign: "center" }}></th>
                <th style={thStyle}>Date</th>
                <th style={thStyle}>Matchup</th>
                <th style={thStyle}>Flags</th>
                <th style={thStyle}>Value Play</th>
                <th style={thStyle}>Forced Pick</th>
                <th style={thStyle}>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {games.map((game) => {
                const gk = makeGameKey(game);
                const isTracked = tracker.isTracked(gk);
                return (
                  <tr
                    key={gk}
                    style={{
                      borderLeft: isTracked ? "3px solid var(--good-fg)" : "3px solid transparent",
                      background: isTracked ? "rgba(127,224,168,0.03)" : "transparent",
                      transition: "background 0.15s, border-color 0.15s",
                    }}
                  >
                    <td style={{ ...tdStyle, textAlign: "center", paddingLeft: 8, paddingRight: 4 }}>
                      <TrackerCheckbox
                        checked={isTracked}
                        onChange={() => handleToggle(game)}
                      />
                    </td>
                    <td style={{ ...tdStyle, fontSize: 12, whiteSpace: "nowrap" }}>{game.pipeline_date}</td>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{game.matchup}</td>
                    <td style={tdStyle}>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                        {game.is_play_of_day ? statusBadge("POTD", "good") : null}
                        {game.status === "pick" ? statusBadge("Value", "good") : null}
                        {game.paper_fallback ? statusBadge("Paper", "warn") : null}
                        {game.status === "error" ? statusBadge("Error", "bad") : null}
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
                      <div style={{ color: "var(--text-h)", fontSize: 13 }}>
                        {game.actual_f5_home_score !== null && game.actual_f5_away_score !== null
                          ? `${game.actual_f5_away_score}-${game.actual_f5_home_score} F5`
                          : game.actual_status ?? "Pending"}
                      </div>
                      <div style={{ color: "var(--muted)", fontSize: 12 }}>
                        {game.no_pick_reason ?? game.error_message ?? "\u2014"}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
};

export default LiveSeasonPage;
