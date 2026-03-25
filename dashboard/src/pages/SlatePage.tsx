import React, { useEffect, useMemo, useState } from "react";
import { getSlate } from "../api";
import type { SlateGame, SlateResponse } from "../api";
import { TooltipLabel } from "../components/TooltipLabel";

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
  gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
  gap: 12,
};

const summaryCardStyle: React.CSSProperties = {
  border: "1px solid var(--border)",
  borderRadius: 12,
  background: "var(--bg-panel)",
  padding: 14,
};

const gamesGridStyle: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))",
  gap: 14,
};

const gameCardStyle: React.CSSProperties = {
  border: "1px solid var(--border)",
  borderRadius: 14,
  background: "var(--bg-panel)",
  padding: 16,
  display: "flex",
  flexDirection: "column",
  gap: 12,
};

const cardTopStyle: React.CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "flex-start",
  gap: 12,
};

const badgeRowStyle: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: 6,
};

const sectionGridStyle: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "1fr 1fr",
  gap: 10,
};

const statBlockStyle: React.CSSProperties = {
  border: "1px solid var(--border)",
  borderRadius: 10,
  padding: 10,
  background: "var(--bg-elevated)",
};

const statusBoxStyle: React.CSSProperties = {
  border: "1px solid var(--border)",
  borderRadius: 10,
  padding: 10,
  background: "var(--bg-elevated)",
};

const decisionBoxStyle: React.CSSProperties = {
  borderRadius: 14,
  padding: "20px 20px 20px 24px",
  background: "linear-gradient(135deg, rgba(16,50,36,0.97) 0%, rgba(11,38,28,0.98) 50%, rgba(8,30,22,0.99) 100%)",
  borderLeft: "4px solid #34d399",
  border: "1px solid rgba(52,211,153,0.18)",
  borderLeftWidth: 4,
  borderLeftColor: "#34d399",
  boxShadow: "0 4px 24px rgba(16,50,36,0.45), inset 0 1px 0 rgba(52,211,153,0.08)",
  position: "relative" as const,
  overflow: "hidden",
};

const estimatedDecisionBoxStyle: React.CSSProperties = {
  borderRadius: 14,
  padding: "20px 20px 20px 24px",
  background: "linear-gradient(135deg, rgba(50,42,16,0.97) 0%, rgba(38,32,11,0.98) 50%, rgba(30,25,8,0.99) 100%)",
  borderLeft: "4px solid #fbbf24",
  border: "1px solid rgba(251,191,36,0.18)",
  borderLeftWidth: 4,
  borderLeftColor: "#fbbf24",
  boxShadow: "0 4px 24px rgba(50,42,16,0.45), inset 0 1px 0 rgba(251,191,36,0.08)",
  position: "relative" as const,
  overflow: "hidden",
};

const forcedBoxStyle: React.CSSProperties = {
  borderRadius: 12,
  padding: "14px 16px 14px 20px",
  background: "linear-gradient(135deg, rgba(16,25,43,0.95), rgba(13,20,33,0.97))",
  border: "1px solid var(--border)",
  borderLeft: "3px solid var(--muted)",
  borderLeftWidth: 3,
  borderLeftColor: "var(--muted)",
};

const mutedStyle: React.CSSProperties = {
  color: "var(--text)",
  fontSize: 13,
};

const emptyStyle: React.CSSProperties = {
  border: "1px dashed var(--border)",
  borderRadius: 10,
  padding: 20,
  color: "var(--text)",
  textAlign: "center",
};

function todayLocalDate(): string {
  const now = new Date();
  const year = now.getFullYear();
  const month = `${now.getMonth() + 1}`.padStart(2, "0");
  const day = `${now.getDate()}`.padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function fmtPct(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function fmtEdgePct(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function fmtOdds(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value > 0 ? `+${value}` : `${value}`;
}

function fmtPoint(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value > 0 ? `+${value}` : `${value}`;
}


function fmtRuns(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return value.toFixed(2);
}

function formatBookLabel(value: string | null | undefined): string {
  if (!value) return "—";
  if (value.startsWith("estimate:full-game:")) {
    return `Estimate via ${value.replace("estimate:full-game:", "")}`;
  }
  if (value.startsWith("sbr:")) {
    return `SBR ${value.replace("sbr:", "")}`;
  }
  return value;
}

function fairAmericanOdds(probability: number | null | undefined): string {
  if (
    probability === null ||
    probability === undefined ||
    Number.isNaN(probability) ||
    probability <= 0 ||
    probability >= 1
  ) {
    return "—";
  }
  const odds =
    probability >= 0.5
      ? -Math.round((probability / (1 - probability)) * 100)
      : Math.round(((1 - probability) / probability) * 100);
  return fmtOdds(odds);
}

function splitMatchup(matchup: string): { awayTeam: string; homeTeam: string } {
  const parts = matchup.split(" @ ");
  if (parts.length === 2) {
    return { awayTeam: parts[0], homeTeam: parts[1] };
  }
  return { awayTeam: "Away", homeTeam: "Home" };
}

function teamNameForSide(game: SlateGame, side: "home" | "away"): string {
  const teams = splitMatchup(game.matchup);
  return side === "home" ? teams.homeTeam : teams.awayTeam;
}

function projectedSpreadLabel(game: SlateGame): string {
  const margin = game.prediction?.projected_f5_home_margin;
  if (margin === null || margin === undefined || Number.isNaN(margin) || Math.abs(margin) < 0.01) {
    return "Pick'em";
  }
  if (margin > 0) {
    return `${teamNameForSide(game, "home")} ${fmtPoint(-margin)}`;
  }
  return `${teamNameForSide(game, "away")} ${fmtPoint(margin)}`;
}

function marketLabel(marketType: string): string {
  if (marketType === "f5_ml") return "First 5 Moneyline";
  if (marketType === "f5_rl") return "First 5 Run Line";
  return marketType.toUpperCase();
}

function sourceModelLabel(value: string | null | undefined): string {
  if (!value) return "Unknown";
  if (value === "legacy_f5_ml") return "Legacy F5 ML";
  if (value === "legacy_f5_ml_equiv") return "Legacy F5 ML Equivalent";
  if (value === "legacy_f5_rl") return "Legacy F5 RL";
  if (value === "rlv2_direct") return "RL V2 Direct";
  if (value === "rlv2_margin") return "RL V2 Margin";
  if (value === "rlv2_blend") return "RL V2 Blend";
  return value;
}
void sourceModelLabel;

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

function recommendedSide(game: SlateGame, market: "ml" | "rl"): { side: string; probability: number } | null {
  if (!game.prediction) return null;
  if (market === "ml") {
    return game.prediction.f5_ml_home_prob >= game.prediction.f5_ml_away_prob
      ? { side: "home", probability: game.prediction.f5_ml_home_prob }
      : { side: "away", probability: game.prediction.f5_ml_away_prob };
  }
  return game.prediction.f5_rl_home_prob >= game.prediction.f5_rl_away_prob
    ? { side: "home", probability: game.prediction.f5_rl_home_prob }
    : { side: "away", probability: game.prediction.f5_rl_away_prob };
}

function GameCard({ game }: { game: SlateGame }) {
  const mlLean = recommendedSide(game, "ml");
  const rlLean = recommendedSide(game, "rl");
  const decision = game.selected_decision;
  const forcedDecision = game.forced_decision;
  const input = game.input_status;
  const estimatedDecision = Boolean(decision?.book_name?.startsWith("estimate:"));
  const recommendedTeam =
    decision?.side === "home" || decision?.side === "away"
      ? teamNameForSide(game, decision.side)
      : null;
  const displayedLine =
    decision?.market_type === "f5_rl" && decision?.line_at_bet !== null && decision?.line_at_bet !== undefined
      ? `${recommendedTeam ?? decision.side.toUpperCase()} ${fmtPoint(decision.line_at_bet)}`
      : `${recommendedTeam ?? decision?.side?.toUpperCase() ?? "—"}`;

  const forcedDisplayedLine =
    forcedDecision?.side === "home" || forcedDecision?.side === "away"
      ? forcedDecision.market_type === "f5_rl" && forcedDecision.line_at_bet !== null && forcedDecision.line_at_bet !== undefined
        ? `${teamNameForSide(game, forcedDecision.side)} ${fmtPoint(forcedDecision.line_at_bet)}`
        : `${teamNameForSide(game, forcedDecision.side)}`
      : "—";

  return (
    <article style={gameCardStyle}>
      <div style={cardTopStyle}>
        <div>
          <h3 style={{ margin: 0, fontSize: 18, color: "var(--text-h)" }}>{game.matchup}</h3>
          <div style={{ ...mutedStyle, marginTop: 4 }}>Game #{game.game_pk}</div>
        </div>
        <div style={badgeRowStyle}>
          {game.status === "pick" && statusBadge("Pick", "good")}
          {game.status === "no_pick" && statusBadge("No pick", "warn")}
          {game.status === "error" && statusBadge("Error", "bad")}
          {game.paper_fallback && statusBadge("Paper fallback", "neutral")}
          {estimatedDecision && statusBadge("Estimated market", "warn")}
        </div>
      </div>

      {decision ? (
        <div style={estimatedDecision ? estimatedDecisionBoxStyle : decisionBoxStyle}>
          {/* Subtle glow overlay */}
          <div style={{
            position: "absolute",
            top: 0,
            right: 0,
            width: 120,
            height: 120,
            background: estimatedDecision
              ? "radial-gradient(circle at top right, rgba(251,191,36,0.06), transparent 70%)"
              : "radial-gradient(circle at top right, rgba(52,211,153,0.06), transparent 70%)",
            pointerEvents: "none",
          }} />

          {/* Market type label */}
          <div style={{
            fontSize: 11,
            fontWeight: 700,
            textTransform: "uppercase",
            letterSpacing: "1.2px",
            color: estimatedDecision ? "rgba(251,191,36,0.7)" : "rgba(52,211,153,0.7)",
            marginBottom: 8,
          }}>
            {estimatedDecision ? "Preview Lean" : "Recommended Bet"} — {marketLabel(decision.market_type)}
          </div>

          {/* Hero row: Team name + Odds */}
          <div style={{
            display: "flex",
            alignItems: "baseline",
            gap: 12,
            flexWrap: "wrap",
          }}>
            <span style={{
              fontSize: 26,
              fontWeight: 800,
              color: "#f3f6fb",
              letterSpacing: "-0.3px",
              lineHeight: 1.15,
            }}>
              {displayedLine}
            </span>
            <span style={{
              fontSize: 22,
              fontWeight: 700,
              color: estimatedDecision ? "#fbbf24" : "#34d399",
              fontFamily: "var(--mono)",
              letterSpacing: "-0.5px",
            }}>
              {fmtOdds(decision.odds_at_bet)}
            </span>
          </div>

          {/* Edge badge + Confidence bar row */}
          <div style={{
            display: "flex",
            alignItems: "center",
            gap: 14,
            marginTop: 14,
            flexWrap: "wrap",
          }}>
            {/* Edge pill */}
            <span style={{
              display: "inline-flex",
              alignItems: "center",
              padding: "4px 12px",
              borderRadius: 999,
              background: estimatedDecision
                ? "rgba(251,191,36,0.12)"
                : "rgba(52,211,153,0.12)",
              border: estimatedDecision
                ? "1px solid rgba(251,191,36,0.28)"
                : "1px solid rgba(52,211,153,0.28)",
              color: estimatedDecision ? "#fbbf24" : "#34d399",
              fontSize: 13,
              fontWeight: 700,
              letterSpacing: "0.3px",
            }}>
              {fmtEdgePct(decision.edge_pct)} EDGE
            </span>

            {/* Confidence bar */}
            <div style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              flex: 1,
              minWidth: 140,
            }}>
              <div style={{
                flex: 1,
                height: 6,
                borderRadius: 3,
                background: "rgba(255,255,255,0.07)",
                overflow: "hidden",
              }}>
                <div style={{
                  width: `${Math.min(Math.max((decision.model_probability ?? 0) * 100, 0), 100)}%`,
                  height: "100%",
                  borderRadius: 3,
                  background: estimatedDecision
                    ? "linear-gradient(90deg, rgba(251,191,36,0.5), #fbbf24)"
                    : "linear-gradient(90deg, rgba(52,211,153,0.5), #34d399)",
                  transition: "width 0.4s ease",
                }} />
              </div>
              <span style={{
                fontSize: 12,
                fontWeight: 600,
                color: "rgba(243,246,251,0.65)",
                whiteSpace: "nowrap",
                fontFamily: "var(--mono)",
              }}>
                {fmtPct(decision.model_probability)}
              </span>
            </div>
          </div>

          {/* Projected F5 Score */}
          {game.prediction ? (
            <div style={{
              marginTop: 14,
              display: "flex",
              alignItems: "center",
              gap: 10,
            }}>
              <span style={{
                fontSize: 11,
                fontWeight: 700,
                textTransform: "uppercase",
                letterSpacing: "0.8px",
                color: "rgba(243,246,251,0.35)",
              }}>
                Proj F5
              </span>
              <span style={{
                fontSize: 15,
                fontWeight: 600,
                color: "rgba(243,246,251,0.85)",
                fontFamily: "var(--mono)",
                letterSpacing: "0.5px",
              }}>
                {teamNameForSide(game, "away")} {fmtRuns(game.prediction.projected_f5_away_runs)}
                {" "}&ndash;{" "}
                {fmtRuns(game.prediction.projected_f5_home_runs)} {teamNameForSide(game, "home")}
              </span>
            </div>
          ) : null}

          {/* Estimated market disclaimer */}
          {estimatedDecision ? (
            <div style={{
              marginTop: 12,
              fontSize: 11,
              color: "rgba(251,191,36,0.45)",
              fontStyle: "italic",
              lineHeight: 1.4,
            }}>
              Estimated F5 market from full-game odds — not a posted sportsbook line.
            </div>
          ) : null}

          {/* Narrative section */}
          {(game as SlateGame & { narrative?: string | null }).narrative ? (
            <div style={{
              marginTop: 14,
              paddingTop: 12,
              borderTop: estimatedDecision
                ? "1px solid rgba(251,191,36,0.1)"
                : "1px solid rgba(52,211,153,0.1)",
            }}>
              <p style={{
                fontSize: 13,
                lineHeight: 1.55,
                color: "rgba(243,246,251,0.55)",
                fontStyle: "italic",
                margin: 0,
              }}>
                {(game as SlateGame & { narrative?: string | null }).narrative}
              </p>
            </div>
          ) : null}
        </div>
      ) : (
        <div style={{
          borderRadius: 12,
          padding: "16px 18px",
          background: "linear-gradient(135deg, rgba(16,25,43,0.6), rgba(13,20,33,0.7))",
          border: "1px solid var(--border)",
        }}>
          <div style={{
            fontSize: 11,
            fontWeight: 700,
            textTransform: "uppercase",
            letterSpacing: "1.2px",
            color: "var(--muted)",
            marginBottom: 6,
          }}>
            No Pick
          </div>
          <div style={{
            fontSize: 14,
            color: "rgba(243,246,251,0.6)",
            lineHeight: 1.5,
          }}>
            {game.no_pick_reason ?? game.error_message ?? "No decision"}
          </div>
        </div>
      )}

      {forcedDecision ? (
        <div style={forcedBoxStyle}>
          <div style={{
            fontSize: 11,
            fontWeight: 700,
            textTransform: "uppercase",
            letterSpacing: "1px",
            color: "var(--muted)",
            marginBottom: 8,
          }}>
            Forced Pick
          </div>
          <div style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            flexWrap: "wrap",
          }}>
            <span style={{
              fontSize: 16,
              fontWeight: 700,
              color: "var(--text-h)",
            }}>
              {forcedDisplayedLine}
            </span>
            <span style={{
              fontSize: 15,
              fontWeight: 600,
              color: "var(--accent)",
              fontFamily: "var(--mono)",
            }}>
              {fmtOdds(forcedDecision.odds_at_bet)}
            </span>
            <span style={{
              fontSize: 12,
              color: "var(--muted)",
            }}>
              {marketLabel(forcedDecision.market_type)}
            </span>
            <span style={{
              display: "inline-flex",
              alignItems: "center",
              padding: "2px 8px",
              borderRadius: 999,
              background: "rgba(124,199,255,0.08)",
              border: "1px solid rgba(124,199,255,0.18)",
              color: "var(--accent)",
              fontSize: 11,
              fontWeight: 700,
            }}>
              {fmtEdgePct(forcedDecision.edge_pct)} EDGE
            </span>
          </div>
        </div>
      ) : null}

      <div style={sectionGridStyle}>
        <div style={statBlockStyle}>
          <TooltipLabel
            label="F5 Moneyline"
            as="div"
            style={{ fontSize: 12, color: "var(--text)", fontWeight: 700, textTransform: "uppercase" }}
          />
          <div style={{ marginTop: 8, fontSize: 14, color: "var(--text-h)" }}>
            {teamNameForSide(game, "home")} {fmtPct(game.prediction?.f5_ml_home_prob)}
          </div>
          <div style={{ fontSize: 14, color: "var(--text-h)" }}>
            {teamNameForSide(game, "away")} {fmtPct(game.prediction?.f5_ml_away_prob)}
          </div>
          <div style={{ ...mutedStyle, marginTop: 8 }}>
            Lean {mlLean ? `${teamNameForSide(game, mlLean.side as "home" | "away")} (${fmtPct(mlLean.probability)})` : "—"}
          </div>
          <div style={{ ...mutedStyle }}>
            Fair {mlLean ? fairAmericanOdds(mlLean.probability) : "—"}
          </div>
        </div>

        <div style={statBlockStyle}>
          <TooltipLabel
            label="F5 Run Line"
            as="div"
            style={{ fontSize: 12, color: "var(--text)", fontWeight: 700, textTransform: "uppercase" }}
          />
          <div style={{ marginTop: 8, fontSize: 14, color: "var(--text-h)" }}>
            {teamNameForSide(game, "home")} {fmtPct(game.prediction?.f5_rl_home_prob)}
          </div>
          <div style={{ fontSize: 14, color: "var(--text-h)" }}>
            {teamNameForSide(game, "away")} {fmtPct(game.prediction?.f5_rl_away_prob)}
          </div>
          <div style={{ ...mutedStyle, marginTop: 8 }}>
            Lean {rlLean ? `${teamNameForSide(game, rlLean.side as "home" | "away")} (${fmtPct(rlLean.probability)})` : "—"}
          </div>
          <div style={{ ...mutedStyle }}>
            Fair {rlLean ? fairAmericanOdds(rlLean.probability) : "—"}
          </div>
        </div>
      </div>

      <div style={statBlockStyle}>
        <TooltipLabel
          label="Model Read"
          as="div"
          style={{ fontSize: 12, color: "var(--text)", fontWeight: 700, textTransform: "uppercase" }}
        />
        <div style={{ marginTop: 8, fontSize: 14, color: "var(--text-h)" }}>
          Estimated F5 score: {teamNameForSide(game, "away")} {fmtRuns(game.prediction?.projected_f5_away_runs)} - {teamNameForSide(game, "home")} {fmtRuns(game.prediction?.projected_f5_home_runs)}
        </div>
        <div style={{ ...mutedStyle, marginTop: 6 }}>
          Projected F5 total: {fmtRuns(game.prediction?.projected_f5_total_runs)}
        </div>
        <div style={mutedStyle}>
          Projected F5 spread: {projectedSpreadLabel(game)}
        </div>
      </div>

      <div style={statBlockStyle}>
        <TooltipLabel
          label="Full Game Market"
          as="div"
          style={{ fontSize: 12, color: "var(--text)", fontWeight: 700, textTransform: "uppercase" }}
        />
        <div style={{ marginTop: 8, fontSize: 14, color: "var(--text-h)" }}>
          Home ML {fmtOdds(input?.full_game_home_ml)} {input?.full_game_home_ml_book ? `(${input.full_game_home_ml_book})` : ""}
        </div>
        <div style={{ fontSize: 14, color: "var(--text-h)" }}>
          Away ML {fmtOdds(input?.full_game_away_ml)} {input?.full_game_away_ml_book ? `(${input.full_game_away_ml_book})` : ""}
        </div>
        <div style={{ ...mutedStyle, marginTop: 8 }}>
          Home RL {input?.full_game_home_spread ?? "—"} ({fmtOdds(input?.full_game_home_spread_odds)}) {input?.full_game_home_spread_book ? `• ${input.full_game_home_spread_book}` : ""}
        </div>
        <div style={mutedStyle}>
          Away RL {input?.full_game_away_spread ?? "—"} ({fmtOdds(input?.full_game_away_spread_odds)}) {input?.full_game_away_spread_book ? `• ${input.full_game_away_spread_book}` : ""}
        </div>
      </div>

      <div style={statusBoxStyle}>
        <TooltipLabel
          label="Inputs"
          as="div"
          style={{ fontSize: 12, color: "var(--text)", fontWeight: 700, textTransform: "uppercase" }}
        />
        <div style={{ marginTop: 8, ...mutedStyle }}>
          Away lineup: {input?.away_lineup_available ? "available" : "missing"} • {input?.away_lineup_source ?? "—"} • {input?.away_lineup_confirmed ? "confirmed" : "projected/schedule"}
        </div>
        <div style={mutedStyle}>
          Home lineup: {input?.home_lineup_available ? "available" : "missing"} • {input?.home_lineup_source ?? "—"} • {input?.home_lineup_confirmed ? "confirmed" : "projected/schedule"}
        </div>
        <div style={mutedStyle}>
          F5 odds: {input?.odds_available ? `available (${(input?.odds_books ?? []).map((book) => formatBookLabel(book)).join(", ") || "book(s)"})` : "missing"}
        </div>
        <div style={mutedStyle}>
          F5 source: {input?.odds_available ? `${(input?.f5_odds_sources ?? []).join(", ") || "—"}${input?.f5_odds_estimated ? " • preview estimate" : ""}` : "missing"}
        </div>
        <div style={mutedStyle}>
          Full-game odds: {input?.full_game_odds_available ? `available (${input?.full_game_odds_books.join(", ") || "book(s)"})` : "missing"}
        </div>
        <div style={mutedStyle}>
          Weather: {input?.weather_available ? "available" : "missing"}
        </div>
      </div>
    </article>
  );
}

const SlatePage: React.FC = () => {
  const [selectedDate, setSelectedDate] = useState<string>(todayLocalDate());
  const [queryDate, setQueryDate] = useState<string>(todayLocalDate());
  const [data, setData] = useState<SlateResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getSlate(queryDate)
      .then((response) => {
        if (!cancelled) {
          setData(response);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load slate");
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [queryDate]);

  const summary = useMemo(() => {
    const games = data?.games ?? [];
    const picks = games.filter((game) => game.status === "pick").length;
    const projectedLineups = games.filter(
      (game) =>
        game.input_status?.home_lineup_source === "rotowire" ||
        game.input_status?.away_lineup_source === "rotowire"
    ).length;
    const confirmedLineups = games.filter(
      (game) =>
        game.input_status?.home_lineup_confirmed || game.input_status?.away_lineup_confirmed
    ).length;
    return { games: games.length, picks, projectedLineups, confirmedLineups };
  }, [data]);

  return (
    <div className="page" style={pageStyle}>
      <div style={topRowStyle}>
        <div>
          <TooltipLabel label="Live Slate" as="h2" style={{ margin: 0, color: "var(--text-h)" }} />
          <div style={{ ...mutedStyle, marginTop: 4 }}>
            Dry-run predictions for the selected date using the current model bundle.
          </div>
        </div>
        <div style={controlsStyle}>
          <input
            type="date"
            value={selectedDate}
            onChange={(event) => setSelectedDate(event.target.value)}
            style={inputStyle}
          />
          <button type="button" style={buttonStyle} onClick={() => setQueryDate(selectedDate)}>
            Run Slate
          </button>
        </div>
      </div>

      <div style={summaryGridStyle}>
        <div style={summaryCardStyle}>
          <TooltipLabel label="Games" as="div" style={mutedStyle} helpText="How many games were returned for the selected date." />
          <div style={{ marginTop: 6, fontSize: 26, fontWeight: 700, color: "var(--text-h)" }}>{summary.games}</div>
        </div>
        <div style={summaryCardStyle}>
          <TooltipLabel label="Picks" as="div" style={mutedStyle} helpText="How many games currently have a selected decision after odds and risk rules." />
          <div style={{ marginTop: 6, fontSize: 26, fontWeight: 700, color: "var(--text-h)" }}>{summary.picks}</div>
        </div>
        <div style={summaryCardStyle}>
          <TooltipLabel label="Projected Lineups" as="div" style={mutedStyle} helpText="Games where at least one side has a projected lineup source rather than schedule-only data." />
          <div style={{ marginTop: 6, fontSize: 26, fontWeight: 700, color: "var(--text-h)" }}>{summary.projectedLineups}</div>
        </div>
        <div style={summaryCardStyle}>
          <TooltipLabel label="Confirmed Lineups" as="div" style={mutedStyle} helpText="Games where MLB has posted an official batting order for at least one side." />
          <div style={{ marginTop: 6, fontSize: 26, fontWeight: 700, color: "var(--text-h)" }}>{summary.confirmedLineups}</div>
        </div>
      </div>

      {data ? (
        <div style={{ ...mutedStyle, marginTop: -8 }}>
          Run {data.run_id} • Model {data.model_version} • Date {data.pipeline_date}
        </div>
      ) : null}

      {loading ? <div style={emptyStyle}>Running dry-run slate...</div> : null}
      {error ? <div style={emptyStyle}>Failed to load slate: {error}</div> : null}
      {!loading && !error && data && data.games.length === 0 ? (
        <div style={emptyStyle}>No games returned for this date.</div>
      ) : null}

      {!loading && !error && data && data.games.length > 0 ? (
        <div style={gamesGridStyle}>
          {data.games.map((game) => (
            <GameCard key={`${game.game_pk}-${game.matchup}`} game={game} />
          ))}
        </div>
      ) : null}
    </div>
  );
};

export default SlatePage;
