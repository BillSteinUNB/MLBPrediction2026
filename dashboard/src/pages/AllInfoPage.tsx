/**
 * AllInfoPage — Screenshot-optimized dense stats display showing EVERY stat per game.
 *
 * Renders one card per game with organized sections:
 *   - Market probabilities (win %, cover %)
 *   - Implied probabilities from odds (ratios)
 *   - Edges (both sides, color-coded)
 *   - Signal indicators
 *   - Projected runs
 *   - Selected pick summary
 *
 * Fixed-width layout (~800px) for consistent screenshot capture.
 * No animations, no responsive breakpoints.
 */
import { useDailyPics } from '../hooks';
import { GlassCard, LoadingState, ErrorState, EmptyState } from '../components';
import type { GamePick, MarketView } from '../types/pics';

/* ── Formatting helpers ─────────────────────────────────── */

/** Format a 0–1 probability as a percentage with 1 decimal (e.g., 0.58 → "58.0%"). */
function fmtPct(v: number | null): string {
  if (v == null) return '—';
  return `${(v * 100).toFixed(1)}%`;
}

/** Format an edge value as a signed percentage (e.g., 0.056 → "+5.6%"). */
function fmtEdge(v: number | null): string {
  if (v == null) return '—';
  const pct = v * 100;
  const sign = pct >= 0 ? '+' : '';
  return `${sign}${pct.toFixed(1)}%`;
}

/** Format American odds with explicit sign (e.g., -110, +130). */
function fmtOdds(v: number | null): string {
  if (v == null) return '—';
  return v >= 0 ? `+${v}` : String(v);
}

/** Color class for edge values: positive = green, negative = red, null = dim. */
function edgeColor(v: number | null): string {
  if (v == null) return 'text-ink-dim';
  return v > 0 ? 'text-positive' : v < 0 ? 'text-negative' : 'text-ink-dim';
}

/* ── Signal Dot ─────────────────────────────────────────── */

function SignalDot({ value }: { value: number | null }) {
  if (value == null) {
    return <span className="inline-block h-2.5 w-2.5 rounded-full bg-stroke/40" title="No signal" />;
  }
  // Signal > 0.5 is considered YES / active
  const active = value > 0.5;
  return (
    <span
      className={`inline-block h-2.5 w-2.5 rounded-full ${active ? 'bg-positive' : 'bg-stroke/40'}`}
      title={active ? 'Signal: YES' : 'Signal: NO'}
    />
  );
}

/* ── Stat Row — a single label + value pair in the grid ── */

function StatCell({
  label,
  value,
  className = '',
}: {
  label: string;
  value: string;
  className?: string;
}) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[10px] font-bold uppercase tracking-widest text-ink-dim">
        {label}
      </span>
      <span className={`font-heading text-sm font-extrabold ${className || 'text-ink'}`}>
        {value}
      </span>
    </div>
  );
}

/* ── Section header inside a card ────────────────────────── */

function SectionHeader({ children }: { children: string }) {
  return (
    <div className="mb-2 mt-4 border-b border-stroke/20 pb-1 first:mt-0">
      <span className="text-[10px] font-bold uppercase tracking-widest text-accent">
        {children}
      </span>
    </div>
  );
}

/* ── Game Stats Card ─────────────────────────────────────── */

function GameStatsCard({ game }: { game: GamePick }) {
  const mv: MarketView = game.market_view;

  return (
    <GlassCard className="w-[800px]">
      {/* Game header */}
      <div className="mb-4 flex items-baseline justify-between">
        <div className="flex items-baseline gap-2">
          <span className="font-heading text-lg font-extrabold text-ink">
            {game.away_team}
          </span>
          <span className="text-sm text-ink-dim">@</span>
          <span className="font-heading text-lg font-extrabold text-ink">
            {game.home_team}
          </span>
        </div>
        <span className="text-xs font-medium text-ink-dim">
          {game.game_date} · {game.game_status}
        </span>
      </div>

      {/* ── Win % (ML Probabilities) ─────────────────────── */}
      <SectionHeader>Win % (Model ML)</SectionHeader>
      <div className="grid grid-cols-4 gap-4">
        <StatCell label="Home ML" value={fmtPct(mv.ml_home_prob)} />
        <StatCell label="Away ML" value={fmtPct(mv.ml_away_prob)} />
        <StatCell label="Home RL" value={fmtPct(mv.rl_home_prob)} />
        <StatCell label="Away RL" value={fmtPct(mv.rl_away_prob)} />
      </div>

      {/* ── Cover % (Run Line Probabilities) ─────────────── */}
      <SectionHeader>Cover %</SectionHeader>
      <div className="grid grid-cols-2 gap-4">
        <StatCell label="Home Cover" value={fmtPct(game.cover_home_prob)} />
        <StatCell label="Away Cover" value={fmtPct(game.cover_away_prob)} />
      </div>

      {/* ── Implied Probabilities (Ratios from Odds) ─────── */}
      <SectionHeader>Implied Probs (Ratios)</SectionHeader>
      <div className="grid grid-cols-4 gap-4">
        <StatCell label="Home Implied" value={fmtPct(mv.home_implied_prob)} />
        <StatCell label="Away Implied" value={fmtPct(mv.away_implied_prob)} />
        <StatCell label="Home Odds" value={fmtOdds(mv.home_odds)} />
        <StatCell label="Away Odds" value={fmtOdds(mv.away_odds)} />
      </div>

      {/* ── Edges (Both Sides) ───────────────────────────── */}
      <SectionHeader>Edges</SectionHeader>
      <div className="grid grid-cols-4 gap-4">
        <StatCell
          label="Home Edge"
          value={fmtEdge(mv.home_edge_pct)}
          className={edgeColor(mv.home_edge_pct)}
        />
        <StatCell
          label="Away Edge"
          value={fmtEdge(mv.away_edge_pct)}
          className={edgeColor(mv.away_edge_pct)}
        />
        <StatCell
          label="Home EV"
          value={mv.home_ev != null ? mv.home_ev.toFixed(3) : '—'}
        />
        <StatCell
          label="Away EV"
          value={mv.away_ev != null ? mv.away_ev.toFixed(3) : '—'}
        />
      </div>

      {/* ── Signal ───────────────────────────────────────── */}
      <SectionHeader>Signal</SectionHeader>
      <div className="grid grid-cols-2 gap-4">
        <div className="flex items-center gap-2">
          <SignalDot value={mv.home_signal} />
          <span className="text-[10px] font-bold uppercase tracking-widest text-ink-dim">
            Home
          </span>
        </div>
        <div className="flex items-center gap-2">
          <SignalDot value={mv.away_signal} />
          <span className="text-[10px] font-bold uppercase tracking-widest text-ink-dim">
            Away
          </span>
        </div>
      </div>

      {/* ── Projected Runs ───────────────────────────────── */}
      <SectionHeader>Projected Runs</SectionHeader>
      <div className="grid grid-cols-3 gap-4">
        <StatCell
          label="Home"
          value={game.projected_home_runs != null ? game.projected_home_runs.toFixed(1) : '—'}
        />
        <StatCell
          label="Away"
          value={game.projected_away_runs != null ? game.projected_away_runs.toFixed(1) : '—'}
        />
        <StatCell
          label="Total"
          value={game.projected_total_runs != null ? game.projected_total_runs.toFixed(1) : '—'}
        />
      </div>

      {/* ── Selected Pick ────────────────────────────────── */}
      {game.selected_market_type != null && (
        <>
          <SectionHeader>Selected Pick</SectionHeader>
          <div className="grid grid-cols-5 gap-4">
            <StatCell label="Market" value={game.selected_market_type ?? '—'} />
            <StatCell label="Side" value={game.selected_side ?? '—'} />
            <StatCell label="Odds" value={fmtOdds(game.selected_odds)} />
            <StatCell
              label="Edge"
              value={fmtEdge(game.selected_edge_pct)}
              className={edgeColor(game.selected_edge_pct)}
            />
            <StatCell
              label="Kelly"
              value={game.selected_kelly_stake != null ? `${(game.selected_kelly_stake * 100).toFixed(1)}%` : '—'}
            />
          </div>
        </>
      )}
    </GlassCard>
  );
}

/* ── Page Component ──────────────────────────────────────── */

export default function AllInfoPage() {
  const { data, loading, error, refetch } = useDailyPics();

  if (loading) return <LoadingState rows={8} />;
  if (error) return <ErrorState message={error} onRetry={refetch} />;
  if (!data || data.games.length === 0) {
    return <EmptyState message="No daily data available — run the pipeline first." />;
  }

  return (
    <div className="mx-auto w-[800px] space-y-6">
      {/* Page header */}
      <header className="space-y-1">
        <h1 className="font-heading text-2xl font-extrabold tracking-tight text-ink">
          All Info
        </h1>
        <p className="text-xs text-ink-dim">
          {data.generated_date} · {data.model_version} · {data.total_games} games
        </p>
      </header>

      {/* One card per game */}
      {data.games.map((game) => (
        <GameStatsCard key={game.game_pk} game={game} />
      ))}
    </div>
  );
}
