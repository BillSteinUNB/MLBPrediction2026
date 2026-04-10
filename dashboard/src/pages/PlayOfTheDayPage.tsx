/**
 * PlayOfTheDayPage — Screenshot-optimized card for the single best pick.
 *
 * Fixed-width (~640px) dark card centered on dark background.
 * Designed for manual screenshot cropping — no animations, no responsive layout.
 *
 * Data flow:
 *   usePlayOfTheDay() → play_of_the_day.json (core pick data)
 *   useDailyPics()    → daily.json (additional game projections for the matched game_pk)
 */
import { usePlayOfTheDay, useDailyPics } from '../hooks';
import {
  GlassCard,
  MetricCard,
  LoadingState,
  ErrorState,
  EmptyState,
} from '../components';
import type { GamePick } from '../types/pics';

/* ── Formatting helpers ──────────────────────────────────── */

/** Format American odds with + prefix for positive values. */
function fmtOdds(odds: number): string {
  return odds > 0 ? `+${String(odds)}` : String(odds);
}

/** Format a probability (0.0–1.0) as a percentage string. */
function fmtPct(value: number, decimals = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

/** Format ISO date string to short readable form. */
function fmtDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString('en-US', {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
  });
}

/** Map market_type codes to human-readable labels. */
function fmtMarket(marketType: string): string {
  const MAP: Record<string, string> = {
    f5_ml: 'F5 Moneyline',
    f5_rl: 'F5 Run Line',
    ml: 'Full Game ML',
    rl: 'Full Game RL',
  };
  return MAP[marketType] ?? marketType.toUpperCase();
}

/** Derive confidence tier from edge percentage. */
function confidenceTier(edgePct: number): { label: string; classes: string } {
  if (edgePct >= 0.08)
    return { label: 'Elite Edge', classes: 'bg-accent/20 text-accent border-accent/30' };
  if (edgePct >= 0.05)
    return { label: 'Strong Edge', classes: 'bg-positive/20 text-positive border-positive/30' };
  if (edgePct >= 0.03)
    return { label: 'Solid Edge', classes: 'bg-caution/20 text-caution border-caution/30' };
  return { label: 'Slim Edge', classes: 'bg-stroke/20 text-ink-dim border-stroke/30' };
}

/* ── Page Component ──────────────────────────────────────── */

export default function PlayOfTheDayPage() {
  const { data: potd, loading, error, refetch } = usePlayOfTheDay();
  const { data: daily } = useDailyPics();

  /* ── Early-return states ──────────────────────────────── */
  if (loading) return <LoadingState rows={4} />;
  if (error) return <ErrorState message={error} onRetry={refetch} />;
  if (!potd)
    return (
      <EmptyState message="No Play of the Day set — edit play_of_the_day.json" />
    );

  /* ── Derived values ───────────────────────────────────── */
  const game: GamePick | undefined = daily?.games.find(
    (g) => g.game_pk === potd.game_pk,
  );
  const teamOnSide =
    potd.side === 'home' ? potd.home_team : potd.away_team;
  const headline = `${teamOnSide} ${fmtMarket(potd.market_type)}`;
  const confidence = confidenceTier(potd.edge_pct);

  return (
    <div className="flex min-h-[80vh] items-start justify-center pt-8">
      <div className="w-[640px]">
        <GlassCard className="border-accent/40">
          {/* ── Header row ──────────────────────────────── */}
          <div className="mb-5 flex items-center justify-between">
            <span className="text-[11px] font-bold uppercase tracking-widest text-accent">
              Play of the Day
            </span>
            <span className="text-[11px] font-bold uppercase tracking-widest text-ink-dim">
              {fmtDate(potd.pick_date)}
            </span>
          </div>

          {/* ── Headline pick ───────────────────────────── */}
          <div className="mb-6">
            <h1 className="font-heading text-4xl font-extrabold tracking-tight text-ink">
              {headline}
            </h1>
            <div className="mt-3 flex items-center gap-3">
              <span className="font-heading text-3xl font-extrabold text-accent">
                {fmtOdds(potd.odds)}
              </span>
              <span
                className={`inline-flex items-center rounded-full border px-3 py-1 text-[11px] font-bold uppercase tracking-widest ${confidence.classes}`}
              >
                {confidence.label}
              </span>
            </div>
          </div>

          {/* ── Matchup block ───────────────────────────── */}
          <div className="mb-6 rounded-xl border border-stroke/20 bg-well/60 px-5 py-4">
            <div className="flex items-center justify-between">
              <span className="font-heading text-xl font-extrabold text-ink">
                {potd.away_team}
              </span>
              <span className="text-sm font-bold text-ink-dim">@</span>
              <span className="font-heading text-xl font-extrabold text-ink">
                {potd.home_team}
              </span>
            </div>
            <p className="mt-1 text-center text-xs font-medium text-ink-dim">
              {fmtDate(potd.game_date)}
            </p>
          </div>

          {/* ── Market / Side / Odds row ────────────────── */}
          <div className="mb-6 flex items-center gap-4">
            <div className="flex flex-col gap-1">
              <span className="text-[11px] font-bold uppercase tracking-widest text-ink-dim">
                Market
              </span>
              <span className="text-sm font-bold text-ink">
                {fmtMarket(potd.market_type)}
              </span>
            </div>
            <div className="h-8 w-px bg-stroke/30" />
            <div className="flex flex-col gap-1">
              <span className="text-[11px] font-bold uppercase tracking-widest text-ink-dim">
                Side
              </span>
              <span className="text-sm font-bold capitalize text-ink">
                {potd.side} — {teamOnSide}
              </span>
            </div>
            <div className="h-8 w-px bg-stroke/30" />
            <div className="flex flex-col gap-1">
              <span className="text-[11px] font-bold uppercase tracking-widest text-ink-dim">
                Odds
              </span>
              <span className="text-sm font-bold text-accent">
                {fmtOdds(potd.odds)}
              </span>
            </div>
          </div>

          {/* ── Key stats ───────────────────────────────── */}
          <div className="mb-6 grid grid-cols-3 gap-3">
            <MetricCard
              label="Model Prob"
              value={fmtPct(potd.model_probability)}
            />
            <MetricCard
              label="Edge"
              value={fmtPct(potd.edge_pct)}
            />
            <MetricCard
              label="Kelly Stake"
              value={fmtPct(potd.kelly_stake_pct)}
            />
          </div>

          {/* ── Game projections from daily data ─────────── */}
          {game && (
            <div className="mb-6 grid grid-cols-3 gap-3">
              <MetricCard
                label="Proj Home"
                value={game.projected_home_runs}
                precision={2}
              />
              <MetricCard
                label="Proj Away"
                value={game.projected_away_runs}
                precision={2}
              />
              <MetricCard
                label="Proj Total"
                value={game.projected_total_runs}
                precision={2}
              />
            </div>
          )}

          {/* ── Implied vs Model probability bar ─────────── */}
          <div className="mb-6 rounded-xl border border-stroke/20 bg-well/60 px-5 py-4">
            <div className="mb-3 flex justify-between text-[11px] font-bold uppercase tracking-widest text-ink-dim">
              <span>Implied: {fmtPct(potd.implied_probability)}</span>
              <span>Model: {fmtPct(potd.model_probability)}</span>
            </div>
            <div className="relative h-2.5 w-full overflow-hidden rounded-full bg-stroke/30">
              <div
                className="absolute inset-y-0 left-0 rounded-full bg-ink-dim/40"
                style={{ width: fmtPct(potd.implied_probability, 0) }}
              />
              <div
                className="absolute inset-y-0 left-0 rounded-full bg-accent"
                style={{ width: fmtPct(potd.model_probability, 0) }}
              />
            </div>
            <p className="mt-2 text-center text-[11px] font-bold uppercase tracking-widest text-ink-dim">
              Edge = Model − Implied
            </p>
          </div>

          {/* ── Narrative / Commentary ────────────────────── */}
          {potd.narrative && (
            <div className="rounded-xl border border-accent/15 bg-deep/30 px-5 py-4">
              <span className="mb-2 block text-[11px] font-bold uppercase tracking-widest text-accent">
                Analysis
              </span>
              <p className="text-sm font-medium leading-relaxed text-ink">
                {potd.narrative}
              </p>
            </div>
          )}
        </GlassCard>
      </div>
    </div>
  );
}
