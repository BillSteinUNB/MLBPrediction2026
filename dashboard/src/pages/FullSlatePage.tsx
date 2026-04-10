import { useDailyPics } from '../hooks';
import {
  GlassCard,
  MetricCard,
  LoadingState,
  ErrorState,
  EmptyState,
} from '../components';
import type { GamePick } from '../types/pics';

/** Format American odds with explicit sign: +130, -110, EVEN */
function fmtOdds(odds: number | null): string {
  if (odds == null) return '—';
  if (odds === 100) return 'EVEN';
  return odds > 0 ? `+${odds}` : String(odds);
}

/** Format a probability (0–1) as a percentage string: "58.0%" */
function fmtPct(value: number | null): string {
  if (value == null) return '—';
  return `${(value * 100).toFixed(1)}%`;
}

/** Format edge % for display: "+5.6%" */
function fmtEdge(value: number | null): string {
  if (value == null) return '—';
  const pct = value * 100;
  return `${pct >= 0 ? '+' : ''}${pct.toFixed(1)}%`;
}

/** Format kelly stake: "4.5%" */
function fmtKelly(value: number | null): string {
  if (value == null) return '—';
  return `${(value * 100).toFixed(1)}%`;
}

/** Format market type label: "f5_ml" → "F5 ML", "f5_rl" → "F5 RL" */
function fmtMarket(market: string | null): string {
  if (!market) return '—';
  return market
    .replace(/_/g, ' ')
    .replace(/\bf5\b/gi, 'F5')
    .replace(/\bml\b/gi, 'ML')
    .replace(/\brl\b/gi, 'RL')
    .toUpperCase();
}

/** Resolve selected side to team abbreviation */
function resolveSide(game: GamePick): string {
  if (!game.selected_side) return '—';
  return game.selected_side === 'home' ? game.home_team : game.away_team;
}

/** Sort games by game_date (proxy for scheduled_start since that's what we have) */
function sortByDate(games: GamePick[]): GamePick[] {
  return [...games].sort((a, b) => {
    const da = a.game_date;
    const db = b.game_date;
    if (da < db) return -1;
    if (da > db) return 1;
    return a.game_pk - b.game_pk;
  });
}

/** Single game card with pick details */
function GameCard({ game }: { game: GamePick }) {
  const hasPick = game.selected_market_type != null && game.selected_side != null;

  return (
    <GlassCard>
      {/* Matchup header */}
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-baseline gap-2">
          <span className="font-heading text-xl font-extrabold tracking-tight text-ink">
            {game.away_team}
          </span>
          <span className="text-sm font-medium text-ink-dim">@</span>
          <span className="font-heading text-xl font-extrabold tracking-tight text-ink">
            {game.home_team}
          </span>
        </div>
        <span className="rounded-lg bg-well/80 px-3 py-1 text-[11px] font-bold uppercase tracking-widest text-ink-dim">
          {game.game_status}
        </span>
      </div>

      {hasPick ? (
        <>
          {/* Pick callout */}
          <div className="mb-4 rounded-xl border border-accent/20 bg-accent/5 px-5 py-4">
            <div className="flex items-center justify-between">
              <div className="flex flex-col gap-1">
                <span className="text-[11px] font-bold uppercase tracking-widest text-accent">
                  {fmtMarket(game.selected_market_type)}
                </span>
                <span className="font-heading text-2xl font-extrabold text-ink">
                  {resolveSide(game)}{' '}
                  <span className="text-accent">{fmtOdds(game.selected_odds)}</span>
                </span>
              </div>
              <div className="flex flex-col items-end gap-1">
                <span
                  className={`font-heading text-lg font-extrabold ${
                    (game.selected_edge_pct ?? 0) >= 0.03
                      ? 'text-positive'
                      : 'text-ink'
                  }`}
                >
                  {fmtEdge(game.selected_edge_pct)}
                </span>
                <span className="text-[11px] font-bold uppercase tracking-widest text-ink-dim">
                  Edge
                </span>
              </div>
            </div>
          </div>

          {/* Stats row */}
          <div className="grid grid-cols-3 gap-3">
            <MetricCard
              label="Model Prob"
              value={
                game.selected_side === 'home'
                  ? fmtPct(game.market_view.ml_home_prob)
                  : fmtPct(game.market_view.ml_away_prob)
              }
              precision={1}
            />
            <MetricCard
              label="Kelly Stake"
              value={fmtKelly(game.selected_kelly_stake)}
              precision={1}
            />
            <MetricCard
              label="Proj Total"
              value={game.projected_total_runs}
              precision={1}
            />
          </div>
        </>
      ) : (
        /* No pick state */
        <div className="flex items-center justify-center rounded-xl border border-stroke/20 bg-well/40 px-5 py-6">
          <span className="text-sm font-medium text-ink-dim">No Pick</span>
        </div>
      )}
    </GlassCard>
  );
}

export default function FullSlatePage() {
  const { data: daily, loading, error, refetch } = useDailyPics();

  if (loading) return <LoadingState rows={4} />;
  if (error) return <ErrorState message={error} onRetry={refetch} />;
  if (!daily) return <EmptyState message="No picks available — run the pipeline first." />;

  const sortedGames = sortByDate(daily.games);

  return (
    <div className="mx-auto w-[680px] space-y-5 py-6">
      {/* Header */}
      <header className="space-y-2">
        <h1 className="font-heading text-3xl font-extrabold tracking-tight text-ink">
          Full Slate
        </h1>
        <div className="flex items-center gap-3">
          <span className="rounded-lg bg-accent/10 px-3 py-1 text-xs font-bold text-accent">
            {daily.generated_date}
          </span>
          <span className="text-xs font-medium text-ink-dim">
            {daily.picks_count} pick{daily.picks_count !== 1 ? 's' : ''} / {daily.total_games} game{daily.total_games !== 1 ? 's' : ''}
          </span>
          <span className="text-xs font-medium text-ink-dim">
            {daily.model_version}
          </span>
        </div>
      </header>

      {/* Game cards */}
      {sortedGames.length === 0 ? (
        <EmptyState message="No games on today's slate." />
      ) : (
        <div className="space-y-4">
          {sortedGames.map((game) => (
            <GameCard key={game.game_pk} game={game} />
          ))}
        </div>
      )}
    </div>
  );
}
