import { useState } from 'react';
import { useLiveSeasonDashboard } from '../hooks';
import { EmptyState, ErrorState, GlassCard, LoadingState, MetricCard } from '../components';
import type { LiveSeasonGame, LiveSeasonSummary } from '../types/liveSeason';

const EMPTY_SUMMARY: LiveSeasonSummary = {
  season: 2026,
  tracked_games: 0,
  settled_games: 0,
  picks: 0,
  graded_picks: 0,
  wins: 0,
  losses: 0,
  pushes: 0,
  no_picks: 0,
  errors: 0,
  paper_fallback_picks: 0,
  flat_profit_units: 0,
  flat_roi: null,
  official_units_risked: 0,
  official_profit_units: 0,
  official_roi: null,
  latest_capture_at: null,
};

function fmtPct(value: number | null | undefined): string {
  if (value == null) return '—';
  return `${(value * 100).toFixed(2)}%`;
}

function fmtOdds(value: number | null | undefined): string {
  if (value == null) return '—';
  return value > 0 ? `+${value}` : String(value);
}

function fmtUnits(value: number | null | undefined): string {
  if (value == null) return '—';
  return `${value.toFixed(1)}u`;
}

function fmtLine(value: number | null | undefined): string {
  if (value == null) return '';
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

function isBackfilled(game: LiveSeasonGame): boolean {
  return (game.tracking_source ?? 'live') !== 'live';
}

function trackingSourceLabel(game: LiveSeasonGame): string {
  return (game.tracking_source ?? 'live').replace(/_/g, ' ');
}

function fmtPick(game: LiveSeasonGame): string {
  const market = game.selected_market_type ?? '';
  const side = game.selected_side ?? '';
  const [awayTeam, homeTeam] = game.matchup.split(' @ ');
  if (!market || !side) return 'No Pick';
  if (market.endsWith('_ml')) {
    return `ML ${side === 'home' ? homeTeam : awayTeam}`;
  }
  if (market.endsWith('_rl')) {
    const team = side === 'home' ? homeTeam : awayTeam;
    if (game.line_at_bet == null) return `RL ${team}`;
    const sign = game.line_at_bet > 0 ? '+' : '';
    return `RL ${team} ${sign}${fmtLine(game.line_at_bet)}`;
  }
  if (market.endsWith('_total')) {
    const label = side === 'over' ? 'Over' : 'Under';
    if (game.line_at_bet == null) return `Total ${label}`;
    return `Total ${label} ${fmtLine(game.line_at_bet)}`;
  }
  return side;
}

function fmtResult(game: LiveSeasonGame): string {
  if (!game.settled_result) return 'Pending';
  const amount =
    game.flat_profit_loss == null
      ? ''
      : ` ${game.flat_profit_loss >= 0 ? '+' : ''}${game.flat_profit_loss.toFixed(2)}u`;
  return `${game.settled_result}${amount}`;
}

function sortHistoricalGames(games: LiveSeasonGame[] | null | undefined): LiveSeasonGame[] {
  return [...(games ?? [])].sort((a, b) => {
    if (a.pipeline_date > b.pipeline_date) return -1;
    if (a.pipeline_date < b.pipeline_date) return 1;
    return a.game_pk - b.game_pk;
  });
}

function canRemoveManualBet(game: LiveSeasonGame): boolean {
  return game.id != null;
}

function SummaryStrip({
  title,
  summary,
}: {
  title: string;
  summary: LiveSeasonSummary;
}) {
  return (
    <GlassCard title={title}>
      <div className="grid gap-4 md:grid-cols-5">
        <MetricCard label="Wins" value={summary.wins} precision={0} />
        <MetricCard label="Losses" value={summary.losses} precision={0} />
        <MetricCard label="Pushes" value={summary.pushes} precision={0} />
        <MetricCard
          label="Units Risked"
          value={summary.official_units_risked}
          unit="u"
          precision={2}
        />
        <MetricCard
          label="ROI"
          value={summary.official_roi == null ? '—' : `${(summary.official_roi * 100).toFixed(2)}%`}
        />
      </div>
    </GlassCard>
  );
}

function TrackingTable({
  title,
  games,
  emptyMessage,
  allowRemoval = false,
  onRemove,
  deletingIds,
}: {
  title: string;
  games: LiveSeasonGame[];
  emptyMessage: string;
  allowRemoval?: boolean;
  onRemove?: (game: LiveSeasonGame) => void;
  deletingIds?: Set<number>;
}) {
  return (
    <GlassCard title={title}>
      {games.length === 0 ? (
        <EmptyState
          message={emptyMessage}
          className="border-0 bg-transparent px-0 py-8"
        />
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full border-collapse text-sm">
            <thead>
              <tr className="border-b border-stroke/20 text-left text-[11px] font-bold uppercase tracking-widest text-ink-dim">
                <th className="px-3 py-3">Date</th>
                <th className="px-3 py-3">Game</th>
                <th className="px-3 py-3">Pick</th>
                <th className="px-3 py-3">Confidence</th>
                <th className="px-3 py-3">Edge</th>
                <th className="px-3 py-3">Value</th>
                <th className="px-3 py-3">Units</th>
                <th className="px-3 py-3">Result</th>
                {allowRemoval ? <th className="px-3 py-3 text-right">Edit</th> : null}
              </tr>
            </thead>
            <tbody>
              {games.map((game) => (
                <tr
                  key={`${title}-${game.pipeline_date}-${game.game_pk}-${game.selected_market_type}-${game.selected_side}-${game.line_at_bet ?? 'none'}`}
                  className="border-b border-stroke/10 text-ink"
                >
                  <td className="px-3 py-3 text-ink-dim">{game.pipeline_date}</td>
                  <td className="px-3 py-3 font-medium">{game.matchup}</td>
                  <td className="px-3 py-3">
                    {fmtPick(game)}
                    {isBackfilled(game) ? (
                      <span
                        className="ml-1 align-top text-xs font-bold text-amber-300"
                        title={`Backfilled from ${trackingSourceLabel(game)}`}
                      >
                        *
                      </span>
                    ) : null}
                  </td>
                  <td className="px-3 py-3">{fmtPct(game.model_probability)}</td>
                  <td className="px-3 py-3">{fmtPct(game.edge_pct)}</td>
                  <td className="px-3 py-3">{fmtOdds(game.odds_at_bet)}</td>
                  <td className="px-3 py-3">{fmtUnits(game.bet_units)}</td>
                  <td className="px-3 py-3">{fmtResult(game)}</td>
                  {allowRemoval ? (
                    <td className="px-3 py-3 text-right">
                      <button
                        type="button"
                        onClick={() => onRemove?.(game)}
                        disabled={
                          !canRemoveManualBet(game) ||
                          (game.id != null && deletingIds?.has(game.id) === true)
                        }
                        className="inline-flex h-7 w-7 items-center justify-center rounded-full border border-rose-500/30 bg-rose-500/10 text-sm font-bold text-rose-300 transition hover:border-rose-400/50 hover:bg-rose-500/20 disabled:cursor-not-allowed disabled:opacity-50"
                        title={
                          canRemoveManualBet(game)
                            ? 'Remove from My Bets'
                            : 'Unavailable'
                        }
                      >
                        {game.id != null && deletingIds?.has(game.id) ? '…' : '-'}
                      </button>
                    </td>
                  ) : null}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </GlassCard>
  );
}

export default function SeasonTrackingPage() {
  const { data, loading, error, refetch } = useLiveSeasonDashboard();
  const [deletingIds, setDeletingIds] = useState<Set<number>>(new Set());

  const removeManualBet = async (game: LiveSeasonGame) => {
    if (!canRemoveManualBet(game)) return;
    setDeletingIds((current) => new Set(current).add(game.id as number));
    try {
      const response = await fetch('/api/live-season/manual-bets', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          season: game.season,
          pipeline_date: data?.pipeline_date ?? game.pipeline_date,
          manual_bet_id: game.id,
        }),
      });
      if (!response.ok) {
        const text = await response.text();
        let detail = text || `HTTP ${response.status}`;
        try {
          const parsed = JSON.parse(text) as { detail?: string };
          if (parsed.detail) detail = parsed.detail;
        } catch {
          // Use the raw response body when the server does not return JSON.
        }
        window.alert(detail);
        return;
      }
      void refetch();
    } finally {
      setDeletingIds((current) => {
        const next = new Set(current);
        next.delete(game.id as number);
        return next;
      });
    }
  };

  if (loading) return <LoadingState rows={5} />;
  if (error && !data) return <ErrorState message={error} onRetry={refetch} />;
  if (!data) return <EmptyState message="No tracked games available." />;

  const modelSummary = data.summary ?? EMPTY_SUMMARY;
  const manualSummary = data.manual_summary ?? EMPTY_SUMMARY;
  const forcedSummary = data.forced_summary ?? EMPTY_SUMMARY;
  const historicalModelGames = sortHistoricalGames(data.historical_games);
  const todayModelGames = sortHistoricalGames(data.today_games);
  const historicalManualGames = sortHistoricalGames(data.manual_historical_games);
  const todayManualGames = sortHistoricalGames(data.manual_today_games);
  const historicalForcedGames = sortHistoricalGames(data.forced_historical_games);
  const todayForcedGames = sortHistoricalGames(data.forced_today_games);

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <header className="space-y-2">
        <p className="text-[11px] font-bold uppercase tracking-widest text-accent">Live Season</p>
        <h1 className="font-heading text-3xl font-extrabold tracking-tight text-ink">
          Tracking
        </h1>
        <p className="max-w-3xl text-sm leading-relaxed text-ink-dim">
          Separate ledgers for the machine’s frozen auto picks and the bets you actually submitted
          from the slate.
        </p>
        <p className="text-xs font-medium text-ink-dim">
          {data.release_name} · {data.model_display_name} · {data.strategy_name} {data.strategy_version}
        </p>
        <p className="text-xs text-ink-dim">* marks a backfilled model entry, not a live capture.</p>
      </header>

      <SummaryStrip title="My Bets" summary={manualSummary} />
      <TrackingTable
        title="My Bets Today"
        games={todayManualGames}
        emptyMessage="No manual bets submitted for today."
        allowRemoval
        onRemove={(game) => void removeManualBet(game)}
        deletingIds={deletingIds}
      />
      <TrackingTable
        title="My Historical Bets"
        games={historicalManualGames}
        emptyMessage="No manual bets have been submitted yet."
        allowRemoval
        onRemove={(game) => void removeManualBet(game)}
        deletingIds={deletingIds}
      />

      <SummaryStrip title="Machine POTD" summary={modelSummary} />
      <TrackingTable
        title="Machine POTD Today"
        games={todayModelGames}
        emptyMessage="No frozen model bets for today."
      />
      <TrackingTable
        title="Machine POTD History"
        games={historicalModelGames}
        emptyMessage="No frozen model bets yet."
      />

      <SummaryStrip title="All Machine Picks" summary={forcedSummary} />
      <TrackingTable
        title="All Machine Picks Today"
        games={todayForcedGames}
        emptyMessage="No machine picks for today."
      />
      <TrackingTable
        title="All Machine Pick History"
        games={historicalForcedGames}
        emptyMessage="No all-machine picks available yet."
      />
    </div>
  );
}
