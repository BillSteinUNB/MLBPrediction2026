import { useState } from 'react';
import { useLiveSeasonDashboard } from '../hooks';
import { EmptyState, ErrorState, GlassCard, LoadingState, MetricCard } from '../components';
import type { LiveSeasonGame } from '../types/liveSeason';

interface ManualFormState {
  odds: string;
  units: string;
  submitting: boolean;
  message: string | null;
  error: string | null;
}

const TEAM_NAMES: Record<string, string> = {
  ARI: 'Arizona Diamondbacks',
  ATL: 'Atlanta Braves',
  BAL: 'Baltimore Orioles',
  BOS: 'Boston Red Sox',
  CHC: 'Chicago Cubs',
  CHW: 'Chicago White Sox',
  CIN: 'Cincinnati Reds',
  CLE: 'Cleveland Guardians',
  COL: 'Colorado Rockies',
  DET: 'Detroit Tigers',
  HOU: 'Houston Astros',
  KC: 'Kansas City Royals',
  LAA: 'Los Angeles Angels',
  LAD: 'Los Angeles Dodgers',
  MIA: 'Miami Marlins',
  MIL: 'Milwaukee Brewers',
  MIN: 'Minnesota Twins',
  NYM: 'New York Mets',
  NYY: 'New York Yankees',
  OAK: 'Athletics',
  PHI: 'Philadelphia Phillies',
  PIT: 'Pittsburgh Pirates',
  SD: 'San Diego Padres',
  SEA: 'Seattle Mariners',
  SF: 'San Francisco Giants',
  STL: 'St. Louis Cardinals',
  TB: 'Tampa Bay Rays',
  TEX: 'Texas Rangers',
  TOR: 'Toronto Blue Jays',
  WSH: 'Washington Nationals',
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

function sortTodayGames(games: LiveSeasonGame[]): LiveSeasonGame[] {
  return [...games].sort((a, b) => a.game_pk - b.game_pk);
}

function matchupFullNames(matchup: string): string {
  const [awayTeam, homeTeam] = matchup.split(' @ ');
  const awayName = TEAM_NAMES[awayTeam] ?? awayTeam;
  const homeName = TEAM_NAMES[homeTeam] ?? homeTeam;
  return `${awayName} at ${homeName}`;
}

function TodayPickCard({
  game,
  form,
  onChange,
  onSubmit,
}: {
  game: LiveSeasonGame;
  form: ManualFormState;
  onChange: (field: 'odds' | 'units', value: string) => void;
  onSubmit: () => void;
}) {
  const gameStarted = (game.actual_status ?? 'scheduled').toLowerCase() !== 'scheduled';

  return (
    <GlassCard className="border-accent/15">
      <div className="mb-4 flex items-start justify-between gap-4">
        <div className="space-y-1">
          <p className="text-[11px] font-bold uppercase tracking-widest text-ink-dim">
            {game.pipeline_date}
          </p>
          <h3 className="font-heading text-2xl font-extrabold tracking-tight text-ink">
            {game.matchup}
          </h3>
          <p className="text-[11px] text-ink-dim">{matchupFullNames(game.matchup)}</p>
          <p className="text-sm font-medium text-ink-dim">{game.book_name ?? 'Unknown book'}</p>
        </div>
        <div className="rounded-xl bg-accent/10 px-3 py-2 text-right">
          <div className="text-[11px] font-bold uppercase tracking-widest text-accent">Pick</div>
          <div className="font-heading text-lg font-extrabold text-ink">{fmtPick(game)}</div>
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-4">
        <MetricCard label="Confidence" value={fmtPct(game.model_probability)} />
        <MetricCard label="Odds" value={fmtOdds(game.odds_at_bet)} />
        <MetricCard label="Units" value={fmtUnits(game.bet_units)} />
        <MetricCard label="Captured" value={new Date(game.captured_at).toLocaleTimeString()} />
      </div>

      <div className="mt-4 grid gap-3 rounded-xl bg-well/70 px-4 py-3 md:grid-cols-[1fr_1fr_auto]">
        <label className="grid gap-1">
          <span className="text-[11px] font-bold uppercase tracking-widest text-ink-dim">
            My Odds
          </span>
          <input
            type="number"
            value={form.odds}
            onChange={(event) => onChange('odds', event.target.value)}
            className="rounded-lg border border-stroke/30 bg-panel px-3 py-2 text-sm text-ink"
            disabled={gameStarted || form.submitting}
          />
        </label>
        <label className="grid gap-1">
          <span className="text-[11px] font-bold uppercase tracking-widest text-ink-dim">
            My Units
          </span>
          <input
            type="number"
            step="0.5"
            min="0.5"
            max="5"
            value={form.units}
            onChange={(event) => onChange('units', event.target.value)}
            className="rounded-lg border border-stroke/30 bg-panel px-3 py-2 text-sm text-ink"
            disabled={gameStarted || form.submitting}
          />
        </label>
        <button
          type="button"
          onClick={onSubmit}
          disabled={gameStarted || form.submitting}
          className="self-end rounded-xl border border-accent/30 bg-accent px-4 py-2.5 text-[10px] font-bold uppercase tracking-widest text-slab transition-colors hover:bg-accent-strong disabled:cursor-not-allowed disabled:opacity-60"
        >
          {form.submitting ? 'Submitting…' : 'Track This Bet'}
        </button>
      </div>

      {gameStarted ? (
        <p className="mt-2 text-xs text-ink-dim">Manual tracking locks once the game has started.</p>
      ) : null}
      {form.message ? <p className="mt-2 text-xs font-medium text-positive">{form.message}</p> : null}
      {form.error ? <p className="mt-2 text-xs font-medium text-negative">{form.error}</p> : null}
    </GlassCard>
  );
}

export default function SeasonPicksPage() {
  const { data, loading, capturing, error, refetch, captureToday } = useLiveSeasonDashboard();
  const [forms, setForms] = useState<Record<number, ManualFormState>>({});

  if (loading) return <LoadingState rows={4} />;
  if (error && !data) return <ErrorState message={error} onRetry={refetch} />;
  if (!data) return <EmptyState message="No live season data available." />;

  const todayGames = sortTodayGames(data.today_games).filter((game) => game.selected_market_type);

  const ensureForm = (game: LiveSeasonGame): ManualFormState =>
    forms[game.game_pk] ?? {
      odds: String(game.odds_at_bet ?? ''),
      units: String(game.bet_units ?? 1),
      submitting: false,
      message: null,
      error: null,
    };

  const updateForm = (game: LiveSeasonGame, field: 'odds' | 'units', value: string) => {
    setForms((current) => ({
      ...current,
      [game.game_pk]: {
        ...ensureForm(game),
        [field]: value,
        submitting: false,
        message: null,
        error: null,
      },
    }));
  };

  const submitGame = async (game: LiveSeasonGame) => {
    const form = ensureForm(game);
    setForms((current) => ({
      ...current,
      [game.game_pk]: { ...form, submitting: true, message: null, error: null },
    }));
    try {
      const response = await fetch('/api/live-season/manual-bets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          season: game.season,
          pipeline_date: game.pipeline_date,
          game_pk: game.game_pk,
          matchup: game.matchup,
          market_type: game.selected_market_type,
          side: game.selected_side,
          odds_at_bet: Number(form.odds),
          line_at_bet: game.line_at_bet,
          fair_probability: game.fair_probability,
          model_probability: game.model_probability,
          edge_pct: game.edge_pct,
          ev: game.ev,
          kelly_stake: game.kelly_stake,
          bet_units: Number(form.units),
          book_name: game.book_name,
          model_version: game.model_version,
          source_model: game.source_model,
          source_model_version: game.source_model_version,
          narrative: game.narrative,
        }),
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `HTTP ${response.status}`);
      }
      setForms((current) => ({
        ...current,
        [game.game_pk]: {
          ...ensureForm(game),
          odds: form.odds,
          units: form.units,
          submitting: false,
          message: 'Saved to My Bets.',
          error: null,
        },
      }));
      void refetch();
    } catch (submitError) {
      setForms((current) => ({
        ...current,
        [game.game_pk]: {
          ...ensureForm(game),
          odds: form.odds,
          units: form.units,
          submitting: false,
          message: null,
          error: submitError instanceof Error ? submitError.message : 'Failed to save manual bet',
        },
      }));
    }
  };

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <header className="flex flex-col gap-4 rounded-2xl border border-accent/15 bg-panel/40 p-6 md:flex-row md:items-end md:justify-between">
        <div className="space-y-2">
          <p className="text-[11px] font-bold uppercase tracking-widest text-accent">Live Season</p>
          <h1 className="font-heading text-3xl font-extrabold tracking-tight text-ink">
            Today&apos;s Picks
          </h1>
          <p className="max-w-2xl text-sm leading-relaxed text-ink-dim">
            Pull today&apos;s slate once, freeze the official picks, and keep yesterday&apos;s results
            settled without letting old picks change.
          </p>
          <p className="text-xs font-medium text-ink-dim">
            {data.release_name} · {data.model_display_name} · {data.strategy_name} {data.strategy_version}
          </p>
        </div>
        <div className="flex flex-col items-start gap-3 md:items-end">
          <button
            type="button"
            onClick={() => void captureToday()}
            disabled={capturing}
            className="rounded-xl border border-accent/30 bg-accent px-5 py-3 text-sm font-bold uppercase tracking-widest text-slab transition-colors hover:bg-accent-strong disabled:cursor-wait disabled:opacity-70"
          >
            {capturing ? 'Pulling Slate…' : 'Pull Today’s Slate'}
          </button>
          {data.capture && (
            <p className="text-xs font-medium text-ink-dim">
              {data.capture.already_captured
                ? 'Today is already frozen; returning saved picks.'
                : 'Today was captured and frozen.'}{' '}
              Settled rows: {data.capture.settled_rows}
            </p>
          )}
          {error && <p className="text-xs font-medium text-negative">{error}</p>}
        </div>
      </header>

      <section className="grid gap-4 md:grid-cols-4">
        <MetricCard label="Today Picks" value={todayGames.length} precision={0} />
        <MetricCard label="Release" value={`v${data.release_version}`} />
        <MetricCard label="Tracked Picks" value={data.summary.picks} precision={0} />
        <MetricCard
          label="Official Profit"
          value={data.summary.official_profit_units}
          unit="u"
          precision={2}
        />
        <MetricCard
          label="Official ROI"
          value={
            data.summary.official_roi == null ? '—' : `${(data.summary.official_roi * 100).toFixed(2)}%`
          }
        />
      </section>

      <section className="space-y-3">
        <h2 className="font-heading text-xl font-extrabold tracking-tight text-ink">Today</h2>
        {todayGames.length === 0 ? (
          <EmptyState message="No Bet365-qualified frozen picks for today. Pull today’s slate again if market availability changes." />
        ) : (
          <div className="grid gap-4">
            {todayGames.map((game) => (
              <TodayPickCard
                key={game.game_pk}
                game={game}
                form={ensureForm(game)}
                onChange={(field, value) => updateForm(game, field, value)}
                onSubmit={() => void submitGame(game)}
              />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
