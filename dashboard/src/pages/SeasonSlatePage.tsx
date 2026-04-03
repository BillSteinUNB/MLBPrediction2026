import { useMemo, useState } from 'react';
import { EmptyState, ErrorState, GlassCard, LoadingState, MetricCard } from '../components';
import { useSeasonSlate } from '../hooks';
import type { SeasonSlateDecision, SeasonSlateGame } from '../types/seasonSlate';

type MarketType = 'full_game_ml' | 'full_game_rl' | 'full_game_total';

interface SlateOpinion {
  marketType: MarketType;
  side: string;
  description: string;
  edgePct: number | null;
  oddsAtBet: number | null;
  confidence: number | null;
  stake: number | null;
  note: string | null;
  lineAtBet: number | null;
  fairProbability: number | null;
  modelProbability: number | null;
  ev: number | null;
  bookName: string | null;
}

interface MarketChoice {
  key: string;
  marketType: MarketType;
  side: string;
  description: string;
  oddsAtBet: number | null;
  lineAtBet: number | null;
  fairProbability: number | null;
  modelProbability: number | null;
  edgePct: number | null;
  ev: number | null;
  stake: number | null;
  bookName: string | null;
}

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
  if (value == null) return '-';
  return `${(value * 100).toFixed(2)}%`;
}

function fmtOdds(value: number | null | undefined): string {
  if (value == null) return '-';
  return value > 0 ? `+${value}` : String(value);
}

function fmtLine(value: number | null | undefined): string {
  if (value == null) return '';
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

function fmtUnits(value: number | null | undefined): string {
  if (value == null) return '-';
  return `${value.toFixed(2)}u`;
}

function teamForSide(matchup: string, side: string): string {
  const [awayTeam, homeTeam] = matchup.split(' @ ');
  return side === 'home' ? homeTeam : awayTeam;
}

function matchupFullNames(matchup: string): string {
  const [awayTeam, homeTeam] = matchup.split(' @ ');
  const awayName = TEAM_NAMES[awayTeam] ?? awayTeam;
  const homeName = TEAM_NAMES[homeTeam] ?? homeTeam;
  return `${awayName} at ${homeName}`;
}

function describeDecision(matchup: string, decision: SeasonSlateDecision): string {
  if (decision.market_type.endsWith('_ml')) {
    return `${teamForSide(matchup, decision.side)} ML`;
  }
  if (decision.market_type.endsWith('_rl')) {
    const team = teamForSide(matchup, decision.side);
    const sign = decision.line_at_bet != null && decision.line_at_bet > 0 ? '+' : '';
    return `${team} ${sign}${fmtLine(decision.line_at_bet)}`;
  }
  if (decision.market_type.endsWith('_total')) {
    const label = decision.side === 'over' ? 'Over' : 'Under';
    return `${label} ${fmtLine(decision.line_at_bet)}`;
  }
  return decision.side;
}

function impliedProbability(odds: number): number {
  return odds > 0 ? 100 / (odds + 100) : Math.abs(odds) / (Math.abs(odds) + 100);
}

function devigPair(firstOdds: number, secondOdds: number): [number, number] {
  const first = impliedProbability(firstOdds);
  const second = impliedProbability(secondOdds);
  const total = first + second;
  return total > 0 ? [first / total, second / total] : [0.5, 0.5];
}

function roundStake(edgePct: number): number {
  if (!Number.isFinite(edgePct) || edgePct <= 0) return 0.5;
  const scaled = Math.max(0.5, Math.min(5.0, edgePct * 20));
  return Math.round(scaled * 2) / 2;
}

function marketChoiceKey(marketType: MarketType, side: string, lineAtBet: number | null): string {
  return [
    marketType,
    side,
    lineAtBet == null ? 'none' : lineAtBet.toFixed(3),
  ].join(':');
}

function decisionToChoice(matchup: string, decision: SeasonSlateDecision): MarketChoice {
  const stake =
    decision.kelly_stake != null && decision.kelly_stake > 0
      ? decision.kelly_stake
      : roundStake(decision.edge_pct ?? 0);
  return {
    key: marketChoiceKey(
      decision.market_type as MarketType,
      decision.side,
      decision.line_at_bet,
    ),
    marketType: decision.market_type as MarketType,
    side: decision.side,
    description: describeDecision(matchup, decision),
    oddsAtBet: decision.odds_at_bet,
    lineAtBet: decision.line_at_bet,
    fairProbability: decision.fair_probability,
    modelProbability: decision.model_probability,
    edgePct: decision.edge_pct,
    ev: decision.ev,
    stake,
    bookName: decision.book_name,
  };
}

function bestChoice(choices: MarketChoice[]): MarketChoice | null {
  if (choices.length === 0) return null;
  return [...choices].sort((a, b) => {
    const edgeDiff = (b.edgePct ?? -1) - (a.edgePct ?? -1);
    if (edgeDiff !== 0) return edgeDiff;
    return (b.modelProbability ?? -1) - (a.modelProbability ?? -1);
  })[0];
}

function choiceToOpinion(choice: MarketChoice, note: string | null = null): SlateOpinion {
  return {
    marketType: choice.marketType,
    side: choice.side,
    description: choice.description,
    edgePct: choice.edgePct,
    oddsAtBet: choice.oddsAtBet,
    confidence: choice.modelProbability,
    stake: choice.stake,
    note,
    lineAtBet: choice.lineAtBet,
    fairProbability: choice.fairProbability,
    modelProbability: choice.modelProbability,
    ev: choice.ev,
    bookName: choice.bookName,
  };
}

function explicitBet365Choices(game: SeasonSlateGame, marketType: MarketType): MarketChoice[] {
  return game.candidate_decisions
    .filter(
      (decision) =>
        decision.market_type === marketType &&
        decision.book_name != null &&
        ['bet365', 'Bet365'].includes(decision.book_name),
    )
    .map((decision) => decisionToChoice(game.matchup, decision));
}

function buildMoneylineChoices(game: SeasonSlateGame): MarketChoice[] {
  const explicit = explicitBet365Choices(game, 'full_game_ml');
  if (explicit.length > 0) return explicit;

  const prediction = game.prediction;
  const input = game.input_status;
  if (!input || input.bet365_full_game_home_ml == null || input.bet365_full_game_away_ml == null) {
    return [];
  }

  const [homeFair, awayFair] = devigPair(
    input.bet365_full_game_home_ml,
    input.bet365_full_game_away_ml,
  );
  return [
    {
      key: marketChoiceKey('full_game_ml', 'away', null),
      marketType: 'full_game_ml',
      side: 'away',
      description: `${teamForSide(game.matchup, 'away')} ML`,
      oddsAtBet: input.bet365_full_game_away_ml,
      lineAtBet: null,
      fairProbability: awayFair,
      modelProbability: prediction?.full_game_ml_away_prob ?? null,
      edgePct:
        prediction?.full_game_ml_away_prob != null
          ? prediction.full_game_ml_away_prob - awayFair
          : null,
      ev: null,
      stake:
        prediction?.full_game_ml_away_prob != null
          ? roundStake(prediction.full_game_ml_away_prob - awayFair)
          : null,
      bookName: 'Bet365',
    },
    {
      key: marketChoiceKey('full_game_ml', 'home', null),
      marketType: 'full_game_ml',
      side: 'home',
      description: `${teamForSide(game.matchup, 'home')} ML`,
      oddsAtBet: input.bet365_full_game_home_ml,
      lineAtBet: null,
      fairProbability: homeFair,
      modelProbability: prediction?.full_game_ml_home_prob ?? null,
      edgePct:
        prediction?.full_game_ml_home_prob != null
          ? prediction.full_game_ml_home_prob - homeFair
          : null,
      ev: null,
      stake:
        prediction?.full_game_ml_home_prob != null
          ? roundStake(prediction.full_game_ml_home_prob - homeFair)
          : null,
      bookName: 'Bet365',
    },
  ];
}

function buildRunLineChoices(game: SeasonSlateGame): MarketChoice[] {
  const explicit = explicitBet365Choices(game, 'full_game_rl');
  if (explicit.length > 0) return explicit;

  const prediction = game.prediction;
  const input = game.input_status;
  if (
    !input ||
    input.bet365_full_game_home_spread == null ||
    input.bet365_full_game_home_spread_odds == null ||
    input.bet365_full_game_away_spread == null ||
    input.bet365_full_game_away_spread_odds == null
  ) {
    return [];
  }

  const [homeFair, awayFair] = devigPair(
    input.bet365_full_game_home_spread_odds,
    input.bet365_full_game_away_spread_odds,
  );
  return [
    {
      key: marketChoiceKey('full_game_rl', 'away', input.bet365_full_game_away_spread),
      marketType: 'full_game_rl',
      side: 'away',
      description: `${teamForSide(game.matchup, 'away')} ${
        input.bet365_full_game_away_spread > 0 ? '+' : ''
      }${fmtLine(input.bet365_full_game_away_spread)}`,
      oddsAtBet: input.bet365_full_game_away_spread_odds,
      lineAtBet: input.bet365_full_game_away_spread,
      fairProbability: awayFair,
      modelProbability: prediction?.full_game_rl_away_prob ?? null,
      edgePct:
        prediction?.full_game_rl_away_prob != null
          ? prediction.full_game_rl_away_prob - awayFair
          : null,
      ev: null,
      stake:
        prediction?.full_game_rl_away_prob != null
          ? roundStake(prediction.full_game_rl_away_prob - awayFair)
          : null,
      bookName: 'Bet365',
    },
    {
      key: marketChoiceKey('full_game_rl', 'home', input.bet365_full_game_home_spread),
      marketType: 'full_game_rl',
      side: 'home',
      description: `${teamForSide(game.matchup, 'home')} ${
        input.bet365_full_game_home_spread > 0 ? '+' : ''
      }${fmtLine(input.bet365_full_game_home_spread)}`,
      oddsAtBet: input.bet365_full_game_home_spread_odds,
      lineAtBet: input.bet365_full_game_home_spread,
      fairProbability: homeFair,
      modelProbability: prediction?.full_game_rl_home_prob ?? null,
      edgePct:
        prediction?.full_game_rl_home_prob != null
          ? prediction.full_game_rl_home_prob - homeFair
          : null,
      ev: null,
      stake:
        prediction?.full_game_rl_home_prob != null
          ? roundStake(prediction.full_game_rl_home_prob - homeFair)
          : null,
      bookName: 'Bet365',
    },
  ];
}

function buildTotalChoices(game: SeasonSlateGame): MarketChoice[] {
  const explicit = explicitBet365Choices(game, 'full_game_total');
  if (explicit.length > 0) return explicit;

  const prediction = game.prediction;
  const input = game.input_status;
  const line = input?.bet365_full_game_total;
  if (
    line == null ||
    input?.bet365_full_game_total_over_odds == null ||
    input?.bet365_full_game_total_under_odds == null
  ) {
    return [];
  }

  const [overFair, underFair] = devigPair(
    input.bet365_full_game_total_over_odds,
    input.bet365_full_game_total_under_odds,
  );
  return [
    {
      key: marketChoiceKey('full_game_total', 'over', line),
      marketType: 'full_game_total',
      side: 'over',
      description: `Over ${fmtLine(line)}`,
      oddsAtBet: input.bet365_full_game_total_over_odds,
      lineAtBet: line,
      fairProbability: overFair,
      modelProbability: prediction?.full_game_total_over_prob ?? null,
      edgePct:
        prediction?.full_game_total_over_prob != null
          ? prediction.full_game_total_over_prob - overFair
          : null,
      ev: null,
      stake:
        prediction?.full_game_total_over_prob != null
          ? roundStake(prediction.full_game_total_over_prob - overFair)
          : null,
      bookName: 'Bet365',
    },
    {
      key: marketChoiceKey('full_game_total', 'under', line),
      marketType: 'full_game_total',
      side: 'under',
      description: `Under ${fmtLine(line)}`,
      oddsAtBet: input.bet365_full_game_total_under_odds,
      lineAtBet: line,
      fairProbability: underFair,
      modelProbability: prediction?.full_game_total_under_prob ?? null,
      edgePct:
        prediction?.full_game_total_under_prob != null
          ? prediction.full_game_total_under_prob - underFair
          : null,
      ev: null,
      stake:
        prediction?.full_game_total_under_prob != null
          ? roundStake(prediction.full_game_total_under_prob - underFair)
          : null,
      bookName: 'Bet365',
    },
  ];
}

function buildMoneylineOpinion(game: SeasonSlateGame): SlateOpinion | null {
  const choice = bestChoice(buildMoneylineChoices(game));
  if (!choice) return null;
  return choiceToOpinion(
    choice,
    choice.edgePct == null ? 'Bet365 moneyline available for manual tracking without a model edge.' : null,
  );
}

function buildRunLineOpinion(game: SeasonSlateGame): SlateOpinion | null {
  const choice = bestChoice(buildRunLineChoices(game));
  if (!choice) return null;
  return choiceToOpinion(
    choice,
    choice.edgePct == null ? 'Bet365 run line available for manual tracking without a model edge.' : null,
  );
}

function buildTotalOpinion(game: SeasonSlateGame): SlateOpinion | null {
  const prediction = game.prediction;
  const choices = buildTotalChoices(game);
  const choice = bestChoice(choices);
  if (choice) {
    return choiceToOpinion(
      choice,
      prediction?.projected_full_game_total_runs != null
        ? `Projection ${prediction.projected_full_game_total_runs.toFixed(2)} runs.`
        : choice.edgePct == null
          ? 'Bet365 total available for manual tracking without a model edge.'
          : null,
    );
  }
  if (prediction?.projected_full_game_total_runs == null) {
    return null;
  }
  return {
    marketType: 'full_game_total',
    side: 'over',
    description: `Projection ${prediction.projected_full_game_total_runs.toFixed(2)} runs`,
    edgePct: null,
    oddsAtBet: null,
    confidence: null,
    stake: null,
    note: 'No Bet365 full-game total line is available for this slate.',
    lineAtBet: null,
    fairProbability: null,
    modelProbability: null,
    ev: null,
    bookName: null,
  };
}

function bestSlateOpinion(opinions: Array<SlateOpinion | null>): SlateOpinion | null {
  const resolved = opinions.filter((opinion): opinion is SlateOpinion => opinion !== null);
  if (resolved.length === 0) return null;
  return [...resolved].sort((a, b) => {
    const edgeDiff = (b.edgePct ?? -1) - (a.edgePct ?? -1);
    if (edgeDiff !== 0) return edgeDiff;
    return (b.confidence ?? -1) - (a.confidence ?? -1);
  })[0];
}

function buildInitialForm(choice: MarketChoice | null): ManualFormState | null {
  if (!choice || choice.oddsAtBet == null) return null;
  return {
    odds: String(choice.oddsAtBet),
    units: String(choice.stake ?? 1),
    submitting: false,
    message: null,
    error: null,
  };
}

function OpinionRow({
  game,
  label,
  opinion,
  choices,
  selectedChoiceKey,
  form,
  onSelectChoice,
  onChange,
  onSubmit,
}: {
  game: SeasonSlateGame;
  label: string;
  opinion: SlateOpinion | null;
  choices: MarketChoice[];
  selectedChoiceKey: string | null;
  form: ManualFormState | null;
  onSelectChoice: (choiceKey: string) => void;
  onChange: (field: 'odds' | 'units', value: string) => void;
  onSubmit: () => void;
}) {
  const selectedChoice =
    choices.find((choice) => choice.key === selectedChoiceKey) ?? choices[0] ?? null;
  const canSubmit = selectedChoice?.oddsAtBet != null && form != null;
  const gameStarted = (game.game_status ?? game.status).toLowerCase() !== 'scheduled';

  if (!opinion && choices.length === 0) {
    return (
      <div className="grid grid-cols-[120px_1fr] gap-3 rounded-xl bg-well/50 px-4 py-3">
        <div className="text-[11px] font-bold uppercase tracking-widest text-ink-dim">{label}</div>
        <div className="text-sm font-medium text-ink-dim">Bet365 unavailable</div>
      </div>
    );
  }

  const displayDescription = selectedChoice?.description ?? opinion?.description ?? 'Bet365 available';
  const displayEdge = selectedChoice?.edgePct ?? opinion?.edgePct ?? null;
  const displayOdds = selectedChoice?.oddsAtBet ?? opinion?.oddsAtBet ?? null;
  const displayConfidence =
    selectedChoice?.modelProbability ?? opinion?.confidence ?? opinion?.modelProbability ?? null;
  const displayStake = selectedChoice?.stake ?? opinion?.stake ?? null;
  const displayBook = selectedChoice?.bookName ?? opinion?.bookName ?? null;
  const displayNote =
    opinion?.note ??
    (selectedChoice && selectedChoice.edgePct == null
      ? 'Bet365 line available for manual tracking even without a model edge.'
      : null);

  return (
    <div className="grid gap-2 rounded-xl bg-well/70 px-3 py-2.5">
      <div className="grid grid-cols-[96px_1fr] gap-2">
        <div className="text-[11px] font-bold uppercase tracking-widest text-ink-dim">{label}</div>
        <div className="grid gap-1.5 xl:grid-cols-5">
          <div>
            <div className="text-sm font-medium text-ink">{displayDescription}</div>
            {displayNote ? <div className="text-xs text-ink-dim">{displayNote}</div> : null}
          </div>
          <div className="text-xs text-ink-dim">Edge {fmtPct(displayEdge)}</div>
          <div className="text-xs text-ink-dim">Odds {fmtOdds(displayOdds)}</div>
          <div className="text-xs text-ink-dim">Conf {fmtPct(displayConfidence)}</div>
          <div className="text-xs text-ink-dim">
            Stake {fmtUnits(displayStake)}
            <div>{displayBook ?? 'Bet365 unavailable'}</div>
          </div>
        </div>
      </div>

      {choices.length > 1 ? (
        <div className="flex flex-wrap gap-2">
          {choices.map((choice) => {
            const active = choice.key === selectedChoiceKey;
            return (
              <button
                key={choice.key}
                type="button"
                onClick={() => onSelectChoice(choice.key)}
                disabled={gameStarted}
                className={`rounded-full border px-3 py-1.5 text-[10px] font-bold uppercase tracking-widest transition-colors ${
                  active
                    ? 'border-accent bg-accent text-slab'
                    : 'border-stroke/30 bg-panel text-ink hover:border-accent/40'
                } disabled:cursor-not-allowed disabled:opacity-60`}
              >
                {choice.description}
              </button>
            );
          })}
        </div>
      ) : null}

      <div className="grid gap-2 xl:grid-cols-[1fr_1fr_auto] xl:items-end">
        <label className="grid gap-1 text-[10px] font-semibold uppercase tracking-widest text-ink-dim">
          My Odds
          <input
            type="number"
            value={form?.odds ?? ''}
            onChange={(event) => onChange('odds', event.target.value)}
            className="rounded-lg border border-stroke/30 bg-panel px-2.5 py-2 text-sm text-ink"
            disabled={!canSubmit || gameStarted}
          />
        </label>
        <label className="grid gap-1 text-[10px] font-semibold uppercase tracking-widest text-ink-dim">
          My Units
          <input
            type="number"
            min="0.5"
            max="5"
            step="0.5"
            value={form?.units ?? ''}
            onChange={(event) => onChange('units', event.target.value)}
            className="rounded-lg border border-stroke/30 bg-panel px-2.5 py-2 text-sm text-ink"
            disabled={!canSubmit || gameStarted}
          />
        </label>
        <button
          type="button"
          onClick={onSubmit}
          disabled={!canSubmit || !!form?.submitting || gameStarted}
          className="rounded-xl border border-accent/30 bg-accent px-3.5 py-2.5 text-[10px] font-bold uppercase tracking-widest text-slab transition-colors hover:bg-accent-strong disabled:cursor-not-allowed disabled:opacity-60"
        >
          {form?.submitting ? 'Submitting...' : 'Track This Bet'}
        </button>
      </div>

      {gameStarted ? (
        <div className="text-xs text-ink-dim">Manual tracking locks once the game has started.</div>
      ) : null}
      {form?.message ? <div className="text-xs font-medium text-positive">{form.message}</div> : null}
      {form?.error ? <div className="text-xs font-medium text-negative">{form.error}</div> : null}
      {!canSubmit ? (
        <div className="text-xs text-ink-dim">
          This opinion cannot be submitted until a real Bet365 market line is available.
        </div>
      ) : null}
    </div>
  );
}

function SlateGameCard({
  game,
  pipelineDate,
}: {
  game: SeasonSlateGame;
  pipelineDate: string;
}) {
  const mlChoices = useMemo(() => buildMoneylineChoices(game), [game]);
  const rlChoices = useMemo(() => buildRunLineChoices(game), [game]);
  const totalChoices = useMemo(() => buildTotalChoices(game), [game]);

  const ml = useMemo(() => buildMoneylineOpinion(game), [game]);
  const rl = useMemo(() => buildRunLineOpinion(game), [game]);
  const total = useMemo(() => buildTotalOpinion(game), [game]);
  const bestOpinion = bestSlateOpinion([ml, rl, total]);

  const season = useMemo(() => Number.parseInt(pipelineDate.slice(0, 4), 10), [pipelineDate]);

  const [selectedChoices, setSelectedChoices] = useState<Record<MarketType, string | null>>({
    full_game_ml: bestChoice(mlChoices)?.key ?? mlChoices[0]?.key ?? null,
    full_game_rl: bestChoice(rlChoices)?.key ?? rlChoices[0]?.key ?? null,
    full_game_total: bestChoice(totalChoices)?.key ?? totalChoices[0]?.key ?? null,
  });

  const [forms, setForms] = useState<Record<MarketType, ManualFormState | null>>({
    full_game_ml: buildInitialForm(bestChoice(mlChoices) ?? mlChoices[0] ?? null),
    full_game_rl: buildInitialForm(bestChoice(rlChoices) ?? rlChoices[0] ?? null),
    full_game_total: buildInitialForm(bestChoice(totalChoices) ?? totalChoices[0] ?? null),
  });

  const selectedChoiceFor = (marketType: MarketType, choices: MarketChoice[]) =>
    choices.find((choice) => choice.key === selectedChoices[marketType]) ?? choices[0] ?? null;

  const updateSelectedChoice = (marketType: MarketType, choices: MarketChoice[], choiceKey: string) => {
    const selected = choices.find((choice) => choice.key === choiceKey) ?? null;
    setSelectedChoices((current) => ({ ...current, [marketType]: choiceKey }));
    setForms((current) => ({
      ...current,
      [marketType]: selected
        ? {
            odds: String(selected.oddsAtBet ?? ''),
            units: String(selected.stake ?? 1),
            submitting: false,
            message: null,
            error: null,
          }
        : current[marketType],
    }));
  };

  const updateForm = (marketType: MarketType, field: 'odds' | 'units', value: string) => {
    setForms((current) => ({
      ...current,
      [marketType]: {
        ...(current[marketType] ?? {
          odds: '',
          units: '1',
          submitting: false,
          message: null,
          error: null,
        }),
        [field]: value,
        message: null,
        error: null,
      },
    }));
  };

  const submitChoice = async (marketType: MarketType, choice: MarketChoice | null) => {
    const form = forms[marketType];
    if (!choice || !form) return;
    setForms((current) => ({
      ...current,
      [marketType]: { ...form, submitting: true, message: null, error: null },
    }));

    try {
      const response = await fetch('/api/live-season/manual-bets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          season,
          pipeline_date: pipelineDate,
          game_pk: game.game_pk,
          matchup: game.matchup,
          market_type: marketType,
          side: choice.side,
          odds_at_bet: Number(form.odds),
          line_at_bet: choice.lineAtBet,
          fair_probability: choice.fairProbability,
          model_probability: choice.modelProbability,
          edge_pct: choice.edgePct,
          ev: choice.ev,
          kelly_stake: choice.stake,
          bet_units: Number(form.units),
          book_name: choice.bookName,
          model_version: game.prediction?.model_version ?? null,
          source_model: 'slate_manual_submit',
          source_model_version: game.prediction?.model_version ?? null,
          input_status: game.input_status ?? null,
          narrative: game.narrative,
        }),
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `HTTP ${response.status}`);
      }

      setForms((current) => ({
        ...current,
        [marketType]: current[marketType]
          ? {
              ...current[marketType],
              submitting: false,
              message: 'Saved to My Bets.',
              error: null,
            }
          : current[marketType],
      }));
    } catch (error) {
      setForms((current) => ({
        ...current,
        [marketType]: current[marketType]
          ? {
              ...current[marketType],
              submitting: false,
              message: null,
              error: error instanceof Error ? error.message : 'Failed to save manual bet',
            }
          : current[marketType],
      }));
    }
  };

  return (
    <GlassCard className="border-stroke/20">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div>
          <h3 className="font-heading text-xl font-extrabold tracking-tight text-ink">
            {game.matchup}
          </h3>
          <p className="text-[11px] text-ink-dim">{matchupFullNames(game.matchup)}</p>
          <p className="text-xs font-medium text-ink-dim">
            {game.selected_decision
              ? `Official pick: ${describeDecision(game.matchup, game.selected_decision)}`
              : game.no_pick_reason || 'No official pick'}
          </p>
        </div>
        <div className="rounded-xl bg-panel/70 px-2.5 py-2 text-right">
          <div className="text-[11px] font-bold uppercase tracking-widest text-ink-dim">Status</div>
          <div className="font-heading text-base font-extrabold text-ink">
            {game.game_status ?? game.status}
          </div>
        </div>
      </div>

      <div className="mb-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label="Projected Total"
          value={game.prediction?.projected_full_game_total_runs ?? null}
          precision={2}
        />
        <MetricCard label="Best Slate Edge" value={bestOpinion ? fmtPct(bestOpinion.edgePct) : '-'} />
        <MetricCard label="Best Slate Odds" value={bestOpinion ? fmtOdds(bestOpinion.oddsAtBet) : '-'} />
        <MetricCard label="Best Slate Stake" value={bestOpinion ? fmtUnits(bestOpinion.stake) : '-'} />
      </div>

      <div className="space-y-3">
        <OpinionRow
          game={game}
          label="Moneyline"
          opinion={ml}
          choices={mlChoices}
          selectedChoiceKey={selectedChoices.full_game_ml}
          form={forms.full_game_ml}
          onSelectChoice={(choiceKey) => updateSelectedChoice('full_game_ml', mlChoices, choiceKey)}
          onChange={(field, value) => updateForm('full_game_ml', field, value)}
          onSubmit={() => void submitChoice('full_game_ml', selectedChoiceFor('full_game_ml', mlChoices))}
        />
        <OpinionRow
          game={game}
          label="Run Line"
          opinion={rl}
          choices={rlChoices}
          selectedChoiceKey={selectedChoices.full_game_rl}
          form={forms.full_game_rl}
          onSelectChoice={(choiceKey) => updateSelectedChoice('full_game_rl', rlChoices, choiceKey)}
          onChange={(field, value) => updateForm('full_game_rl', field, value)}
          onSubmit={() => void submitChoice('full_game_rl', selectedChoiceFor('full_game_rl', rlChoices))}
        />
        <OpinionRow
          game={game}
          label="Total"
          opinion={total}
          choices={totalChoices}
          selectedChoiceKey={selectedChoices.full_game_total}
          form={forms.full_game_total}
          onSelectChoice={(choiceKey) =>
            updateSelectedChoice('full_game_total', totalChoices, choiceKey)
          }
          onChange={(field, value) => updateForm('full_game_total', field, value)}
          onSubmit={() =>
            void submitChoice('full_game_total', selectedChoiceFor('full_game_total', totalChoices))
          }
        />
      </div>
    </GlassCard>
  );
}

export default function SeasonSlatePage() {
  const { data, loading, error, refetch } = useSeasonSlate();

  if (loading) return <LoadingState rows={6} />;
  if (error) return <ErrorState message={error} onRetry={refetch} />;
  if (!data) return <EmptyState message="No slate data available." />;

  const games = [...data.games].sort((a, b) => a.game_pk - b.game_pk);

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <header className="space-y-2">
        <p className="text-[11px] font-bold uppercase tracking-widest text-accent">Live Season</p>
        <h1 className="font-heading text-3xl font-extrabold tracking-tight text-ink">Slate</h1>
        <p className="max-w-3xl text-sm leading-relaxed text-ink-dim">
          Full-game view of every matchup. Review the Bet365 moneyline, run line, and total for
          each game, see the model's strongest view where it exists, and track either side into
          your own bets.
        </p>
        <p className="text-xs font-medium text-ink-dim">
          {data.release_name} · {data.model_display_name} · {data.strategy_name} {data.strategy_version}
        </p>
      </header>

      <section className="grid gap-4 md:grid-cols-4">
        <MetricCard label="Games" value={data.games.length} precision={0} />
        <MetricCard label="Official Picks" value={data.pick_count} precision={0} />
        <MetricCard label="No Picks" value={data.no_pick_count} precision={0} />
        <MetricCard label="Release" value={`v${data.release_version}`} />
      </section>

      {games.length === 0 ? (
        <EmptyState message="No games on the current slate." />
      ) : (
        <div className="grid gap-4 xl:grid-cols-2">
          {games.map((game) => (
            <SlateGameCard key={game.game_pk} game={game} pipelineDate={data.pipeline_date} />
          ))}
        </div>
      )}
    </div>
  );
}
