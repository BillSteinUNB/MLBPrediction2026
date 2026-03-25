import { useCallback, useEffect, useState } from "react";
import type { LiveSeasonGameResponse } from "../api";

const STORAGE_KEY = "mlb-tracked-bets";

export interface TrackedBet {
  gameKey: string;
  pipeline_date: string;
  game_pk: number;
  matchup: string;
  market_type: string;
  side: string;
  odds: number | null;
  edge_pct: number | null;
  is_play_of_day: boolean;
  trackedAt: string;
}

export interface BetTrackerSummary {
  totalTracked: number;
  wins: number;
  losses: number;
  pushes: number;
  pending: number;
  totalUnits: number;
  roi: number | null;
}

export interface SplitSummary {
  potd: BetTrackerSummary;
  value: BetTrackerSummary;
  all: BetTrackerSummary;
}

export interface UseBetTrackerReturn {
  trackedBets: Map<string, TrackedBet>;
  isTracked: (gameKey: string) => boolean;
  toggleBet: (bet: TrackedBet) => void;
  removeTracked: (gameKey: string) => void;
  clearAll: () => void;
  computeSplitSummary: (games: LiveSeasonGameResponse[]) => SplitSummary;
}

function emptySum(): BetTrackerSummary {
  return { totalTracked: 0, wins: 0, losses: 0, pushes: 0, pending: 0, totalUnits: 0, roi: null };
}

function accumulateGame(sum: BetTrackerSummary, game: LiveSeasonGameResponse): void {
  sum.totalTracked++;
  const result = game.settled_result;
  const profit = game.flat_profit_loss ?? 0;
  if (result === "WIN") { sum.wins++; sum.totalUnits += profit; }
  else if (result === "LOSS") { sum.losses++; sum.totalUnits += profit; }
  else if (result === "PUSH") { sum.pushes++; sum.totalUnits += profit; }
  else { sum.pending++; }
}

function finalizeRoi(sum: BetTrackerSummary): void {
  const graded = sum.wins + sum.losses + sum.pushes;
  sum.roi = graded > 0 ? sum.totalUnits / graded : null;
}

function loadFromStorage(): Map<string, TrackedBet> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return new Map();
    const arr: TrackedBet[] = JSON.parse(raw);
    const map = new Map<string, TrackedBet>();
    for (const bet of arr) {
      // Backwards-compat: old entries may lack is_play_of_day
      if (bet.is_play_of_day === undefined) {
        (bet as TrackedBet).is_play_of_day = false;
      }
      map.set(bet.gameKey, bet);
    }
    return map;
  } catch {
    return new Map();
  }
}

function saveToStorage(map: Map<string, TrackedBet>): void {
  try {
    const arr = Array.from(map.values());
    localStorage.setItem(STORAGE_KEY, JSON.stringify(arr));
  } catch {
    // silently fail if storage is full
  }
}

export function useBetTracker(): UseBetTrackerReturn {
  const [trackedBets, setTrackedBets] = useState<Map<string, TrackedBet>>(() => loadFromStorage());

  useEffect(() => {
    saveToStorage(trackedBets);
  }, [trackedBets]);

  const isTracked = useCallback(
    (gameKey: string): boolean => trackedBets.has(gameKey),
    [trackedBets],
  );

  const toggleBet = useCallback((bet: TrackedBet): void => {
    setTrackedBets((prev) => {
      const next = new Map(prev);
      if (next.has(bet.gameKey)) {
        next.delete(bet.gameKey);
      } else {
        next.set(bet.gameKey, bet);
      }
      return next;
    });
  }, []);

  const removeTracked = useCallback((gameKey: string): void => {
    setTrackedBets((prev) => {
      const next = new Map(prev);
      next.delete(gameKey);
      return next;
    });
  }, []);

  const clearAll = useCallback((): void => {
    setTrackedBets(new Map());
  }, []);

  const computeSplitSummary = useCallback(
    (games: LiveSeasonGameResponse[]): SplitSummary => {
      const potd = emptySum();
      const value = emptySum();
      const all = emptySum();

      for (const game of games) {
        const key = `${game.pipeline_date}-${game.game_pk}`;
        const tracked = trackedBets.get(key);
        if (!tracked) continue;

        accumulateGame(all, game);
        if (tracked.is_play_of_day) {
          accumulateGame(potd, game);
        } else {
          accumulateGame(value, game);
        }
      }

      finalizeRoi(potd);
      finalizeRoi(value);
      finalizeRoi(all);
      return { potd, value, all };
    },
    [trackedBets],
  );

  return {
    trackedBets,
    isTracked,
    toggleBet,
    removeTracked,
    clearAll,
    computeSplitSummary,
  };
}
