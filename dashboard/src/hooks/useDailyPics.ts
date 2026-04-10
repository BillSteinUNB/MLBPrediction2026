import { useFetch } from './useFetch';
import type { DailyPicsData } from '../types/pics';

/**
 * Hook for fetching daily picks (PICS) from the betting engine.
 *
 * Loads all game predictions, market data, and play-of-the-day selection.
 *
 * @param filename - Name of the daily picks file (e.g., "daily.json")
 * @returns UseFetchResult<DailyPicsData> with daily picks data.
 *
 * @example
 * ```tsx
 * const { data: dailyPics, loading, error } = useDailyPics('daily.json');
 * if (loading) return <p>Loading daily picks...</p>;
 * if (error) return <p>Error: {error}</p>;
 * return <div>{dailyPics?.picks_count} picks found</div>;
 * ```
 */
export function useDailyPics(filename: string = 'daily.json') {
  return useFetch<DailyPicsData>(`/api/pics/${filename}`);
}
