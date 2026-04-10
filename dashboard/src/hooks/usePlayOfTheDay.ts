import { useFetch } from './useFetch';
import type { PlayOfTheDayData } from '../types/pics';

/**
 * Hook for fetching the play-of-the-day (best pick) from the betting engine.
 *
 * Loads the single highest-confidence, highest-edge pick from the daily slate.
 *
 * @param filename - Name of the play-of-the-day file (e.g., "play_of_the_day.json")
 * @returns UseFetchResult<PlayOfTheDayData> with play-of-the-day pick data.
 *
 * @example
 * ```tsx
 * const { data: potd, loading, error } = usePlayOfTheDay('play_of_the_day.json');
 * if (loading) return <p>Loading play of the day...</p>;
 * if (error) return <p>Error: {error}</p>;
 * return <div>{potd?.home_team} vs {potd?.away_team} - {potd?.edge_pct}%</div>;
 * ```
 */
export function usePlayOfTheDay(filename: string = 'play_of_the_day.json') {
  return useFetch<PlayOfTheDayData>(`/api/pics/${filename}`);
}
