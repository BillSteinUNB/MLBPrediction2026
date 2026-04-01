import { useFetch } from './useFetch';
import type { RunHistoryIndex } from '../types';

/**
 * Hook for fetching the run history index.
 *
 * Loads metadata for all historical runs, enabling navigation and comparison across runs.
 *
 * @returns UseFetchResult<RunHistoryIndex> with historical run metadata.
 *
 * @example
 * ```tsx
 * const { data: history, loading, error } = useRunHistory();
 * if (loading) return <p>Loading run history...</p>;
 * if (error) return <p>Error: {error}</p>;
 * return <div>{history?.runs.length} runs available</div>;
 * ```
 */
export function useRunHistory() {
  return useFetch<RunHistoryIndex>('/api/tracker/run_history_index.json');
}
