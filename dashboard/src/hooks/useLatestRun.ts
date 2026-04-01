import { useFetch } from './useFetch';
import type { LatestRun } from '../types';

/**
 * Hook for fetching the latest run summary from the research tracker.
 *
 * Loads workflow metadata, stage summaries, and promotion recommendations.
 *
 * @returns UseFetchResult<LatestRun> with workflow summary data.
 *
 * @example
 * ```tsx
 * const { data: latestRun, loading, error } = useLatestRun();
 * if (loading) return <p>Loading latest run...</p>;
 * if (error) return <p>Error: {error}</p>;
 * return <div>{latestRun?.workflow_summary.headline_result}</div>;
 * ```
 */
export function useLatestRun() {
  return useFetch<LatestRun>('/api/tracker/latest_run.json');
}
