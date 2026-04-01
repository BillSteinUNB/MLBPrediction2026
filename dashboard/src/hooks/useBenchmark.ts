import { useFetch } from './useFetch';
import type { BenchmarkFile } from '../types';

/**
 * Hook for fetching benchmark comparison data.
 *
 * Loads performance metrics for the active benchmark model used for comparison.
 *
 * @returns UseFetchResult<BenchmarkFile> with benchmark performance data.
 *
 * @example
 * ```tsx
 * const { data: benchmark, loading, error } = useBenchmark();
 * if (loading) return <p>Loading benchmark...</p>;
 * if (error) return <p>Error: {error}</p>;
 * return <div>Benchmark ROI: {benchmark?.performance.roi}</div>;
 * ```
 */
export function useBenchmark() {
  return useFetch<BenchmarkFile>('/api/tracker/benchmark.json');
}
