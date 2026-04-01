import { useFetch } from './useFetch';
import type { DualView } from '../types';

/**
 * Hook for fetching the dual-view comparison report.
 *
 * Loads side-by-side control vs. research lane comparisons including calibration metrics and performance deltas.
 *
 * @returns UseFetchResult<DualView> with control vs. research lane comparison data.
 *
 * @example
 * ```tsx
 * const { data: dualView, loading, error } = useDualView();
 * if (loading) return <p>Loading dual view...</p>;
 * if (error) return <p>Error: {error}</p>;
 * return <div>Control CRPS: {dualView?.control_lane.mean_crps}</div>;
 * ```
 */
export function useDualView() {
  return useFetch<DualView>('/api/dual-view/current_dual_view.json');
}
