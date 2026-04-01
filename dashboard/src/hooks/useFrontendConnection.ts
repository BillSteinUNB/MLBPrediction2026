import { useFetch } from './useFetch';
import type { FrontendConnection } from '../types';

/**
 * Hook for fetching frontend connection status and metadata.
 *
 * Loads information about the frontend's connection to the data pipeline and API availability.
 *
 * @returns UseFetchResult<FrontendConnection> with connection status data.
 *
 * @example
 * ```tsx
 * const { data: connection, loading, error } = useFrontendConnection();
 * if (loading) return <p>Checking connection...</p>;
 * if (error) return <p>Error: {error}</p>;
 * return <div>API Status: {connection?.api_available ? 'OK' : 'DOWN'}</div>;
 * ```
 */
export function useFrontendConnection() {
  return useFetch<FrontendConnection>('/api/tracker/frontend_connection.json');
}
