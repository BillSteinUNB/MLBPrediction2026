import { useEffect, useState } from 'react';
import type { LiveSeasonDashboardData } from '../types/liveSeason';

interface LiveSeasonDashboardResult {
  data: LiveSeasonDashboardData | null;
  loading: boolean;
  capturing: boolean;
  error: string | null;
  refetch: () => void;
  captureToday: () => Promise<void>;
}

const DEFAULT_SEASON = 2026;

export function useLiveSeasonDashboard(
  pipelineDate?: string,
  season: number = DEFAULT_SEASON,
): LiveSeasonDashboardResult {
  const [data, setData] = useState<LiveSeasonDashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [capturing, setCapturing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reloadToken, setReloadToken] = useState(0);

  const query = new URLSearchParams();
  query.set('season', String(season));
  if (pipelineDate) {
    query.set('pipeline_date', pipelineDate);
  }
  const baseUrl = `/api/live-season/dashboard?${query.toString()}`;
  const captureUrl = `/api/live-season/capture-today?${query.toString()}`;

  useEffect(() => {
    let mounted = true;

    const load = async () => {
      try {
        setLoading(true);
        setError(null);

        const response = await fetch(baseUrl);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const payload = (await response.json()) as LiveSeasonDashboardData;
        if (mounted) {
          setData(payload);
        }
      } catch (err) {
        if (mounted) {
          setData(null);
          setError(err instanceof Error ? err.message : 'Failed to load live season data');
        }
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    };

    void load();
    return () => {
      mounted = false;
    };
  }, [baseUrl, reloadToken]);

  const refetch = () => {
    setReloadToken((value) => value + 1);
  };

  const captureToday = async () => {
    try {
      setCapturing(true);
      setError(null);
      const response = await fetch(captureUrl, { method: 'POST' });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const payload = (await response.json()) as LiveSeasonDashboardData;
      setData(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to capture today');
    } finally {
      setCapturing(false);
    }
  };

  return { data, loading, capturing, error, refetch, captureToday };
}
