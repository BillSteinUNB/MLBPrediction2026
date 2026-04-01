import { useEffect, useState } from 'react';

export interface UseFetchResult<T> {
  /** The fetched data, or null if loading or error occurred. */
  data: T | null;
  /** True while the fetch is in progress. */
  loading: boolean;
  /** Human-readable error message, or null if no error. */
  error: string | null;
  /** Function to manually trigger a re-fetch. */
  refetch: () => void;
}

/**
 * Generic fetch hook for loading JSON data from any URL.
 *
 * Handles network errors, JSON parse errors, and 404 responses with user-friendly messages.
 * Provides a refetch function to manually re-trigger the fetch.
 *
 * @template T The type of data being fetched.
 * @param url The URL to fetch from.
 * @returns UseFetchResult<T> containing data, loading state, error message, and refetch function.
 *
 * @example
 * ```tsx
 * const { data, loading, error, refetch } = useFetch<MyData>('/api/my-endpoint');
 * if (loading) return <p>Loading...</p>;
 * if (error) return <p>Error: {error}</p>;
 * return <div>{data?.id}</div>;
 * ```
 */
export function useFetch<T>(url: string): UseFetchResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refetchCount, setRefetchCount] = useState(0);

  useEffect(() => {
    let isMounted = true;

    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);

        const response = await fetch(url);

        if (!response.ok) {
          if (response.status === 404) {
            throw new Error('Data not found — run the pipeline first');
          }
          throw new Error(`HTTP ${response.status}`);
        }

        const json = await response.json();

        if (isMounted) {
          setData(json);
          setError(null);
        }
      } catch (err) {
        if (isMounted) {
          if (err instanceof SyntaxError) {
            setError('Invalid data format');
          } else if (err instanceof TypeError) {
            // Network error (fetch failed)
            setError('Failed to load data');
          } else if (err instanceof Error) {
            setError(err.message);
          } else {
            setError('Unknown error occurred');
          }
          setData(null);
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    fetchData();

    return () => {
      isMounted = false;
    };
  }, [url, refetchCount]);

  const refetch = () => {
    setRefetchCount((prev) => prev + 1);
  };

  return { data, loading, error, refetch };
}
