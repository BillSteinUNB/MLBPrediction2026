import { useFetch } from './useFetch';
import type { SeasonSlateData } from '../types/seasonSlate';

export function useSeasonSlate(pipelineDate?: string) {
  const query = new URLSearchParams();
  if (pipelineDate) {
    query.set('pipeline_date', pipelineDate);
  }
  return useFetch<SeasonSlateData>(`/api/slate?${query.toString()}`);
}
