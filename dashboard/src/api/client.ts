/**
 * Fetch wrapper functions for all API endpoints
 * All requests proxied to /api/* by Vite dev server
 * Production uses backend CORS or same-origin requests
 */

import * as t from './types'

const BASE_URL = '/api'

/**
 * Base fetch wrapper that handles errors
 * Throws on non-2xx responses
 */
async function fetchAPI<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${BASE_URL}${endpoint}`
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  })

  if (!response.ok) {
    const error = await response.text()
    throw new Error(`API Error [${response.status}]: ${error || response.statusText}`)
  }

  return response.json() as Promise<T>
}

/**
 * GET request wrapper
 */
function fetchGet<T>(endpoint: string): Promise<T> {
  return fetchAPI<T>(endpoint, { method: 'GET' })
}

/**
 * POST request wrapper
 */
function fetchPost<T>(endpoint: string, body: unknown): Promise<T> {
  return fetchAPI<T>(endpoint, {
    method: 'POST',
    body: JSON.stringify(body),
  })
}



// ============================================================================
// HEALTH & OVERVIEW
// ============================================================================

/**
 * GET /health — Health check
 */
export function getHealth(): Promise<t.HealthResponse> {
  return fetchGet('/health')
}

/**
 * GET /overview — Dashboard overview
 */
export function getOverview(): Promise<t.OverviewResponse> {
  return fetchGet('/overview')
}

// ============================================================================
// RUNS
// ============================================================================

/**
 * GET /runs — List all runs (paginated)
 * @param skip Number of runs to skip (default: 0)
 * @param limit Number of runs to return (default: 20)
 */
export function listRuns(skip: number = 0, limit: number = 20): Promise<t.RunSummary[]> {
  return fetchGet(`/runs?skip=${skip}&limit=${limit}`)
}

/**
 * GET /runs/detail — Get run detail by summary path
 * @param summaryPath Path to the run summary JSON file
 */
export function getRunDetail(summaryPath: string): Promise<t.RunDetail> {
  return fetchGet(`/runs/detail?summary_path=${encodeURIComponent(summaryPath)}`)
}



// ============================================================================
// LANES
// ============================================================================

/**
 * GET /lanes — List all lanes
 */
export function listLanes(): Promise<t.Lane[]> {
  return fetchGet('/lanes')
}

// ============================================================================
// PROMOTIONS
// ============================================================================

/**
 * GET /promotions — List all promotions (paginated)
 * @param skip Number of promotions to skip (default: 0)
 * @param limit Number of promotions to return (default: 20)
 */
export function listPromotions(
  skip: number = 0,
  limit: number = 20
): Promise<t.Promotion[]> {
  return fetchGet(`/promotions?skip=${skip}&limit=${limit}`)
}

/**
 * POST /promotions — Promote a run
 */
export function promoteRun(request: t.PromotionRequest): Promise<t.Promotion> {
  return fetchPost('/promotions', request)
}

// ============================================================================
// COMPARISON
// ============================================================================

/**
 * GET /compare — Compare two runs
 * @param runAId First run summary path
 * @param runBId Second run summary path
 */
export function compareRuns(runAId: string, runBId: string): Promise<t.CompareResult> {
  return fetchGet(`/compare?run_a=${encodeURIComponent(runAId)}&run_b=${encodeURIComponent(runBId)}`)
}

// ============================================================================
// SEARCH & FILTER
// ============================================================================

/**
 * GET /lanes/:lane_id/runs — Get all runs in a lane (paginated)
 * @param laneId Lane ID
 * @param skip Pagination skip
 * @param limit Pagination limit
 */
export function getLaneRuns(
  laneId: string,
  skip: number = 0,
  limit: number = 20
): Promise<t.RunSummary[]> {
  return fetchGet(`/lanes/${laneId}/runs?skip=${skip}&limit=${limit}`)
}

/**
 * GET /slate - Run the dry-run daily pipeline for a target date.
 */
export function getSlate(pipelineDate?: string): Promise<t.SlateResponse> {
  const params = new URLSearchParams()
  if (pipelineDate) {
    params.set('pipeline_date', pipelineDate)
  }
  const query = params.toString()
  return fetchGet(`/slate${query ? `?${query}` : ''}`)
}

export function pullSlateFromMac(pipelineDate?: string): Promise<t.MacSyncResponse> {
  const params = new URLSearchParams()
  if (pipelineDate) {
    params.set('pipeline_date', pipelineDate)
  }
  const query = params.toString()
  return fetchAPI<t.MacSyncResponse>(`/slate/pull-from-mac${query ? `?${query}` : ''}`, {
    method: 'POST',
  })
}

export function getLiveSeasonSummary(
  season: number = 2026
): Promise<t.LiveSeasonSummaryResponse> {
  return fetchGet(`/live-season/summary?season=${season}`)
}

export function getLiveSeasonGames(
  season: number = 2026,
  pipelineDate?: string
): Promise<t.LiveSeasonGameResponse[]> {
  const params = new URLSearchParams()
  params.set('season', String(season))
  if (pipelineDate) {
    params.set('pipeline_date', pipelineDate)
  }
  return fetchGet(`/live-season/games?${params.toString()}`)
}
