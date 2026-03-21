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
 * GET /runs/:run_id — Get run detail
 */
export function getRun(runId: string): Promise<t.RunDetail> {
  return fetchGet(`/runs/${runId}`)
}

/**
 * GET /runs/:run_id/summary — Get run summary (lightweight)
 */
export function getRunSummary(runId: string): Promise<t.RunSummary> {
  return fetchGet(`/runs/${runId}/summary`)
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

/**
 * GET /lanes/:lane_id — Get lane detail
 */
export function getLane(laneId: string): Promise<t.Lane> {
  return fetchGet(`/lanes/${laneId}`)
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
 * GET /promotions/:promotion_id — Get promotion detail
 */
export function getPromotion(promotionId: string): Promise<t.Promotion> {
  return fetchGet(`/promotions/${promotionId}`)
}

/**
 * POST /promotions — Promote a run
 */
export function promoteRun(request: t.PromotionRequest): Promise<t.Promotion> {
  return fetchPost('/promotions', request)
}

/**
 * GET /runs/:run_id/promotions — Get promotions for a run
 */
export function getRunPromotions(runId: string): Promise<t.Promotion[]> {
  return fetchGet(`/runs/${runId}/promotions`)
}

// ============================================================================
// COMPARISON
// ============================================================================

/**
 * POST /compare — Compare two runs
 * @param runAId First run ID
 * @param runBId Second run ID
 */
export function compareRuns(runAId: string, runBId: string): Promise<t.CompareResult> {
  return fetchPost('/compare', {
    run_a_id: runAId,
    run_b_id: runBId,
  })
}

// ============================================================================
// SEARCH & FILTER
// ============================================================================

/**
 * GET /runs/search — Search runs by criteria
 * @param modelName Filter by model name
 * @param variant Filter by variant
 * @param minMetric Minimum metric value (e.g., roc_auc)
 * @param skip Pagination skip
 * @param limit Pagination limit
 */
export function searchRuns(params: {
  modelName?: string
  variant?: string
  minMetric?: number
  skip?: number
  limit?: number
}): Promise<t.RunSummary[]> {
  const queryParams = new URLSearchParams()
  if (params.modelName) queryParams.append('model_name', params.modelName)
  if (params.variant) queryParams.append('variant', params.variant)
  if (params.minMetric !== undefined) queryParams.append('min_metric', params.minMetric.toString())
  if (params.skip !== undefined) queryParams.append('skip', params.skip.toString())
  if (params.limit !== undefined) queryParams.append('limit', params.limit.toString())

  return fetchGet(`/runs/search?${queryParams.toString()}`)
}

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

// ============================================================================
// EXPORT & REPORTING
// ============================================================================

/**
 * GET /runs/:run_id/export/json — Export run as JSON
 */
export async function exportRunJson(runId: string): Promise<string> {
  const response = await fetch(`${BASE_URL}/runs/${runId}/export/json`)
  if (!response.ok) {
    throw new Error(`Export failed [${response.status}]: ${response.statusText}`)
  }
  return response.text()
}

/**
 * GET /runs/:run_id/export/csv — Export run as CSV
 */
export async function exportRunCsv(runId: string): Promise<string> {
  const response = await fetch(`${BASE_URL}/runs/${runId}/export/csv`)
  if (!response.ok) {
    throw new Error(`Export failed [${response.status}]: ${response.statusText}`)
  }
  return response.text()
}
