import apiClient from './apiClient'

/**
 * Normalize Axios errors into a clean UI-friendly message without fabricating data.
 */
function formatApiError(error, fallbackMessage) {
  if (error?.response?.data?.detail) {
    const detail = error.response.data.detail
    return typeof detail === 'string' ? detail : JSON.stringify(detail)
  }
  if (error?.code === 'ECONNABORTED') {
    return 'Request timed out while contacting the FastAPI backend.'
  }
  if (error?.message === 'Network Error' || !error?.response) {
    return 'Backend unreachable. Ensure the FastAPI server is running on Laptop 1.'
  }
  return fallbackMessage || `Request failed (HTTP ${error.response.status}).`
}

/**
 * GET /api/health
 * Checks whether Laptop 1's FastAPI service is reachable and ready.
 */
export async function fetchBackendHealth() {
  const startedAt = performance.now()
  try {
    const response = await apiClient.get('/api/health')
    const latencyMs = Math.round(performance.now() - startedAt)
    return {
      ok: true,
      latencyMs,
      data: response.data,
      error: null,
    }
  } catch (error) {
    return {
      ok: false,
      latencyMs: null,
      data: null,
      error: formatApiError(error, 'Failed to check backend health.'),
    }
  }
}

/**
 * GET /api/trips
 * Dynamically retrieves available evaluation trip IDs from the backend.
 * Trip IDs are never hardcoded on the frontend.
 */
export async function fetchAvailableTrips() {
  try {
    const response = await apiClient.get('/api/trips')
    const payload = response.data
    const trips = Array.isArray(payload)
      ? payload
      : Array.isArray(payload?.trips)
        ? payload.trips
        : []
    return { ok: true, trips, error: null }
  } catch (error) {
    return {
      ok: false,
      trips: [],
      error: formatApiError(error, 'Unable to load available trips from backend.'),
    }
  }
}

/**
 * GET /api/models
 * Retrieves available ML models (KNN, MLP, final integrated model, HMM status).
 */
export async function fetchAvailableModels() {
  try {
    const response = await apiClient.get('/api/models')
    const payload = response.data
    const models = Array.isArray(payload)
      ? payload
      : Array.isArray(payload?.models)
        ? payload.models
        : []
    return { ok: true, models, meta: payload, error: null }
  } catch (error) {
    return {
      ok: false,
      models: [],
      meta: null,
      error: formatApiError(error, 'Unable to load model registry from backend.'),
    }
  }
}

/**
 * GET /api/models/comparison
 * Retrieves recorded experimental comparison metrics (never fabricated).
 */
export async function fetchModelComparison() {
  try {
    const response = await apiClient.get('/api/models/comparison')
    const payload = response.data
    const metrics = Array.isArray(payload)
      ? payload
      : Array.isArray(payload?.models)
        ? payload.models
        : []
    return { ok: true, metrics, meta: payload, error: null }
  } catch (error) {
    return {
      ok: false,
      metrics: [],
      meta: null,
      error: formatApiError(
        error,
        'Unable to load recorded model comparison metrics.'
      ),
    }
  }
}

/**
 * POST /api/predict
 * Sends the selected trip_id and model identifier to Laptop 1 for live inference.
 * Ground-truth labels remain strictly server-side and are never returned to the UI.
 */
export async function runRoutePrediction({ tripId, modelKey }) {
  try {
    const response = await apiClient.post('/api/predict', {
      trip_id: tripId,
      model: modelKey,
    })
    return { ok: true, data: response.data, error: null }
  } catch (error) {
    return {
      ok: false,
      data: null,
      error: formatApiError(error, 'Route prediction request failed.'),
    }
  }
}
