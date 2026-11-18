import { useCallback, useEffect, useState } from 'react'
import {
  getStoredApiBaseUrl,
  resetApiBaseUrl,
  setStoredApiBaseUrl,
} from '../services/apiClient'
import {
  fetchAvailableModels,
  fetchAvailableTrips,
  fetchBackendHealth,
  fetchModelComparison,
} from '../services/pathsnapApi'
import { SystemContext } from './systemContextValue'

async function loadSystemSnapshot() {
  const healthResult = await fetchBackendHealth()

  if (!healthResult.ok) {
    return {
      health: {
        connected: false,
        latencyMs: null,
        data: null,
        error: healthResult.error,
      },
      trips: [],
      tripsError: healthResult.error,
      models: [],
      modelsMeta: null,
      modelsError: healthResult.error,
      comparison: [],
      comparisonError: healthResult.error,
      checkedAt: new Date().toISOString(),
    }
  }

  const [tripsResult, modelsResult, comparisonResult] = await Promise.all([
    fetchAvailableTrips(),
    fetchAvailableModels(),
    fetchModelComparison(),
  ])

  return {
    health: {
      connected: true,
      latencyMs: healthResult.latencyMs,
      data: healthResult.data,
      error: null,
    },
    trips: tripsResult.trips,
    tripsError: tripsResult.error,
    models: modelsResult.models,
    modelsMeta: modelsResult.meta,
    modelsError: modelsResult.error,
    comparison: comparisonResult.metrics,
    comparisonError: comparisonResult.error,
    checkedAt: new Date().toISOString(),
  }
}

export function SystemProvider({ children }) {
  const [apiBaseUrl, setApiBaseUrlState] = useState(getStoredApiBaseUrl)
  const [loading, setLoading] = useState(true)
  const [lastCheckedAt, setLastCheckedAt] = useState(null)

  const [health, setHealth] = useState({
    connected: false,
    latencyMs: null,
    data: null,
    error: null,
  })
  const [trips, setTrips] = useState([])
  const [tripsError, setTripsError] = useState(null)
  const [models, setModels] = useState([])
  const [modelsMeta, setModelsMeta] = useState(null)
  const [modelsError, setModelsError] = useState(null)
  const [comparison, setComparison] = useState([])
  const [comparisonError, setComparisonError] = useState(null)

  const applySnapshot = useCallback((snapshot) => {
    setHealth(snapshot.health)
    setTrips(snapshot.trips)
    setTripsError(snapshot.tripsError)
    setModels(snapshot.models)
    setModelsMeta(snapshot.modelsMeta)
    setModelsError(snapshot.modelsError)
    setComparison(snapshot.comparison)
    setComparisonError(snapshot.comparisonError)
    setLastCheckedAt(snapshot.checkedAt)
    setLoading(false)
  }, [])

  useEffect(() => {
    let cancelled = false
    loadSystemSnapshot().then((snapshot) => {
      if (!cancelled) {
        applySnapshot(snapshot)
      }
    })
    return () => {
      cancelled = true
    }
  }, [applySnapshot])

  const refreshSystemStatus = useCallback(async () => {
    setLoading(true)
    const snapshot = await loadSystemSnapshot()
    applySnapshot(snapshot)
  }, [applySnapshot])

  const updateApiBaseUrl = useCallback(
    async (nextUrl) => {
      const saved = setStoredApiBaseUrl(nextUrl)
      setApiBaseUrlState(saved)
      await refreshSystemStatus()
    },
    [refreshSystemStatus]
  )

  const restoreDefaultApiBaseUrl = useCallback(async () => {
    const defaultUrl = resetApiBaseUrl()
    setApiBaseUrlState(defaultUrl)
    await refreshSystemStatus()
  }, [refreshSystemStatus])

  const value = {
    apiBaseUrl,
    loading,
    lastCheckedAt,
    health,
    trips,
    tripsError,
    models,
    modelsMeta,
    modelsError,
    comparison,
    comparisonError,
    refreshSystemStatus,
    updateApiBaseUrl,
    restoreDefaultApiBaseUrl,
  }

  return (
    <SystemContext.Provider value={value}>{children}</SystemContext.Provider>
  )
}
