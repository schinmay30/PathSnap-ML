import { useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import {
  AlertCircle,
  CheckCircle2,
  Cpu,
  Map as MapIcon,
  Play,
  ShieldCheck,
} from 'lucide-react'
import { useSystem } from '../context/systemContextValue'
import { runRoutePrediction } from '../services/pathsnapApi'
import RouteMap from '../components/RouteMap'
import EmptyState from '../components/EmptyState'
import StatusBadge from '../components/StatusBadge'

const DEFAULT_MODEL_OPTIONS = [
  {
    key: 'knn',
    label: 'KNN',
    description: 'K-Nearest Neighbors (Instance-based)',
  },
  {
    key: 'mlp',
    label: 'MLP',
    description: 'Multi-Layer Perceptron (Neural Network)',
  },
]

function extractTripId(trip) {
  if (typeof trip === 'string') return trip
  return trip?.trip_id ?? trip?.id ?? ''
}

function formatMetric(value, { percent = false, suffix = '', decimals = 4 } = {}) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—'
  if (percent) return `${(value * 100).toFixed(2)}%`
  return `${value.toFixed(decimals)}${suffix}`
}

export default function RouteAnalysisPage() {
  const [searchParams] = useSearchParams()
  const { health, trips, models } = useSystem()

  const tripIds = useMemo(
    () => trips.map(extractTripId).filter(Boolean),
    [trips]
  )

  const initialTripFromQuery = searchParams.get('trip') || ''
  const [selectedTripId, setSelectedTripId] = useState(initialTripFromQuery)
  const [selectedModel, setSelectedModel] = useState('knn')
  const [predicting, setPredicting] = useState(false)
  const [predictionError, setPredictionError] = useState(null)
  const [predictionResult, setPredictionResult] = useState(null)

  const activeTripId = tripIds.includes(selectedTripId)
    ? selectedTripId
    : tripIds[0] || ''

  const modelOptions = useMemo(() => {
    const options = [...DEFAULT_MODEL_OPTIONS]
    models.forEach((m) => {
      const key = String(m?.key ?? m?.id ?? '').toLowerCase()
      if (!key || options.some((opt) => opt.key === key)) return
      options.push({
        key,
        label: m?.name || key.toUpperCase(),
        description: m?.family || 'Registered backend model',
      })
    })
    return options
  }, [models])

  async function handleRunPrediction(e) {
    e.preventDefault()
    if (!activeTripId) return

    setPredicting(true)
    setPredictionError(null)

    const result = await runRoutePrediction({
      tripId: activeTripId,
      modelKey: selectedModel,
    })

    if (result.ok) {
      setPredictionResult(result.data)
    } else {
      setPredictionResult(null)
      setPredictionError(result.error)
    }
    setPredicting(false)
  }

  const metrics = predictionResult?.metrics || predictionResult?.evaluation || {}

  return (
    <div className="route-analysis-grid">
      {/* Left Control Panel */}
      <section className="card">
        <div className="card-header">
          <div className="card-title-row">
            <Cpu size={17} color="#1d4ed8" />
            <h3 className="card-title">Inference Controls</h3>
          </div>
          <StatusBadge
            variant={health.connected ? 'success' : 'error'}
            label={health.connected ? 'API Ready' : 'Offline'}
          />
        </div>

        <form className="card-body" onSubmit={handleRunPrediction}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
            {/* Trip Selection */}
            <div className="form-group">
              <label className="form-label" htmlFor="trip-select">
                Evaluation Trip
              </label>
              <select
                id="trip-select"
                className="form-select mono"
                value={activeTripId}
                onChange={(e) => setSelectedTripId(e.target.value)}
                disabled={!health.connected || tripIds.length === 0 || predicting}
              >
                {tripIds.length === 0 ? (
                  <option value="">
                    {health.connected
                      ? 'No trips returned by backend'
                      : 'Connect backend to load trips'}
                  </option>
                ) : (
                  tripIds.map((id) => (
                    <option key={id} value={id}>
                      Trip {id}
                    </option>
                  ))
                )}
              </select>
              <span className="form-hint">
                Trip list is retrieved dynamically from{' '}
                <span className="mono">GET /api/trips</span>.
              </span>
            </div>

            {/* Model Selection */}
            <div className="form-group">
              <span className="form-label">Classifier Selection</span>
              <div className="model-option-grid">
                {modelOptions.map((opt) => (
                  <button
                    key={opt.key}
                    type="button"
                    className={`model-option-btn ${
                      selectedModel === opt.key ? 'selected' : ''
                    }`}
                    onClick={() => setSelectedModel(opt.key)}
                    disabled={predicting}
                  >
                    <span className="model-option-name">{opt.label}</span>
                    <span className="model-option-desc">{opt.description}</span>
                  </button>
                ))}
              </div>
              <span className="form-hint">
                Selected model identifier is sent with{' '}
                <span className="mono">POST /api/predict</span>.
              </span>
            </div>

            <button
              type="submit"
              className="btn btn-primary btn-full"
              disabled={!health.connected || !activeTripId || predicting}
            >
              <Play size={15} />
              <span>
                {predicting ? 'Running Inference…' : 'Run Route Prediction'}
              </span>
            </button>

            {predictionError && (
              <div className="callout callout-error" role="alert">
                <AlertCircle size={16} style={{ flexShrink: 0, marginTop: 2 }} />
                <span>{predictionError}</span>
              </div>
            )}

            <div className="callout">
              <ShieldCheck size={16} style={{ flexShrink: 0, marginTop: 2 }} />
              <span>
                <strong>Server-Side Ground-Truth Isolation:</strong> Reference{' '}
                <span className="mono">.route</span> files stay strictly on Laptop
                1 and are never passed to the model or returned to this UI.
              </span>
            </div>
          </div>
        </form>
      </section>

      {/* Right Visualization & Results Column */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
        {/* GPS Route Visualization Area */}
        <section className="card">
          <div className="card-header">
            <div className="card-title-row">
              <MapIcon size={17} color="#1d4ed8" />
              <div>
                <h3 className="card-title">GPS Route Visualisation</h3>
                <p className="card-subtitle">
                  Interactive Leaflet map rendering raw GPS observations, road
                  corridor, and reconstructed route
                </p>
              </div>
            </div>
            {predictionResult?.trip_id && (
              <StatusBadge
                variant="info"
                label={`Trip ${predictionResult.trip_id} · ${String(
                  predictionResult.model || selectedModel
                ).toUpperCase()}`}
                showDot={false}
              />
            )}
          </div>
          <div className="card-body">
            <RouteMap
              gpsPoints={predictionResult?.gps_points || []}
              roadNetwork={predictionResult?.road_network || []}
              predictedRoute={predictionResult?.predicted_route || []}
              emptyTitle={
                health.connected
                  ? 'Awaiting Route Prediction'
                  : 'Backend Not Connected'
              }
              emptyDescription={
                health.connected
                  ? 'Choose an evaluation trip and model on the left, then click "Run Route Prediction" to fetch and visualize the real trajectory.'
                  : 'Connect Laptop 1 FastAPI backend to load evaluation trips and display predicted GPS routes. No synthetic routes are plotted.'
              }
            />
          </div>
        </section>

        {/* Prediction Results Area */}
        <section className="card">
          <div className="card-header">
            <div className="card-title-row">
              <CheckCircle2 size={17} color="#1d4ed8" />
              <div>
                <h3 className="card-title">Prediction &amp; Evaluation Results</h3>
                <p className="card-subtitle">
                  Metrics computed server-side by Laptop 1 upon live inference
                </p>
              </div>
            </div>
            <StatusBadge
              variant={predictionResult ? 'success' : 'neutral'}
              label={predictionResult ? 'Prediction Complete' : 'No Results Yet'}
            />
          </div>

          <div className="card-body">
            {!predictionResult ? (
              <EmptyState
                icon={CheckCircle2}
                title="No Prediction Executed Yet"
                description="Once the FastAPI backend processes a selected trip through the ML classifier and HMM/Viterbi decoder, server-computed accuracy, route continuity, and latency metrics will appear here."
              />
            ) : (
              <div className="metrics-mini-grid">
                <div className="metric-tile">
                  <div className="metric-tile-label">Prediction Accuracy</div>
                  <div className="metric-tile-val mono">
                    {formatMetric(
                      metrics.accuracy ?? metrics.route_accuracy,
                      { percent: true }
                    )}
                  </div>
                </div>
                <div className="metric-tile">
                  <div className="metric-tile-label">Observation Top-1 Accuracy</div>
                  <div className="metric-tile-val mono">
                    {formatMetric(metrics.top1_accuracy, { percent: true })}
                  </div>
                </div>
                <div className="metric-tile">
                  <div className="metric-tile-label">Route Continuity</div>
                  <div className="metric-tile-val mono">
                    {formatMetric(metrics.route_continuity, { percent: true })}
                  </div>
                </div>
                <div className="metric-tile">
                  <div className="metric-tile-label">GPS Observations</div>
                  <div className="metric-tile-val mono">
                    {typeof predictionResult.num_gps_points === 'number'
                      ? predictionResult.num_gps_points.toLocaleString()
                      : Array.isArray(predictionResult.gps_points)
                        ? predictionResult.gps_points.length.toLocaleString()
                        : '—'}
                  </div>
                </div>
                <div className="metric-tile">
                  <div className="metric-tile-label">Predicted Route Arcs</div>
                  <div className="metric-tile-val mono">
                    {typeof predictionResult.num_predicted_arcs === 'number'
                      ? predictionResult.num_predicted_arcs.toLocaleString()
                      : Array.isArray(predictionResult.predicted_arc_ids)
                        ? predictionResult.predicted_arc_ids.length.toLocaleString()
                        : '—'}
                  </div>
                </div>
                <div className="metric-tile">
                  <div className="metric-tile-label">Inference Latency</div>
                  <div className="metric-tile-val mono">
                    {formatMetric(
                      predictionResult.inference_time_ms ?? metrics.inference_time_ms,
                      { decimals: 1, suffix: ' ms' }
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  )
}
