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

function extractTripId(trip) {
  if (typeof trip === 'string') return trip
  return trip?.trip_id ?? trip?.id ?? ''
}

export default function RouteAnalysisPage() {
  const [searchParams] = useSearchParams()
  const { health, trips, tripsError, models, modelsError, loading } = useSystem()

  const tripIds = useMemo(
    () => trips.map(extractTripId).filter(Boolean),
    [trips]
  )

  const initialTripFromQuery = searchParams.get('trip') || ''
  const [selectedTripId, setSelectedTripId] = useState(initialTripFromQuery)
  const [selectedModel, setSelectedModel] = useState('final')
  const [predicting, setPredicting] = useState(false)
  const [predictionError, setPredictionError] = useState(null)
  const [predictionResult, setPredictionResult] = useState(null)

  const activeTripId = tripIds.includes(selectedTripId)
    ? selectedTripId
    : tripIds[0] || ''

  const selectedReady = models.some((m) => m.key === selectedModel && m.loaded)
  const matches = predictionResult?.matches
  const matched = matches?.filter((m) => m.arc_id != null).length
  const unmatched = matches && matches.length - matched
  const jumps = matches?.filter((m) => m.status === 'soft_jump').length
  const resets = matches?.filter((m) => m.status.endsWith('_reset')).length

  function clearResult() {
    setPredictionResult(null)
    setPredictionError(null)
  }

  async function handleRunPrediction(e) {
    e.preventDefault()
    if (!activeTripId || !selectedReady || predicting) return

    setPredicting(true)
    clearResult()

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

  const resultStatus = predicting ? 'Running inference' : predictionError ? 'Prediction failed'
    : !predictionResult ? 'Awaiting prediction' : !matched ? 'No matched observations'
    : unmatched ? 'Partially matched' : 'Prediction complete'
  const tiles = predictionResult ? [
    ['Matched observations', matches ? `${matched.toLocaleString()} / ${matches.length.toLocaleString()}` : '—'],
    ['Unmatched observations', unmatched], ['Soft jumps', jumps], ['Decoder resets', resets],
    ['Route components', predictionResult.predicted_route?.length],
    ['Inference latency', Number.isFinite(predictionResult.inference_time_ms)
      ? `${predictionResult.inference_time_ms.toFixed(1)} ms` : '—'],
  ] : []

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
                onChange={(e) => { clearResult(); setSelectedTripId(e.target.value) }}
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
                {models.map((opt) => (
                  <button
                    key={opt.key}
                    type="button"
                    className={`model-option-btn ${
                      selectedModel === opt.key ? 'selected' : ''
                    }`}
                    onClick={() => { clearResult(); setSelectedModel(opt.key) }}
                    disabled={predicting || !opt.loaded}
                  >
                    <span className="model-option-name">{opt.key === 'final' ? 'HistGradientBoosting + Viterbi' : opt.name}</span>
                    <span className="model-option-desc">{opt.loaded ? 'Ready' : 'Unavailable'}</span>
                  </button>
                ))}
              </div>
              <span className="form-hint">
                {loading ? 'Checking model availability…' : !selectedReady ? 'Select a ready model to run inference.' : 'The selected model runs on the backend.'}
              </span>
            </div>

            <button
              type="submit"
              className="btn btn-primary btn-full"
              disabled={!health.connected || !activeTripId || !selectedReady || predicting}
            >
              <Play size={15} />
              <span>
                {predicting ? 'Running Inference…' : 'Run Route Prediction'}
              </span>
            </button>

            {(predictionError || tripsError || modelsError) && (
              <div className="callout callout-error" role="alert">
                <AlertCircle size={16} style={{ flexShrink: 0, marginTop: 2 }} />
                <span>{predictionError || tripsError || modelsError}</span>
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
                predictionResult ? 'No route geometry returned' : predicting ? 'Running inference…' : health.connected
                  ? 'Awaiting Route Prediction'
                  : 'Backend Not Connected'
              }
              emptyDescription={
                predictionResult ? 'This trip returned no drawable coordinates.' : predicting ? 'Processing the selected trip.' : health.connected
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
                <h3 className="card-title">Live Prediction Results</h3>
                <p className="card-subtitle">
                  Observation assignments and route breaks for the selected trip
                </p>
              </div>
            </div>
            <StatusBadge
              variant={predictionError ? 'error' : matched ? 'success' : 'neutral'}
              label={resultStatus}
            />
          </div>

          <div className="card-body">
            {!predictionResult ? (
              <EmptyState
                icon={CheckCircle2}
                title={resultStatus}
                description={predicting ? `Processing trip ${activeTripId}.` : "Run a prediction to see observation coverage and inference time."}
              />
            ) : (
              <div className="metrics-mini-grid">
                {tiles.map(([label, value]) => (
                  <div className="metric-tile" key={label}>
                    <div className="metric-tile-label">{label}</div>
                    <div className="metric-tile-val mono">{value ?? '—'}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  )
}
