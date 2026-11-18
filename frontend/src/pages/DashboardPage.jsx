import { Link, useNavigate } from 'react-router-dom'
import {
  Activity,
  ArrowRight,
  Compass,
  Cpu,
  Database,
  MapPin,
  Server,
  ShieldCheck,
} from 'lucide-react'
import { useSystem } from '../context/systemContextValue'
import StatCard from '../components/StatCard'
import StatusBadge from '../components/StatusBadge'
import EmptyState from '../components/EmptyState'

function extractTripId(trip) {
  if (typeof trip === 'string') return trip
  return trip?.trip_id ?? trip?.id ?? ''
}

function findModelStatus(models, keyword) {
  const match = models.find((m) => {
    const text = `${m?.key ?? ''} ${m?.id ?? ''} ${m?.name ?? ''}`.toLowerCase()
    return text.includes(keyword)
  })
  if (!match) return null
  return match
}

export default function DashboardPage() {
  const navigate = useNavigate()
  const { apiBaseUrl, health, trips, models, modelsMeta, loading, lastCheckedAt } =
    useSystem()

  const knnModel = findModelStatus(models, 'knn')
  const mlpModel = findModelStatus(models, 'mlp')
  const hmmLoaded = Boolean(
    modelsMeta?.hmm_decoder_loaded ??
      health.data?.hmm_decoder_loaded ??
      models.some((m) => m?.hmm_enabled)
  )

  return (
    <>
      {/* Branding & Project Overview */}
      <section className="card">
        <div className="card-header">
          <div className="card-title-row">
            <Compass size={18} color="#1d4ed8" />
            <div>
              <h3 className="card-title">
                PathSnap — Distributed Map-Matching &amp; Route Reconstruction
              </h3>
              <p className="card-subtitle">
                UE24CS352A: Machine Learning · Two-Laptop Real-Time Inference &amp;
                Evaluation Console
              </p>
            </div>
          </div>
          <StatusBadge
            variant="info"
            label="Laptop 2 · Evaluator UI"
            showDot={false}
          />
        </div>

        <div className="card-body">
          <p style={{ color: 'var(--text-secondary)', maxWidth: 920 }}>
            PathSnap reconstructs realistic vehicle trajectories by snapping noisy
            1 Hz GPS observations onto directed road-network segments. Machine
            learning classifiers score candidate arcs per observation, while an
            calibrated Viterbi decoder selects a road segment for each matched
            observation and flags jumps and sampling gaps.
          </p>

          <div className="pipeline-steps">
            <div className="pipeline-step">
              <div className="pipeline-step-num">Stage 1</div>
              <div className="pipeline-step-title">GPS &amp; Graph Ingestion</div>
              <div className="pipeline-step-desc">
                Loads per-trip <span className="mono">.track</span>,{' '}
                <span className="mono">.nodes</span>, and{' '}
                <span className="mono">.arcs</span> files on Laptop 1.
              </div>
            </div>
            <div className="pipeline-step">
              <div className="pipeline-step-num">Stage 2</div>
              <div className="pipeline-step-title">Candidate Scoring</div>
              <div className="pipeline-step-desc">
                Extracts geospatial &amp; kinematic features and scores candidates
                via KNN, MLP, or the final tuned classifier.
              </div>
            </div>
            <div className="pipeline-step">
              <div className="pipeline-step-num">Stage 3</div>
              <div className="pipeline-step-title">HMM / Viterbi Decoding</div>
              <div className="pipeline-step-desc">
                Returns ordered matches and separate route components across jumps and gaps.
              </div>
            </div>
            <div className="pipeline-step">
              <div className="pipeline-step-num">Stage 4</div>
              <div className="pipeline-step-title">Route Visualisation</div>
              <div className="pipeline-step-desc">
                Displays the predicted road segments, matched observations and inference time.
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Live Status Summary Cards */}
      <section className="stats-grid" aria-label="System summary">
        <StatCard
          label="Backend Connection"
          value={loading ? 'Checking…' : health.connected ? 'Connected' : 'Offline'}
          helper={
            health.connected
              ? `FastAPI reachable (${health.latencyMs ?? 0} ms)`
              : 'Awaiting FastAPI server on Laptop 1'
          }
          icon={Server}
        />
        <StatCard
          label="Available Evaluation Trips"
          value={health.connected ? String(trips.length) : '—'}
          helper={
            health.connected
              ? 'Loaded dynamically from GET /api/trips'
              : 'No trips loaded while backend is offline'
          }
          icon={MapPin}
        />
        <StatCard
          label="Registered Models"
          value={health.connected ? String(models.length) : '—'}
          helper={
            health.connected
              ? 'Reported by GET /api/models'
              : 'KNN, MLP & final model awaiting backend'
          }
          icon={Cpu}
        />
        <StatCard
          label="HMM / Viterbi Decoder"
          value={
            !health.connected ? '—' : hmmLoaded ? 'Ready' : 'Not Loaded'
          }
          helper={
            health.connected
              ? hmmLoaded
                ? 'Sequential decoder active on Laptop 1'
                : 'Awaiting Person 3 decoder integration'
              : 'Status unavailable while backend is offline'
          }
          icon={Activity}
        />
      </section>

      {/* Two-Column Details: Model Status & Backend Connection */}
      <section className="two-col-grid">
        <div className="card">
          <div className="card-header">
            <div className="card-title-row">
              <Cpu size={17} color="#1d4ed8" />
              <h3 className="card-title">Model Status</h3>
            </div>
            <Link to="/model-comparison" className="btn btn-secondary">
              <span>Compare Models</span>
              <ArrowRight size={14} />
            </Link>
          </div>
          <div className="card-body">
            <div className="kv-list">
              <div className="kv-row">
                <div>
                  <div className="kv-label" style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
                    K-Nearest Neighbors (KNN)
                  </div>
                  <div className="form-hint">
                    Instance-based classifier · Person 4
                  </div>
                </div>
                <StatusBadge
                  variant={knnModel?.loaded ? 'success' : 'neutral'}
                  label={
                    !health.connected
                      ? 'Awaiting Backend'
                      : knnModel?.loaded
                        ? 'Loaded'
                        : 'Not Loaded'
                  }
                />
              </div>

              <div className="kv-row">
                <div>
                  <div className="kv-label" style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
                    Multi-Layer Perceptron (MLP)
                  </div>
                  <div className="form-hint">
                    Neural network classifier · Person 4
                  </div>
                </div>
                <StatusBadge
                  variant={mlpModel?.loaded ? 'success' : 'neutral'}
                  label={
                    !health.connected
                      ? 'Awaiting Backend'
                      : mlpModel?.loaded
                        ? 'Loaded'
                        : 'Not Loaded'
                  }
                />
              </div>

              <div className="kv-row">
                <div>
                  <div className="kv-label" style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
                    Final Tuned Model + HMM / Viterbi
                  </div>
                  <div className="form-hint">
                    Integrated sequential route reconstruction · Person 3 &amp; 4
                  </div>
                </div>
                <StatusBadge
                  variant={findModelStatus(models, 'final')?.loaded ? 'success' : 'neutral'}
                  label={
                    !health.connected
                      ? 'Awaiting Backend'
                      : findModelStatus(models, 'final')?.loaded
                        ? 'Ready'
                        : 'Unavailable'
                  }
                />
              </div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="card-title-row">
              <Server size={17} color="#1d4ed8" />
              <h3 className="card-title">Backend Connection Status</h3>
            </div>
            <Link to="/system-info" className="btn btn-secondary">
              <span>Configure API</span>
              <ArrowRight size={14} />
            </Link>
          </div>
          <div className="card-body">
            <div className="kv-list">
              <div className="kv-row">
                <span className="kv-label">Endpoint URL (Laptop 1)</span>
                <span className="kv-value mono">{apiBaseUrl}</span>
              </div>
              <div className="kv-row">
                <span className="kv-label">Reachability</span>
                <StatusBadge
                  variant={health.connected ? 'success' : 'error'}
                  label={health.connected ? 'Online' : 'Offline'}
                />
              </div>
              <div className="kv-row">
                <span className="kv-label">Response Latency</span>
                <span className="kv-value mono">
                  {health.latencyMs !== null ? `${health.latencyMs} ms` : '—'}
                </span>
              </div>
              <div className="kv-row">
                <span className="kv-label">Last Checked</span>
                <span className="kv-value mono">
                  {lastCheckedAt
                    ? new Date(lastCheckedAt).toLocaleTimeString()
                    : '—'}
                </span>
              </div>
            </div>

            {!health.connected && health.error && (
              <div className="callout callout-warn" style={{ marginTop: 14 }}>
                <ShieldCheck size={16} style={{ flexShrink: 0, marginTop: 2 }} />
                <span>{health.error}</span>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Available Trips Section */}
      <section className="card">
        <div className="card-header">
          <div className="card-title-row">
            <Database size={17} color="#1d4ed8" />
            <div>
              <h3 className="card-title">Available Evaluation Trips</h3>
              <p className="card-subtitle">
                Trips are fetched live from Laptop 1 (<span className="mono">GET /api/trips</span>) — never hardcoded
              </p>
            </div>
          </div>
          <Link to="/route-analysis" className="btn btn-primary">
            <span>Open Route Analysis</span>
            <ArrowRight size={14} />
          </Link>
        </div>

        <div className="card-body">
          {trips.length === 0 ? (
            <EmptyState
              icon={MapPin}
              title={
                health.connected
                  ? 'No Trips Returned by Backend'
                  : 'Trip Inventory Unavailable (Backend Offline)'
              }
              description={
                health.connected
                  ? 'The FastAPI server is reachable, but no evaluation trips were returned by GET /api/trips yet.'
                  : 'Start the FastAPI server on Laptop 1 to dynamically list the 10 available evaluation trips. Evaluator trip IDs are never hardcoded in the frontend.'
              }
            />
          ) : (
            <div className="trip-grid">
              {trips.map((trip) => {
                const tripId = extractTripId(trip)
                const pointCount = trip?.track_points ?? trip?.num_points
                return (
                  <button
                    key={tripId}
                    type="button"
                    className="trip-card"
                    onClick={() =>
                      navigate(`/route-analysis?trip=${encodeURIComponent(tripId)}`)
                    }
                  >
                    <span className="trip-card-id mono">Trip {tripId}</span>
                    <span className="trip-card-meta">
                      {typeof pointCount === 'number'
                        ? `${pointCount.toLocaleString()} GPS points`
                        : 'Click to inspect & predict'}
                    </span>
                  </button>
                )
              })}
            </div>
          )}
        </div>
      </section>
    </>
  )
}
