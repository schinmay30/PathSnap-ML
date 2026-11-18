import { useState } from 'react'
import {
  Cpu,
  Globe,
  Info,
  RotateCcw,
  Save,
  Server,
  ShieldCheck,
} from 'lucide-react'
import { useSystem } from '../context/systemContextValue'
import StatusBadge from '../components/StatusBadge'
import EmptyState from '../components/EmptyState'

const API_CONTRACT_ENDPOINTS = [
  { method: 'GET', path: '/api/health', purpose: 'Backend readiness & latency check' },
  { method: 'GET', path: '/api/trips', purpose: 'Dynamically list available evaluation trip IDs' },
  { method: 'GET', path: '/api/models', purpose: 'List loaded ML models & HMM/Viterbi status' },
  {
    method: 'GET',
    path: '/api/models/comparison',
    purpose: 'Serve recorded KNN, MLP & team experiment metrics',
  },
  {
    method: 'POST',
    path: '/api/predict',
    purpose: 'Run live inference & server-side evaluation for selected trip_id',
  },
]

export default function SystemInfoPage() {
  const {
    apiBaseUrl,
    health,
    models,
    loading,
    lastCheckedAt,
    updateApiBaseUrl,
    restoreDefaultApiBaseUrl,
  } = useSystem()

  const [urlInput, setUrlInput] = useState(apiBaseUrl)

  async function handleSaveUrl(e) {
    e.preventDefault()
    await updateApiBaseUrl(urlInput)
  }

  async function handleResetUrl() {
    await restoreDefaultApiBaseUrl()
    setUrlInput(import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000')
  }

  return (
    <>
      {/* Backend Status & Endpoint Configuration */}
      <section className="card">
        <div className="card-header">
          <div className="card-title-row">
            <Server size={17} color="#1d4ed8" />
            <div>
              <h3 className="card-title">Backend Status &amp; Connection</h3>
              <p className="card-subtitle">
                Configure Laptop 1 FastAPI endpoint URL for local or two-laptop
                LAN evaluation
              </p>
            </div>
          </div>
          <StatusBadge
            variant={loading ? 'neutral' : health.connected ? 'success' : 'error'}
            label={
              loading ? 'Checking…' : health.connected ? 'Connected' : 'Offline'
            }
          />
        </div>

        <div className="card-body">
          <form onSubmit={handleSaveUrl} style={{ marginBottom: 18 }}>
            <div className="form-group">
              <label className="form-label" htmlFor="api-base-url-input">
                FastAPI Base URL (Laptop 1)
              </label>
              <div className="url-config-row">
                <input
                  id="api-base-url-input"
                  type="url"
                  className="form-input mono"
                  value={urlInput}
                  onChange={(e) => setUrlInput(e.target.value)}
                  placeholder="http://127.0.0.1:8000"
                  required
                />
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={loading}
                >
                  <Save size={14} />
                  <span>Save &amp; Test</span>
                </button>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={handleResetUrl}
                  disabled={loading}
                >
                  <RotateCcw size={14} />
                  <span>Reset</span>
                </button>
              </div>
              <span className="form-hint">
                Saved in browser local storage so Laptop 2 can connect to Laptop
                1&apos;s LAN IP (e.g., <span className="mono">http://192.168.x.x:8000</span>)
                without rebuilding.
              </span>
            </div>
          </form>

          <div className="kv-list">
            <div className="kv-row">
              <span className="kv-label">Active Endpoint</span>
              <span className="kv-value mono">{apiBaseUrl}</span>
            </div>
            <div className="kv-row">
              <span className="kv-label">Health Check Latency</span>
              <span className="kv-value mono">
                {health.latencyMs !== null ? `${health.latencyMs} ms` : '—'}
              </span>
            </div>
            <div className="kv-row">
              <span className="kv-label">Backend Service Status</span>
              <span className="kv-value">
                {health.connected
                  ? health.data?.status || 'Healthy'
                  : health.error || 'Unreachable'}
              </span>
            </div>
            <div className="kv-row">
              <span className="kv-label">Last Checked</span>
              <span className="kv-value mono">
                {lastCheckedAt ? new Date(lastCheckedAt).toLocaleString() : '—'}
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* Available Models */}
      <section className="card">
        <div className="card-header">
          <div className="card-title-row">
            <Cpu size={17} color="#1d4ed8" />
            <div>
              <h3 className="card-title">Available Models</h3>
              <p className="card-subtitle">
                Live model registry reported by <span className="mono">GET /api/models</span>
              </p>
            </div>
          </div>
          <StatusBadge
            variant={models.length > 0 ? 'success' : 'neutral'}
            label={`${models.length} Loaded`}
          />
        </div>

        <div className="card-body">
          {models.length === 0 ? (
            <EmptyState
              icon={Cpu}
              title={
                health.connected
                  ? 'No Models Registered Yet'
                  : 'Model Registry Unavailable'
              }
              description={
                health.connected
                  ? 'The FastAPI server has not reported any loaded KNN, MLP, or final HMM-integrated model artifacts yet.'
                  : 'Connect to the FastAPI backend to inspect available models, artifact versions, and HMM/Viterbi decoder readiness.'
              }
            />
          ) : (
            <div className="table-wrapper">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Key</th>
                    <th>Model Name</th>
                    <th>Family</th>
                    <th>HMM / Viterbi</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {models.map((m) => (
                    <tr key={m.key || m.id || m.name}>
                      <td className="mono">{m.key || m.id || '—'}</td>
                      <td style={{ fontWeight: 600 }}>{m.name || '—'}</td>
                      <td>{m.family || '—'}</td>
                      <td>{m.hmm_enabled ? 'Enabled' : 'Classifier Only'}</td>
                      <td>
                        <StatusBadge
                          variant={m.loaded ? 'success' : 'neutral'}
                          label={m.loaded ? 'Loaded' : 'Unavailable'}
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </section>

      {/* Application Information & API Contract */}
      <section className="two-col-grid">
        <div className="card">
          <div className="card-header">
            <div className="card-title-row">
              <Info size={17} color="#1d4ed8" />
              <h3 className="card-title">Application Information</h3>
            </div>
          </div>
          <div className="card-body">
            <div className="kv-list">
              <div className="kv-row">
                <span className="kv-label">Application</span>
                <span className="kv-value">PathSnap Evaluation Console</span>
              </div>
              <div className="kv-row">
                <span className="kv-label">Course</span>
                <span className="kv-value">UE24CS352A · Machine Learning</span>
              </div>
              <div className="kv-row">
                <span className="kv-label">Deployment Role</span>
                <span className="kv-value">Laptop 2 · Frontend UI</span>
              </div>
              <div className="kv-row">
                <span className="kv-label">Frontend Stack</span>
                <span className="kv-value">
                  React, Vite, React Router, Axios, Lucide, Leaflet
                </span>
              </div>
              <div className="kv-row">
                <span className="kv-label">Ground-Truth Policy</span>
                <span className="kv-value">Strictly Server-Side (Laptop 1)</span>
              </div>
            </div>

            <div className="callout" style={{ marginTop: 14 }}>
              <ShieldCheck size={16} style={{ flexShrink: 0, marginTop: 2 }} />
              <span>
                No evaluator trip IDs, GPS coordinates, or accuracy scores are
                hardcoded in this client bundle.
              </span>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="card-title-row">
              <Globe size={17} color="#1d4ed8" />
              <h3 className="card-title">Expected FastAPI Endpoints</h3>
            </div>
          </div>
          <div className="table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Method</th>
                  <th>Endpoint</th>
                  <th>Purpose</th>
                </tr>
              </thead>
              <tbody>
                {API_CONTRACT_ENDPOINTS.map((ep) => (
                  <tr key={ep.path}>
                    <td>
                      <StatusBadge
                        variant={ep.method === 'POST' ? 'info' : 'neutral'}
                        label={ep.method}
                        showDot={false}
                      />
                    </td>
                    <td className="mono">{ep.path}</td>
                    <td>{ep.purpose}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </>
  )
}
