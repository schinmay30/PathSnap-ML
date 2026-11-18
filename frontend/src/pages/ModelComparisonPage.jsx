import { BarChart3, Info, Table as TableIcon } from 'lucide-react'
import { useSystem } from '../context/systemContextValue'
import ComparisonChart from '../components/ComparisonChart'
import StatusBadge from '../components/StatusBadge'

const PLACEHOLDER_MODELS = [
  {
    model: 'K-Nearest Neighbors (KNN)',
    family: 'Instance-based',
    cv_macro_f1: null,
    test_top1_accuracy: null,
    test_macro_f1: null,
    log_loss: null,
    route_continuity: null,
    training_time_s: null,
    inference_time_s: null,
  },
  {
    model: 'Multi-Layer Perceptron (MLP)',
    family: 'Neural Networks',
    cv_macro_f1: null,
    test_top1_accuracy: null,
    test_macro_f1: null,
    log_loss: null,
    route_continuity: null,
    training_time_s: null,
    inference_time_s: null,
  },
]

function fmtNumber(val, decimals = 4, suffix = '') {
  if (typeof val !== 'number' || !Number.isFinite(val)) return '—'
  return `${val.toFixed(decimals)}${suffix}`
}

function fmtPercent(val) {
  if (typeof val !== 'number' || !Number.isFinite(val)) return '—'
  return `${(val * 100).toFixed(2)}%`
}

export default function ModelComparisonPage() {
  const { health, comparison } = useSystem()

  const hasRecordedResults = comparison.length > 0
  const rows = hasRecordedResults ? comparison : PLACEHOLDER_MODELS

  const cvF1Items = hasRecordedResults
    ? comparison.map((m) => ({
        label: m.model || m.name,
        value: m.cv_macro_f1,
      }))
    : []

  const top1Items = hasRecordedResults
    ? comparison.map((m) => ({
        label: m.model || m.name,
        value: m.test_top1_accuracy,
      }))
    : []

  const testF1Items = hasRecordedResults
    ? comparison.map((m) => ({
        label: m.model || m.name,
        value: m.test_macro_f1,
      }))
    : []

  const continuityItems = hasRecordedResults
    ? comparison.map((m) => ({
        label: m.model || m.name,
        value: m.route_continuity,
      }))
    : []

  return (
    <>
      {!hasRecordedResults && (
        <div className="callout callout-warn">
          <Info size={16} style={{ flexShrink: 0, marginTop: 2 }} />
          <span>
            <strong>Awaiting Experimental Results:</strong>{' '}
            {health.connected
              ? 'No recorded model metrics have been exported to GET /api/models/comparison yet.'
              : 'FastAPI backend is currently offline.'}{' '}
            Table cells and charts intentionally remain empty rather than
            displaying synthetic accuracy values.
          </span>
        </div>
      )}

      {/* Performance Metrics Table */}
      <section className="card">
        <div className="card-header">
          <div className="card-title-row">
            <TableIcon size={17} color="#1d4ed8" />
            <div>
              <h3 className="card-title">KNN &amp; MLP Performance Comparison</h3>
              <p className="card-subtitle">
                Group-aware cross-validation (80 train trips) and development
                comparison (10 student_test trips)
              </p>
            </div>
          </div>
          <StatusBadge
            variant={hasRecordedResults ? 'success' : 'neutral'}
            label={
              hasRecordedResults
                ? `${comparison.length} Models Recorded`
                : 'Awaiting Experiment Logs'
            }
          />
        </div>

        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Algorithm Family</th>
                <th>CV Macro F1</th>
                <th>Student-Test Top-1 Acc</th>
                <th>Student-Test Macro F1</th>
                <th>Log Loss</th>
                <th>Route Continuity</th>
                <th>Train Time</th>
                <th>Inference Time</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.model || row.name}>
                  <td style={{ fontWeight: 600 }}>{row.model || row.name}</td>
                  <td>{row.family || '—'}</td>
                  <td className="mono">{fmtNumber(row.cv_macro_f1)}</td>
                  <td className="mono">{fmtPercent(row.test_top1_accuracy)}</td>
                  <td className="mono">{fmtNumber(row.test_macro_f1)}</td>
                  <td className="mono">{fmtNumber(row.log_loss)}</td>
                  <td className="mono">{fmtPercent(row.route_continuity)}</td>
                  <td className="mono">
                    {fmtNumber(row.training_time_s, 2, ' s')}
                  </td>
                  <td className="mono">
                    {fmtNumber(row.inference_time_s, 3, ' s')}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Comparison Charts */}
      <section className="card">
        <div className="card-header">
          <div className="card-title-row">
            <BarChart3 size={17} color="#1d4ed8" />
            <div>
              <h3 className="card-title">Comparison Charts</h3>
              <p className="card-subtitle">
                Visual comparison generated strictly from measured experimental
                outputs
              </p>
            </div>
          </div>
        </div>

        <div className="card-body">
          <div className="chart-grid">
            <ComparisonChart
              title="Cross-Validation Macro F1-Score"
              subtitle="Trip-grouped CV across 80 training trips"
              items={cvF1Items}
              maxValue={1}
            />
            <ComparisonChart
              title="Student-Test Observation Top-1 Accuracy"
              subtitle="Highest-scoring candidate accuracy across 10 student_test trips"
              items={top1Items}
              maxValue={1}
              unit="%"
            />
            <ComparisonChart
              title="Student-Test Macro F1-Score"
              subtitle="Candidate classification quality on unseen student_test trips"
              items={testF1Items}
              maxValue={1}
            />
            <ComparisonChart
              title="Route Continuity"
              subtitle="Fraction of consecutive predicted arcs connected in the road graph"
              items={continuityItems}
              maxValue={1}
              unit="%"
            />
          </div>

          <div className="callout" style={{ marginTop: 18 }}>
            <Info size={16} style={{ flexShrink: 0, marginTop: 2 }} />
            <span>
              <strong>Methodological Note:</strong> Epoch-wise training and
              validation loss curves are recorded exclusively for iterative neural
              models (MLP) and are never artificially generated for non-iterative
              models such as KNN, in compliance with PathSnap deliverable
              guidelines.
            </span>
          </div>
        </div>
      </section>
    </>
  )
}
