import { BarChart3 } from 'lucide-react'
import EmptyState from './EmptyState'

const BAR_COLORS = ['#1d4ed8', '#0f766e', '#7c3aed', '#b45309', '#be123c']

export default function ComparisonChart({
  title,
  subtitle,
  items = [],
  maxValue = 1,
  unit = '',
  decimals = 4,
}) {
  const validItems = items.filter(
    (item) => typeof item.value === 'number' && Number.isFinite(item.value)
  )

  const computedMax =
    maxValue === 'auto'
      ? Math.max(...validItems.map((item) => item.value), 1e-6)
      : maxValue

  return (
    <div className="chart-box">
      <div className="chart-box-title">{title}</div>
      {subtitle && <div className="chart-box-sub">{subtitle}</div>}

      {validItems.length === 0 ? (
        <EmptyState
          icon={BarChart3}
          title="No Recorded Metrics Yet"
          description="This chart populates automatically once genuine KNN / MLP evaluation results are served by the backend."
        />
      ) : (
        <div>
          {validItems.map((item, idx) => {
            const pct = Math.max(
              0,
              Math.min(100, (item.value / computedMax) * 100)
            )
            const formatted =
              unit === '%'
                ? `${(item.value * 100).toFixed(2)}%`
                : `${item.value.toFixed(decimals)}${unit ? ` ${unit}` : ''}`
            return (
              <div className="bar-row" key={item.label}>
                <div className="bar-row-head">
                  <span>{item.label}</span>
                  <span className="mono">{formatted}</span>
                </div>
                <div className="bar-track">
                  <div
                    className="bar-fill"
                    style={{
                      width: `${pct}%`,
                      backgroundColor: BAR_COLORS[idx % BAR_COLORS.length],
                    }}
                  />
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
