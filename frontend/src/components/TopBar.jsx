import { Menu, RefreshCw } from 'lucide-react'
import { useSystem } from '../context/systemContextValue'
import StatusBadge from './StatusBadge'

export default function TopBar({ title, subtitle, onMenuClick }) {
  const { health, loading, refreshSystemStatus } = useSystem()

  const badgeVariant = loading
    ? 'neutral'
    : health.connected
      ? 'success'
      : 'error'

  const badgeLabel = loading
    ? 'Checking Backend…'
    : health.connected
      ? `API Connected${health.latencyMs !== null ? ` (${health.latencyMs} ms)` : ''}`
      : 'Backend Offline'

  return (
    <header className="topbar">
      <div className="topbar-left">
        <button
          type="button"
          className="mobile-menu-btn"
          onClick={onMenuClick}
          aria-label="Toggle navigation menu"
        >
          <Menu size={19} />
        </button>
        <div className="topbar-title-group">
          <h2>{title}</h2>
          {subtitle && <p>{subtitle}</p>}
        </div>
      </div>

      <div className="topbar-right">
        <StatusBadge variant={badgeVariant} label={badgeLabel} />
        <button
          type="button"
          className="btn btn-secondary"
          onClick={refreshSystemStatus}
          disabled={loading}
          title="Re-check FastAPI backend status"
        >
          <RefreshCw size={14} className={loading ? 'spin' : ''} />
          <span>Refresh</span>
        </button>
      </div>
    </header>
  )
}
