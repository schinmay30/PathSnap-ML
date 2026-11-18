export default function StatusBadge({ variant = 'neutral', label, showDot = true }) {
  return (
    <span className={`status-badge status-${variant}`}>
      {showDot && <span className="status-dot" aria-hidden="true" />}
      {label}
    </span>
  )
}
