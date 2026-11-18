export default function EmptyState({
  icon: Icon,
  title,
  description,
  action = null,
}) {
  return (
    <div className="empty-state" role="status">
      {Icon && (
        <div className="empty-state-icon" aria-hidden="true">
          <Icon size={20} />
        </div>
      )}
      <div className="empty-state-title">{title}</div>
      {description && <p className="empty-state-desc">{description}</p>}
      {action && <div style={{ marginTop: 6 }}>{action}</div>}
    </div>
  )
}
