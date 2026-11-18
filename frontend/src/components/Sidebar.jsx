import { NavLink } from 'react-router-dom'
import {
  BarChart3,
  Compass,
  LayoutDashboard,
  Map,
  Server,
} from 'lucide-react'

const NAV_ITEMS = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard, end: true },
  { to: '/route-analysis', label: 'Route Analysis', icon: Map },
  { to: '/model-comparison', label: 'Model Comparison', icon: BarChart3 },
  { to: '/system-info', label: 'System Information', icon: Server },
]

export default function Sidebar({ isOpen, onClose }) {
  return (
    <>
      <div
        className={`sidebar-backdrop ${isOpen ? 'open' : ''}`}
        onClick={onClose}
        aria-hidden="true"
      />
      <aside className={`sidebar ${isOpen ? 'open' : ''}`} aria-label="Main navigation">
        <div className="sidebar-brand">
          <div className="sidebar-logo" aria-hidden="true">
            <Compass size={21} />
          </div>
          <div className="sidebar-brand-text">
            <h1>PathSnap</h1>
            <span>Map-Matching ML Suite</span>
          </div>
        </div>

        <nav className="sidebar-nav">
          <div className="sidebar-section-label">Workspace</div>
          {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              onClick={onClose}
              className={({ isActive }) =>
                `sidebar-link ${isActive ? 'active' : ''}`
              }
            >
              <Icon size={17} aria-hidden="true" />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="sidebar-footer-title">PES University · UE24CS352A</div>
          <div className="sidebar-footer-meta">
            Distributed Map-Matching &amp; Route Reconstruction
            <br />
            Laptop 2 · Evaluator Interface
          </div>
        </div>
      </aside>
    </>
  )
}
