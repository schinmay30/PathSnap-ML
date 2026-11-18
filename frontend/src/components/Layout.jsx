import { useState } from 'react'
import { Outlet, useLocation } from 'react-router-dom'
import Sidebar from './Sidebar'
import TopBar from './TopBar'

const ROUTE_META = {
  '/': {
    title: 'Dashboard',
    subtitle: 'PathSnap system overview, trip inventory, and live backend readiness',
  },
  '/route-analysis': {
    title: 'Route Analysis',
    subtitle: 'Select an evaluation trip, run live ML + HMM inference, and inspect reconstructed trajectories',
  },
  '/model-comparison': {
    title: 'Model Comparison',
    subtitle: 'Recorded KNN and MLP experimentation metrics and comparative charts',
  },
  '/system-info': {
    title: 'System Information',
    subtitle: 'Backend endpoint configuration, loaded model artifacts, and deployment architecture',
  },
}

export default function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const location = useLocation()
  const meta = ROUTE_META[location.pathname] || ROUTE_META['/']

  return (
    <div className="app-shell">
      <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      <div className="main-area">
        <TopBar
          title={meta.title}
          subtitle={meta.subtitle}
          onMenuClick={() => setSidebarOpen((prev) => !prev)}
        />
        <main className="page-content">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
