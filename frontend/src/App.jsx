import { Suspense, lazy } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { SystemProvider } from './context/SystemProvider'
import Layout from './components/Layout'
import DashboardPage from './pages/DashboardPage'
import ModelComparisonPage from './pages/ModelComparisonPage'
import SystemInfoPage from './pages/SystemInfoPage'
import './styles/layout.css'

const RouteAnalysisPage = lazy(() => import('./pages/RouteAnalysisPage'))

export default function App() {
  return (
    <SystemProvider>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<DashboardPage />} />
          <Route
            path="route-analysis"
            element={
              <Suspense
                fallback={
                  <div className="card">
                    <div className="card-body">Loading Route Analysis…</div>
                  </div>
                }
              >
                <RouteAnalysisPage />
              </Suspense>
            }
          />
          <Route path="model-comparison" element={<ModelComparisonPage />} />
          <Route path="system-info" element={<SystemInfoPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </SystemProvider>
  )
}
