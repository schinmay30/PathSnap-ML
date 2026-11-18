import { useEffect, useMemo, useRef, useState } from 'react'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { MapPin } from 'lucide-react'

function normalizeLatLngPoint(pt) {
  if (!pt) return null
  if (Array.isArray(pt) && pt.length >= 2) {
    const lat = Number(pt[0])
    const lon = Number(pt[1])
    if (Number.isFinite(lat) && Number.isFinite(lon)) return [lat, lon]
  }
  if (typeof pt === 'object') {
    const lat = Number(pt.lat ?? pt.latitude)
    const lon = Number(pt.lon ?? pt.lng ?? pt.longitude)
    if (Number.isFinite(lat) && Number.isFinite(lon)) return [lat, lon]
  }
  return null
}

function normalizeSegments(segments) {
  if (!Array.isArray(segments)) return []
  return segments
    .map((seg) => {
      const points = Array.isArray(seg) ? seg : seg?.coordinates
      if (!Array.isArray(points)) return []
      const coords = points.map(normalizeLatLngPoint)
      // Dropping a bad point inside a component would bridge its neighbours.
      return coords.every(Boolean) ? coords : []
    })
    .filter((coords) => coords.length >= 2)
}

export default function RouteMap({
  gpsPoints = [],
  roadNetwork = [],
  predictedRoute = [],
  emptyTitle = 'No GPS Trajectory Loaded',
  emptyDescription = 'Select an available trip and run prediction to visualize raw GPS observations, road-network geometry, and the reconstructed route returned by the FastAPI backend.',
}) {
  const containerRef = useRef(null)
  const mapRef = useRef(null)
  const layersRef = useRef({
    road: null,
    gps: null,
    route: null,
  })

  const [showGps, setShowGps] = useState(true)
  const [showRoad, setShowRoad] = useState(true)
  const [showRoute, setShowRoute] = useState(true)

  const parsedGps = useMemo(
    () => (Array.isArray(gpsPoints) ? gpsPoints.map(normalizeLatLngPoint).filter(Boolean) : []),
    [gpsPoints]
  )
  const parsedRoad = useMemo(() => normalizeSegments(roadNetwork), [roadNetwork])
  const parsedRoute = useMemo(() => normalizeSegments(predictedRoute), [predictedRoute])

  const hasGeometry =
    parsedGps.length > 0 || parsedRoad.length > 0 || parsedRoute.length > 0

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return

    const map = L.map(containerRef.current, {
      center: [20, 0],
      zoom: 2,
      zoomControl: true,
      attributionControl: true,
    })

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors',
    }).addTo(map)

    layersRef.current = {
      road: L.layerGroup().addTo(map),
      gps: L.layerGroup().addTo(map),
      route: L.layerGroup().addTo(map),
    }
    mapRef.current = map

    return () => {
      map.remove()
      mapRef.current = null
    }
  }, [])

  useEffect(() => {
    const map = mapRef.current
    const layers = layersRef.current
    if (!map || !layers.road || !layers.gps || !layers.route) return

    layers.road.clearLayers()
    layers.gps.clearLayers()
    layers.route.clearLayers()

    const boundsCoords = []

    if (showRoad && parsedRoad.length > 0) {
      parsedRoad.forEach((coords) => {
        L.polyline(coords, {
          color: '#94a3b8',
          weight: 2,
          opacity: 0.6,
        }).addTo(layers.road)
        coords.forEach((c) => boundsCoords.push(c))
      })
    }

    if (showGps && parsedGps.length > 0) {
      parsedGps.forEach((coord) => {
        L.circleMarker(coord, {
          radius: 3.5,
          color: '#1d4ed8',
          fillColor: '#2563eb',
          fillOpacity: 0.85,
          weight: 1,
        }).addTo(layers.gps)
        boundsCoords.push(coord)
      })
    }

    if (showRoute && parsedRoute.length > 0) {
      parsedRoute.forEach((coords) => {
        L.polyline(coords, {
          color: '#059669',
          weight: 4,
          opacity: 0.92,
        }).addTo(layers.route)
        coords.forEach((c) => boundsCoords.push(c))
      })
    }

    if (boundsCoords.length > 0) {
      const bounds = L.latLngBounds(boundsCoords)
      if (bounds.isValid()) {
        map.fitBounds(bounds, { padding: [32, 32], maxZoom: 16 })
      }
    }
  }, [parsedGps, parsedRoad, parsedRoute, showGps, showRoad, showRoute])

  return (
    <div className="map-shell">
      <div className="map-toolbar">
        <div className="map-legend" aria-label="Map layer legend">
          <span className="legend-item">
            <span className="legend-swatch legend-swatch-gps" />
            Raw GPS Points ({parsedGps.length})
          </span>
          <span className="legend-item">
            <span className="legend-swatch legend-swatch-road" />
            Road Network ({parsedRoad.length})
          </span>
          <span className="legend-item">
            <span className="legend-swatch legend-swatch-route" />
            Predicted Route ({parsedRoute.length})
          </span>
        </div>

        <div className="map-toggles">
          <label className="map-toggle-label">
            <input
              type="checkbox"
              checked={showGps}
              onChange={(e) => setShowGps(e.target.checked)}
            />
            GPS
          </label>
          <label className="map-toggle-label">
            <input
              type="checkbox"
              checked={showRoad}
              onChange={(e) => setShowRoad(e.target.checked)}
            />
            Road Network
          </label>
          <label className="map-toggle-label">
            <input
              type="checkbox"
              checked={showRoute}
              onChange={(e) => setShowRoute(e.target.checked)}
            />
            Predicted Route
          </label>
        </div>
      </div>

      <div className="map-stage">
        <div
          ref={containerRef}
          className="leaflet-canvas"
          aria-label="Interactive GPS route map"
        />
        {!hasGeometry && (
          <div className="map-empty-overlay">
            <div className="map-empty-card">
              <div className="empty-state-icon" aria-hidden="true">
                <MapPin size={20} />
              </div>
              <div className="empty-state-title">{emptyTitle}</div>
              <p className="empty-state-desc">{emptyDescription}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
