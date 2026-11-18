"""Interactive Folium map visualizations for PathSnap.

Generates multi-layer geographical maps illustrating:
- Raw GPS observation points and trajectory paths
- Local road-network segments
- Road-segment candidate sets with distance annotations
- Ground-truth reference routes (where permitted)
- Anomaly callouts for zero-candidate observations
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Set

import folium
import numpy as np

from pathsnap.candidates import ObservationCandidates
from pathsnap.geometry import haversine_distance
from pathsnap.preprocess import TripData


def create_trip_folium_map(
    trip: TripData,
    obs_candidates: List[ObservationCandidates],
    max_network_arcs_to_render: int = 5000,
    show_candidates_for_sample_obs: int = 150,
    output_html_path: Optional[Path] = None,
) -> folium.Map:
    """Generate an interactive Folium map for one trip."""
    track_pts = [(point[1], point[0]) for point in trip.track]  # (lat, lon) for Folium
    if not track_pts:
        raise ValueError(f"Trip {trip.trip_id} has no track points.")

    center_lat = float(np.mean([p[0] for p in track_pts]))
    center_lon = float(np.mean([p[1] for p in track_pts]))

    # Initialize map with OpenStreetMap tiles
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # 1. FeatureGroup: Road Network (Local buffer around trajectory)
    fg_network = folium.FeatureGroup(name="Road Network (Nearby Arcs)", show=True)
    lats = [p[0] for p in track_pts]
    lons = [p[1] for p in track_pts]
    min_lat, max_lat = min(lats) - 0.01, max(lats) + 0.01
    min_lon, max_lon = min(lons) - 0.015, max(lons) + 0.015

    rendered_arcs = 0
    candidate_arc_ids: Set[int] = set()
    for oc in obs_candidates:
        for c in oc.candidates:
            candidate_arc_ids.add(c.candidate_arc_id)

    nodes = trip.nodes
    arcs = trip.arcs

    for arc_id, (u, v) in enumerate(arcs):
        if arc_id in candidate_arc_ids:
            continue  # Candidates rendered in their own dedicated layer
        u_lon, u_lat = nodes[u]
        v_lon, v_lat = nodes[v]
        if (
            min_lat <= u_lat <= max_lat
            and min_lon <= u_lon <= max_lon
            and min_lat <= v_lat <= max_lat
            and min_lon <= v_lon <= max_lon
        ):
            folium.PolyLine(
                locations=[(u_lat, u_lon), (v_lat, v_lon)],
                color="#888888",
                weight=1.5,
                opacity=0.4,
                tooltip=f"Arc #{arc_id} ({u}->{v})",
            ).add_to(fg_network)
            rendered_arcs += 1
            if rendered_arcs >= max_network_arcs_to_render:
                break
    fg_network.add_to(m)

    # 2. FeatureGroup: Ground-Truth Reference Route (Permitted dev partitions only)
    if trip.route:
        fg_route = folium.FeatureGroup(name="Ground-Truth Reference Route", show=True)
        for route_idx, arc_id in enumerate(trip.route):
            u, v = arcs[arc_id]
            u_lon, u_lat = nodes[u]
            v_lon, v_lat = nodes[v]
            folium.PolyLine(
                locations=[(u_lat, u_lon), (v_lat, v_lon)],
                color="#198754",  # Green
                weight=4.0,
                opacity=0.85,
                tooltip=f"Route Step #{route_idx}: Arc #{arc_id}",
            ).add_to(fg_route)
        fg_route.add_to(m)

    # 3. FeatureGroup: Candidate Segments
    fg_candidates = folium.FeatureGroup(name="Generated Candidate Arcs", show=True)
    route_set = set(trip.route) if trip.route else set()
    for arc_id in candidate_arc_ids:
        u, v = arcs[arc_id]
        u_lon, u_lat = nodes[u]
        v_lon, v_lat = nodes[v]
        is_in_route = arc_id in route_set
        color = "#fd7e14" if not is_in_route else "#20c997"
        folium.PolyLine(
            locations=[(u_lat, u_lon), (v_lat, v_lon)],
            color=color,
            weight=3.0,
            opacity=0.8,
            tooltip=f"Candidate Arc #{arc_id} (In Route: {is_in_route})",
        ).add_to(fg_candidates)
    fg_candidates.add_to(m)

    # 4. FeatureGroup: GPS Trajectory Path
    fg_traj = folium.FeatureGroup(name="GPS Trajectory Line", show=True)
    folium.PolyLine(
        locations=track_pts,
        color="#0d6efd",
        weight=2.0,
        dash_array="4, 4",
        opacity=0.7,
    ).add_to(fg_traj)
    fg_traj.add_to(m)

    # 5. FeatureGroup: GPS Observation Markers
    fg_obs = folium.FeatureGroup(name="GPS Observations", show=True)
    step_sample = max(1, len(obs_candidates) // show_candidates_for_sample_obs)

    for i in range(0, len(obs_candidates), step_sample):
        oc = obs_candidates[i]
        lon, lat = oc.longitude, oc.latitude
        popup_html = (
            f"<b>Observation #{oc.observation_id}</b><br>"
            f"Time: {oc.timestamp:.1f}s<br>"
            f"Lat/Lon: {lat:.6f}, {lon:.6f}<br>"
            f"Candidates found: {oc.candidate_count}<br>"
        )
        if oc.has_candidates:
            nearest = oc.candidates[0]
            popup_html += (
                f"Nearest Arc: #{nearest.candidate_arc_id} ({nearest.distance_m:.1f}m)"
            )
            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color="#0d6efd",
                fill=True,
                fill_color="#0d6efd",
                fill_opacity=0.8,
                popup=folium.Popup(popup_html, max_width=300),
            ).add_to(fg_obs)
    fg_obs.add_to(m)

    # 6. FeatureGroup: Zero-Candidate Anomalies
    fg_zeros = folium.FeatureGroup(name="Zero-Candidate Observations", show=True)
    zero_count = 0
    for oc in obs_candidates:
        if not oc.has_candidates:
            zero_count += 1
            popup_html = (
                f"<b style='color:red;'>ZERO CANDIDATES</b><br>"
                f"Observation #{oc.observation_id}<br>"
                f"Time: {oc.timestamp:.1f}s<br>"
                f"No road segment within search radius."
            )
            folium.CircleMarker(
                location=[oc.latitude, oc.longitude],
                radius=8,
                color="#dc3545",
                fill=True,
                fill_color="#dc3545",
                fill_opacity=1.0,
                popup=folium.Popup(popup_html, max_width=300),
            ).add_to(fg_zeros)
    if zero_count > 0:
        fg_zeros.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    if output_html_path:
        output_html_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_html_path))

    return m
