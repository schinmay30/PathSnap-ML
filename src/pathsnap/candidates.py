"""Road-segment candidate generation for PathSnap.

Constructs spatial road-network geometries, computes exact metric distances from
GPS observations to directed road segments using local metric projection, and
generates ranked candidate road segments for each GPS observation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from pathsnap.geometry import (
    angle_difference_degrees,
    bearing_degrees,
    haversine_distance,
    point_to_segment_projection,
    project_local,
)
from pathsnap.preprocess import TripData


@dataclass(frozen=True)
class Candidate:
    """A road segment candidate for a specific GPS observation."""

    observation_id: int
    candidate_arc_id: int
    distance_m: float
    projection_t: float
    segment_length_m: float
    heading_deg: float
    from_node_id: int
    to_node_id: int


@dataclass(frozen=True)
class ObservationCandidates:
    """All candidates generated for a single GPS observation."""

    observation_id: int
    longitude: float
    latitude: float
    timestamp: float
    candidates: List[Candidate]

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def has_candidates(self) -> bool:
        return len(self.candidates) > 0


class RoadNetworkIndex:
    """Spatial index over a trip's road network using local metric projection."""

    def __init__(
        self,
        nodes: List[Tuple[float, float]],
        arcs: List[Tuple[int, int]],
        sample_step_m: float = 25.0,
    ) -> None:
        if not nodes:
            raise ValueError("Road network nodes list cannot be empty.")
        if not arcs:
            raise ValueError("Road network arcs list cannot be empty.")

        self.num_nodes = len(nodes)
        self.num_arcs = len(arcs)
        self.sample_step_m = sample_step_m

        # 1. Local metric projection origin (network centroid)
        nodes_arr = np.array(nodes, dtype=float)
        self.lon0 = float(np.mean(nodes_arr[:, 0]))
        self.lat0 = float(np.mean(nodes_arr[:, 1]))

        # Project nodes to local meters
        self.node_x, self.node_y = project_local(
            nodes_arr[:, 0], nodes_arr[:, 1], self.lon0, self.lat0
        )

        # 2. Extract directed arc geometries
        arcs_arr = np.array(arcs, dtype=int)
        self.arcs = arcs
        self.u_idx = arcs_arr[:, 0]
        self.v_idx = arcs_arr[:, 1]

        self.ax = self.node_x[self.u_idx]
        self.ay = self.node_y[self.u_idx]
        self.bx = self.node_x[self.v_idx]
        self.by = self.node_y[self.v_idx]

        self.vx = self.bx - self.ax
        self.vy = self.by - self.ay
        self.lengths = np.hypot(self.vx, self.vy)
        self.lens_sq = np.where(self.lengths == 0.0, 1.0, self.vx**2 + self.vy**2)

        # Arc headings in degrees [0, 360)
        self.headings = (np.degrees(np.arctan2(self.vx, self.vy)) + 360.0) % 360.0

        # 3. Discretize segments into spatial index sample points
        # Long segments (e.g. up to 3.5 km) are subdivided with sample_step_m <= 25m
        # so they are never missed by a radius query.
        sample_x: List[float] = []
        sample_y: List[float] = []
        sample_arc_ids: List[int] = []

        for arc_id in range(self.num_arcs):
            length = self.lengths[arc_id]
            if length <= sample_step_m:
                # Store segment midpoint
                sample_x.append(0.5 * (self.ax[arc_id] + self.bx[arc_id]))
                sample_y.append(0.5 * (self.ay[arc_id] + self.by[arc_id]))
                sample_arc_ids.append(arc_id)
            else:
                num_samples = int(math.ceil(length / sample_step_m)) + 1
                alphas = np.linspace(0.0, 1.0, num_samples)
                for a in alphas:
                    sample_x.append(self.ax[arc_id] + a * self.vx[arc_id])
                    sample_y.append(self.ay[arc_id] + a * self.vy[arc_id])
                    sample_arc_ids.append(arc_id)

        self._sample_arc_ids = np.array(sample_arc_ids, dtype=int)
        self._tree = cKDTree(np.column_stack([sample_x, sample_y]))

    def query_observation(
        self,
        observation_id: int,
        longitude: float,
        latitude: float,
        radius_m: float = 50.0,
        max_candidates: int = 10,
    ) -> List[Candidate]:
        """Find road-segment candidates within radius_m of a single GPS observation."""
        px, py = project_local(longitude, latitude, self.lon0, self.lat0)
        search_radius = radius_m + (0.5 * self.sample_step_m)

        neighbor_indices = self._tree.query_ball_point([px, py], r=search_radius)
        if not neighbor_indices:
            return []

        cand_arcs = np.unique(self._sample_arc_ids[neighbor_indices])
        ax_sub = self.ax[cand_arcs]
        ay_sub = self.ay[cand_arcs]
        vx_sub = self.vx[cand_arcs]
        vy_sub = self.vy[cand_arcs]
        lens_sq_sub = self.lens_sq[cand_arcs]

        # Vectorized projection
        t = ((px - ax_sub) * vx_sub + (py - ay_sub) * vy_sub) / lens_sq_sub
        t_clamped = np.clip(t, 0.0, 1.0)
        qx = ax_sub + t_clamped * vx_sub
        qy = ay_sub + t_clamped * vy_sub
        dist = np.hypot(px - qx, py - qy)

        # Filter within exact radius
        within_radius = dist <= radius_m
        valid_arcs = cand_arcs[within_radius]
        valid_dist = dist[within_radius]
        valid_t = t_clamped[within_radius]

        if len(valid_arcs) == 0:
            return []

        # Sort ascending by distance
        sort_order = np.argsort(valid_dist)[:max_candidates]
        candidates: List[Candidate] = []
        for idx in sort_order:
            arc_id = int(valid_arcs[idx])
            candidates.append(
                Candidate(
                    observation_id=observation_id,
                    candidate_arc_id=arc_id,
                    distance_m=float(valid_dist[idx]),
                    projection_t=float(valid_t[idx]),
                    segment_length_m=float(self.lengths[arc_id]),
                    heading_deg=float(self.headings[arc_id]),
                    from_node_id=int(self.u_idx[arc_id]),
                    to_node_id=int(self.v_idx[arc_id]),
                )
            )
        return candidates

    def query_trip(
        self,
        trip: TripData,
        radius_m: float = 50.0,
        max_candidates: int = 10,
    ) -> List[ObservationCandidates]:
        """Vectorized query for all GPS observations in a trip.

        Preserves observation ordering and explicitly retains zero-candidate observations.
        """
        track_arr = np.array(trip.track, dtype=float)
        num_obs = len(track_arr)
        px_all, py_all = project_local(
            track_arr[:, 0], track_arr[:, 1], self.lon0, self.lat0
        )
        search_radius = radius_m + (0.5 * self.sample_step_m)

        pts = np.column_stack([px_all, py_all])
        all_neighbors = self._tree.query_ball_point(pts, r=search_radius)

        results: List[ObservationCandidates] = []
        for obs_id in range(num_obs):
            nbrs = all_neighbors[obs_id]
            lon = float(track_arr[obs_id, 0])
            lat = float(track_arr[obs_id, 1])
            ts = float(track_arr[obs_id, 2])

            if not nbrs:
                results.append(
                    ObservationCandidates(
                        observation_id=obs_id,
                        longitude=lon,
                        latitude=lat,
                        timestamp=ts,
                        candidates=[],
                    )
                )
                continue

            cand_arcs = np.unique(self._sample_arc_ids[nbrs])
            px = px_all[obs_id]
            py = py_all[obs_id]

            ax_sub = self.ax[cand_arcs]
            ay_sub = self.ay[cand_arcs]
            vx_sub = self.vx[cand_arcs]
            vy_sub = self.vy[cand_arcs]
            lens_sq_sub = self.lens_sq[cand_arcs]

            t = ((px - ax_sub) * vx_sub + (py - ay_sub) * vy_sub) / lens_sq_sub
            t_clamped = np.clip(t, 0.0, 1.0)
            qx = ax_sub + t_clamped * vx_sub
            qy = ay_sub + t_clamped * vy_sub
            dist = np.hypot(px - qx, py - qy)

            within_radius = dist <= radius_m
            valid_arcs = cand_arcs[within_radius]
            valid_dist = dist[within_radius]
            valid_t = t_clamped[within_radius]

            if len(valid_arcs) == 0:
                results.append(
                    ObservationCandidates(
                        observation_id=obs_id,
                        longitude=lon,
                        latitude=lat,
                        timestamp=ts,
                        candidates=[],
                    )
                )
                continue

            sort_order = np.argsort(valid_dist)[:max_candidates]
            cands: List[Candidate] = []
            for idx in sort_order:
                arc_id = int(valid_arcs[idx])
                cands.append(
                    Candidate(
                        observation_id=obs_id,
                        candidate_arc_id=arc_id,
                        distance_m=float(valid_dist[idx]),
                        projection_t=float(valid_t[idx]),
                        segment_length_m=float(self.lengths[arc_id]),
                        heading_deg=float(self.headings[arc_id]),
                        from_node_id=int(self.u_idx[arc_id]),
                        to_node_id=int(self.v_idx[arc_id]),
                    )
                )
            results.append(
                ObservationCandidates(
                    observation_id=obs_id,
                    longitude=lon,
                    latitude=lat,
                    timestamp=ts,
                    candidates=cands,
                )
            )

        return results


def generate_candidate_records_for_trip(
    trip: TripData,
    partition: str,
    radius_m: float = 50.0,
    max_candidates: int = 10,
    include_zero_candidates: bool = True,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Generate candidate records for a trip.

    Returns:
        (candidate_records, zero_candidate_records)
    """
    index = RoadNetworkIndex(trip.nodes, trip.arcs)
    obs_candidates = index.query_trip(
        trip, radius_m=radius_m, max_candidates=max_candidates
    )

    candidate_records: List[Dict[str, object]] = []
    zero_candidate_records: List[Dict[str, object]] = []

    for oc in obs_candidates:
        if oc.candidate_count == 0:
            zero_candidate_records.append(
                {
                    "partition": partition,
                    "trip_id": trip.trip_id,
                    "observation_id": oc.observation_id,
                    "longitude": oc.longitude,
                    "latitude": oc.latitude,
                    "timestamp": oc.timestamp,
                }
            )
            if include_zero_candidates:
                candidate_records.append(
                    {
                        "partition": partition,
                        "trip_id": trip.trip_id,
                        "observation_id": oc.observation_id,
                        "candidate_arc_id": -1,
                        "distance_m": float("nan"),
                        "projection_t": float("nan"),
                        "segment_length_m": float("nan"),
                        "heading_deg": float("nan"),
                        "from_node_id": -1,
                        "to_node_id": -1,
                    }
                )
        else:
            for c in oc.candidates:
                candidate_records.append(
                    {
                        "partition": partition,
                        "trip_id": trip.trip_id,
                        "observation_id": c.observation_id,
                        "candidate_arc_id": c.candidate_arc_id,
                        "distance_m": c.distance_m,
                        "projection_t": c.projection_t,
                        "segment_length_m": c.segment_length_m,
                        "heading_deg": c.heading_deg,
                        "from_node_id": c.from_node_id,
                        "to_node_id": c.to_node_id,
                    }
                )

    return candidate_records, zero_candidate_records


def align_route_to_observations(
    trip: TripData,
    search_window: int = 15,
) -> List[Optional[int]]:
    """Align GPS observations to the ground-truth reference route.

    Uses a forward-progressing search window over the chronological sequence
    of route arcs. For each observation, identifies the active route arc minimizing
    orthogonal distance.

    Returns:
        List of reference arc_id values for each observation index (or None if unaligned).
    """
    if not trip.route or not trip.track:
        return [None] * len(trip.track)

    nodes = np.array(trip.nodes, dtype=float)
    arcs = np.array(trip.arcs, dtype=int)
    route = np.array(trip.route, dtype=int)
    track = np.array(trip.track, dtype=float)

    lon0 = float(np.mean(nodes[:, 0]))
    lat0 = float(np.mean(nodes[:, 1]))

    node_x, node_y = project_local(nodes[:, 0], nodes[:, 1], lon0, lat0)
    track_x, track_y = project_local(track[:, 0], track[:, 1], lon0, lat0)

    u_r = arcs[route, 0]
    v_r = arcs[route, 1]
    Ax = node_x[u_r]
    Ay = node_y[u_r]
    Bx = node_x[v_r]
    By = node_y[v_r]
    Vx = Bx - Ax
    Vy = By - Ay
    L_sq = np.where(Vx**2 + Vy**2 == 0.0, 1.0, Vx**2 + Vy**2)

    num_obs = len(track)
    num_route = len(route)
    aligned_arcs: List[Optional[int]] = []

    curr_k = 0
    for i in range(num_obs):
        px = track_x[i]
        py = track_y[i]

        k_start = max(0, curr_k - 2)
        k_end = min(num_route, curr_k + search_window + 1)

        sub_Ax = Ax[k_start:k_end]
        sub_Ay = Ay[k_start:k_end]
        sub_Vx = Vx[k_start:k_end]
        sub_Vy = Vy[k_start:k_end]
        sub_Lsq = L_sq[k_start:k_end]

        t = np.clip(((px - sub_Ax) * sub_Vx + (py - sub_Ay) * sub_Vy) / sub_Lsq, 0.0, 1.0)
        qx = sub_Ax + t * sub_Vx
        qy = sub_Ay + t * sub_Vy
        dists = np.hypot(px - qx, py - qy)

        best_local_idx = int(np.argmin(dists))
        best_k = k_start + best_local_idx
        curr_k = best_k
        aligned_arcs.append(int(route[best_k]))

    return aligned_arcs


def extract_observation_features(
    trip: TripData,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate trajectory instantaneous speed (m/s) and heading (deg) for all GPS points."""
    track = trip.track
    num_obs = len(track)
    speeds = np.zeros(num_obs, dtype=float)
    headings = np.zeros(num_obs, dtype=float)

    if num_obs <= 1:
        return speeds, headings

    for i in range(num_obs - 1):
        p1 = track[i]
        p2 = track[i + 1]
        dist = haversine_distance(p1[0], p1[1], p2[0], p2[1])
        dt = max(0.001, p2[2] - p1[2])
        sp = dist / dt
        brg = bearing_degrees(p1[0], p1[1], p2[0], p2[1])
        speeds[i] = sp
        headings[i] = brg

    speeds[-1] = speeds[-2]
    headings[-1] = headings[-2]
    return speeds, headings


def generate_candidate_features_for_trip(
    trip: TripData,
    partition: str,
    radius_m: float = 50.0,
    max_candidates: int = 10,
    align_labels: bool = True,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Generate candidate feature rows matching the shared data contract.

    Produces:
    - partition, trip_id, observation_id, candidate_arc_id, label
    - 6 geometric features: distance_m, projection_t, segment_length_m,
      heading_diff_deg, speed_mps, log_distance
    """
    index = RoadNetworkIndex(trip.nodes, trip.arcs)
    obs_candidates = index.query_trip(
        trip, radius_m=radius_m, max_candidates=max_candidates
    )
    speeds, headings = extract_observation_features(trip)

    aligned_route_arcs = (
        align_route_to_observations(trip) if align_labels else [None] * len(trip.track)
    )

    feature_rows: List[Dict[str, object]] = []
    zero_obs_rows: List[Dict[str, object]] = []

    for oc in obs_candidates:
        obs_id = oc.observation_id
        if not oc.has_candidates:
            zero_obs_rows.append(
                {
                    "partition": partition,
                    "trip_id": trip.trip_id,
                    "observation_id": obs_id,
                    "timestamp": oc.timestamp,
                    "longitude": oc.longitude,
                    "latitude": oc.latitude,
                }
            )
            continue

        ref_arc = aligned_route_arcs[obs_id]
        obs_speed = speeds[obs_id]
        obs_heading = headings[obs_id]

        for c in oc.candidates:
            label = 1 if (ref_arc is not None and c.candidate_arc_id == ref_arc) else 0
            heading_diff = angle_difference_degrees(obs_heading, c.heading_deg)
            log_dist = float(math.log1p(c.distance_m))

            feature_rows.append(
                {
                    "partition": partition,
                    "trip_id": trip.trip_id,
                    "observation_id": obs_id,
                    "candidate_arc_id": c.candidate_arc_id,
                    "label": label,
                    "distance_m": round(c.distance_m, 4),
                    "projection_t": round(c.projection_t, 4),
                    "segment_length_m": round(c.segment_length_m, 4),
                    "heading_diff_deg": round(heading_diff, 2),
                    "speed_mps": round(obs_speed, 4),
                    "log_distance": round(log_dist, 4),
                }
            )

    return feature_rows, zero_obs_rows
