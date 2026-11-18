"""Baseline candidate generation + features (stand-in for the shared pipeline).

For every GPS observation, candidates are all road arcs within `radius_m`
(nearest `max_candidates` kept). Features use ONLY the GPS track and the road
network. The reference route is never used here, so there is no label leakage;
labels are attached afterwards from `pathsnap.labels`.

This exists so the Person 1 models can be run end to end now. Person 2's
candidate generation and Person 3's feature engineering replace it once ready.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from pathsnap.geo import pairs_point_segment_m, project_local
from pathsnap.preprocess import TripData

FEATURE_COLUMNS = [
    "dist_m", "dist_rel_m", "dist_rank", "n_candidates", "arc_length_m",
    "along_fraction", "heading_cos", "moving", "speed_ms",
]


def _motion(points: np.ndarray, ts: np.ndarray, max_dt: float = 5.0):
    """Per-observation motion vector and speed from the neighbouring fix."""
    n = len(points)
    vec = np.zeros((n, 2))
    dt = np.ones(n)
    if n > 1:
        fwd = points[1:] - points[:-1]
        dts = ts[1:] - ts[:-1]
        vec[:-1], dt[:-1] = fwd, dts
        # last point (or one followed by a gap) uses the previous fix instead
        use_back = np.r_[dts > max_dt, True]
        vec[use_back], dt[use_back] = (
            np.vstack([fwd[:1] * 0, fwd])[use_back],
            np.r_[1.0, dts][use_back],
        )
    speed = np.hypot(vec[:, 0], vec[:, 1]) / np.where(dt > 0, dt, 1.0)
    return vec, speed


def build_trip_candidates(
    trip: TripData, radius_m: float = 50.0, max_candidates: int = 20, spacing_m: float = 10.0
) -> pd.DataFrame:
    track = np.asarray(trip.track, dtype=float)
    nodes = np.asarray(trip.nodes, dtype=float)
    arcs = np.asarray(trip.arcs, dtype=int)
    lat0 = float(track[:, 1].mean())
    pts = project_local(track[:, 0], track[:, 1], lat0)
    nxy = project_local(nodes[:, 0], nodes[:, 1], lat0)
    a_xy, b_xy = nxy[arcs[:, 0]], nxy[arcs[:, 1]]
    length = np.hypot(*(b_xy - a_xy).T)

    # Densify arcs so a KD-tree can find every arc passing near a GPS point.
    counts = np.maximum(2, np.ceil(length / spacing_m).astype(int) + 1)
    arc_of_dense = np.repeat(np.arange(len(arcs)), counts)
    offset = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    frac = offset / np.repeat(counts - 1, counts)
    dense = a_xy[arc_of_dense] + frac[:, None] * (b_xy - a_xy)[arc_of_dense]

    coo = cKDTree(pts).sparse_distance_matrix(
        cKDTree(dense), radius_m + spacing_m / 2, output_type="coo_matrix"
    )
    obs = coo.row.astype(np.int64)
    arc = arc_of_dense[coo.col].astype(np.int64)
    key = np.unique(obs * len(arcs) + arc)
    obs, arc = key // len(arcs), key % len(arcs)

    dist, t = pairs_point_segment_m(pts[obs], a_xy[arc], b_xy[arc])
    keep = dist <= radius_m
    obs, arc, dist, t = obs[keep], arc[keep], dist[keep], t[keep]

    order = np.lexsort((dist, obs))
    obs, arc, dist, t = obs[order], arc[order], dist[order], t[order]
    first = np.r_[True, obs[1:] != obs[:-1]]
    group_start = np.maximum.accumulate(np.where(first, np.arange(len(obs)), 0))
    rank = np.arange(len(obs)) - group_start
    keep = rank < max_candidates
    obs, arc, dist, t, rank = obs[keep], arc[keep], dist[keep], t[keep], rank[keep]

    n_cand = np.bincount(obs, minlength=len(pts))[obs]
    min_dist = np.minimum.reduceat(dist, np.flatnonzero(np.r_[True, obs[1:] != obs[:-1]]))
    first_idx = np.cumsum(np.r_[True, obs[1:] != obs[:-1]]) - 1
    dist_rel = dist - min_dist[first_idx]

    vec, speed = _motion(pts, track[:, 2])
    moving = speed[obs] > 1.0
    arc_vec = (b_xy - a_xy)[arc]
    denom = np.hypot(*arc_vec.T) * np.hypot(*vec[obs].T)
    cos = np.where(denom > 0, np.einsum("ij,ij->i", arc_vec, vec[obs]) / np.where(denom > 0, denom, 1), 0.0)
    heading_cos = np.where(moving, cos, 0.0)

    return pd.DataFrame(
        {
            "trip_id": trip.trip_id,
            "observation_id": obs,
            "candidate_arc_id": arc,
            "dist_m": dist,
            "dist_rel_m": dist_rel,
            "dist_rank": rank,
            "n_candidates": n_cand,
            "arc_length_m": length[arc],
            "along_fraction": t,
            "heading_cos": heading_cos,
            "moving": moving.astype(int),
            "speed_ms": speed[obs],
        }
    )
