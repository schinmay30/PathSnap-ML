"""Geometry helpers: metric distances for lon/lat data.

Longitude/latitude are angles, so Euclidean distance in degrees is not a
distance. Each trip covers only a few kilometres, so we project it with a
local equirectangular projection centred on the trip: x = R*lon*cos(lat0),
y = R*lat. Over a few km this is accurate to well under a metre, which is
far below GPS noise (several metres). Haversine is used for step lengths.
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_000.0


def project_local(lon, lat, lat0: float) -> np.ndarray:
    """Project lon/lat (degrees) to local metres, shape (n, 2)."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    x = np.radians(lon) * EARTH_RADIUS_M * np.cos(np.radians(lat0))
    y = np.radians(lat) * EARTH_RADIUS_M
    return np.column_stack([x, y])


def haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Great-circle distance in metres (vectorised)."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def point_to_segments_m(point: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray) -> np.ndarray:
    """Distance from one point (2,) to many segments A->B (m, 2), in metres."""
    ab = seg_b - seg_a
    length_sq = np.einsum("ij,ij->i", ab, ab)
    safe = np.where(length_sq == 0.0, 1.0, length_sq)
    t = np.clip(np.einsum("ij,ij->i", point - seg_a, ab) / safe, 0.0, 1.0)
    t = np.where(length_sq == 0.0, 0.0, t)
    closest = seg_a + t[:, None] * ab
    diff = point - closest
    return np.sqrt(np.einsum("ij,ij->i", diff, diff))


def pairs_point_segment_m(points: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray):
    """Distance and projection fraction for aligned (point_k, segment_k) pairs.

    All three arrays have shape (k, 2). Returns (distance_m, t) with t in [0, 1]
    the position of the closest point along the segment A->B.
    """
    ab = seg_b - seg_a
    length_sq = np.einsum("ij,ij->i", ab, ab)
    safe = np.where(length_sq == 0.0, 1.0, length_sq)
    t = np.clip(np.einsum("ij,ij->i", points - seg_a, ab) / safe, 0.0, 1.0)
    t = np.where(length_sq == 0.0, 0.0, t)
    diff = points - (seg_a + t[:, None] * ab)
    return np.sqrt(np.einsum("ij,ij->i", diff, diff)), t
