"""Geodesic and projected geometric calculations for PathSnap.

All distances are calculated in metric units (meters) using WGS-84 geodesic formulas
or local metric conformal projections. Direct calculations in raw longitude/latitude
degrees are strictly avoided because angular degree scaling varies with latitude.
"""

from __future__ import annotations

import math
from typing import Tuple
import numpy as np

# WGS-84 Ellipsoid Parameters
WGS84_A_METERS: float = 6378137.0  # Semi-major axis (equatorial radius)
WGS84_F: float = 1.0 / 298.257223563  # Flattening
WGS84_B_METERS: float = WGS84_A_METERS * (1.0 - WGS84_F)  # Semi-minor axis (polar radius)
EARTH_RADIUS_METERS: float = 6371000.0  # Standard mean spherical radius


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculate great-circle distance between two (lon, lat) points in meters."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return EARTH_RADIUS_METERS * c


def wgs84_geodesic_distance(
    lon1: float, lat1: float, lon2: float, lat2: float, max_iter: int = 200, tol: float = 1e-12
) -> float:
    """Calculate geodesic distance on the WGS-84 ellipsoid using Vincenty's inverse formula.

    Falls back to Haversine distance in nearly-antipodal or non-convergent cases.
    """
    if abs(lat1 - lat2) < 1e-11 and abs(lon1 - lon2) < 1e-11:
        return 0.0

    a = WGS84_A_METERS
    b = WGS84_B_METERS
    f = WGS84_F

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    u1 = math.atan((1.0 - f) * math.tan(phi1))
    u2 = math.atan((1.0 - f) * math.tan(phi2))
    lon_diff = math.radians(lon2 - lon1)

    sin_u1, cos_u1 = math.sin(u1), math.cos(u1)
    sin_u2, cos_u2 = math.sin(u2), math.cos(u2)

    lam = lon_diff
    for _ in range(max_iter):
        sin_lam = math.sin(lam)
        cos_lam = math.cos(lam)
        sin_sigma = math.hypot(
            cos_u2 * sin_lam,
            cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lam,
        )
        if sin_sigma == 0.0:
            return 0.0  # Coincident points

        cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lam
        sigma = math.atan2(sin_sigma, cos_sigma)

        sin_alpha = (cos_u1 * cos_u2 * sin_lam) / sin_sigma
        cos2_alpha = 1.0 - sin_alpha**2
        cos_2sigma_m = (
            cos_sigma - (2.0 * sin_u1 * sin_u2 / cos2_alpha)
            if cos2_alpha != 0.0
            else 0.0
        )

        c = (f / 16.0) * cos2_alpha * (4.0 + f * (4.0 - 3.0 * cos2_alpha))
        lam_prev = lam
        lam = lon_diff + (1.0 - c) * f * sin_alpha * (
            sigma
            + c * sin_sigma * (cos_2sigma_m + c * cos_sigma * (-1.0 + 2.0 * cos_2sigma_m**2))
        )
        if abs(lam - lam_prev) < tol:
            break
    else:
        # Fallback to spherical Haversine if Vincenty fails to converge
        return haversine_distance(lon1, lat1, lon2, lat2)

    u_sq = cos2_alpha * (a**2 - b**2) / (b**2)
    big_a = 1.0 + (u_sq / 16384.0) * (4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)))
    big_b = (u_sq / 1024.0) * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)))
    delta_sigma = (
        big_b
        * sin_sigma
        * (
            cos_2sigma_m
            + 0.25
            * big_b
            * (
                cos_sigma * (-1.0 + 2.0 * cos_2sigma_m**2)
                - (big_b / 6.0)
                * cos_2sigma_m
                * (-3.0 + 4.0 * sin_sigma**2)
                * (-3.0 + 4.0 * cos_2sigma_m**2)
            )
        )
    )
    s = b * big_a * (sigma - delta_sigma)
    return float(s)


def bearing_degrees(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculate initial compass forward bearing from point 1 to point 2 in degrees [0, 360)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    y = math.sin(dlam) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360.0) % 360.0


def angle_difference_degrees(angle1: float, angle2: float) -> float:
    """Calculate minimal angular difference between two bearings in degrees [0, 180]."""
    diff = abs(angle1 - angle2) % 360.0
    return 360.0 - diff if diff > 180.0 else diff


def project_local(
    lon: float | np.ndarray,
    lat: float | np.ndarray,
    lon0: float,
    lat0: float,
) -> Tuple[float | np.ndarray, float | np.ndarray]:
    """Project geographic (lon, lat) coordinates to local Cartesian (x, y) meters.

    Uses a local conformal / equidistant projection centered at (lon0, lat0).
    For regional extents (< 100 km), scale distortion is < 0.1%.
    """
    cos_lat0 = math.cos(math.radians(lat0))
    x = EARTH_RADIUS_METERS * np.radians(lon - lon0) * cos_lat0
    y = EARTH_RADIUS_METERS * np.radians(lat - lat0)
    return x, y


def unproject_local(
    x: float | np.ndarray,
    y: float | np.ndarray,
    lon0: float,
    lat0: float,
) -> Tuple[float | np.ndarray, float | np.ndarray]:
    """Invert local Cartesian (x, y) meters back to geographic (lon, lat) degrees."""
    cos_lat0 = math.cos(math.radians(lat0))
    lon = lon0 + np.degrees(x / (EARTH_RADIUS_METERS * cos_lat0))
    lat = lat0 + np.degrees(y / EARTH_RADIUS_METERS)
    return lon, lat


def point_to_segment_projection(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> Tuple[float, float, float, float]:
    """Calculate shortest distance from point P to line segment AB in meters.

    Returns:
        (distance_m, projection_t, closest_qx, closest_qy)
        where projection_t is clamped to [0.0, 1.0] representing the relative position
        along segment AB (0.0 = point A, 1.0 = point B).
    """
    vx = bx - ax
    vy = by - ay
    lens_sq = vx * vx + vy * vy
    if lens_sq == 0.0:
        dist = math.hypot(px - ax, py - ay)
        return dist, 0.0, ax, ay

    t = ((px - ax) * vx + (py - ay) * vy) / lens_sq
    t_clamped = max(0.0, min(1.0, t))
    qx = ax + t_clamped * vx
    qy = ay + t_clamped * vy
    dist = math.hypot(px - qx, py - qy)
    return dist, t_clamped, qx, qy
