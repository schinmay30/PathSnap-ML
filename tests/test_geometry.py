"""Unit tests for pathsnap.geometry (Person 2 - Sujeet)."""

import math
import numpy as np
import pytest

from pathsnap.geometry import (
    EARTH_RADIUS_METERS,
    WGS84_A_METERS,
    WGS84_B_METERS,
    angle_difference_degrees,
    bearing_degrees,
    haversine_distance,
    point_to_segment_projection,
    project_local,
    unproject_local,
    wgs84_geodesic_distance,
)


def test_haversine_zero_distance():
    assert haversine_distance(12.9716, 77.5946, 12.9716, 77.5946) == 0.0


def test_haversine_known_distance():
    # 1 degree of latitude is approximately 111.19 km
    dist = haversine_distance(0.0, 0.0, 0.0, 1.0)
    assert 111000.0 < dist < 111500.0


def test_wgs84_geodesic_distance_zero_and_known():
    # Coincident points
    assert wgs84_geodesic_distance(12.9716, 77.5946, 12.9716, 77.5946) == 0.0

    # 1 degree of latitude along prime meridian (equatorial to 1 deg lat)
    dist_wgs84 = wgs84_geodesic_distance(0.0, 0.0, 0.0, 1.0)
    dist_hav = haversine_distance(0.0, 0.0, 0.0, 1.0)
    assert 110500.0 < dist_wgs84 < 111500.0
    # WGS-84 ellipsoidal distance and spherical Haversine agree to within 0.6%
    rel_diff = abs(dist_wgs84 - dist_hav) / dist_wgs84
    assert rel_diff < 0.006


def test_bearing_cardinal_directions():
    # Due North: lat increases
    b_north = bearing_degrees(0.0, 0.0, 0.0, 1.0)
    assert abs(b_north - 0.0) < 1e-4 or abs(b_north - 360.0) < 1e-4

    # Due East: lon increases
    b_east = bearing_degrees(0.0, 0.0, 1.0, 0.0)
    assert abs(b_east - 90.0) < 1e-4

    # Due South: lat decreases
    b_south = bearing_degrees(0.0, 0.0, 0.0, -1.0)
    assert abs(b_south - 180.0) < 1e-4

    # Due West: lon decreases
    b_west = bearing_degrees(0.0, 0.0, -1.0, 0.0)
    assert abs(b_west - 270.0) < 1e-4


def test_angle_difference():
    assert angle_difference_degrees(10.0, 20.0) == 10.0
    assert angle_difference_degrees(10.0, 350.0) == 20.0
    assert angle_difference_degrees(0.0, 180.0) == 180.0
    assert angle_difference_degrees(45.0, 45.0) == 0.0
    assert angle_difference_degrees(350.0, 10.0) == 20.0  # Symmetry


def test_project_and_unproject_fidelity():
    lon0, lat0 = 55.0, 56.0
    test_lons = [54.95, 55.0, 55.05]
    test_lats = [55.95, 56.0, 56.05]

    for lon, lat in zip(test_lons, test_lats):
        x, y = project_local(lon, lat, lon0, lat0)
        rec_lon, rec_lat = unproject_local(x, y, lon0, lat0)
        assert abs(lon - rec_lon) < 1e-6
        assert abs(lat - rec_lat) < 1e-6


def test_point_to_segment_projection_perpendicular():
    # Segment from (0, 0) to (100, 0) along x-axis; Point at (50, 25)
    dist, t_clamped, qx, qy = point_to_segment_projection(50.0, 25.0, 0.0, 0.0, 100.0, 0.0)
    assert abs(dist - 25.0) < 1e-6
    assert abs(t_clamped - 0.5) < 1e-6
    assert abs(qx - 50.0) < 1e-6
    assert abs(qy - 0.0) < 1e-6


def test_point_to_segment_projection_clamped_endpoints():
    # Point before segment start: (-20, 10)
    dist, t_clamped, qx, qy = point_to_segment_projection(-20.0, 10.0, 0.0, 0.0, 100.0, 0.0)
    assert abs(t_clamped - 0.0) < 1e-6
    assert abs(qx - 0.0) < 1e-6
    assert abs(qy - 0.0) < 1e-6
    assert abs(dist - math.hypot(-20.0, 10.0)) < 1e-6

    # Point beyond segment end: (120, 10)
    dist, t_clamped, qx, qy = point_to_segment_projection(120.0, 10.0, 0.0, 0.0, 100.0, 0.0)
    assert abs(t_clamped - 1.0) < 1e-6
    assert abs(qx - 100.0) < 1e-6
    assert abs(qy - 0.0) < 1e-6
    assert abs(dist - math.hypot(20.0, 10.0)) < 1e-6


def test_point_to_segment_projection_zero_length_segment():
    # Degenerate segment A == B
    dist, t_clamped, qx, qy = point_to_segment_projection(10.0, 10.0, 5.0, 5.0, 5.0, 5.0)
    assert abs(t_clamped - 0.0) < 1e-6
    assert abs(qx - 5.0) < 1e-6
    assert abs(qy - 5.0) < 1e-6
    assert abs(dist - math.hypot(5.0, 5.0)) < 1e-6
