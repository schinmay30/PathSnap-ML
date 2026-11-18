"""Unit tests for pathsnap.candidates and candidate_analysis (Person 2 - Sujeet)."""

import math
import numpy as np
import pytest

from pathsnap.candidates import (
    Candidate,
    ObservationCandidates,
    RoadNetworkIndex,
    generate_candidate_records_for_trip,
)
from pathsnap.candidate_analysis import compute_trip_candidate_stats
from pathsnap.geometry import point_to_segment_projection, project_local
from pathsnap.preprocess import TripData


@pytest.fixture
def sample_trip():
    # Synthetic square network:
    # Node 0: (0.0, 0.0), Node 1: (0.001, 0.0), Node 2: (0.001, 0.001), Node 3: (0.0, 0.001)
    nodes = [
        (0.0, 0.0),
        (0.001, 0.0),
        (0.001, 0.001),
        (0.0, 0.001),
    ]
    # Arcs: 0->1, 1->2, 2->3, 3->0
    arcs = [
        (0, 1),  # arc 0
        (1, 2),  # arc 1
        (2, 3),  # arc 2
        (3, 0),  # arc 3
    ]
    # Track points:
    # Point 0: near arc 0
    # Point 1: near arc 1
    # Point 2: far away (~5 km away -> zero candidates)
    track = [
        (0.0005, 0.00005, 1.0),
        (0.00105, 0.0005, 2.0),
        (0.05, 0.05, 3.0),
    ]
    route = [0, 1]
    return TripData(trip_id="test_trip", track=track, nodes=nodes, arcs=arcs, route=route)


def test_road_network_index_query(sample_trip):
    index = RoadNetworkIndex(sample_trip.nodes, sample_trip.arcs, sample_step_m=10.0)
    assert index.num_nodes == 4
    assert index.num_arcs == 4

    cands_p0 = index.query_observation(
        observation_id=0,
        longitude=0.0005,
        latitude=0.00005,
        radius_m=30.0,
        max_candidates=5,
    )
    assert len(cands_p0) >= 1
    assert cands_p0[0].candidate_arc_id == 0
    assert cands_p0[0].distance_m < 15.0


def test_zero_candidates_handling(sample_trip):
    index = RoadNetworkIndex(sample_trip.nodes, sample_trip.arcs)
    # Query point far away
    cands = index.query_observation(
        observation_id=2,
        longitude=0.05,
        latitude=0.05,
        radius_m=50.0,
    )
    assert len(cands) == 0


def test_query_trip_preserves_order(sample_trip):
    index = RoadNetworkIndex(sample_trip.nodes, sample_trip.arcs)
    obs_cands = index.query_trip(sample_trip, radius_m=30.0, max_candidates=5)

    assert len(obs_cands) == 3
    assert obs_cands[0].observation_id == 0
    assert obs_cands[0].has_candidates
    assert obs_cands[1].observation_id == 1
    assert obs_cands[1].has_candidates
    assert obs_cands[2].observation_id == 2
    assert not obs_cands[2].has_candidates  # zero candidates


def test_candidate_generation_records(sample_trip):
    cand_recs, zero_recs = generate_candidate_records_for_trip(
        trip=sample_trip,
        partition="test",
        radius_m=30.0,
        max_candidates=5,
        include_zero_candidates=False,
    )
    assert len(zero_recs) == 1
    assert zero_recs[0]["observation_id"] == 2
    assert len(cand_recs) >= 2


def test_trip_candidate_stats(sample_trip):
    index = RoadNetworkIndex(sample_trip.nodes, sample_trip.arcs)
    obs_cands = index.query_trip(sample_trip, radius_m=30.0, max_candidates=5)
    stats, zero_recs = compute_trip_candidate_stats(sample_trip, "test", obs_cands)

    assert stats.total_observations == 3
    assert stats.covered_observations == 2
    assert stats.zero_candidate_observations == 1
    assert abs(stats.coverage_rate - (2.0 / 3.0)) < 1e-4
    assert stats.route_unique_arcs == 2
    assert stats.route_arcs_recalled == 2
    assert stats.route_arc_recall == 1.0


def test_brute_force_comparison(sample_trip):
    """Verify that KDTree spatial index matches exhaustive brute force with 0 discrepancies."""
    index = RoadNetworkIndex(sample_trip.nodes, sample_trip.arcs, sample_step_m=10.0)
    radius_m = 50.0

    # Test query points
    test_queries = [
        (0.0005, 0.00005),
        (0.001, 0.0005),
        (0.0005, 0.001),
        (0.0, 0.0005),
        (0.0005, 0.0005),
    ]

    nodes_arr = np.array(sample_trip.nodes, dtype=float)
    arcs_arr = np.array(sample_trip.arcs, dtype=int)
    lon0 = float(np.mean(nodes_arr[:, 0]))
    lat0 = float(np.mean(nodes_arr[:, 1]))
    nx, ny = project_local(nodes_arr[:, 0], nodes_arr[:, 1], lon0, lat0)

    for obs_id, (qlon, qlat) in enumerate(test_queries):
        # 1. Spatial index query
        cands_idx = index.query_observation(
            obs_id, qlon, qlat, radius_m=radius_m, max_candidates=10
        )
        idx_arc_ids = {c.candidate_arc_id for c in cands_idx}

        # 2. Exhaustive brute force
        qpx, qpy = project_local(qlon, qlat, lon0, lat0)
        bf_arc_ids = set()
        for arc_id in range(len(arcs_arr)):
            u, v = arcs_arr[arc_id]
            ax, ay = nx[u], ny[u]
            bx, by = nx[v], ny[v]
            d, _, _, _ = point_to_segment_projection(qpx, qpy, ax, ay, bx, by)
            if d <= radius_m:
                bf_arc_ids.add(arc_id)

        assert idx_arc_ids == bf_arc_ids, f"Discrepancy at query {obs_id}: idx={idx_arc_ids} vs bf={bf_arc_ids}"
