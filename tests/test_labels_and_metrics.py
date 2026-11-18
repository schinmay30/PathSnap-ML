"""Run with: python -m pytest tests -q"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pathsnap.geo import haversine_m, point_to_segments_m, project_local  # noqa: E402
from pathsnap.labels import align_trip_to_route  # noqa: E402
from pathsnap.network import network_stats  # noqa: E402
from pathsnap.preprocess import TripData  # noqa: E402
from run_person1_models import observation_scores  # noqa: E402

M_PER_DEG_LAT = 111_194.9  # R * pi / 180 for R = 6371 km


def straight_trip(noise_m=0.0, seed=0) -> TripData:
    """Straight north-going road: 4 nodes, 3 arcs, plus a parallel decoy arc."""
    lat = np.array([0.0, 0.001, 0.002, 0.003])
    nodes = [(0.0, float(v)) for v in lat] + [(0.0002, 0.0), (0.0002, 0.003)]
    arcs = [(0, 1), (1, 2), (2, 3), (4, 5)]  # last arc is a parallel decoy (~22 m east)
    route = [0, 1, 2]
    rng = np.random.default_rng(seed)
    obs_lat = np.linspace(0.00005, 0.00295, 30)
    lon = rng.normal(0, noise_m / M_PER_DEG_LAT, size=obs_lat.size)
    track = [(float(x), float(y), float(i)) for i, (x, y) in enumerate(zip(lon, obs_lat))]
    return TripData("t", track, nodes, arcs, route)


def test_projection_and_haversine_agree_at_trip_scale():
    lon = np.array([77.5, 77.51]); lat = np.array([12.9, 12.91])
    xy = project_local(lon, lat, lat.mean())
    planar = float(np.hypot(*(xy[1] - xy[0])))
    great_circle = float(haversine_m(lon[0], lat[0], lon[1], lat[1]))
    assert abs(planar - great_circle) < 0.5  # under half a metre over ~1.6 km


def test_point_to_segment_distance():
    a = np.array([[0.0, 0.0]]); b = np.array([[10.0, 0.0]])
    assert point_to_segments_m(np.array([5.0, 3.0]), a, b)[0] == 3.0
    assert point_to_segments_m(np.array([-4.0, 3.0]), a, b)[0] == 5.0  # clamps to endpoint


def test_alignment_is_monotone_and_follows_route_order():
    labels = align_trip_to_route(straight_trip())
    assert (labels["route_index"].diff().dropna() >= 0).all()
    assert labels["arc_id"].iloc[0] == 0 and labels["arc_id"].iloc[-1] == 2
    assert set(labels["arc_id"]) == {0, 1, 2}  # decoy arc 3 is never a label
    assert labels["label_valid"].all()


def test_far_points_are_flagged_invalid_not_dropped():
    trip = straight_trip()
    track = list(trip.track)
    track[10] = (0.01, track[10][1], track[10][2])  # ~1.1 km off the road
    labels = align_trip_to_route(TripData("t", track, trip.nodes, trip.arcs, trip.route))
    assert len(labels) == len(track)
    assert not labels.loc[10, "label_valid"] and labels["label_valid"].sum() == len(track) - 1


def test_network_stats_counts_components_and_route_breaks():
    stats = network_stats(straight_trip())
    assert stats["isolated_nodes"] == 0
    assert stats["weak_components"] == 2  # road chain + decoy arc
    assert stats["route_breaks"] == 0 and stats["route_nodes_in_components"] == 1


def test_macro_f1_does_not_merge_arc_ids_across_trips():
    # Arc IDs are per-trip row indices, so "arc 7" in trip A and trip B are
    # different roads. Cases: (trip, true_arc, predicted_arc, n_observations).
    cases = [("A", 7, 7, 3), ("B", 7, 8, 1), ("B", 8, 7, 2), ("A", 9, 9, 1)]
    rows, obs = [], 0
    for trip, truth, pred, count in cases:
        for _ in range(count):
            for arc in {truth, pred, 99}:
                rows.append((trip, obs, arc, int(arc == truth), 0.9 if arc == pred else 0.1))
            obs += 1
    df = pd.DataFrame(rows, columns=["trip_id", "observation_id", "candidate_arc_id", "label", "p"])
    top1, macro_f1, eligible, none_pos, multi_pos = observation_scores(df, df["p"].to_numpy())
    assert (eligible, none_pos, multi_pos) == (7, 0, 0)
    assert abs(top1 - 4 / 7) < 1e-9
    # Trip-qualified classes give 0.5. The old arc-ID-only classes gave 0.5556.
    assert abs(macro_f1 - 0.5) < 1e-9


def test_label_candidates_drops_invalid_and_counts_misses():
    from pathsnap.labels import candidate_miss_rate, label_candidates

    labels = pd.DataFrame(
        {
            "trip_id": ["t"] * 3,
            "observation_id": [0, 1, 2],
            "arc_id": [5, 6, 7],
            "label_valid": [True, True, False],
        }
    )
    cands = pd.DataFrame(
        {
            "trip_id": ["t"] * 6,
            "observation_id": [0, 0, 1, 1, 2, 2],
            "candidate_arc_id": [5, 9, 8, 9, 7, 1],
        }
    )
    out = label_candidates(cands, labels)
    assert set(out["observation_id"]) == {0, 1}  # invalid observation 2 dropped
    assert out.groupby("observation_id")["label"].sum().to_dict() == {0: 1, 1: 0}
    assert candidate_miss_rate(out) == 0.5
    # With the alignment table, a valid observation with no candidates counts as a miss.
    extra = pd.DataFrame({"trip_id": ["t"], "observation_id": [3], "arc_id": [4], "label_valid": [True]})
    both = pd.concat([labels, extra], ignore_index=True)
    assert abs(candidate_miss_rate(out, both) - (1 - 1 / 3)) < 1e-9
