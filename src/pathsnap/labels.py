"""Align GPS observations to the reference route (candidate labels).

The .route file is a chronological list of arc IDs, NOT one entry per GPS
row (e.g. 1315 GPS rows vs 525 route arcs). To get a per-observation label
we find, for every GPS row, which route arc the vehicle was on.

Method: monotone dynamic programming over route positions.
  state      : index j into the route sequence (0 .. m-1)
  emission   : distance (m) from the GPS point to arc route[j], capped
  transition : stay at j, or move forward to j' > j. Skipping arcs that no
               observation falls on costs `skip_penalty_m` per skipped arc.
               Moving backwards is forbidden, so the label sequence follows
               the route order even where GPS noise puts a point closer to
               a different (parallel / reverse-direction) arc.

Output is an observation_id (0-based row index in the .track file), the
route index and arc_id chosen, the distance to that arc, and `label_valid`
which is False when the point is farther than `max_dist_m` from its
aligned arc (label cannot be trusted; do not train on it silently).

The reference route is reference information, not verified truth; see
`label_valid` and the per-trip quality table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pathsnap.geo import point_to_segments_m, project_local
from pathsnap.preprocess import TripData

DEFAULT_MAX_DIST_M = 50.0
DEFAULT_COST_CAP_M = 100.0
DEFAULT_SKIP_PENALTY_M = 1.0


def align_trip_to_route(
    trip: TripData,
    max_dist_m: float = DEFAULT_MAX_DIST_M,
    cost_cap_m: float = DEFAULT_COST_CAP_M,
    skip_penalty_m: float = DEFAULT_SKIP_PENALTY_M,
) -> pd.DataFrame:
    """Return one row per GPS observation with its aligned reference arc."""
    track = np.asarray(trip.track, dtype=float)
    nodes = np.asarray(trip.nodes, dtype=float)
    arcs = np.asarray(trip.arcs, dtype=int)
    route = np.asarray(trip.route, dtype=int)
    n, m = len(track), len(route)

    lat0 = float(track[:, 1].mean())
    points = project_local(track[:, 0], track[:, 1], lat0)
    node_xy = project_local(nodes[:, 0], nodes[:, 1], lat0)
    seg_a = node_xy[arcs[route, 0]]
    seg_b = node_xy[arcs[route, 1]]

    index = np.arange(m)
    lam = float(skip_penalty_m)
    back = np.empty((n, m), dtype=np.int32)
    dist_to_arc = np.empty((n, m), dtype=np.float32)

    # Initial row: the vehicle may start anywhere on the route, no skip cost
    # for arcs before the first observation.
    d = point_to_segments_m(points[0], seg_a, seg_b)
    dist_to_arc[0] = d
    cost = np.minimum(d, cost_cap_m)
    back[0] = index

    for i in range(1, n):
        d = point_to_segments_m(points[i], seg_a, seg_b)
        dist_to_arc[i] = d
        emission = np.minimum(d, cost_cap_m)

        # best predecessor strictly before j: min_{j'<j} cost[j'] + lam*(j-j'-1)
        g = cost - lam * index
        running = np.minimum.accumulate(g)
        arg = np.maximum.accumulate(np.where(g == running, index, 0))
        jump_cost = np.full(m, np.inf)
        jump_arg = np.zeros(m, dtype=np.int32)
        if m > 1:
            jump_cost[1:] = running[:-1] + lam * (index[1:] - 1)
            jump_arg[1:] = arg[:-1]

        stay = cost <= jump_cost
        best = np.where(stay, cost, jump_cost)
        back[i] = np.where(stay, index, jump_arg)
        cost = emission + best

    # Backtrack
    path = np.empty(n, dtype=np.int32)
    path[-1] = int(np.argmin(cost))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]

    dist = dist_to_arc[np.arange(n), path].astype(float)
    return pd.DataFrame(
        {
            "trip_id": trip.trip_id,
            "observation_id": np.arange(n),
            "route_index": path,
            "arc_id": route[path],
            "dist_to_arc_m": dist,
            "label_valid": dist <= max_dist_m,
        }
    )


def summarize_alignment(labels: pd.DataFrame, route_length: int) -> dict[str, float]:
    """Per-trip quality numbers for the alignment."""
    dist = labels["dist_to_arc_m"].to_numpy()
    covered = labels.loc[labels["label_valid"], "route_index"].nunique()
    return {
        "observations": len(labels),
        "valid_fraction": float(labels["label_valid"].mean()),
        "median_dist_m": float(np.median(dist)),
        "p95_dist_m": float(np.percentile(dist, 95)),
        "max_dist_m": float(dist.max()),
        "route_arcs": route_length,
        "route_arcs_with_observation": int(covered),
    }


def label_candidates(candidates: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Attach 0/1 `label` to a candidate table using the alignment table.

    `candidates` needs trip_id, observation_id, candidate_arc_id (trip_id as
    str). Observations whose alignment is not valid (`label_valid` False) are
    dropped, never labelled 0, because their reference arc is unreliable.
    An observation whose reference arc is missing from its candidate set keeps
    all-zero labels; count those with `candidate_miss_rate`.
    """
    keep = labels.loc[labels["label_valid"], ["trip_id", "observation_id", "arc_id"]]
    merged = candidates.merge(keep, on=["trip_id", "observation_id"], how="inner")
    merged["label"] = (merged["candidate_arc_id"] == merged["arc_id"]).astype(int)
    return merged.drop(columns="arc_id")


def candidate_miss_rate(labelled: pd.DataFrame, labels: pd.DataFrame | None = None) -> float:
    """Share of valid observations whose reference arc is not among the candidates.

    Pass the alignment table as `labels` to also count valid observations that
    received no candidates at all (they vanish from `labelled` otherwise).
    """
    hit = labelled.groupby(["trip_id", "observation_id"])["label"].sum()
    if labels is None:
        return float((hit == 0).mean())
    valid = labels.loc[labels["label_valid"] & labels["trip_id"].isin(labelled["trip_id"].unique())]
    return float(1.0 - (hit > 0).sum() / len(valid))
