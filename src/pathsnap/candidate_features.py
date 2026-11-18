"""Additional features for one trip, computed from GPS fixes and road geometry."""

import numpy as np
import pandas as pd

from pathsnap.geo import project_local


ADDED_FEATURES = [
    "nearest_endpoint_m", "previous_step_m", "previous_step_available",
    "previous_bearing_cos", "previous_bearing_available", "zero_length_arc",
    "entry_in_degree", "exit_out_degree", "nearest_runner_up_gap_m",
    "has_runner_up",
]


def additional_features(candidates, track, nodes, arcs):
    """Return features in input row order. Supply all candidates of each observation."""
    track = np.asarray(track, dtype=float)
    nodes = np.asarray(nodes, dtype=float)
    arcs = np.asarray(arcs)
    if not np.isfinite(track).all() or not np.isfinite(nodes).all():
        raise ValueError("Track and node coordinates and timestamps must be finite")
    obs = candidates.observation_id.to_numpy()
    arc = candidates.candidate_arc_id.to_numpy()
    for ids, size in ((obs, len(track)), (arc, len(arcs)), (arcs, len(nodes))):
        if ids.dtype.kind not in "iu" or np.any((ids < 0) | (ids >= size)):
            raise ValueError("Observation, arc and node IDs must be valid integer indices")
    if candidates.duplicated(["observation_id", "candidate_arc_id"]).any():
        raise ValueError("Duplicate candidate key within a trip")
    lat0 = track[:, 1].mean()
    points = project_local(track[:, 0], track[:, 1], lat0)
    xy = project_local(nodes[:, 0], nodes[:, 1], lat0)
    src, dst = arcs[arc].T
    direction = xy[dst] - xy[src]
    length = np.linalg.norm(direction, axis=1)
    endpoint = np.minimum(np.linalg.norm(points[obs] - xy[src], axis=1),
                          np.linalg.norm(points[obs] - xy[dst], axis=1))

    dt = np.diff(track[:, 2], prepend=track[0, 2])[obs]
    valid = (dt > 0) & (dt <= 5)
    step = points[obs] - points[np.maximum(obs - 1, 0)]
    displacement = np.linalg.norm(step, axis=1)
    # Same moving threshold as the baseline, applied to the incoming GPS step.
    bearing_valid = valid & (displacement > dt * 1.0) & (length > 0)
    cosine = np.divide(np.einsum("ij,ij->i", step, direction), displacement * length,
                       where=bearing_valid, out=np.zeros(len(obs)))

    # Count distinct directed neighbours. Self-loops do not add a neighbour.
    edges = np.unique(arcs[arcs[:, 0] != arcs[:, 1]], axis=0)
    indegree = np.bincount(edges[:, 1], minlength=len(nodes))
    outdegree = np.bincount(edges[:, 0], minlength=len(nodes))
    # Retain different arc IDs even when geometries coincide. A tie gives zero gap.
    ordered = candidates.sort_values(["observation_id", "dist_m"])
    nearest_two = ordered.groupby("observation_id", sort=False).head(2)
    distances = nearest_two.groupby("observation_id").dist_m.agg(["min", "max", "size"])
    margins = distances.loc[obs]
    values = [endpoint, np.where(valid, displacement, 0), valid.astype(int),
              np.clip(cosine, -1, 1), bearing_valid.astype(int), (length == 0).astype(int),
              indegree[src], outdegree[dst], (margins["max"] - margins["min"]).to_numpy(),
              (margins["size"] > 1).to_numpy(dtype=int)]
    result = pd.DataFrame(dict(zip(ADDED_FEATURES, values)), index=candidates.index)
    if not np.isfinite(result.to_numpy()).all() or not np.isfinite(candidates.dist_m).all():
        raise ValueError("Candidate features must be finite")
    return result
