"""Classify frozen Phase 2 reset pairs using directed reachability without a speed cap."""
import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from run_route_evaluation import ROOT, RoadTrip
from pathsnap.geo import pairs_point_segment_m, project_local


def diagnose(root, output):
    records = []
    for path in sorted((ROOT / "experiments/person3/route_phase2").glob("*_observations.csv.gz")):
        frame = pd.read_csv(path, dtype={"trip_id": str})
        for trip_id, part in frame.groupby("trip_id"):
            raw = RoadTrip.read(root / "train" / trip_id)
            arcs = np.asarray(raw.arcs)
            graph = coo_matrix((np.ones(len(arcs)), arcs.T), shape=(len(raw.nodes),)*2).tocsr()
            count, component = connected_components(graph, directed=True, connection="strong")
            successors = [set() for _ in range(count)]
            for a, b in component[arcs]:
                if a != b:
                    successors[a].add(b)

            @lru_cache(maxsize=64)
            def reachable(source):
                visited, todo = {source}, [source]
                while todo:
                    new = successors[todo.pop()] - visited
                    visited.update(new)
                    todo.extend(new)
                return visited

            previous = part.hmm.shift()
            resets = part.loc[part.status.eq("transition_reset")]
            structural = 0
            track = np.asarray(raw.track)
            lat0 = track[:, 1].mean()
            xy_nodes = project_local(*np.asarray(raw.nodes).T, lat0)
            xy_track = project_local(*track[:, :2].T, lat0)
            for idx, arc in resets.hmm.items():
                src = arcs[int(previous[idx]), 1]
                dst = arcs[int(arc), 0]
                # Same forward arc can bypass endpoints. Such a reset is possible
                # for an over-budget forward move; distinguish it using projections.
                if int(previous[idx]) == int(arc):
                    i = int(part.loc[idx, "observation_id"])
                    xy = xy_nodes[arcs[int(arc)]]
                    points = xy_track[i-1:i+1]
                    _, frac = pairs_point_segment_m(points, np.repeat(xy[:1],2,axis=0),
                                                     np.repeat(xy[1:],2,axis=0))
                    if frac[1] >= frac[0]:
                        continue
                structural += component[dst] not in reachable(component[src])
            eligible = part[part.eligible]
            greedy = eligible.independent == eligible.arc_id
            hmm = eligible.hmm == eligible.arc_id
            records.append(dict(model=path.name.split("_observations")[0], trip_id=trip_id,
                resets=len(resets), no_directed_path=structural, reachable_over_budget=len(resets)-structural,
                spoiled=int((greedy & ~hmm).sum()), corrected=int((~greedy & hmm).sum()),
                spoiled_at_reset=int((greedy & ~hmm & eligible.status.eq("transition_reset")).sum())))
    pd.DataFrame(records).to_csv(output, mode="x", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    diagnose(args.data_root, args.output)
