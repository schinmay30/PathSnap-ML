"""Road-network connectivity statistics for one trip's nodes/arcs."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from pathsnap.preprocess import TripData


def network_stats(trip: TripData) -> dict[str, float]:
    """Size and connectivity of the directed road graph.

    isolated nodes     : nodes that appear in no arc
    weak components    : components ignoring arc direction (among nodes that
                         appear in at least one arc)
    strong components  : components respecting arc direction
    """
    n_nodes = len(trip.nodes)
    arcs = np.asarray(trip.arcs, dtype=int)
    src, dst = arcs[:, 0], arcs[:, 1]
    used = np.zeros(n_nodes, dtype=bool)
    used[src] = True
    used[dst] = True
    graph = coo_matrix((np.ones(len(arcs)), (src, dst)), shape=(n_nodes, n_nodes)).tocsr()

    weak_n, weak_labels = connected_components(graph, directed=True, connection="weak")
    strong_n, strong_labels = connected_components(graph, directed=True, connection="strong")

    used_weak = weak_labels[used]
    weak_sizes = np.bincount(used_weak) if used_weak.size else np.array([0])
    largest_weak = int(weak_sizes.max())
    n_used = int(used.sum())
    n_weak_used = int((weak_sizes > 0).sum())

    route = np.asarray(trip.route, dtype=int)
    route_nodes = np.unique(arcs[route].ravel())
    route_components = len(set(weak_labels[route_nodes].tolist()))

    self_loops = int((src == dst).sum())
    # Consecutive route arcs should chain: head of arc k == tail of arc k+1.
    route_breaks = int(np.sum(arcs[route[:-1], 1] != arcs[route[1:], 0]))
    route_reverse_pairs = int(
        np.sum((arcs[route[:-1], 0] == arcs[route[1:], 1]) & (arcs[route[:-1], 1] == arcs[route[1:], 0]))
    )
    return {
        "nodes": n_nodes,
        "arcs": len(arcs),
        "isolated_nodes": int(n_nodes - n_used),
        "weak_components": n_weak_used,
        "largest_weak_component_fraction": largest_weak / n_used if n_used else 0.0,
        "strong_components": int(len(set(strong_labels[used].tolist()))),
        "self_loop_arcs": self_loops,
        "duplicate_arcs": int(len(arcs) - len({(a, b) for a, b in arcs.tolist()})),
        "route_nodes_in_components": route_components,
        "route_arcs": len(route),
        "route_breaks": route_breaks,
        "route_reverse_pairs": route_reverse_pairs,
        "mean_out_degree": float(len(arcs) / n_used) if n_used else 0.0,
    }
