"""Offline log-space map matching on one directed trip graph."""

from dataclasses import dataclass
from heapq import heappop, heappush
from math import inf, isfinite, log
from numbers import Integral
from typing import Sequence

from pathsnap.geo import haversine_m


@dataclass(frozen=True)
class Candidate:
    arc_id: int
    along_fraction: float
    probability: float


@dataclass(frozen=True)
class Observation:
    observation_id: int
    longitude: float
    latitude: float
    timestamp: float
    candidates: Sequence[Candidate]


@dataclass(frozen=True)
class Match:
    observation_id: int
    arc_id: int | None
    status: str


def _coordinates(lon, lat):
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        raise ValueError("Invalid longitude or latitude")


class DirectedRoadGraph:
    """Arc IDs are zero-based indices in arcs; parallel arc IDs remain distinct."""

    def __init__(self, nodes, arcs):
        self.arcs = tuple(tuple(edge) for edge in arcs)
        self.lengths = []
        self.outgoing = [[] for _ in nodes]
        for lon, lat in nodes:
            _coordinates(lon, lat)
        for src, dst in self.arcs:
            if any(not isinstance(n, Integral) or not 0 <= n < len(nodes) for n in (src, dst)):
                raise ValueError("Invalid arc endpoint")
            length = float(haversine_m(*nodes[src], *nodes[dst]))
            self.lengths.append(length)
            self.outgoing[src].append((dst, length))

    def _distances(self, source, limit):
        """Dijkstra over the reachable neighbourhood, including zero-weight edges."""
        distances = {source: 0.0}
        queue = [(0.0, source)]
        while queue:
            cost, node = heappop(queue)
            if cost > distances[node]:
                continue
            for target, length in self.outgoing[node]:
                total = cost + length
                if total <= limit and total < distances.get(target, inf):
                    distances[target] = total
                    heappush(queue, (total, target))
        return distances

    def travel(self, a, b, distances):
        length = self.lengths[a.arc_id]
        if a.arc_id == b.arc_id and (b.along_fraction >= a.along_fraction or length == 0):
            return max(0.0, b.along_fraction - a.along_fraction) * length
        middle = distances.get(self.arcs[b.arc_id][0], inf)
        return (1 - a.along_fraction) * length + middle + b.along_fraction * self.lengths[b.arc_id]


def decode(graph: DirectedRoadGraph, observations: Sequence[Observation], *,
           gap_seconds: float = 5.0, max_speed_mps: float = 60.0,
           gps_slack_m: float = 25.0, distance_scale_m: float = 20.0,
           emission_weight: float = 1., transition_weight: float = 1.,
           distance_deadband_m: float = 0., displacement_reset_m: float | None = None,
           unreachable_penalty: float | None = None) -> list[Match]:
    """Find a maximum-score sequence per connected time segment. See docs/viterbi.md."""
    if any(not isfinite(x) or x <= 0 for x in (gap_seconds, max_speed_mps, distance_scale_m)):
        raise ValueError("Gap, speed and distance scale must be finite and positive")
    if not isfinite(gps_slack_m) or gps_slack_m < 0:
        raise ValueError("GPS slack must be finite and nonnegative")
    if not isfinite(emission_weight) or emission_weight <= 0:
        raise ValueError("Emission weight must be positive and finite")
    for value in (transition_weight, distance_deadband_m, displacement_reset_m, unreachable_penalty):
        if value is not None and (not isfinite(value) or value < 0):
            raise ValueError("Weights, thresholds and penalties must be finite and nonnegative")
    matches = []
    # Keep only backpointers for each layer; cumulative scores live at the frontier.
    layers = []
    scores = []

    def flush():
        if layers:
            state = max(range(len(scores)), key=scores.__getitem__)
            segment = []
            for obs_id, states, parents, status, jumps in reversed(layers):
                segment.append(Match(obs_id, int(states[state].arc_id),
                                     "soft_jump" if jumps[state] else status))
                state = parents[state]
            matches.extend(reversed(segment))
            layers.clear()

    previous = None
    seen = set()
    for obs in observations:
        _coordinates(obs.longitude, obs.latitude)
        if obs.observation_id in seen or not isfinite(obs.timestamp):
            raise ValueError("Expected unique observation IDs and finite timestamps")
        seen.add(obs.observation_id)
        dt = obs.timestamp - previous.timestamp if previous else 0
        if previous and dt <= 0:
            raise ValueError("Timestamps must strictly increase")
        states = sorted(obs.candidates, key=lambda c: c.arc_id)
        if len({c.arc_id for c in states}) != len(states):
            raise ValueError("Duplicate candidate arc ID at one observation")
        for c in states:
            if not isinstance(c.arc_id, Integral) or not 0 <= c.arc_id < len(graph.arcs):
                raise ValueError("Invalid candidate arc ID")
            if not (0 <= c.along_fraction <= 1 and 0 <= c.probability <= 1):
                raise ValueError("Invalid candidate fraction or probability")
        if not states:
            flush()
            matches.append(Match(obs.observation_id, None, "no_candidates"))
            previous = obs
            continue
        emissions = [emission_weight * log(max(c.probability, 1e-15)) for c in states]
        status = "start" if previous is None else "restart_after_missing"
        parents = [-1] * len(states)
        jumps = [False] * len(states)
        current = emissions.copy()
        displacement = (float(haversine_m(previous.longitude, previous.latitude,
                                          obs.longitude, obs.latitude)) if layers else 0.)
        if layers and dt > gap_seconds:
            flush()
            status = "gap_reset"
        elif layers and displacement_reset_m is not None and displacement > displacement_reset_m:
            flush()
            status = "displacement_reset"
        elif layers:
            limit = max_speed_mps * dt + gps_slack_m
            current = [-inf] * len(states)
            cache = {}
            for i, a in enumerate(layers[-1][1]):
                if scores[i] == -inf:
                    continue
                tail = graph.arcs[a.arc_id][1]
                if tail not in cache:
                    cache[tail] = graph._distances(tail, limit)
                for j, b in enumerate(states):
                    travel = graph.travel(a, b, cache[tail])
                    jump = travel > limit
                    if jump and unreachable_penalty is None:
                        continue
                    penalty = (unreachable_penalty if jump else
                               max(0., abs(travel - displacement) - distance_deadband_m) / distance_scale_m)
                    score = scores[i] + emissions[j] - transition_weight * penalty
                    if score > current[j]:
                        current[j], parents[j], jumps[j] = score, i, jump
            status = "matched"
            if max(current) == -inf:
                flush()
                current = emissions.copy()
                status = "transition_reset"
        scores = current
        layers.append((obs.observation_id, states, parents, status, jumps))
        previous = obs
    flush()
    return matches
