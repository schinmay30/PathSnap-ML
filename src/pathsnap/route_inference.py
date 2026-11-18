"""Full-track inference. No reference route, alignment or label is an input."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from pathsnap.candidate_features import ADDED_FEATURES, additional_features
from pathsnap.candidates_baseline import FEATURE_COLUMNS, build_trip_candidates
from pathsnap.data_io import read_arcs, read_nodes, read_track
from pathsnap.geo import project_local, point_to_segments_m
from pathsnap.viterbi import Candidate, DirectedRoadGraph, Observation, decode

FEATURES = FEATURE_COLUMNS + ADDED_FEATURES
POLICY = dict(gap_seconds=5., max_speed_mps=60., gps_slack_m=25., distance_scale_m=20.)


@dataclass
class RoadTrip:
    trip_id: str
    track: list
    nodes: list
    arcs: list

    @classmethod
    def read(cls, folder):
        stem = folder / folder.name
        return cls(folder.name, read_track(stem.with_suffix(".track")),
                   read_nodes(stem.with_suffix(".nodes")), read_arcs(stem.with_suffix(".arcs")))


def feature_table(trip):
    empty = pd.DataFrame(columns=["trip_id", "observation_id", "candidate_arc_id", *FEATURES])
    if not trip.track or not trip.arcs:
        return empty
    try:
        candidates = build_trip_candidates(trip)
    except IndexError:
        # The baseline indexes an empty reduceat array on a wholly unmatched track.
        # Confirm geometrically before treating any IndexError as an empty result.
        track = np.asarray(trip.track)
        nodes = np.asarray(trip.nodes)
        lat0 = track[:, 1].mean()
        xy = project_local(*nodes.T, lat0)
        src, dst = xy[np.asarray(trip.arcs)].transpose(1, 0, 2)
        for point in project_local(track[:, 0], track[:, 1], lat0):
            if np.any(point_to_segments_m(point, src, dst) <= 50):
                raise
        return empty
    return candidates.join(additional_features(candidates, trip.track, trip.nodes, trip.arcs))


def predict_trip(trip, graph, table, model, *, policy=None):
    if list(getattr(model, "feature_names_in_", [])) != FEATURES or list(model.classes_) != [0, 1]:
        raise ValueError("Expected a fitted binary classifier with the ordered 19-feature schema")
    start = perf_counter()
    groups = [[] for _ in trip.track]
    if len(table):
        probability = model.predict_proba(table[FEATURES])[:, 1]
        for obs, arc, frac, p in zip(table.observation_id, table.candidate_arc_id,
                                     table.along_fraction, probability):
            groups[obs].append(Candidate(arc, frac, p))
    observations = [Observation(i, *point, groups[i]) for i, point in enumerate(trip.track)]
    independent = [max(cs, key=lambda c: c.probability, default=None) for cs in groups]
    emission_seconds = perf_counter() - start
    start = perf_counter()
    matches = decode(graph, observations, **(POLICY if policy is None else policy))
    decode_seconds = perf_counter() - start
    output = []
    prior = {}
    for obs, chosen, match in zip(observations, independent, matches):
        states = {c.arc_id: c for c in obs.candidates}
        row = dict(trip_id=trip.trip_id, observation_id=obs.observation_id,
                   candidates=tuple(states), status=match.status)
        dt = obs.timestamp - trip.track[max(obs.observation_id - 1, 0)][2]
        for method, state in (("independent", chosen), ("hmm", states.get(match.arc_id))):
            row[method] = None if state is None else state.arc_id
            previous = prior.get(method)
            valid = None
            if previous is not None and state is not None and 0 < dt <= POLICY["gap_seconds"]:
                limit = POLICY["max_speed_mps"] * dt + POLICY["gps_slack_m"]
                distances = graph._distances(graph.arcs[previous.arc_id][1], limit)
                valid = graph.travel(previous, state, distances) <= limit
            row[method + "_transition"] = valid
            prior[method] = state
        output.append(row)
    return pd.DataFrame(output), dict(emission_seconds=emission_seconds, decode_seconds=decode_seconds)
