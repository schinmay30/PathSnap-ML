# Phase 1 decoder interface

Call `pathsnap.viterbi.decode(graph, observations)` once per trip with the complete, time-ordered track. It returns one `Match(observation_id, arc_id, status)` per input fix in the original order. Arc IDs are local to the trip. Repeated IDs across observations are meaningful. Phase 1 returns sampled arc assignments; intermediate connecting arcs are not emitted.

## FastAPI integration

1. Load the trip's nodes, directed arcs and track using `read_nodes`, `read_arcs` and `read_track` in `pathsnap.data_io`. Build `DirectedRoadGraph(nodes, arcs)` once. Use the unlabeled candidate generator on the original track. The training loader `load_trip` requires a reference route and is unsuitable for serving.
2. Preserve the original candidate row order through scoring. For a fitted 19-feature pipeline, assemble `FEATURE_COLUMNS + ADDED_FEATURES` using `pathsnap.candidates_baseline` and `pathsnap.candidate_features.additional_features`. Select exactly the names and order in the fitted pipeline's `feature_names_in_`. Saved nine-feature models require just their nine inputs. Resolve the positive probability column from the fitted classifier's `classes_`, as in `positive_probability` in `scripts/run_person1_models.py`. Skip model prediction on an empty candidate table.
3. Group the scored rows by observation ID. Pass the original zero-based track row ID, longitude and latitude in degrees, timestamp in seconds, and all available candidates. Each candidate uses the generator's directed `along_fraction` from source (0) to destination (1). Supply an empty list for a fix with zero candidates.

```python
from dataclasses import asdict
from pathsnap.viterbi import Candidate, DirectedRoadGraph, Observation, decode

# scored_rows has observation_id, candidate_arc_id, along_fraction, probability.
by_fix = {}
for row in scored_rows:
    by_fix.setdefault(row.observation_id, []).append(
        Candidate(int(row.candidate_arc_id), row.along_fraction, row.probability)
    )
observations = [
    Observation(i, lon, lat, ts, by_fix.get(i, []))
    for i, (lon, lat, ts) in enumerate(track)
]
response = [asdict(match) for match in decode(DirectedRoadGraph(nodes, arcs), observations)]
```

Use the returned dictionaries as the response array. Map malformed requests to HTTP 422. Validate field types in the request schema. The decoder raises `ValueError` for duplicate observation IDs, non-increasing timestamps, invalid coordinates, out-of-range fractions, probabilities or arc references, and duplicate arc IDs within one fix.

| Status | Meaning |
|---|---|
| `start` | First fix starts a path. |
| `matched` | The selected state has a feasible predecessor. |
| `no_candidates` | Null arc ID. The preceding segment ends. |
| `restart_after_missing` | First available states after an empty fix. |
| `gap_reset` | Elapsed time exceeds the configured gap between populated fixes. |
| `transition_reset` | Every reachable predecessor failed to connect to every current candidate within the travel budget. A new segment starts here. |

## Scoring and defaults

Viterbi maximizes the sum of log positive-class probabilities (floored at 1e-15) and transition scores. Start states have equal prior weight. The transition score is `-abs(road_metres - gps_metres) / distance_scale_m`. The GPS displacement and endpoint-to-endpoint arc lengths use haversine distance. Travel exceeding `max_speed_mps * elapsed_seconds + gps_slack_m` or requiring an unreachable directed path is impossible. Defaults are 5 s, 60 m/s, 25 m slack and 20 m distance scale. These are initial operating choices.

Forward movement on the same arc uses the difference in projected positions. Otherwise travel includes the remainder of the previous arc, the shortest directed connector, and the beginning of the next arc. A return to an earlier position on the same arc needs a directed cycle. Distinct parallel arcs retain their IDs, zero-length arcs contribute zero metres, and overlapping geometry alone does not connect nodes. Exact ties favour the smaller terminal arc ID, then the smaller predecessor ID during backtracking. Resets delimit independent paths. Classifier scores are emission potentials; the output does not include calibrated sequence confidence.
