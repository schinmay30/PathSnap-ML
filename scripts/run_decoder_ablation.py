"""Six predeclared decoder policies, one seeded eight-trip inner holdout per outer fold.

Select by pooled conditional Top-1, breaking ties by fixed-policy transition
validity, then declaration order. Outer trips are scored once after selection.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from run_route_evaluation import (ROOT, POLICY, FEATURES, RoadTrip, DirectedRoadGraph,
                                  feature_table, predict_trip, attach_reference, summarize, build_models)

POLICIES = {
    "phase2": POLICY,
    "weaker_distance": {**POLICY, "transition_weight": .25},
    "deadband": {**POLICY, "transition_weight": .25, "distance_deadband_m": 10.},
    "jump_reset": {**POLICY, "displacement_reset_m": 65.},
    "soft_jump": {**POLICY, "unreachable_penalty": 2.},
    "soft_emission": {**POLICY, "unreachable_penalty": 2., "emission_weight": 3.},
}


def train_rows(path, columns):
    chunks = pd.read_csv(path, usecols=["partition", "trip_id", *columns],
                         dtype={"trip_id": str}, chunksize=100000)
    return pd.concat([chunk.loc[chunk.partition.eq("train")] for chunk in chunks], ignore_index=True)


def partition_ids(fold, seed):
    fit, outer = sorted(fold["fit_trips"]), sorted(fold["held_out_trips"])
    if set(fit) & set(outer) or len(fit) <= 8:
        raise ValueError("Invalid Phase 2 manifest")
    inner = sorted(np.random.default_rng(seed).choice(fit, size=8, replace=False).tolist())
    return sorted(set(fit)-set(inner)), inner, outer


def evaluate(raw, ids, models, choices, labels, fold, stage, timings):
    collected = {}
    for trip_id in ids:
        start = perf_counter()
        trip = RoadTrip.read(raw / trip_id)
        graph = DirectedRoadGraph(trip.nodes, trip.arcs)
        table = feature_table(trip)
        prep = perf_counter() - start
        for name, model in models.items():
            for policy in choices[name]:
                pred, timing = predict_trip(trip, graph, table, model, policy=POLICIES[policy])
                scored = attach_reference(pred, labels.get_group(trip_id))
                collected.setdefault((name, policy), []).append(scored)
                timings.append(dict(stage=stage, fold=fold, model=name, policy=policy,
                                    trip_id=trip_id, prepare_seconds=prep, **timing))
    return {key: pd.concat(rows, ignore_index=True) for key, rows in collected.items()}


def metrics(frame):
    row = summarize(frame)[1]
    row.update(soft_jumps=int(frame.status.eq("soft_jump").sum()),
               forced_resets=int(frame.status.eq("transition_reset").sum()),
               displacement_resets=int(frame.status.eq("displacement_reset").sum()),
               gap_resets=int(frame.status.eq("gap_reset").sum()),
               restarts_after_missing=int(frame.status.eq("restart_after_missing").sum()))
    return row


def run(root, output):
    phase2 = ROOT / "experiments/person3/route_phase2"
    folds = json.loads((phase2 / "manifest.json").read_text())["folds"]
    train = train_rows(ROOT / "data/processed/person3_extended_candidates.csv.gz", ["label", *FEATURES])
    labels = train_rows(ROOT / "results/labels/observation_labels.csv.gz",
                        ["observation_id", "arc_id", "label_valid"]).groupby("trip_id")
    output.mkdir(parents=True, exist_ok=False)
    inner_results, timings, manifest, outer_frames = [], [], [], {}
    for fold in folds:
        number = fold["fold"]
        fit, inner, held = partition_ids(fold, 42 + number)
        models = build_models()
        def refit(ids):
            subset = train.loc[train.trip_id.isin(ids)]
            for model in models.values():
                model.fit(subset[FEATURES], subset.label)
        refit(fit)
        inner_frames = evaluate(root / "train", inner, models, {m: POLICIES for m in models},
                                labels, number, "inner", timings)
        selected = {}
        for name in models:
            trials = [dict(model=name, fold=number, policy=p, **metrics(inner_frames[name, p]))
                      for p in POLICIES]
            selected[name] = max(trials, key=lambda r: (r["conditional_top1"], r["transition_validity"]))["policy"]
            inner_results.extend(trials)
        del inner_frames
        manifest.append(dict(fold=number, inner_fit=fit, inner_validation=inner, outer=held,
                             selected=selected))
        print(f"Outer fold {number}: {selected}", flush=True)
        refit(sorted(fit + inner))
        outer = evaluate(root / "train", held, models, {m: [p] for m,p in selected.items()},
                         labels, number, "outer", timings)
        for (name, policy), frame in outer.items():
            frame["fold"], frame["policy"] = number, policy
            outer_frames.setdefault(name, []).append(frame)
    results, trips = [], []
    for name, frames in outer_frames.items():
        selected = pd.concat(frames, ignore_index=True)
        baseline = pd.read_csv(phase2 / f"{name}_observations.csv.gz", dtype={"trip_id": str})
        keys = ["trip_id", "observation_id"]
        for frame in (baseline, selected):
            frame.sort_values(keys, inplace=True, ignore_index=True)
        shared = keys + ["independent", "eligible", "label_valid", "arc_id"]
        pd.testing.assert_frame_equal(baseline[shared], selected[shared], check_dtype=False)
        selected.to_csv(output / f"{name}_observations.csv.gz", index=False)
        for variant, frame in [("phase2", baseline), ("selected", selected)]:
            results.append(dict(model=name, variant=variant, **metrics(frame)))
            for trip_id, part in frame.groupby("trip_id"):
                trips.append(dict(model=name, variant=variant, trip_id=trip_id, **metrics(part)))
    for file, rows in [("inner_trials", inner_results), ("latencies", timings),
                       ("per_trip", trips), ("summary", results)]:
        pd.DataFrame(rows).to_csv(output / f"{file}.csv", index=False)
    (output / "manifest.json").write_text(json.dumps(dict(policies=POLICIES, folds=manifest,
        inner_seed="42 + outer fold number", objective="conditional_top1; ties by transition_validity then order"),
        indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/person3/decoder_ablation")
    args = parser.parse_args()
    run(args.data_root, args.output)
