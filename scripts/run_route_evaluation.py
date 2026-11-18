"""Out-of-fold independent and Viterbi predictions on complete training tracks."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from run_person3_models import ROOT, build_models
from pathsnap.route_inference import (FEATURES, POLICY, RoadTrip, DirectedRoadGraph,
                                      feature_table, predict_trip)


def attach_reference(predictions, reference):
    keys = ["trip_id", "observation_id"]
    scored = predictions.merge(reference[keys + ["arc_id", "label_valid"]], on=keys,
                               how="outer", validate="one_to_one", indicator=True)
    if not scored["_merge"].eq("both").all():
        raise ValueError("Reference and full-track observation IDs differ")
    scored["eligible"] = [valid and truth in states for valid, truth, states in
                          zip(scored.label_valid, scored.arc_id, scored.candidates)]
    return scored.drop(columns=["_merge", "candidates"])


def summarize(scored):
    eligible = scored.loc[scored.eligible]
    rows = []
    for method in ("independent", "hmm"):
        qualified = eligible.trip_id + "::"
        correct = (eligible.arc_id == eligible[method]).sum()
        valid = int(scored.label_valid.sum())
        rows.append(dict(method=method, observations=len(scored), valid_labels=valid,
                         matched_observations=len(eligible), correct=int(correct),
                         prediction_coverage=scored[method].notna().mean(),
                         candidate_recall=len(eligible)/valid if valid else np.nan,
                         conditional_top1=correct/len(eligible) if len(eligible) else np.nan,
                         all_valid_top1=correct/valid if valid else np.nan,
                         macro_f1=f1_score(qualified + eligible.arc_id.astype(int).astype(str),
                                           qualified + eligible[method].astype(int).astype(str),
                                           average="macro", zero_division=0) if len(eligible) else np.nan,
                         transition_pairs=scored[method + "_transition"].count(),
                         feasible_pairs=scored[method + "_transition"].sum(),
                         transition_validity=scored[method + "_transition"].mean()))
    return rows


def run(root, output, smoke):
    raw = root / "train"
    trip_ids = sorted(p.name for p in raw.iterdir() if p.is_dir())
    if smoke:
        trip_ids = trip_ids[:4]
    def read_train(path, columns):
        chunks = pd.read_csv(path, usecols=["partition", "trip_id", *columns],
                             dtype={"trip_id": str}, chunksize=100000)
        return pd.concat([c.loc[c.partition.eq("train") & c.trip_id.isin(trip_ids)]
                          for c in chunks], ignore_index=True)

    train = read_train(ROOT / "data/processed/person3_extended_candidates.csv.gz", ["label", *FEATURES])
    if set(train.trip_id) != set(trip_ids):
        raise ValueError("Training table and raw trip roster disagree")
    output.mkdir(parents=True, exist_ok=False)
    predictions, latencies, manifest = [], [], []
    for fold, (fit, held) in enumerate(GroupKFold(2).split(train, groups=train.trip_id), 1):
        fit_ids = sorted(train.trip_id.iloc[fit].unique())
        held_ids = sorted(train.trip_id.iloc[held].unique())
        manifest.append(dict(fold=fold, fit_trips=fit_ids, held_out_trips=held_ids))
        models = {}
        for name, model in build_models().items():
            model.fit(train.iloc[fit][FEATURES], train.label.iloc[fit])
            path = output / f"fold{fold}_{name}.joblib"
            joblib.dump(model, path)
            models[name] = joblib.load(path)
        for trip_id in held_ids:
            start = perf_counter()
            trip = RoadTrip.read(raw / trip_id)
            graph = DirectedRoadGraph(trip.nodes, trip.arcs)
            table = feature_table(trip)
            prepare_seconds = perf_counter() - start
            for name, model in models.items():
                frame, timing = predict_trip(trip, graph, table, model)
                predictions.append((name, frame))
                latencies.append(dict(model=name, fold=fold, trip_id=trip_id, observations=len(trip.track),
                                      prepare_seconds=prepare_seconds, **timing))
            print(f"Completed fold {fold}, trip {trip_id}", flush=True)
    # All predictions are fixed before opening the reference table.
    reference = read_train(ROOT / "results/labels/observation_labels.csv.gz",
                           ["observation_id", "arc_id", "label_valid"])
    summary = []
    for name in models:
        frame = pd.concat([p for model, p in predictions if model == name], ignore_index=True)
        scored = attach_reference(frame, reference)
        scored.to_csv(output / f"{name}_observations.csv.gz", index=False)
        summary.extend(dict(model=name, **row) for row in summarize(scored))
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(latencies).to_csv(output / "latencies.csv", index=False)
    (output / "manifest.json").write_text(json.dumps(dict(features=FEATURES, policy=POLICY,
                                                         folds=manifest), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Two grouped folds on the first four train trips")
    args = parser.parse_args()
    run(args.data_root, args.output, args.smoke)
