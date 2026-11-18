"""Append route-blind features while retaining baseline values and candidate order."""

import argparse
from pathlib import Path

import pandas as pd

from run_person3_models import ROOT
from pathsnap.candidate_features import additional_features
from pathsnap.data_io import find_dataset_root, read_arcs, read_nodes, read_track


EXTENDED = ROOT / "data/processed/person3_extended_candidates.csv.gz"


def build(data_root, baseline, output):
    root = find_dataset_root(data_root)
    # Read original fields as text to preserve even their serialized precision.
    table = pd.read_csv(baseline, dtype=str, keep_default_na=False)
    additions = []
    for (partition, trip), rows in table.groupby(["partition", "trip_id"], sort=False):
        if partition not in {"train", "student_test"} or not trip.isdecimal():
            raise ValueError(f"Unexpected trip key {partition}/{trip}")
        stem = root / partition / trip / trip
        candidates = rows[["observation_id", "candidate_arc_id", "dist_m"]].apply(pd.to_numeric)
        additions.append(additional_features(candidates, read_track(stem.with_suffix(".track")),
                                             read_nodes(stem.with_suffix(".nodes")),
                                             read_arcs(stem.with_suffix(".arcs"))))
    extended = table.join(pd.concat(additions), validate="one_to_one")
    if extended.isna().any().any():
        raise ValueError("Missing extended candidate values")
    extended.to_csv(output, index=False, mode="x", compression={"method": "gzip", "compresslevel": 1})
    print(f"Wrote {len(extended)} rows to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=ROOT / "data/processed/baseline_candidates.csv.gz")
    parser.add_argument("--output", type=Path, default=EXTENDED)
    args = parser.parse_args()
    build(args.data_root, args.baseline, args.output)
