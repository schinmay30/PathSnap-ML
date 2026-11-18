"""Build the baseline candidate-feature table for train and student_test.

Writes data/processed/baseline_candidates.csv.gz (gitignored: it is derived
from the dataset) in the shared contract, and results/candidate_generation_summary.csv.
Requires results/labels/observation_labels.csv.gz (run build_labels.py first).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathsnap.candidates_baseline import build_trip_candidates  # noqa: E402
from pathsnap.data_io import find_dataset_root  # noqa: E402
from pathsnap.labels import candidate_miss_rate, label_candidates  # noqa: E402
from pathsnap.preprocess import load_trip  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "results/labels/observation_labels.csv.gz")
    parser.add_argument("--radius-m", type=float, default=50.0)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/processed/baseline_candidates.csv.gz")
    args = parser.parse_args()

    root = find_dataset_root(args.data_root)
    labels = pd.read_csv(args.labels, dtype={"trip_id": str})
    frames, summary = [], []
    for partition in ("train", "student_test"):
        part_frames = []
        for trip_dir in sorted(p for p in (root / partition).iterdir() if p.is_dir()):
            trip = load_trip(trip_dir)
            cands = build_trip_candidates(trip, args.radius_m, args.max_candidates)
            trip_labels = labels[labels["trip_id"] == trip.trip_id]
            labelled = label_candidates(cands, trip_labels)
            labelled.insert(0, "partition", partition)
            part_frames.append(labelled)
            print(f"{partition}/{trip.trip_id}: {len(labelled)} candidate rows")
        part = pd.concat(part_frames, ignore_index=True)
        part_labels = labels[labels["partition"] == partition]
        valid_obs = int(part_labels["label_valid"].sum())
        summary.append(
            {
                "partition": partition,
                "radius_m": args.radius_m,
                "max_candidates": args.max_candidates,
                "valid_observations": valid_obs,
                "candidate_rows": len(part),
                "mean_candidates_per_labelled_observation": len(part) / part[["trip_id", "observation_id"]].drop_duplicates().shape[0],
                "candidate_miss_rate": candidate_miss_rate(part, part_labels),
            }
        )
        frames.append(part)

    table = pd.concat(frames, ignore_index=True)
    meta = ["partition", "trip_id", "observation_id", "candidate_arc_id", "label"]
    table = table[meta + [c for c in table.columns if c not in meta]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    out = PROJECT_ROOT / "results" / "candidate_generation_summary.csv"
    pd.DataFrame(summary).to_csv(out, index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
