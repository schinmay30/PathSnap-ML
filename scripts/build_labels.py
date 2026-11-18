"""Build per-observation reference-arc labels for train and student_test trips.

Output (one row per GPS observation) is the shared label table that candidate
generation / feature engineering should join on (trip_id, observation_id):

    trip_id, observation_id, route_index, arc_id, dist_to_arc_m, label_valid

`observation_id` is the 0-based row index in the trip's .track file.
A candidate row gets label=1 when its arc equals `arc_id` of that observation
and `label_valid` is True. Evaluator data is never read.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathsnap.data_io import find_dataset_root  # noqa: E402
from pathsnap.labels import (  # noqa: E402
    DEFAULT_MAX_DIST_M,
    align_trip_to_route,
    summarize_alignment,
)
from pathsnap.preprocess import load_trip  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--max-dist-m", type=float, default=DEFAULT_MAX_DIST_M)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "results" / "labels"
    )
    args = parser.parse_args()

    root = find_dataset_root(args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_labels, quality = [], []
    for partition in ("train", "student_test"):
        for trip_dir in sorted(p for p in (root / partition).iterdir() if p.is_dir()):
            trip = load_trip(trip_dir)
            labels = align_trip_to_route(trip, max_dist_m=args.max_dist_m)
            labels.insert(0, "partition", partition)
            all_labels.append(labels)
            row = {"partition": partition, "trip_id": trip.trip_id}
            row.update(summarize_alignment(labels, len(trip.route)))
            quality.append(row)
            print(f"{partition}/{trip.trip_id}: valid={row['valid_fraction']:.3f}")

    labels = pd.concat(all_labels, ignore_index=True)
    labels.to_csv(args.output_dir / "observation_labels.csv.gz", index=False)
    pd.DataFrame(quality).to_csv(args.output_dir / "label_quality_per_trip.csv", index=False)
    print(f"Saved labels for {labels['trip_id'].nunique()} trips, {len(labels)} observations")


if __name__ == "__main__":
    main()
