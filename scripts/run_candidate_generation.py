"""CLI script to run Road-Segment Candidate Generation & Analysis for PathSnap (Person 2 - Sujeet).

Processes training and student_test partitions:
- Constructs road-network geometries and local metric spatial indexes
- Generates ranked candidate road segments for each GPS observation
- Computes candidate coverage, distance distributions, and route-set recall
- Generates candidate features and aligned reference labels
- Produces diagnostic charts and interactive Folium HTML maps
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathsnap.candidate_analysis import (  # noqa: E402
    compute_trip_candidate_stats,
    plot_candidate_distributions,
)
from pathsnap.candidates import (  # noqa: E402
    ObservationCandidates,
    RoadNetworkIndex,
    generate_candidate_features_for_trip,
    generate_candidate_records_for_trip,
)
from pathsnap.data_io import find_dataset_root  # noqa: E402
from pathsnap.preprocess import TripData, load_trip  # noqa: E402
from pathsnap.visualize import create_trip_folium_map  # noqa: E402


def resolve_partitions(data_root: Path) -> dict[str, Path]:
    """Identify train and student_test partitions; strictly excludes evaluator data."""
    root = data_root.expanduser().resolve()
    train_dir = None
    test_dir = None

    candidates_train = [
        root / "train",
        root / "train-20261118T042214Z-1-001" / "train",
    ]
    for c in candidates_train:
        if c.is_dir():
            train_dir = c
            break

    candidates_test = [
        root / "student_test",
        root / "student_test-20261118T042236Z-1-001" / "student_test",
    ]
    for c in candidates_test:
        if c.is_dir():
            test_dir = c
            break

    if not train_dir or not test_dir:
        # Fall back to recursive search for standard names
        for path in root.rglob("*"):
            if path.is_dir():
                if path.name == "train" and (path.parent / "train").is_dir() and not train_dir:
                    train_dir = path
                if path.name == "student_test" and not test_dir:
                    test_dir = path

    if not train_dir or not test_dir:
        raise FileNotFoundError(
            f"Could not locate both 'train' and 'student_test' partitions under {data_root}"
        )

    return {"train": train_dir, "student_test": test_dir}


def run_pipeline(
    data_root: Path,
    output_dir: Path,
    radius_m: float = 50.0,
    max_candidates: int = 10,
    max_trips_per_partition: int | None = None,
    create_maps: bool = True,
) -> None:
    t_start = time.perf_counter()
    partitions = resolve_partitions(data_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = PROJECT_ROOT / "results" / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("PATHSNAP CANDIDATE GENERATION PIPELINE (PERSON 2 - SUJEET)")
    print("==================================================================")
    print(f"Data Root: {data_root.resolve()}")
    print(f"Training Partition: {partitions['train']}")
    print(f"Student Test Partition: {partitions['student_test']}")
    print(f"Parameters: search_radius={radius_m}m, max_candidates={max_candidates}")
    print(f"Output Directory: {output_dir.resolve()}")

    all_candidate_records: list[dict] = []
    all_feature_records: list[dict] = []
    all_trip_stats: list[dict] = []
    all_zero_records: list[dict] = []

    maps_generated = 0
    max_maps_to_generate = 4 if create_maps else 0

    for part_name, part_path in partitions.items():
        trip_dirs = sorted([d for d in part_path.iterdir() if d.is_dir()])
        if max_trips_per_partition:
            trip_dirs = trip_dirs[:max_trips_per_partition]

        print(f"\nProcessing partition '{part_name}' ({len(trip_dirs)} trips)...")

        for trip_idx, trip_folder in enumerate(trip_dirs, start=1):
            trip_id = trip_folder.name
            t0 = time.perf_counter()
            trip = load_trip(trip_folder)

            # 1. Generate candidates & zero records
            cand_recs, zero_recs = generate_candidate_records_for_trip(
                trip=trip,
                partition=part_name,
                radius_m=radius_m,
                max_candidates=max_candidates,
                include_zero_candidates=False,
            )
            all_candidate_records.extend(cand_recs)
            all_zero_records.extend(zero_recs)

            # 2. Generate candidate features & aligned labels
            feat_recs, _ = generate_candidate_features_for_trip(
                trip=trip,
                partition=part_name,
                radius_m=radius_m,
                max_candidates=max_candidates,
                align_labels=True,
            )
            all_feature_records.extend(feat_recs)

            # 3. Compute detailed trip stats
            index = RoadNetworkIndex(trip.nodes, trip.arcs)
            obs_cands = index.query_trip(
                trip, radius_m=radius_m, max_candidates=max_candidates
            )
            stats, _ = compute_trip_candidate_stats(trip, part_name, obs_cands)
            all_trip_stats.append(stats.__dict__)

            # 4. Generate sample Folium map if within limit
            if maps_generated < max_maps_to_generate and (
                trip_idx == 1 or stats.zero_candidate_observations > 0
            ):
                map_file = viz_dir / f"{part_name}_{trip_id}_candidates.html"
                create_trip_folium_map(
                    trip=trip,
                    obs_candidates=obs_cands,
                    output_html_path=map_file,
                )
                print(f"  [Map] Generated interactive map: {map_file.name}")
                maps_generated += 1

            t_elapsed = time.perf_counter() - t0
            print(
                f"  [{trip_idx}/{len(trip_dirs)}] Trip {trip_id}: {stats.total_observations} obs, "
                f"cov={stats.coverage_rate*100:.1f}%, mean_cands={stats.mean_candidates:.1f}, "
                f"route_recall={stats.route_arc_recall*100:.1f}%, time={t_elapsed:.2f}s"
            )

    # Export CSV tables
    print("\nWriting output CSV files...")
    df_cands = pd.DataFrame(all_candidate_records)
    df_cands.to_csv(output_dir / "candidate_records.csv", index=False)

    df_feats = pd.DataFrame(all_feature_records)
    features_csv = output_dir / "candidate_features.csv"
    df_feats.to_csv(features_csv, index=False)
    print(f"  Saved candidate-feature table: {features_csv} ({len(df_feats)} rows)")

    df_stats = pd.DataFrame(all_trip_stats)
    df_stats.to_csv(output_dir / "trip_candidate_summary.csv", index=False)

    df_zeros = pd.DataFrame(all_zero_records)
    df_zeros.to_csv(output_dir / "zero_candidate_observations.csv", index=False)
    print(f"  Zero-candidate observations recorded: {len(df_zeros)}")

    # Generate summary diagnostic plot
    plot_file = output_dir / "candidate_distributions.png"
    plot_candidate_distributions(df_cands, df_stats, plot_file)
    print(f"  Saved distribution plots: {plot_file}")

    total_time = time.perf_counter() - t_start
    print("\n==================================================================")
    print("CANDIDATE GENERATION PIPELINE COMPLETE")
    print("==================================================================")
    print(f"Total Trips Processed: {len(df_stats)}")
    print(f"Total GPS Observations: {df_stats['total_observations'].sum():,}")
    print(f"Total Road Segment Candidates: {len(df_cands):,}")
    overall_cov = (
        df_stats["covered_observations"].sum()
        / df_stats["total_observations"].sum()
        * 100.0
    )
    print(f"Overall Candidate Coverage: {overall_cov:.2f}%")
    print(f"Zero-Candidate Observations: {len(df_zeros)}")
    print(f"Average Candidates per GPS Point: {df_stats['mean_candidates'].mean():.2f}")
    print(f"Median Candidate Distance: {df_stats['median_distance_m'].median():.2f}m")
    print(f"Average Route Arc Recall: {df_stats['route_arc_recall'].mean()*100:.2f}%")
    print(f"Elapsed Time: {total_time:.2f} seconds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("E:/ML hackathon"),
        help="Root folder containing train and student_test partitions",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "candidates",
        help="Destination directory for candidate tables and reports",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=50.0,
        help="Candidate search radius in meters (default: 50.0)",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=10,
        help="Maximum candidates per GPS observation (default: 10)",
    )
    parser.add_argument(
        "--max-trips",
        type=int,
        default=None,
        help="Limit number of trips per partition (for testing/quick runs)",
    )
    parser.add_argument(
        "--no-maps",
        action="store_true",
        help="Disable interactive Folium map generation",
    )
    args = parser.parse_args()

    run_pipeline(
        data_root=args.data_root,
        output_dir=args.output_dir,
        radius_m=args.radius,
        max_candidates=args.max_candidates,
        max_trips_per_partition=args.max_trips,
        create_maps=not args.no_maps,
    )


if __name__ == "__main__":
    main()
