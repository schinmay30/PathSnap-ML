"""Candidate analysis and coverage evaluation for PathSnap.

Audits candidate generator performance:
- Observation coverage (fraction with >= 1 candidate)
- Candidate count distributions
- Distance to road segment distributions
- Route-set coverage and candidate miss rate
- Identification of zero-candidate observations
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap.candidates import ObservationCandidates
from pathsnap.preprocess import TripData


@dataclass(frozen=True)
class TripCandidateStats:
    """Summary metrics for one trip's candidate generation."""

    partition: str
    trip_id: str
    total_observations: int
    covered_observations: int
    zero_candidate_observations: int
    coverage_rate: float
    mean_candidates: float
    median_candidates: float
    max_candidates: int
    mean_distance_m: float
    median_distance_m: float
    p95_distance_m: float
    route_unique_arcs: int
    route_arcs_recalled: int
    route_arc_recall: float
    route_set_covered_obs: int
    route_set_miss_rate: float


def compute_trip_candidate_stats(
    trip: TripData,
    partition: str,
    obs_candidates: List[ObservationCandidates],
) -> Tuple[TripCandidateStats, List[Dict[str, object]]]:
    """Calculate candidate coverage and distribution statistics for a single trip."""
    total_obs = len(obs_candidates)
    covered_obs = sum(1 for oc in obs_candidates if oc.has_candidates)
    zero_obs = total_obs - covered_obs
    coverage_rate = (covered_obs / total_obs) if total_obs > 0 else 0.0

    cand_counts = [oc.candidate_count for oc in obs_candidates]
    mean_cands = float(np.mean(cand_counts)) if cand_counts else 0.0
    median_cands = float(np.median(cand_counts)) if cand_counts else 0.0
    max_cands = int(max(cand_counts)) if cand_counts else 0

    all_dists: List[float] = []
    nearest_dists: List[float] = []
    for oc in obs_candidates:
        if oc.has_candidates:
            dists = [c.distance_m for c in oc.candidates]
            all_dists.extend(dists)
            nearest_dists.append(min(dists))

    mean_dist = float(np.mean(all_dists)) if all_dists else float("nan")
    median_dist = float(np.median(all_dists)) if all_dists else float("nan")
    p95_dist = float(np.percentile(all_dists, 95)) if all_dists else float("nan")

    # Reference route analysis (without assuming 1-to-1 row alignment)
    route_set = set(trip.route)
    unique_route_arcs = len(route_set)

    all_cand_arcs_set = set()
    route_covered_count = 0
    zero_records: List[Dict[str, object]] = []

    for oc in obs_candidates:
        if not oc.has_candidates:
            zero_records.append(
                {
                    "partition": partition,
                    "trip_id": trip.trip_id,
                    "observation_id": oc.observation_id,
                    "timestamp": oc.timestamp,
                    "longitude": oc.longitude,
                    "latitude": oc.latitude,
                }
            )
        else:
            cand_arc_ids = {c.candidate_arc_id for c in oc.candidates}
            all_cand_arcs_set.update(cand_arc_ids)
            if cand_arc_ids.intersection(route_set):
                route_covered_count += 1

    recalled_route_arcs = len(all_cand_arcs_set.intersection(route_set))
    route_arc_recall = (
        (recalled_route_arcs / unique_route_arcs) if unique_route_arcs > 0 else 0.0
    )
    route_set_miss_rate = (
        (1.0 - (route_covered_count / total_obs)) if total_obs > 0 else 0.0
    )

    stats = TripCandidateStats(
        partition=partition,
        trip_id=trip.trip_id,
        total_observations=total_obs,
        covered_observations=covered_obs,
        zero_candidate_observations=zero_obs,
        coverage_rate=coverage_rate,
        mean_candidates=mean_cands,
        median_candidates=median_cands,
        max_candidates=max_cands,
        mean_distance_m=mean_dist,
        median_distance_m=median_dist,
        p95_distance_m=p95_dist,
        route_unique_arcs=unique_route_arcs,
        route_arcs_recalled=recalled_route_arcs,
        route_arc_recall=route_arc_recall,
        route_set_covered_obs=route_covered_count,
        route_set_miss_rate=route_set_miss_rate,
    )

    return stats, zero_records


def plot_candidate_distributions(
    candidate_records_df: pd.DataFrame,
    trip_stats_df: pd.DataFrame,
    output_png_path: Path,
) -> None:
    """Generate diagnostic visualization plots for candidate generation."""
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Candidate count distribution per observation
    ax1 = axes[0, 0]
    valid_cands = candidate_records_df[candidate_records_df["candidate_arc_id"] >= 0]
    counts_per_obs = valid_cands.groupby(["trip_id", "observation_id"]).size()
    max_c = int(counts_per_obs.max()) if not counts_per_obs.empty else 10
    bins = np.arange(0.5, max_c + 1.5, 1.0)
    ax1.hist(
        counts_per_obs,
        bins=bins,
        color="#2b5c8f",
        edgecolor="white",
        rwidth=0.85,
    )
    ax1.set_title("Candidate Counts per GPS Observation", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Number of Candidates")
    ax1.set_ylabel("Observation Frequency")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Distance distribution to candidates
    ax2 = axes[0, 1]
    dists = valid_cands["distance_m"].dropna()
    ax2.hist(dists, bins=50, color="#198754", edgecolor="white", density=True)
    ax2.axvline(
        float(dists.median()),
        color="red",
        linestyle="--",
        label=f"Median: {dists.median():.1f}m",
    )
    ax2.axvline(
        float(dists.quantile(0.95)),
        color="orange",
        linestyle=":",
        label=f"95th %: {dists.quantile(0.95):.1f}m",
    )
    ax2.set_title("Distance Distribution from GPS to Candidates (m)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Distance (meters)")
    ax2.set_ylabel("Density")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. Trip coverage distribution
    ax3 = axes[1, 0]
    cov_pct = trip_stats_df["coverage_rate"] * 100.0
    ax3.hist(cov_pct, bins=20, color="#0d6efd", edgecolor="white")
    ax3.set_title("Trip-Level GPS Candidate Coverage (%)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Coverage Rate (%)")
    ax3.set_ylabel("Trip Count")
    ax3.grid(True, linestyle="--", alpha=0.5)

    # 4. Route Arc Recall distribution
    ax4 = axes[1, 1]
    recall_pct = trip_stats_df["route_arc_recall"] * 100.0
    ax4.hist(recall_pct, bins=20, color="#6f42c1", edgecolor="white")
    ax4.set_title("Trip Route Arc Recall in Candidate Sets (%)", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Route Arc Recall (%)")
    ax4.set_ylabel("Trip Count")
    ax4.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_png_path, dpi=200)
    plt.close(fig)
