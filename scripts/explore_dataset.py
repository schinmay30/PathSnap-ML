"""Dataset exploration outputs for train and student_test (never evaluator data).

Writes, under --output-dir (default results/):
  trajectory_stats.csv       per-trip GPS length/extent/speed/sampling stats
  network_connectivity.csv   per-trip road-graph size and connectivity
  label_quality_per_trip.csv GPS-to-reference-route alignment quality
  figures/*.png              EDA plots from these tables and sample trips
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathsnap.data_io import find_dataset_root  # noqa: E402
from pathsnap.geo import haversine_m, project_local  # noqa: E402
from pathsnap.labels import align_trip_to_route, summarize_alignment  # noqa: E402
from pathsnap.network import network_stats  # noqa: E402
from pathsnap.preprocess import TripData, load_trip  # noqa: E402

# Colour-blind-safe, no blue: black / orange / green / red / grey.
COLORS = {"gps": "#d95f02", "route": "#1b7837", "net": "#bdbdbd", "bad": "#b2182b", "ink": "#222222"}
MAX_PLAUSIBLE_SPEED_MS = 60.0  # ~216 km/h; faster steps are treated as jumps
GAP_SECONDS = 5.0


def trajectory_stats(trip: TripData) -> dict[str, float]:
    track = np.asarray(trip.track, dtype=float)
    lon, lat, ts = track[:, 0], track[:, 1], track[:, 2]
    dt = np.diff(ts)
    step_m = haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:])
    valid = dt > 0
    speed = np.where(valid, step_m / np.where(valid, dt, 1.0), np.nan)
    lat0 = float(lat.mean())
    xy = project_local(lon, lat, lat0)
    contiguous = dt <= GAP_SECONDS
    return {
        "observations": len(track),
        "duration_s": float(ts[-1] - ts[0]),
        "path_length_km": float(step_m[contiguous].sum() / 1000.0),
        "extent_width_km": float((xy[:, 0].max() - xy[:, 0].min()) / 1000.0),
        "extent_height_km": float((xy[:, 1].max() - xy[:, 1].min()) / 1000.0),
        "share_steps_exactly_1s": float(np.mean(dt == 1.0)),
        "median_step_s": float(np.median(dt)),
        "gaps_over_5s": int((dt > GAP_SECONDS).sum()),
        "max_gap_s": float(dt.max()),
        "median_speed_ms": float(np.nanmedian(speed[contiguous])) if contiguous.any() else np.nan,
        "p99_speed_ms": float(np.nanpercentile(speed[contiguous], 99)) if contiguous.any() else np.nan,
        "jump_steps_over_60ms": int(np.nansum(speed[contiguous] > MAX_PLAUSIBLE_SPEED_MS)),
        "stationary_steps": int(np.sum(step_m[contiguous] < 0.5)),
        "missing_or_nonfinite_values": int((~np.isfinite(track)).sum()),
        "exact_duplicate_rows": int(pd.DataFrame(track).duplicated().sum()),
    }


def sample_trip_figure(trip: TripData, labels: pd.DataFrame, path: Path) -> None:
    track = np.asarray(trip.track)
    nodes = np.asarray(trip.nodes)
    arcs = np.asarray(trip.arcs)
    route = np.asarray(trip.route)
    fig, ax = plt.subplots(figsize=(7, 7))
    # Only draw network arcs inside the (padded) GPS bounding box.
    pad = 0.002
    box = (track[:, 0].min() - pad, track[:, 0].max() + pad, track[:, 1].min() - pad, track[:, 1].max() + pad)
    a, b = nodes[arcs[:, 0]], nodes[arcs[:, 1]]
    inside = (
        (np.minimum(a[:, 0], b[:, 0]) <= box[1]) & (np.maximum(a[:, 0], b[:, 0]) >= box[0])
        & (np.minimum(a[:, 1], b[:, 1]) <= box[3]) & (np.maximum(a[:, 1], b[:, 1]) >= box[2])
    )
    for s, e in zip(a[inside], b[inside]):
        ax.plot([s[0], e[0]], [s[1], e[1]], color=COLORS["net"], lw=0.5, zorder=1)
    for arc in route:
        s, e = nodes[arcs[arc, 0]], nodes[arcs[arc, 1]]
        ax.plot([s[0], e[0]], [s[1], e[1]], color=COLORS["route"], lw=2.2, zorder=2)
    ok = labels["label_valid"].to_numpy()
    ax.scatter(track[ok, 0], track[ok, 1], s=3, color=COLORS["gps"], zorder=3)
    if (~ok).any():
        ax.scatter(track[~ok, 0], track[~ok, 1], s=6, color=COLORS["bad"], zorder=4)
    ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3])
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title(f"Trip {trip.trip_id}: network (grey), reference route (green), GPS (orange)")
    ax.set_aspect(1 / np.cos(np.radians(track[:, 1].mean())))
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def make_figures(traj: pd.DataFrame, net: pd.DataFrame, qual: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    parts = {"train": COLORS["gps"], "student_test": COLORS["route"]}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for part, colour in parts.items():
        sub = traj[traj.partition == part]
        axes[0].hist(sub.observations, bins=25, alpha=0.7, color=colour, label=part)
        axes[1].hist(sub.path_length_km, bins=25, alpha=0.7, color=colour, label=part)
        axes[2].hist(sub.share_steps_exactly_1s, bins=20, alpha=0.7, color=colour, label=part)
    axes[0].set_xlabel("GPS observations per trip"); axes[0].set_yscale("log")
    axes[1].set_xlabel("path length per trip (km, gaps excluded)")
    axes[2].set_xlabel("share of steps exactly 1 s apart")
    for ax in axes:
        ax.set_ylabel("trips"); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "trip_size_length_sampling.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    gaps = traj[traj.gaps_over_5s > 0].sort_values("max_gap_s")
    ax.bar(range(len(gaps)), gaps.max_gap_s, color=COLORS["gps"])
    ax.set_yscale("log"); ax.set_ylabel("largest gap (s, log scale)")
    ax.set_xlabel("trips with at least one gap > 5 s (sorted)")
    ax.set_title("Largest timestamp gap per trip")
    fig.tight_layout(); fig.savefig(fig_dir / "largest_gap_per_trip.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(net.arcs, traj.observations, s=14, color=COLORS["ink"])
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("road arcs in trip network"); axes[0].set_ylabel("GPS observations")
    axes[1].hist(net.weak_components, bins=range(1, int(net.weak_components.max()) + 2), color=COLORS["route"])
    axes[1].set_xlabel("weakly connected components per trip network"); axes[1].set_ylabel("trips")
    fig.tight_layout(); fig.savefig(fig_dir / "network_size_and_components.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    ordered = qual.sort_values("median_dist_m").reset_index(drop=True)
    ax.vlines(ordered.index, ordered.median_dist_m, ordered.p95_dist_m, color=COLORS["net"], lw=2)
    ax.scatter(ordered.index, ordered.median_dist_m, s=14, color=COLORS["route"], label="median")
    ax.scatter(ordered.index, ordered.p95_dist_m, s=14, color=COLORS["gps"], label="95th percentile")
    ax.axhline(50, color=COLORS["bad"], ls="--", lw=1, label="50 m validity threshold")
    ax.set_yscale("log"); ax.set_ylabel("GPS distance to aligned reference arc (m)")
    ax.set_xlabel("trips (sorted by median distance)"); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "gps_to_reference_route_distance.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for part, colour in parts.items():
        sub = traj[traj.partition == part]
        ax.scatter(sub.lon_center, sub.lat_center, s=22, color=colour, label=part)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title("Trip centre locations (geographic spread)"); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "trip_locations.png", dpi=130); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--sample-trips", nargs="*", default=["train/00000001", "student_test/00000000", "train/00000054"])
    args = parser.parse_args()

    root = find_dataset_root(args.data_root)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    traj_rows, net_rows, qual_rows, cache = [], [], [], {}
    for partition in ("train", "student_test"):
        for trip_dir in sorted(p for p in (root / partition).iterdir() if p.is_dir()):
            trip = load_trip(trip_dir)
            key = {"partition": partition, "trip_id": trip.trip_id}
            t = trajectory_stats(trip)
            track = np.asarray(trip.track)
            t.update(lon_center=float(track[:, 0].mean()), lat_center=float(track[:, 1].mean()))
            traj_rows.append({**key, **t})
            net_rows.append({**key, **network_stats(trip)})
            labels = align_trip_to_route(trip)
            qual_rows.append({**key, **summarize_alignment(labels, len(trip.route))})
            if f"{partition}/{trip.trip_id}" in args.sample_trips:
                cache[f"{partition}/{trip.trip_id}"] = (trip, labels)
            print(f"{partition}/{trip.trip_id} done")

    traj, net, qual = map(pd.DataFrame, (traj_rows, net_rows, qual_rows))
    traj.to_csv(out / "trajectory_stats.csv", index=False)
    net.to_csv(out / "network_connectivity.csv", index=False)
    qual.to_csv(out / "label_quality_per_trip.csv", index=False)
    make_figures(traj, net, qual, out / "figures")
    for name, (trip, labels) in cache.items():
        sample_trip_figure(trip, labels, out / "figures" / f"sample_trip_{name.replace('/', '_')}.png")
    print(f"Wrote tables and figures to {out.resolve()}")


if __name__ == "__main__":
    main()
