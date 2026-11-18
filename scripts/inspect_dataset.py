"""Create a per-trip inventory for the permitted development partitions."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathsnap.data_io import (  # noqa: E402
    find_dataset_root,
    read_arcs,
    read_nodes,
    read_route,
    read_track,
)


REQUIRED_EXTENSIONS = (".track", ".nodes", ".arcs", ".route")
LARGE_GAP_SECONDS = 5.0


def _trip_files(trip_dir: Path) -> dict[str, Path | None]:
    found: dict[str, Path | None] = {}
    for extension in REQUIRED_EXTENSIONS:
        candidates = sorted(trip_dir.glob(f"*{extension}"))
        found[extension] = candidates[0] if len(candidates) == 1 else None
    return found


def _analyze_trip(partition: str, trip_dir: Path) -> dict[str, object]:
    files = _trip_files(trip_dir)
    row: dict[str, object] = {
        "partition": partition,
        "trip_id": trip_dir.name,
        "trip_folder": f"{partition}/{trip_dir.name}",
        "file_set_ok": all(files[ext] is not None for ext in REQUIRED_EXTENSIONS),
    }
    for extension, path in files.items():
        key = extension.removeprefix(".")
        row[f"has_{key}"] = path is not None
        row[f"{key}_bytes"] = path.stat().st_size if path else ""

    # Preserve malformed-file information in the inventory instead of stopping
    # the scan; teammates can then inspect the affected trip.
    for key in ("track", "nodes", "arcs", "route"):
        row[f"{key}_rows"] = ""
        row[f"{key}_error"] = ""

    parsed: dict[str, list] = {}
    readers = {
        ".track": ("track", read_track),
        ".nodes": ("nodes", read_nodes),
        ".arcs": ("arcs", read_arcs),
        ".route": ("route", read_route),
    }
    for extension, (key, reader) in readers.items():
        path = files[extension]
        if path is None:
            row[f"{key}_error"] = "missing or multiple matching files"
            continue
        try:
            values = reader(path)
            parsed[key] = values
            row[f"{key}_rows"] = len(values)
        except (OSError, ValueError) as exc:
            row[f"{key}_error"] = str(exc)

    track = parsed.get("track", [])
    if track:
        longitudes = [point[0] for point in track]
        latitudes = [point[1] for point in track]
        timestamps = [point[2] for point in track]
        deltas = [b - a for a, b in zip(timestamps, timestamps[1:])]
        row.update(
            {
                "longitude_min": min(longitudes),
                "longitude_max": max(longitudes),
                "latitude_min": min(latitudes),
                "latitude_max": max(latitudes),
                "timestamps_monotonic": all(delta >= 0 for delta in deltas),
                "nonpositive_time_steps": sum(delta <= 0 for delta in deltas),
                "non_1s_steps": sum(abs(delta - 1.0) > 1e-6 for delta in deltas),
                "gaps_over_5s": sum(delta > LARGE_GAP_SECONDS for delta in deltas),
                "max_time_gap_seconds": max(deltas) if deltas else 0.0,
                "coordinates_in_lon_lat_range": all(
                    -180 <= lon <= 180 and -90 <= lat <= 90
                    for lon, lat in zip(longitudes, latitudes)
                ),
            }
        )
    else:
        for name in (
            "longitude_min", "longitude_max", "latitude_min", "latitude_max",
            "timestamps_monotonic", "nonpositive_time_steps", "non_1s_steps",
            "gaps_over_5s", "max_time_gap_seconds", "coordinates_in_lon_lat_range",
        ):
            row[name] = ""

    nodes = parsed.get("nodes", [])
    arcs = parsed.get("arcs", [])
    route = parsed.get("route", [])
    row["arc_node_ids_in_range"] = (
        all(0 <= node_id < len(nodes) for arc in arcs for node_id in arc)
        if "nodes" in parsed and "arcs" in parsed
        else ""
    )
    row["route_arc_ids_in_range"] = (
        all(0 <= arc_id < len(arcs) for arc_id in route)
        if "arcs" in parsed and "route" in parsed
        else ""
    )
    return row


def inspect_dataset(data_root: Path) -> list[dict[str, object]]:
    root = find_dataset_root(data_root)
    records: list[dict[str, object]] = []
    for partition in ("train", "student_test"):
        partition_dir = root / partition
        for trip_dir in sorted(path for path in partition_dir.iterdir() if path.is_dir()):
            records.append(_analyze_trip(partition, trip_dir))
    return records


def write_csv(records: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for record in records for key in record))
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "results" / "dataset_inventory.csv"
    )
    args = parser.parse_args()

    records = inspect_dataset(args.data_root)
    write_csv(records, args.output)
    counts: dict[str, int] = {}
    for record in records:
        partition = str(record["partition"])
        counts[partition] = counts.get(partition, 0) + 1
    print(f"Dataset inventory saved: {args.output.resolve()}")
    for partition, count in counts.items():
        print(f"{partition}: {count} trips")
    issues = sum(
        bool(record.get("track_error") or record.get("nodes_error")
             or record.get("arcs_error") or record.get("route_error"))
        for record in records
    )
    print(f"Trips with missing or malformed required files: {issues}")


if __name__ == "__main__":
    main()
