"""Load and validate one trip while preserving its original observation order."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from pathsnap.data_io import read_arcs, read_nodes, read_route, read_track


@dataclass(frozen=True)
class TripData:
    trip_id: str
    track: list[tuple[float, float, float]]
    nodes: list[tuple[float, float]]
    arcs: list[tuple[int, int]]
    route: list[int]


def load_trip(trip_dir: str | Path) -> TripData:
    """Load the four same-trip files from a numbered trip directory.

    Route IDs are retained separately. This function does not assume that
    route rows align one-to-one with GPS observations.
    """
    folder = Path(trip_dir)
    trip_id = folder.name
    paths = {suffix: folder / f"{trip_id}{suffix}" for suffix in (
        ".track", ".nodes", ".arcs", ".route"
    )}
    missing = [str(path.name) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{folder}: missing required file(s): {', '.join(missing)}")
    trip = TripData(
        trip_id=trip_id,
        track=read_track(paths[".track"]),
        nodes=read_nodes(paths[".nodes"]),
        arcs=read_arcs(paths[".arcs"]),
        route=read_route(paths[".route"]),
    )
    issues = validate_trip(trip)
    if issues:
        raise ValueError(f"Trip {trip_id} failed validation: {'; '.join(issues)}")
    return trip


def validate_trip(trip: TripData) -> list[str]:
    """Return data-quality issues; do not silently drop or reorder records."""
    issues: list[str] = []
    if not trip.track:
        issues.append("track is empty")
    if not trip.nodes:
        issues.append("nodes are empty")
    if not trip.arcs:
        issues.append("arcs are empty")
    if not trip.route:
        issues.append("route is empty")

    for index, (longitude, latitude, timestamp) in enumerate(trip.track):
        if not all(math.isfinite(value) for value in (longitude, latitude, timestamp)):
            issues.append(f"track row {index} contains a non-finite value")
            break
        if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
            issues.append(f"track row {index} has coordinates outside lon/lat ranges")
            break
    for index, (longitude, latitude) in enumerate(trip.nodes):
        if not (math.isfinite(longitude) and math.isfinite(latitude)):
            issues.append(f"node row {index} contains a non-finite coordinate")
            break
        if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
            issues.append(f"node row {index} has coordinates outside lon/lat ranges")
            break

    timestamps = [point[2] for point in trip.track]
    if any(later <= earlier for earlier, later in zip(timestamps, timestamps[1:])):
        issues.append("timestamps are not strictly increasing")
    if any(
        node_id < 0 or node_id >= len(trip.nodes)
        for arc in trip.arcs
        for node_id in arc
    ):
        issues.append("an arc references a node ID outside the nodes file")
    if any(arc_id < 0 or arc_id >= len(trip.arcs) for arc_id in trip.route):
        issues.append("the route references an arc ID outside the arcs file")
    return issues
