"""Readers for PathSnap's headerless whitespace-separated trip files."""

from __future__ import annotations

from pathlib import Path
def _read_rows(path: str | Path, expected_columns: int, cast) -> list[tuple]:
    """Read a whitespace-separated file and report malformed line numbers."""
    path = Path(path)
    rows: list[tuple] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            values = line.split()
            if not values:
                continue
            if len(values) != expected_columns:
                raise ValueError(
                    f"{path}:{line_number}: expected {expected_columns} columns, "
                    f"found {len(values)}"
                )
            try:
                rows.append(tuple(cast(value) for value in values))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid value: {exc}") from exc
    return rows


def read_track(path: str | Path) -> list[tuple[float, float, float]]:
    """Read GPS observations as (longitude, latitude, timestamp_seconds)."""
    return _read_rows(path, 3, float)


def read_nodes(path: str | Path) -> list[tuple[float, float]]:
    """Read road nodes as (longitude, latitude), indexed by row number."""
    return _read_rows(path, 2, float)


def read_arcs(path: str | Path) -> list[tuple[int, int]]:
    """Read arcs as (from_node_id, to_node_id), indexed by row number."""
    return _read_rows(path, 2, int)


def read_route(path: str | Path) -> list[int]:
    """Read the supplied chronological sequence of arc IDs."""
    values: list[int] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 1:
                raise ValueError(
                    f"{path}:{line_number}: expected one route ID, found {len(fields)}"
                )
            try:
                values.append(int(fields[0]))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid arc ID") from exc
    return values


def find_dataset_root(path: str | Path) -> Path:
    """Resolve an extracted folder containing train/ and student_test/."""
    root = Path(path).expanduser().resolve()
    if (root / "train").is_dir() and (root / "student_test").is_dir():
        return root
    matches = [
        candidate
        for candidate in root.rglob("*")
        if candidate.is_dir()
        and (candidate / "train").is_dir()
        and (candidate / "student_test").is_dir()
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Could not find a folder containing train/ and student_test/ under {root}"
        )
    raise ValueError(
        "More than one dataset root found; pass the exact folder containing "
        "train/ and student_test/"
    )
