from pathlib import Path

from app.core.config import Settings
from app.schemas.trips import TripSummary, TripsListResponse

REQUIRED_INFERENCE_EXTENSIONS = (".track", ".nodes", ".arcs")


def _count_non_empty_lines(file_path: Path) -> int:
    count = 0
    with file_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


class DatasetService:
    """Read-only service for inspecting trip folders on Laptop 1.

    Security & integrity rules:
    - Never modifies original dataset files.
    - Dynamically discovers trip directories without hardcoding trip IDs.
    - Clearly distinguishes `student_test` (development test partition) from
      any reserved final-evaluation partition configured at demo time.
    - Strictly isolates `.route` reference files; trip metadata endpoints
      never read or expose ground-truth routes.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def trip_data_dir(self) -> Path:
        return self._settings.trip_data_dir

    @property
    def partition_name(self) -> str:
        return self.trip_data_dir.name

    @property
    def partition_role(self) -> str:
        name = self.partition_name.lower()
        if name == "student_test":
            return "student_development_test"
        if name == "train":
            return "training"
        return "configured_inference_partition"

    def is_dataset_available(self) -> bool:
        return self.trip_data_dir.is_dir()

    def _is_valid_trip_dir(self, trip_dir: Path) -> bool:
        if not trip_dir.is_dir():
            return False
        trip_id = trip_dir.name
        return all(
            (trip_dir / f"{trip_id}{ext}").is_file()
            for ext in REQUIRED_INFERENCE_EXTENSIONS
        )

    def resolve_trip_dir(self, trip_id: str) -> Path | None:
        """Safely resolve a trip folder strictly inside `trip_data_dir`."""
        if not self.is_dataset_available():
            return None
        base_resolved = self.trip_data_dir.resolve()
        candidate = (base_resolved / trip_id).resolve()
        if candidate.parent != base_resolved:
            return None
        if not self._is_valid_trip_dir(candidate):
            return None
        return candidate

    def trip_exists(self, trip_id: str) -> bool:
        return self.resolve_trip_dir(trip_id) is not None

    def count_available_trips(self) -> int:
        if not self.is_dataset_available():
            return 0
        return sum(
            1
            for entry in self.trip_data_dir.iterdir()
            if self._is_valid_trip_dir(entry)
        )

    def list_trips(self) -> TripsListResponse:
        if not self.is_dataset_available():
            return TripsListResponse(
                count=0,
                partition=self.partition_name,
                partition_role=self.partition_role,
                trips=[],
            )

        summaries: list[TripSummary] = []
        for entry in sorted(self.trip_data_dir.iterdir(), key=lambda p: p.name):
            if not self._is_valid_trip_dir(entry):
                continue
            trip_id = entry.name
            # Intentionally read ONLY .track, .nodes, and .arcs — never .route.
            summaries.append(
                TripSummary(
                    trip_id=trip_id,
                    track_points=_count_non_empty_lines(entry / f"{trip_id}.track"),
                    num_nodes=_count_non_empty_lines(entry / f"{trip_id}.nodes"),
                    num_arcs=_count_non_empty_lines(entry / f"{trip_id}.arcs"),
                )
            )

        return TripsListResponse(
            count=len(summaries),
            partition=self.partition_name,
            partition_role=self.partition_role,
            trips=summaries,
        )
