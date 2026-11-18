from dataclasses import dataclass, field
import os
from pathlib import Path


def _detect_workspace_root() -> Path:
    # backend/app/core/config.py -> parents[3] is workspace root (ML_hacthon)
    return Path(__file__).resolve().parents[3]


def _detect_default_trip_data_dir(workspace_root: Path) -> Path:
    env_override = os.getenv("PATHSNAP_TRIP_DATA_DIR") or os.getenv(
        "PATHSNAP_EVAL_DATA_DIR"
    )
    if env_override:
        return Path(env_override).expanduser().resolve()

    # Development default: student_test (NOT the reserved evaluator partition)
    candidates = [
        workspace_root / "Student folder" / "Student folder" / "student_test",
        workspace_root / "Student folder" / "student_test",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def _parse_cors_origins() -> list[str]:
    raw = os.getenv("PATHSNAP_CORS_ORIGINS", "")
    if raw.strip():
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]


@dataclass(frozen=True)
class Settings:
    app_name: str = "PathSnap FastAPI Backend"
    app_version: str = "0.1.0"
    app_description: str = (
        "Distributed Map-Matching & Route Reconstruction API (Laptop 1). "
        "Serves dynamic trip metadata from the configured development or "
        "deployment trip partition, model readiness status, recorded comparison "
        "metrics, and live route-matching inference with strict server-side "
        "ground-truth isolation."
    )
    workspace_root: Path = field(default_factory=_detect_workspace_root)
    trip_data_dir: Path = field(init=False)
    artifacts_final_dir: Path = field(init=False)
    artifacts_models_dir: Path = field(init=False)
    artifacts_metrics_dir: Path = field(init=False)
    backend_models_dir: Path = field(init=False)
    cors_origins: list[str] = field(default_factory=_parse_cors_origins)
    cors_origin_regex: str = (
        r"^http://(localhost|127\.0\.0\.1|"
        r"10\.\d+\.\d+\.\d+|"
        r"172\.(1[6-9]|2\d|3[01])\.\d+\.\d+|"
        r"192\.168\.\d+\.\d+)(:\d+)?$"
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "trip_data_dir", _detect_default_trip_data_dir(self.workspace_root)
        )
        object.__setattr__(
            self, "artifacts_final_dir", self.workspace_root / "artifacts" / "final"
        )
        object.__setattr__(
            self, "artifacts_models_dir", self.workspace_root / "artifacts" / "models"
        )
        object.__setattr__(
            self, "artifacts_metrics_dir", self.workspace_root / "artifacts" / "metrics"
        )
        object.__setattr__(
            self, "backend_models_dir", self.workspace_root / "backend" / "models"
        )


def get_settings() -> Settings:
    return Settings()
