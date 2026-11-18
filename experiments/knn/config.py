"""Configuration dataclasses for PathSnap Person 4 KNN experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass

VALID_WEIGHTS = {"uniform", "distance"}
VALID_ALGORITHMS = {"auto", "ball_tree", "kd_tree", "brute"}


@dataclass(frozen=True)
class KNNConfig:
    """Configurable hyperparameters and execution settings for KNN."""

    n_neighbors: int = 15
    weights: str = "distance"
    metric: str = "euclidean"
    algorithm: str = "kd_tree"
    n_jobs: int = -1
    leaf_size: int = 30
    p: int = 2
    n_splits: int = 5
    random_state: int = 42
    batch_size: int = 50_000
    max_train_rows_per_fold: int | None = None
    use_float32: bool = True

    def __post_init__(self) -> None:
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")
        if self.weights not in VALID_WEIGHTS:
            raise ValueError(f"weights must be one of {sorted(VALID_WEIGHTS)}")
        if self.algorithm not in VALID_ALGORITHMS:
            raise ValueError(f"algorithm must be one of {sorted(VALID_ALGORITHMS)}")
        if not self.metric or not isinstance(self.metric, str):
            raise ValueError("metric must be a non-empty distance metric name")
        if self.leaf_size < 1:
            raise ValueError("leaf_size must be >= 1")
        if self.p < 1:
            raise ValueError("p must be >= 1")
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2 for GroupKFold")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.max_train_rows_per_fold is not None and self.max_train_rows_per_fold < self.n_neighbors:
            raise ValueError("max_train_rows_per_fold must be >= n_neighbors")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ResourceBudget:
    """Safety limits to guard against loading excessively large CSVs blindly."""

    max_memory_mb: float = 2048.0
    sample_rows: int = 5_000
    chunk_rows: int = 100_000

    def __post_init__(self) -> None:
        if self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be > 0")
        if self.sample_rows < 10:
            raise ValueError("sample_rows must be >= 10")
        if self.chunk_rows < 100:
            raise ValueError("chunk_rows must be >= 100")
