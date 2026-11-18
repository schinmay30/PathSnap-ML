"""Shared candidate-feature CSV validation and memory-safe pre-flight checks.

Follows Person 1's shared contract:
    partition, trip_id, observation_id, candidate_arc_id, label, <numeric features...>

Key guarantees:
- Preserves leading zeros in `trip_id` by forcing `dtype={"trip_id": str}`.
- Dynamically detects numeric feature columns (schema-agnostic).
- Explicitly rejects metadata or ground-truth route leakage columns as features.
- Allows ONLY `train` and `student_test` partitions (never evaluator partitions).
- Estimates memory footprint in streaming chunks before loading millions of rows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ResourceBudget

METADATA_COLUMNS: frozenset[str] = frozenset(
    {"partition", "trip_id", "observation_id", "candidate_arc_id", "label"}
)
ALLOWED_PARTITIONS: frozenset[str] = frozenset({"train", "student_test"})

# Guard against accidental leakage of ground-truth alignment or route columns.
FORBIDDEN_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        "route",
        "route_index",
        "route_arc_id",
        "arc_id",
        "true_arc_id",
        "ground_truth",
        "ground_truth_arc_id",
        "label_valid",
        "dist_to_route_m",
    }
)


class MemoryGuardError(MemoryError):
    """Raised when estimated dataset memory exceeds the configured ResourceBudget."""


@dataclass(frozen=True)
class DatasetResourceEstimate:
    """Streaming pre-flight estimate of dataset size, feature schema, and RAM."""

    csv_path: str
    total_rows: int
    train_rows: int
    student_test_rows: int
    train_trips: int
    student_test_trips: int
    feature_columns: tuple[str, ...]
    estimated_dataframe_mb: float
    estimated_knn_working_mb: float
    max_memory_mb: float
    safe_to_load: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["feature_columns"] = list(self.feature_columns)
        return payload


def detect_feature_columns(columns: list[str] | pd.Index) -> list[str]:
    """Dynamically extract feature columns while enforcing contract & anti-leakage rules."""
    col_list = list(columns)
    missing_meta = METADATA_COLUMNS.difference(col_list)
    if missing_meta:
        raise ValueError(
            f"Candidate CSV is missing required metadata columns: {sorted(missing_meta)}"
        )

    leaked = [
        c for c in col_list if c not in METADATA_COLUMNS and c.lower() in FORBIDDEN_FEATURE_COLUMNS
    ]
    if leaked:
        raise ValueError(
            f"Potential ground-truth leakage column(s) detected in feature schema: {leaked}"
        )

    feature_columns = [c for c in col_list if c not in METADATA_COLUMNS]
    if not feature_columns:
        raise ValueError("No numeric feature columns found after metadata columns.")
    return feature_columns


def estimate_csv_resources(
    csv_path: str | Path,
    budget: ResourceBudget | None = None,
    use_float32: bool = True,
) -> DatasetResourceEstimate:
    """Stream the CSV in chunks to inspect schema and estimate memory before full load."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Candidate-feature CSV not found: {path}")

    cfg = budget or ResourceBudget()

    # 1. Inspect header + small sample for per-row memory measurement
    sample = pd.read_csv(path, nrows=cfg.sample_rows, dtype={"trip_id": str})
    feature_columns = detect_feature_columns(sample.columns)
    bytes_per_row = (
        float(sample.memory_usage(deep=True).sum()) / max(1, len(sample))
    )

    # 2. Stream partition and trip_id columns only to count rows/trips safely
    total_rows = 0
    train_rows = 0
    student_test_rows = 0
    train_trips: set[str] = set()
    student_test_trips: set[str] = set()

    reader = pd.read_csv(
        path,
        usecols=["partition", "trip_id"],
        dtype={"partition": str, "trip_id": str},
        chunksize=cfg.chunk_rows,
    )
    for chunk in reader:
        partitions = set(chunk["partition"].dropna().unique())
        unexpected = partitions.difference(ALLOWED_PARTITIONS)
        if unexpected:
            raise ValueError(
                "Only 'train' and 'student_test' partitions are permitted; "
                f"unexpected partition(s): {sorted(unexpected)}"
            )
        total_rows += len(chunk)
        train_mask = chunk["partition"] == "train"
        test_mask = chunk["partition"] == "student_test"
        train_rows += int(train_mask.sum())
        student_test_rows += int(test_mask.sum())
        if train_mask.any():
            train_trips.update(chunk.loc[train_mask, "trip_id"].dropna().tolist())
        if test_mask.any():
            student_test_trips.update(chunk.loc[test_mask, "trip_id"].dropna().tolist())

    estimated_df_mb = (total_rows * bytes_per_row) / (1024.0 * 1024.0)
    # Estimate working RAM: DataFrame + scaled float matrix + KD/Ball-tree overhead (~2.5x)
    bytes_per_feat = 4.0 if use_float32 else 8.0
    train_matrix_mb = (train_rows * len(feature_columns) * bytes_per_feat) / (1024.0 * 1024.0)
    estimated_knn_mb = estimated_df_mb + 2.5 * train_matrix_mb

    return DatasetResourceEstimate(
        csv_path=str(path),
        total_rows=total_rows,
        train_rows=train_rows,
        student_test_rows=student_test_rows,
        train_trips=len(train_trips),
        student_test_trips=len(student_test_trips),
        feature_columns=tuple(feature_columns),
        estimated_dataframe_mb=round(estimated_df_mb, 2),
        estimated_knn_working_mb=round(estimated_knn_mb, 2),
        max_memory_mb=cfg.max_memory_mb,
        safe_to_load=estimated_knn_mb <= cfg.max_memory_mb,
    )


def load_candidate_dataset(
    csv_path: str | Path,
    budget: ResourceBudget | None = None,
    use_float32: bool = True,
    enforce_memory_budget: bool = True,
) -> tuple[pd.DataFrame, list[str], DatasetResourceEstimate]:
    """Validate resources, then load and verify the candidate-feature CSV."""
    estimate = estimate_csv_resources(
        csv_path, budget=budget, use_float32=use_float32
    )
    if enforce_memory_budget and not estimate.safe_to_load:
        raise MemoryGuardError(
            f"Estimated KNN working memory ({estimate.estimated_knn_working_mb:.1f} MB) "
            f"exceeds configured budget ({estimate.max_memory_mb:.1f} MB) for "
            f"{estimate.total_rows:,} rows. Increase ResourceBudget.max_memory_mb or "
            "configure max_train_rows_per_fold."
        )

    feature_columns = list(estimate.feature_columns)
    dtype_map: dict[str, object] = {
        "partition": str,
        "trip_id": str,
    }
    table = pd.read_csv(csv_path, dtype=dtype_map)

    if table.empty:
        raise ValueError("Candidate-feature CSV is empty.")

    if table[list(METADATA_COLUMNS)].isna().any().any():
        raise ValueError("Metadata columns contain missing values.")

    partitions = set(table["partition"].unique())
    if not partitions.issubset(ALLOWED_PARTITIONS):
        raise ValueError(
            f"Unexpected partition(s) found: {sorted(partitions - ALLOWED_PARTITIONS)}"
        )
    if partitions != ALLOWED_PARTITIONS:
        raise ValueError(
            f"Dataset must contain both 'train' and 'student_test'; found {sorted(partitions)}"
        )

    table["observation_id"] = pd.to_numeric(table["observation_id"], errors="raise").astype(
        np.int64
    )
    table["candidate_arc_id"] = pd.to_numeric(
        table["candidate_arc_id"], errors="raise"
    ).astype(np.int64)
    table["label"] = pd.to_numeric(table["label"], errors="raise").astype(np.int8)
    if not set(table["label"].unique()).issubset({0, 1}):
        raise ValueError("`label` must contain only binary values 0 and 1.")

    float_dtype = np.float32 if use_float32 else np.float64
    for col in feature_columns:
        numeric = pd.to_numeric(table[col], errors="raise").to_numpy(dtype=float_dtype)
        if not np.isfinite(numeric).all():
            raise ValueError(
                f"Feature column '{col}' contains NaN or infinite values."
            )
        table[col] = numeric

    return table, feature_columns, estimate
