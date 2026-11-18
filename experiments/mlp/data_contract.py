"""Shared candidate-feature CSV contract reused directly from experiments.knn.

Re-exporting the exact KNN data contract guarantees that KNN and MLP share
identical metadata validation, zero-padded `trip_id` string handling, dynamic
feature detection, ground-truth leakage rejection, evaluator-partition blocking,
and streaming memory pre-flight checks.
"""

from __future__ import annotations

from experiments.knn.data_contract import (
    ALLOWED_PARTITIONS,
    DatasetResourceEstimate,
    FORBIDDEN_FEATURE_COLUMNS,
    METADATA_COLUMNS,
    MemoryGuardError,
    detect_feature_columns,
    estimate_csv_resources,
    load_candidate_dataset,
)

__all__ = [
    "ALLOWED_PARTITIONS",
    "DatasetResourceEstimate",
    "FORBIDDEN_FEATURE_COLUMNS",
    "METADATA_COLUMNS",
    "MemoryGuardError",
    "detect_feature_columns",
    "estimate_csv_resources",
    "load_candidate_dataset",
]
