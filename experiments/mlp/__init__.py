"""Reusable MLP training, benchmarking, and evaluation package for PathSnap (Person 4)."""

from .config import MLPConfig, ResourceBudget
from .data_contract import (
    ALLOWED_PARTITIONS,
    DatasetResourceEstimate,
    FORBIDDEN_FEATURE_COLUMNS,
    METADATA_COLUMNS,
    MemoryGuardError,
    detect_feature_columns,
    estimate_csv_resources,
    load_candidate_dataset,
)
from .evaluate import (
    benchmark_mlp_latency,
    compute_route_continuity,
    evaluate_mlp_experiment,
    observation_scores,
    save_mlp_artifacts,
    score_predictions,
)
from .model import (
    build_mlp_pipeline,
    extract_training_history,
    predict_classes,
    predict_positive_proba,
)

__all__ = [
    "ALLOWED_PARTITIONS",
    "DatasetResourceEstimate",
    "FORBIDDEN_FEATURE_COLUMNS",
    "METADATA_COLUMNS",
    "MLPConfig",
    "MemoryGuardError",
    "ResourceBudget",
    "benchmark_mlp_latency",
    "build_mlp_pipeline",
    "compute_route_continuity",
    "detect_feature_columns",
    "estimate_csv_resources",
    "evaluate_mlp_experiment",
    "extract_training_history",
    "load_candidate_dataset",
    "observation_scores",
    "predict_classes",
    "predict_positive_proba",
    "save_mlp_artifacts",
    "score_predictions",
]
