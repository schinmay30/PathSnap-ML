"""Reusable KNN training, benchmarking, and evaluation package for PathSnap (Person 4)."""

from .config import KNNConfig, ResourceBudget
from .data_contract import (
    FORBIDDEN_FEATURE_COLUMNS,
    METADATA_COLUMNS,
    MemoryGuardError,
    DatasetResourceEstimate,
    estimate_csv_resources,
    load_candidate_dataset,
)
from .evaluate import (
    evaluate_knn_experiment,
    observation_scores,
    score_predictions,
)
from .model import build_knn_pipeline, predict_positive_proba

__all__ = [
    "DatasetResourceEstimate",
    "FORBIDDEN_FEATURE_COLUMNS",
    "KNNConfig",
    "METADATA_COLUMNS",
    "MemoryGuardError",
    "ResourceBudget",
    "build_knn_pipeline",
    "estimate_csv_resources",
    "evaluate_knn_experiment",
    "load_candidate_dataset",
    "observation_scores",
    "predict_positive_proba",
    "score_predictions",
]
