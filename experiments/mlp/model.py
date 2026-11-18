"""Configurable, leakage-safe scikit-learn MLP pipeline and inference utilities."""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.knn.model import (
    predict_positive_proba as _batched_predict_positive_proba,
    select_training_subset,
)

from .config import MLPConfig


def build_mlp_pipeline(config: MLPConfig) -> Pipeline:
    """Create a fresh StandardScaler + MLPClassifier pipeline.

    Feature scaling is strictly encapsulated inside the Pipeline so that when
    fit on a training fold, validation and student_test statistics can never
    leak across folds.
    """
    mlp_kwargs: dict[str, object] = {
        "hidden_layer_sizes": config.hidden_layer_sizes,
        "activation": config.activation,
        "solver": config.solver,
        "alpha": config.alpha,
        "max_iter": config.max_iter,
        "random_state": config.random_state,
    }

    if config.solver in {"adam", "sgd"}:
        mlp_kwargs.update(
            {
                "batch_size": config.batch_size,
                "learning_rate_init": config.learning_rate_init,
                "early_stopping": config.early_stopping,
                "n_iter_no_change": config.n_iter_no_change,
            }
        )
        if config.early_stopping:
            mlp_kwargs["validation_fraction"] = config.validation_fraction
        if config.solver == "sgd":
            mlp_kwargs["learning_rate"] = config.learning_rate

    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", MLPClassifier(**mlp_kwargs)),
        ]
    )


def predict_positive_proba(
    pipeline: Pipeline,
    X: np.ndarray,
    batch_size: int = 50_000,
) -> np.ndarray:
    """Predict positive-class probabilities in memory-bounded batches."""
    return _batched_predict_positive_proba(pipeline, X, batch_size=batch_size)


def predict_classes(
    pipeline: Pipeline,
    X: np.ndarray,
    threshold: float = 0.5,
    batch_size: int = 50_000,
) -> np.ndarray:
    """Predict binary candidate labels {0, 1} using batched probability inference."""
    probs = predict_positive_proba(pipeline, X, batch_size=batch_size)
    return (probs >= threshold).astype(int)


def extract_training_history(pipeline: Pipeline) -> dict[str, object]:
    """Extract genuine epoch-wise training loss and validation score history.

    Only records real optimization histories produced by the fitted MLPClassifier.
    """
    classifier = pipeline.named_steps["classifier"]
    raw_loss = getattr(classifier, "loss_curve_", None) or []
    raw_val = getattr(classifier, "validation_scores_", None) or []
    loss_curve = [float(v) for v in raw_loss]
    val_scores = [float(v) for v in raw_val]
    return {
        "n_iter": int(getattr(classifier, "n_iter_", len(loss_curve))),
        "loss_curve": loss_curve,
        "validation_scores": val_scores,
        "final_training_loss": float(loss_curve[-1]) if loss_curve else None,
        "best_validation_score": float(max(val_scores)) if val_scores else None,
    }


__all__ = [
    "build_mlp_pipeline",
    "extract_training_history",
    "predict_classes",
    "predict_positive_proba",
    "select_training_subset",
]
