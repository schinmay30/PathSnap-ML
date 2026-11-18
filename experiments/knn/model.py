"""Configurable, leakage-safe KNN pipeline and batched inference utilities."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import KNNConfig


def build_knn_pipeline(config: KNNConfig) -> Pipeline:
    """Create a fresh StandardScaler + KNeighborsClassifier pipeline.

    Scaling is strictly encapsulated inside the Pipeline so that when fit on a
    training fold, validation and student_test statistics can never leak in.
    """
    knn_kwargs: dict[str, object] = {
        "n_neighbors": config.n_neighbors,
        "weights": config.weights,
        "metric": config.metric,
        "algorithm": config.algorithm,
        "leaf_size": config.leaf_size,
        "n_jobs": config.n_jobs,
    }
    if config.metric == "minkowski":
        knn_kwargs["p"] = config.p

    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", KNeighborsClassifier(**knn_kwargs)),
        ]
    )


def select_training_subset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_rows: int | None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Optionally apply deterministic stratified subsampling strictly inside a train fold.

    Used only when explicitly configured via `KNNConfig.max_train_rows_per_fold`
    to bound KNN query latency/memory on multi-million-row training splits.
    Validation and student_test splits are NEVER subsampled.
    """
    n_rows = len(y_train)
    if max_rows is None or n_rows <= max_rows:
        return X_train, y_train, False

    rng = np.random.default_rng(random_state)
    pos_idx = np.flatnonzero(y_train == 1)
    neg_idx = np.flatnonzero(y_train == 0)

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        chosen = rng.choice(n_rows, size=max_rows, replace=False)
    else:
        pos_frac = len(pos_idx) / n_rows
        n_pos = max(1, min(len(pos_idx), int(round(max_rows * pos_frac))))
        n_neg = max(1, min(len(neg_idx), max_rows - n_pos))
        chosen = np.concatenate(
            [
                rng.choice(pos_idx, size=n_pos, replace=False),
                rng.choice(neg_idx, size=n_neg, replace=False),
            ]
        )
    chosen.sort()
    return X_train[chosen], y_train[chosen], True


def predict_positive_proba(
    pipeline: Pipeline,
    X: np.ndarray,
    batch_size: int = 50_000,
) -> np.ndarray:
    """Predict positive-class probabilities in memory-bounded batches."""
    classifier = pipeline.named_steps["classifier"]
    classes = list(classifier.classes_)
    n_samples = len(X)
    if n_samples == 0 or 1 not in classes:
        return np.zeros(n_samples, dtype=np.float64)

    pos_col = classes.index(1)
    if n_samples <= batch_size:
        return np.asarray(pipeline.predict_proba(X)[:, pos_col], dtype=np.float64)

    out = np.empty(n_samples, dtype=np.float64)
    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        out[start:stop] = pipeline.predict_proba(X[start:stop])[:, pos_col]
    return out
