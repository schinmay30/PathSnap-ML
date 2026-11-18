"""Logistic Regression and Gaussian Naive Bayes experiment helpers for Person 1."""

from __future__ import annotations

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def build_models(random_state: int = 42) -> dict[str, Pipeline]:
    """Return the two assigned probability-producing candidate classifiers."""
    return {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "gaussian_naive_bayes": Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", GaussianNB(var_smoothing=1e-9)),
            ]
        ),
    }
