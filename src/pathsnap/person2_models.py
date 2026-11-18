"""Decision Tree and Random Forest experiment helpers for Person 2 (Sujeet).

Implements tree-based and ensemble classifiers for candidate selection:
- DecisionTreeClassifier: interpretable, non-linear decision boundaries
- RandomForestClassifier: ensemble of bagged randomized decision trees
Both models output candidate-level positive probabilities for HMM emission
or top-1 candidate selection.
"""

from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def build_models(random_state: int = 42) -> dict[str, Pipeline]:
    """Return Person 2's assigned Decision Tree and Random Forest baseline pipelines."""
    return {
        "decision_tree": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        max_depth=10,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=12,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def build_pipeline(
    model_type: str,
    params: Dict[str, Any],
    random_state: int = 42,
) -> Pipeline:
    """Build a model pipeline with custom hyperparameters."""
    clf_params = dict(params)
    clf_params["random_state"] = random_state
    clf_params.setdefault("class_weight", "balanced")

    if model_type == "decision_tree":
        classifier = DecisionTreeClassifier(**clf_params)
    elif model_type == "random_forest":
        clf_params.setdefault("n_jobs", -1)
        classifier = RandomForestClassifier(**clf_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return Pipeline([("scale", StandardScaler()), ("classifier", classifier)])
