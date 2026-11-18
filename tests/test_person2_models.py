"""Unit tests for pathsnap.person2_models and bayesian_opt (Person 2 - Sujeet)."""

import numpy as np
import pandas as pd
import pytest

from pathsnap.bayesian_opt import (
    GaussianProcessBayesianOptimizer,
    Hyperparameter,
)
from pathsnap.person2_models import build_models, build_pipeline


def test_build_models_structure():
    models = build_models(random_state=42)
    assert "decision_tree" in models
    assert "random_forest" in models
    assert hasattr(models["decision_tree"], "fit")
    assert hasattr(models["decision_tree"], "predict_proba")
    assert hasattr(models["random_forest"], "fit")
    assert hasattr(models["random_forest"], "predict_proba")


def test_models_fit_and_predict_proba():
    models = build_models(random_state=42)

    # Synthetic candidate features
    np.random.seed(42)
    N = 100
    X = pd.DataFrame(
        {
            "distance_m": np.random.uniform(0.0, 50.0, N),
            "projection_t": np.random.uniform(0.0, 1.0, N),
            "segment_length_m": np.random.uniform(20.0, 200.0, N),
            "heading_diff_deg": np.random.uniform(0.0, 180.0, N),
            "speed_mps": np.random.uniform(5.0, 25.0, N),
            "log_distance": np.random.uniform(0.0, 4.0, N),
        }
    )
    # Synthetic label: closer distance -> higher probability of positive label
    y = (X["distance_m"] < 15.0).astype(int).to_numpy()

    for name, model in models.items():
        model.fit(X, y)
        probs = model.predict_proba(X)
        assert probs.shape == (N, 2)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        assert np.allclose(probs.sum(axis=1), 1.0)


def test_build_pipeline_custom_params():
    pipe_dt = build_pipeline(
        "decision_tree",
        {"max_depth": 7, "min_samples_leaf": 10},
        random_state=123,
    )
    clf = pipe_dt.named_steps["classifier"]
    assert clf.max_depth == 7
    assert clf.min_samples_leaf == 10
    assert clf.random_state == 123


def test_bayesian_optimizer_loop():
    param_space = [
        Hyperparameter("max_depth", "int", low=3, high=12),
        Hyperparameter("min_samples_leaf", "int", low=2, high=10),
    ]
    opt = GaussianProcessBayesianOptimizer(param_space, random_state=42)

    # Synthetic noisy quadratic objective to maximize: -(x-7)^2 - (y-4)^2
    for trial_idx in range(1, 6):
        params, gp_mean, gp_std, ei = opt.suggest_next_params(n_candidates=100)
        score = -(params["max_depth"] - 7) ** 2 - (params["min_samples_leaf"] - 4) ** 2
        opt.register_trial(
            trial_idx=trial_idx,
            params=params,
            score=score,
            top1_accuracy=0.8,
            log_loss=0.3,
            gp_mean=gp_mean,
            gp_std=gp_std,
            ei=ei,
            duration=0.01,
        )

    best = opt.get_best_trial()
    assert best.score <= 0.0
    df = opt.export_trials_dataframe()
    assert len(df) == 5
    assert "param_max_depth" in df.columns
    assert "cv_macro_f1" in df.columns
