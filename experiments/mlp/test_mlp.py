"""Unit tests for Person 4's reusable MLP module using small synthetic fixtures ONLY.

Run with:
    .\\backend\\.venv\\Scripts\\python.exe -m pytest experiments\\mlp\\test_mlp.py -v
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.mlp.config import MLPConfig, ResourceBudget  # noqa: E402
from experiments.mlp.data_contract import (  # noqa: E402
    MemoryGuardError,
    detect_feature_columns,
    estimate_csv_resources,
    load_candidate_dataset,
)
from experiments.mlp.evaluate import (  # noqa: E402
    benchmark_mlp_latency,
    compute_route_continuity,
    evaluate_mlp_experiment,
    observation_scores,
    save_mlp_artifacts,
    score_predictions,
)
from experiments.mlp.model import (  # noqa: E402
    build_mlp_pipeline,
    extract_training_history,
    predict_classes,
    predict_positive_proba,
)


def _make_synthetic_candidate_df(
    feature_names: tuple[str, ...] = ("feat_x1", "feat_x2", "feat_x3"),
    n_train_trips: int = 5,
    n_test_trips: int = 2,
    obs_per_trip: int = 8,
    cands_per_obs: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a small synthetic candidate table with arbitrary feature names."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    trip_specs = [
        ("train", f"0000000{i}") for i in range(1, n_train_trips + 1)
    ] + [
        ("student_test", f"0000009{i}") for i in range(n_test_trips)
    ]

    for partition, trip_id in trip_specs:
        for obs_id in range(obs_per_trip):
            true_arc = obs_id
            for c_idx in range(cands_per_obs):
                arc_id = true_arc if c_idx == 0 else true_arc + 100 + c_idx
                is_pos = int(c_idx == 0)
                row: dict[str, object] = {
                    "partition": partition,
                    "trip_id": trip_id,
                    "observation_id": obs_id,
                    "candidate_arc_id": arc_id,
                    "label": is_pos,
                }
                for f_i, fname in enumerate(feature_names):
                    base = -2.2 if is_pos else 2.2
                    row[fname] = float(base + 0.1 * f_i + rng.normal(0.0, 0.35))
                rows.append(row)

    return pd.DataFrame(rows)


def test_shared_data_contract_preserves_trip_ids_and_rejects_leakage(tmp_path: Path):
    custom_features = ("mlp_f1", "mlp_f2", "mlp_f3", "mlp_f4")
    df = _make_synthetic_candidate_df(feature_names=custom_features)
    csv_path = tmp_path / "mlp_synth.csv"
    df.to_csv(csv_path, index=False)

    loaded, detected, estimate = load_candidate_dataset(csv_path)
    assert detected == list(custom_features)
    assert all(isinstance(t, str) and len(t) == 8 for t in loaded["trip_id"].unique())
    assert estimate.safe_to_load is True

    # Leakage columns must be rejected
    for forbidden in ("route", "route_index", "arc_id", "label_valid"):
        with pytest.raises(ValueError, match="leakage"):
            detect_feature_columns(
                ["partition", "trip_id", "observation_id", "candidate_arc_id", "label", forbidden]
            )

    # Evaluator partition must be rejected
    df.loc[0, "partition"] = "evaluator_test"
    bad_csv = tmp_path / "eval_leak.csv"
    df.to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="unexpected partition"):
        estimate_csv_resources(bad_csv)


def test_memory_guard_blocks_oversized_load(tmp_path: Path):
    df = _make_synthetic_candidate_df()
    csv_path = tmp_path / "tiny.csv"
    df.to_csv(csv_path, index=False)

    tiny_budget = ResourceBudget(max_memory_mb=0.0001, sample_rows=20, chunk_rows=100)
    with pytest.raises(MemoryGuardError):
        load_candidate_dataset(csv_path, budget=tiny_budget, enforce_memory_budget=True)


def test_configurable_mlp_hyperparameters_and_validation():
    cfg = MLPConfig(
        hidden_layer_sizes=(32, 16),
        activation="tanh",
        solver="sgd",
        alpha=5e-4,
        learning_rate="adaptive",
        learning_rate_init=5e-3,
        max_iter=25,
        early_stopping=True,
        validation_fraction=0.15,
        batch_size=16,
        random_state=123,
    )
    pipe = build_mlp_pipeline(cfg)
    clf = pipe.named_steps["classifier"]
    assert clf.hidden_layer_sizes == (32, 16)
    assert clf.activation == "tanh"
    assert clf.solver == "sgd"
    assert clf.alpha == 5e-4
    assert clf.learning_rate == "adaptive"
    assert clf.learning_rate_init == 5e-3
    assert clf.max_iter == 25
    assert clf.early_stopping is True
    assert clf.batch_size == 16
    assert clf.random_state == 123

    # Invalid combination: lbfgs + early_stopping=True must raise ValueError
    with pytest.raises(ValueError, match="early_stopping"):
        MLPConfig(solver="lbfgs", early_stopping=True)


def test_reproducibility_probability_bounds_and_batched_inference():
    cfg = MLPConfig(
        hidden_layer_sizes=(16, 8),
        solver="adam",
        max_iter=20,
        early_stopping=False,
        batch_size=16,
        inference_batch_size=11,
        random_state=42,
    )
    rng = np.random.default_rng(11)
    X = rng.normal(size=(80, 4)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)

    pipe_a = build_mlp_pipeline(cfg)
    pipe_b = build_mlp_pipeline(cfg)
    pipe_a.fit(X, y)
    pipe_b.fit(X, y)

    probs_a = predict_positive_proba(pipe_a, X, batch_size=11)
    probs_b = predict_positive_proba(pipe_b, X, batch_size=1000)
    preds_a = predict_classes(pipe_a, X, batch_size=11)

    assert np.all((probs_a >= 0.0) & (probs_a <= 1.0))
    assert np.allclose(probs_a, probs_b, atol=1e-12)
    assert set(preds_a.tolist()).issubset({0, 1})

    history = extract_training_history(pipe_a)
    assert history["n_iter"] >= 1
    assert len(history["loss_curve"]) == history["n_iter"]


def test_group_kfold_isolation_and_scaler_no_cross_fold_leakage():
    df = _make_synthetic_candidate_df(n_train_trips=5, n_test_trips=2)
    train_df = df[df["partition"] == "train"].copy()
    feat_cols = ["feat_x1", "feat_x2", "feat_x3"]

    # Inject extreme outlier into one trip to confirm scaler never sees held-out trip
    train_df.loc[train_df["trip_id"] == "00000005", "feat_x1"] = 50_000.0

    groups = train_df["trip_id"].to_numpy()
    splitter = GroupKFold(n_splits=5)
    for fit_idx, val_idx in splitter.split(train_df[feat_cols], train_df["label"], groups):
        fit_trips = set(train_df.iloc[fit_idx]["trip_id"])
        val_trips = set(train_df.iloc[val_idx]["trip_id"])
        assert fit_trips.isdisjoint(val_trips)

        pipe = build_mlp_pipeline(
            MLPConfig(hidden_layer_sizes=(8,), max_iter=10, early_stopping=False, batch_size=16)
        )
        pipe.fit(
            train_df.iloc[fit_idx][feat_cols].to_numpy(dtype=np.float32),
            train_df.iloc[fit_idx]["label"].to_numpy(dtype=int),
        )
        scaler_mean = float(pipe.named_steps["scale"].mean_[0])
        if "00000005" in val_trips:
            assert abs(scaler_mean) < 50.0


def test_macro_f1_trip_qualification_and_optional_route_continuity():
    df = pd.DataFrame(
        {
            "trip_id": ["00000001"] * 4,
            "observation_id": [0, 1, 2, 3],
            "candidate_arc_id": [0, 1, 1, 3],
            "label": [1, 1, 1, 1],
        }
    )
    probs = np.array([0.92, 0.88, 0.91, 0.85])
    top1, macro_f1, eligible, no_pos, multi_pos = observation_scores(df, probs)
    assert (eligible, no_pos, multi_pos) == (4, 0, 0)
    assert top1 == 1.0 and macro_f1 == 1.0

    # Without graph interface, route_continuity MUST be None
    assert compute_route_continuity(df, probs, trip_arcs_by_id=None) is None
    assert score_predictions(df, probs, model_name="mlp")["route_continuity"] is None

    # With graph interface: 0->1 connected, 1->1 connected, 1->3 disconnected => 2/3
    arcs_map = {"00000001": [(0, 1), (1, 2), (2, 3), (9, 10)]}
    cont = compute_route_continuity(df, probs, trip_arcs_by_id=arcs_map)
    assert cont is not None and abs(cont - (2.0 / 3.0)) < 1e-9


def test_end_to_end_synthetic_mlp_evaluation_and_save_authorization_gate(tmp_path: Path):
    df = _make_synthetic_candidate_df(n_train_trips=5, n_test_trips=2)
    csv_path = tmp_path / "mlp_e2e.csv"
    df.to_csv(csv_path, index=False)

    table, feat_cols, _ = load_candidate_dataset(csv_path)
    cfg = MLPConfig(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        max_iter=20,
        early_stopping=True,
        validation_fraction=0.2,
        batch_size=16,
        n_splits=5,
        random_state=42,
    )

    probe = benchmark_mlp_latency(
        table[feat_cols].to_numpy(dtype=np.float32),
        table["label"].to_numpy(dtype=int),
        cfg,
        probe_query_size=10,
    )
    assert probe["epochs_run"] >= 1
    assert probe["ms_per_1000_queries"] >= 0.0

    model, fold_df, cv_summary, test_metrics = evaluate_mlp_experiment(
        table, feat_cols, cfg
    )
    assert len(fold_df) == 5
    assert 0.0 <= cv_summary["observation_top1_accuracy_mean"] <= 1.0
    assert 0.0 <= test_metrics["observation_top1_accuracy"] <= 1.0
    assert test_metrics["route_continuity"] is None
    assert test_metrics["metrics_scope"] == "candidate_classifier_only_not_full_route_reconstruction"
    assert test_metrics["epochs_run"] >= 1
    assert len(test_metrics["training_history"]["loss_curve"]) >= 1

    # Saving MUST be blocked unless authorize_save=True
    with pytest.raises(PermissionError):
        save_mlp_artifacts(
            model,
            fold_df,
            cv_summary,
            test_metrics,
            output_dir=tmp_path / "mlp_out",
            authorize_save=False,
        )
    assert not (tmp_path / "mlp_out").exists()
