"""Unit tests for Person 4's reusable KNN module using small synthetic fixtures ONLY.

Run with:
    .\\backend\\.venv\\Scripts\\python.exe -m pytest experiments\\knn\\test_knn.py -v
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

from experiments.knn.config import KNNConfig, ResourceBudget  # noqa: E402
from experiments.knn.data_contract import (  # noqa: E402
    MemoryGuardError,
    detect_feature_columns,
    estimate_csv_resources,
    load_candidate_dataset,
)
from experiments.knn.evaluate import (  # noqa: E402
    benchmark_knn_latency,
    compute_route_continuity,
    evaluate_knn_experiment,
    observation_scores,
    save_knn_artifacts,
    score_predictions,
)
from experiments.knn.model import (  # noqa: E402
    build_knn_pipeline,
    predict_positive_proba,
)


def _make_synthetic_candidate_df(
    feature_names: tuple[str, ...] = ("feat_alpha", "feat_beta", "feat_gamma"),
    n_train_trips: int = 5,
    n_test_trips: int = 2,
    obs_per_trip: int = 6,
    cands_per_obs: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a tiny synthetic candidate table with arbitrary feature names."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    trip_specs = [
        ("train", f"0000000{i}") for i in range(1, n_train_trips + 1)
    ] + [
        ("student_test", f"0000009{i}") for i in range(n_test_trips)
    ]

    for partition, trip_id in trip_specs:
        for obs_id in range(obs_per_trip):
            true_arc = obs_id  # chained arc sequence 0 -> 1 -> 2 ...
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
                    # Separable synthetic signal for unit-testing pipeline wiring
                    base = -2.0 if is_pos else 2.0
                    row[fname] = float(base + 0.15 * f_i + rng.normal(0.0, 0.35))
                rows.append(row)

    return pd.DataFrame(rows)


def test_preserves_leading_zeros_in_trip_id_and_detects_custom_features(tmp_path: Path):
    custom_features = ("custom_f1", "custom_f2", "custom_f3", "custom_f4")
    df = _make_synthetic_candidate_df(feature_names=custom_features)
    csv_path = tmp_path / "synthetic_candidates.csv"
    df.to_csv(csv_path, index=False)

    loaded, detected_features, estimate = load_candidate_dataset(csv_path)

    assert detected_features == list(custom_features)
    assert set(loaded["trip_id"].unique()) >= {"00000001", "00000090"}
    assert all(isinstance(tid, str) and len(tid) == 8 for tid in loaded["trip_id"].unique())
    assert estimate.train_trips == 5
    assert estimate.student_test_trips == 2
    assert estimate.safe_to_load is True


def test_rejects_leakage_columns_and_evaluator_partition(tmp_path: Path):
    # 1. Ground-truth route leakage column must be rejected
    with pytest.raises(ValueError, match="leakage"):
        detect_feature_columns(
            ["partition", "trip_id", "observation_id", "candidate_arc_id", "label", "route_index"]
        )

    # 2. Evaluator partition must be rejected during streaming pre-flight
    df = _make_synthetic_candidate_df()
    df.loc[0, "partition"] = "evaluator_test"
    bad_csv = tmp_path / "bad_partition.csv"
    df.to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="unexpected partition"):
        estimate_csv_resources(bad_csv)


def test_memory_guard_blocks_oversized_load(tmp_path: Path):
    df = _make_synthetic_candidate_df()
    csv_path = tmp_path / "tiny.csv"
    df.to_csv(csv_path, index=False)

    tiny_budget = ResourceBudget(max_memory_mb=0.0001, sample_rows=20, chunk_rows=100)
    estimate = estimate_csv_resources(csv_path, budget=tiny_budget)
    assert estimate.safe_to_load is False

    with pytest.raises(MemoryGuardError):
        load_candidate_dataset(csv_path, budget=tiny_budget, enforce_memory_budget=True)


def test_configurable_knn_hyperparameters_and_batched_inference():
    cfg = KNNConfig(
        n_neighbors=5,
        weights="uniform",
        metric="manhattan",
        algorithm="ball_tree",
        n_jobs=1,
        batch_size=7,
    )
    pipe = build_knn_pipeline(cfg)
    clf = pipe.named_steps["classifier"]
    assert clf.n_neighbors == 5
    assert clf.weights == "uniform"
    assert clf.metric == "manhattan"
    assert clf.algorithm == "ball_tree"
    assert clf.n_jobs == 1

    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    pipe.fit(X, y)

    batched = predict_positive_proba(pipe, X, batch_size=7)
    unbatched = predict_positive_proba(pipe, X, batch_size=1000)
    assert np.allclose(batched, unbatched, atol=1e-12)


def test_macro_f1_qualifies_classes_by_trip_id():
    # Mirrors Person 1's contract test: identical local arc_id across trips A and B
    cases = [("00000001", 7, 7, 3), ("00000002", 7, 8, 1), ("00000002", 8, 7, 2), ("00000001", 9, 9, 1)]
    rows = []
    obs_id = 0
    for trip, truth, pred, count in cases:
        for _ in range(count):
            for arc in {truth, pred, 99}:
                rows.append(
                    (trip, obs_id, arc, int(arc == truth), 0.9 if arc == pred else 0.1)
                )
            obs_id += 1
    df = pd.DataFrame(
        rows, columns=["trip_id", "observation_id", "candidate_arc_id", "label", "p"]
    )
    top1, macro_f1, eligible, no_pos, multi_pos = observation_scores(
        df, df["p"].to_numpy()
    )
    assert (eligible, no_pos, multi_pos) == (7, 0, 0)
    assert abs(top1 - 4.0 / 7.0) < 1e-9
    assert abs(macro_f1 - 0.5) < 1e-9


def test_group_kfold_and_scaler_have_no_trip_leakage():
    df = _make_synthetic_candidate_df(n_train_trips=5, n_test_trips=2)
    train_df = df[df["partition"] == "train"].copy()
    feat_cols = ["feat_alpha", "feat_beta", "feat_gamma"]

    # Inject extreme values into one specific trip to confirm scaler never sees held-out trip
    train_df.loc[train_df["trip_id"] == "00000005", "feat_alpha"] = 10_000.0

    groups = train_df["trip_id"].to_numpy()
    splitter = GroupKFold(n_splits=5)
    for fit_idx, val_idx in splitter.split(train_df[feat_cols], train_df["label"], groups):
        fit_trips = set(train_df.iloc[fit_idx]["trip_id"])
        val_trips = set(train_df.iloc[val_idx]["trip_id"])
        assert fit_trips.isdisjoint(val_trips)

        pipe = build_knn_pipeline(KNNConfig(n_neighbors=3, n_jobs=1))
        pipe.fit(
            train_df.iloc[fit_idx][feat_cols].to_numpy(dtype=np.float32),
            train_df.iloc[fit_idx]["label"].to_numpy(dtype=int),
        )
        scaler_mean_alpha = float(pipe.named_steps["scale"].mean_[0])
        if "00000005" in val_trips:
            assert abs(scaler_mean_alpha) < 50.0


def test_route_continuity_none_without_graph_and_exact_with_graph():
    df = pd.DataFrame(
        {
            "trip_id": ["00000001"] * 4,
            "observation_id": [0, 1, 2, 3],
            "candidate_arc_id": [0, 1, 1, 3],
            "label": [1, 1, 1, 1],
        }
    )
    probs = np.array([0.9, 0.9, 0.9, 0.9])

    # Without a connectivity interface, route_continuity MUST be None
    assert compute_route_continuity(df, probs, trip_arcs_by_id=None) is None
    metrics_no_graph = score_predictions(df, probs, trip_arcs_by_id=None)
    assert metrics_no_graph["route_continuity"] is None

    # With a valid road-graph arc table:
    # arc 0=(0->1), arc 1=(1->2), arc 2=(2->3), arc 3=(9->10 disconnected)
    # Transitions: 0->1 (connected), 1->1 (same arc, connected), 1->3 (disconnected) => 2/3
    arcs_map = {"00000001": [(0, 1), (1, 2), (2, 3), (9, 10)]}
    cont = compute_route_continuity(df, probs, trip_arcs_by_id=arcs_map)
    assert cont is not None and abs(cont - (2.0 / 3.0)) < 1e-9


def test_end_to_end_synthetic_evaluation_and_save_authorization_gate(tmp_path: Path):
    df = _make_synthetic_candidate_df(n_train_trips=5, n_test_trips=2)
    csv_path = tmp_path / "synth.csv"
    df.to_csv(csv_path, index=False)

    table, feat_cols, _ = load_candidate_dataset(csv_path)
    cfg = KNNConfig(n_neighbors=3, weights="distance", algorithm="kd_tree", n_jobs=1, n_splits=5)

    # Probe benchmark works on synthetic sample
    probe = benchmark_knn_latency(
        table[feat_cols].to_numpy(dtype=np.float32),
        table["label"].to_numpy(dtype=int),
        cfg,
        probe_query_size=10,
    )
    assert probe["ms_per_1000_queries"] >= 0.0

    model, fold_df, cv_summary, test_metrics = evaluate_knn_experiment(
        table, feat_cols, cfg
    )
    assert len(fold_df) == 5
    assert 0.0 <= cv_summary["observation_top1_accuracy_mean"] <= 1.0
    assert 0.0 <= test_metrics["observation_top1_accuracy"] <= 1.0
    assert test_metrics["route_continuity"] is None
    assert test_metrics["fit_seconds_on_training"] >= 0.0
    assert test_metrics["student_test_inference_seconds"] >= 0.0

    # Saving MUST be refused without explicit authorize_save=True
    with pytest.raises(PermissionError):
        save_knn_artifacts(
            model,
            fold_df,
            cv_summary,
            test_metrics,
            output_dir=tmp_path / "out",
            authorize_save=False,
        )
    assert not (tmp_path / "out").exists()
