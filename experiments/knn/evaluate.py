"""Leakage-free evaluation, benchmarking, and artifact saving for PathSnap KNN."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold

from .config import KNNConfig
from .model import (
    build_knn_pipeline,
    predict_positive_proba,
    select_training_subset,
)

PROB_EPS = 1e-15


def observation_scores(
    frame: pd.DataFrame, probabilities: np.ndarray
) -> tuple[float, float, int, int, int]:
    """Compute observation-level Top-1 Accuracy and Macro F1 (`trip_id::candidate_arc_id`).

    Matches Person 1's shared evaluation contract:
    - Groups candidate rows by `(trip_id, observation_id)`.
    - Evaluates only observations with exactly one positive candidate (`label == 1`).
    - Reports zero-positive (candidate miss) and multi-positive counts separately.
    - Qualifies class labels as `trip_id::candidate_arc_id` because arc IDs are
      local 0-based indices inside each trip's road network.
    """
    keys = ["trip_id", "observation_id"]
    scored = frame[keys + ["candidate_arc_id", "label"]].copy()
    scored["trip_id"] = scored["trip_id"].astype(str)
    scored["p"] = np.asarray(probabilities, dtype=np.float64)

    grouped = scored.groupby(keys, sort=False)
    n_positive = grouped["label"].sum()
    no_positive = int((n_positive == 0).sum())
    multiple_positive = int((n_positive > 1).sum())
    eligible_keys = n_positive.index[n_positive == 1]

    if len(eligible_keys) == 0:
        return float("nan"), float("nan"), 0, no_positive, multiple_positive

    best = scored.loc[grouped["p"].idxmax()].set_index(keys).loc[eligible_keys]
    truth = scored[scored["label"] == 1].set_index(keys).loc[eligible_keys]
    trip_part = eligible_keys.get_level_values("trip_id").astype(str)

    y_true = (
        trip_part + "::" + truth["candidate_arc_id"].astype(str).to_numpy()
    ).to_numpy()
    y_pred = (
        trip_part + "::" + best["candidate_arc_id"].astype(str).to_numpy()
    ).to_numpy()

    return (
        float(accuracy_score(y_true, y_pred)),
        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        len(y_true),
        no_positive,
        multiple_positive,
    )


def compute_route_continuity(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    trip_arcs_by_id: Mapping[str, Sequence[tuple[int, int]]] | None,
) -> float | None:
    """Compute consecutive predicted-arc continuity when road-graph arcs are supplied.

    Returns `None` if no road connectivity / decoder interface (`trip_arcs_by_id`)
    is provided. When provided, for each trip's chronologically ordered predicted
    observations, a transition `(arc_prev -> arc_next)` is continuous if:
      - `arc_prev == arc_next` (vehicle stayed on the same road segment), OR
      - `arcs[arc_prev][1] == arcs[arc_next][0]` (directed road-graph connection).
    """
    if trip_arcs_by_id is None:
        return None

    keys = ["trip_id", "observation_id"]
    scored = frame[keys + ["candidate_arc_id"]].copy()
    scored["trip_id"] = scored["trip_id"].astype(str)
    scored["p"] = np.asarray(probabilities, dtype=np.float64)

    grouped = scored.groupby(keys, sort=False)
    best = scored.loc[grouped["p"].idxmax()].sort_values(keys)

    connected_transitions = 0
    total_transitions = 0

    for trip_id, trip_df in best.groupby("trip_id", sort=False):
        arcs = trip_arcs_by_id.get(str(trip_id))
        if arcs is None:
            return None
        pred_arcs = trip_df["candidate_arc_id"].to_numpy(dtype=np.int64)
        if len(pred_arcs) < 2:
            continue
        for prev_arc, next_arc in zip(pred_arcs[:-1], pred_arcs[1:]):
            total_transitions += 1
            if prev_arc == next_arc:
                connected_transitions += 1
            elif (
                0 <= prev_arc < len(arcs)
                and 0 <= next_arc < len(arcs)
                and arcs[prev_arc][1] == arcs[next_arc][0]
            ):
                connected_transitions += 1

    if total_transitions == 0:
        return None
    return float(connected_transitions / total_transitions)


def score_predictions(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    model_name: str = "knn",
    trip_arcs_by_id: Mapping[str, Sequence[tuple[int, int]]] | None = None,
) -> dict[str, object]:
    """Compute full PathSnap candidate + observation metrics for predicted probabilities."""
    y_candidate = frame["label"].to_numpy(dtype=int)
    probs = np.asarray(probabilities, dtype=np.float64)
    clipped = np.clip(probs, PROB_EPS, 1.0 - PROB_EPS)
    y_hat_binary = (probs >= 0.5).astype(int)

    obs_top1, obs_macro_f1, eligible, no_pos, multi_pos = observation_scores(
        frame, probs
    )
    continuity = compute_route_continuity(frame, probs, trip_arcs_by_id)

    return {
        "model": model_name,
        "candidate_binary_log_loss": float(
            log_loss(y_candidate, clipped, labels=[0, 1])
        ),
        "candidate_precision": float(
            precision_score(y_candidate, y_hat_binary, zero_division=0)
        ),
        "candidate_recall": float(
            recall_score(y_candidate, y_hat_binary, zero_division=0)
        ),
        "candidate_f1": float(
            f1_score(y_candidate, y_hat_binary, zero_division=0)
        ),
        "observation_top1_accuracy": obs_top1,
        "observation_macro_f1": obs_macro_f1,
        "route_continuity": continuity,
        "eligible_observations": eligible,
        "observations_without_positive_candidate_label": no_pos,
        "observations_with_multiple_positive_labels": multi_pos,
    }


def benchmark_knn_latency(
    X_sample: np.ndarray,
    y_sample: np.ndarray,
    config: KNNConfig,
    probe_query_size: int = 500,
) -> dict[str, float | int | str]:
    """Measure KNN fit and query latency on a small probe sample before full runs."""
    n_total = len(y_sample)
    if n_total < config.n_neighbors + 2:
        raise ValueError(
            "Sample size must exceed n_neighbors + 1 to benchmark KNN latency."
        )

    n_query = min(probe_query_size, max(1, n_total // 5))
    X_fit, y_fit = X_sample[:-n_query], y_sample[:-n_query]
    X_query = X_sample[-n_query:]

    model = build_knn_pipeline(config)
    t0 = time.perf_counter()
    model.fit(X_fit, y_fit)
    fit_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    _ = predict_positive_proba(model, X_query, batch_size=config.batch_size)
    query_s = time.perf_counter() - t1

    return {
        "fit_rows": int(len(X_fit)),
        "query_rows": int(len(X_query)),
        "n_features": int(X_sample.shape[1]),
        "algorithm": config.algorithm,
        "metric": config.metric,
        "n_neighbors": config.n_neighbors,
        "probe_fit_seconds": round(fit_s, 6),
        "probe_query_seconds": round(query_s, 6),
        "ms_per_1000_queries": round((query_s / max(1, len(X_query))) * 1_000_000.0, 3),
    }


def evaluate_knn_experiment(
    table: pd.DataFrame,
    feature_columns: list[str],
    config: KNNConfig,
    trip_arcs_by_id: Mapping[str, Sequence[tuple[int, int]]] | None = None,
) -> tuple[object, pd.DataFrame, dict[str, object], dict[str, object]]:
    """Run leakage-free trip-grouped CV on `train`, then evaluate on `student_test`.

    Does NOT write any artifacts to disk; artifact saving is gated separately via
    `save_knn_artifacts(..., authorize_save=True)`.
    """
    train_df = table[table["partition"] == "train"].copy()
    test_df = table[table["partition"] == "student_test"].copy()

    n_train_trips = int(train_df["trip_id"].nunique())
    if n_train_trips < config.n_splits:
        raise ValueError(
            f"GroupKFold requires at least {config.n_splits} distinct training trips, "
            f"but found {n_train_trips}."
        )

    float_dtype = np.float32 if config.use_float32 else np.float64
    X_train_all = train_df[feature_columns].to_numpy(dtype=float_dtype)
    y_train_all = train_df["label"].to_numpy(dtype=int)
    groups = train_df["trip_id"].astype(str).to_numpy()

    X_test_all = test_df[feature_columns].to_numpy(dtype=float_dtype)

    splitter = GroupKFold(n_splits=config.n_splits)
    fold_rows: list[dict[str, object]] = []

    for fold_idx, (fit_idx, val_idx) in enumerate(
        splitter.split(X_train_all, y_train_all, groups), start=1
    ):
        X_fit_raw = X_train_all[fit_idx]
        y_fit_raw = y_train_all[fit_idx]
        X_fit, y_fit, was_subsampled = select_training_subset(
            X_fit_raw,
            y_fit_raw,
            max_rows=config.max_train_rows_per_fold,
            random_state=config.random_state + fold_idx,
        )

        fold_model = build_knn_pipeline(config)
        t_fit0 = time.perf_counter()
        fold_model.fit(X_fit, y_fit)
        fit_seconds = time.perf_counter() - t_fit0

        val_df = train_df.iloc[val_idx]
        X_val = X_train_all[val_idx]
        t_inf0 = time.perf_counter()
        val_probs = predict_positive_proba(
            fold_model, X_val, batch_size=config.batch_size
        )
        inference_seconds = time.perf_counter() - t_inf0

        metrics = score_predictions(
            val_df, val_probs, model_name="knn", trip_arcs_by_id=trip_arcs_by_id
        )
        metrics.update(
            {
                "fold": fold_idx,
                "fit_rows": int(len(X_fit)),
                "subsampled_train_fold": was_subsampled,
                "validation_rows": int(len(val_df)),
                "validation_trip_count": int(val_df["trip_id"].nunique()),
                "fit_seconds": fit_seconds,
                "inference_seconds": inference_seconds,
            }
        )
        fold_rows.append(metrics)

    fold_df = pd.DataFrame(fold_rows)

    # Refit on full train partition (strictly excluding student_test)
    X_final_fit, y_final_fit, final_subsampled = select_training_subset(
        X_train_all,
        y_train_all,
        max_rows=config.max_train_rows_per_fold,
        random_state=config.random_state,
    )
    final_model = build_knn_pipeline(config)
    t_full0 = time.perf_counter()
    final_model.fit(X_final_fit, y_final_fit)
    full_fit_seconds = time.perf_counter() - t_full0

    t_test0 = time.perf_counter()
    test_probs = predict_positive_proba(
        final_model, X_test_all, batch_size=config.batch_size
    )
    test_inference_seconds = time.perf_counter() - t_test0

    student_test_metrics = score_predictions(
        test_df, test_probs, model_name="knn", trip_arcs_by_id=trip_arcs_by_id
    )
    student_test_metrics.update(
        {
            "cv_splits": config.n_splits,
            "train_trip_count": n_train_trips,
            "student_test_trip_count": int(test_df["trip_id"].nunique()),
            "fit_rows_final": int(len(X_final_fit)),
            "subsampled_final_train": final_subsampled,
            "fit_seconds_on_training": full_fit_seconds,
            "student_test_inference_seconds": test_inference_seconds,
            "feature_count": len(feature_columns),
            "feature_columns": list(feature_columns),
            "random_state": config.random_state,
            "sklearn_version": sklearn.__version__,
            "configuration": config.to_dict(),
        }
    )

    cv_summary: dict[str, object] = {
        "model": "knn",
        "cv_splits": config.n_splits,
        "observation_top1_accuracy_mean": float(
            fold_df["observation_top1_accuracy"].mean()
        ),
        "observation_top1_accuracy_std": float(
            fold_df["observation_top1_accuracy"].std(ddof=1)
        ),
        "observation_macro_f1_mean": float(fold_df["observation_macro_f1"].mean()),
        "observation_macro_f1_std": float(
            fold_df["observation_macro_f1"].std(ddof=1)
        ),
        "candidate_binary_log_loss_mean": float(
            fold_df["candidate_binary_log_loss"].mean()
        ),
        "candidate_binary_log_loss_std": float(
            fold_df["candidate_binary_log_loss"].std(ddof=1)
        ),
        "fit_seconds_mean": float(fold_df["fit_seconds"].mean()),
        "inference_seconds_mean": float(fold_df["inference_seconds"].mean()),
        "route_continuity_mean": (
            float(fold_df["route_continuity"].mean())
            if fold_df["route_continuity"].notna().any()
            else None
        ),
    }

    return final_model, fold_df, cv_summary, student_test_metrics


def save_knn_artifacts(
    model: object,
    fold_df: pd.DataFrame,
    cv_summary: dict[str, object],
    student_test_metrics: dict[str, object],
    output_dir: str | Path,
    model_output_path: str | Path | None = None,
    authorize_save: bool = False,
) -> dict[str, str]:
    """Persist trained KNN model and metrics ONLY when explicitly authorized."""
    if not authorize_save:
        raise PermissionError(
            "Saving KNN models or metrics is blocked unless authorize_save=True "
            "(CLI flag --authorize-save). Wait for the approved shared feature CSV."
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    folds_path = out_dir / "knn_cv_folds.csv"
    summary_path = out_dir / "knn_cv_summary.json"
    test_path = out_dir / "knn_student_test_results.json"

    fold_df.to_csv(folds_path, index=False)
    summary_path.write_text(json.dumps(cv_summary, indent=2), encoding="utf-8")
    test_path.write_text(
        json.dumps(student_test_metrics, indent=2), encoding="utf-8"
    )

    saved = {
        "cv_folds_csv": str(folds_path),
        "cv_summary_json": str(summary_path),
        "student_test_json": str(test_path),
    }

    if model_output_path is not None:
        m_path = Path(model_output_path)
        m_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, m_path)
        saved["model_joblib"] = str(m_path)

    return saved
