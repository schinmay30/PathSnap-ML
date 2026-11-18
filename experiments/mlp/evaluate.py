"""Leakage-free evaluation, benchmarking, and artifact saving for PathSnap MLP.

Note on metric scope:
All metrics computed here measure candidate-level and observation-level
classifier performance (Top-1 Accuracy, `trip_id::candidate_arc_id` Macro F1,
Candidate Binary Log Loss, and optional raw-classifier Route Continuity).
Classifier metrics are NOT full route-reconstruction accuracy; sequential
route-reconstruction accuracy requires Person 3's HMM/Viterbi decoder.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold

from experiments.knn.evaluate import (
    compute_route_continuity,
    observation_scores,
    score_predictions,
)

from .config import MLPConfig
from .model import (
    build_mlp_pipeline,
    extract_training_history,
    predict_positive_proba,
    select_training_subset,
)


def benchmark_mlp_latency(
    X_sample: np.ndarray,
    y_sample: np.ndarray,
    config: MLPConfig,
    probe_query_size: int = 500,
) -> dict[str, float | int | str | list[int]]:
    """Measure MLP fit and query latency on a small probe sample before full runs."""
    n_total = len(y_sample)
    if n_total < 20:
        raise ValueError("Sample size must be >= 20 to benchmark MLP latency.")

    n_query = min(probe_query_size, max(2, n_total // 5))
    X_fit, y_fit = X_sample[:-n_query], y_sample[:-n_query]
    X_query = X_sample[-n_query:]

    model = build_mlp_pipeline(config)
    t0 = time.perf_counter()
    model.fit(X_fit, y_fit)
    fit_s = time.perf_counter() - t0
    history = extract_training_history(model)

    t1 = time.perf_counter()
    _ = predict_positive_proba(
        model, X_query, batch_size=config.inference_batch_size
    )
    query_s = time.perf_counter() - t1

    return {
        "fit_rows": int(len(X_fit)),
        "query_rows": int(len(X_query)),
        "n_features": int(X_sample.shape[1]),
        "hidden_layer_sizes": list(config.hidden_layer_sizes),
        "activation": config.activation,
        "solver": config.solver,
        "epochs_run": int(history["n_iter"]),
        "probe_fit_seconds": round(fit_s, 6),
        "probe_query_seconds": round(query_s, 6),
        "ms_per_1000_queries": round(
            (query_s / max(1, len(X_query))) * 1_000_000.0, 3
        ),
    }


def evaluate_mlp_experiment(
    table: pd.DataFrame,
    feature_columns: list[str],
    config: MLPConfig,
    trip_arcs_by_id: Mapping[str, Sequence[tuple[int, int]]] | None = None,
) -> tuple[object, pd.DataFrame, dict[str, object], dict[str, object]]:
    """Run leakage-free trip-grouped CV on `train`, then evaluate on `student_test`.

    Does NOT write any artifacts to disk; artifact saving is strictly gated via
    `save_mlp_artifacts(..., authorize_save=True)`.
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
        X_fit, y_fit, was_subsampled = select_training_subset(
            X_train_all[fit_idx],
            y_train_all[fit_idx],
            max_rows=config.max_train_rows_per_fold,
            random_state=config.random_state + fold_idx,
        )

        fold_model = build_mlp_pipeline(config)
        t_fit0 = time.perf_counter()
        fold_model.fit(X_fit, y_fit)
        fit_seconds = time.perf_counter() - t_fit0
        fold_history = extract_training_history(fold_model)

        val_df = train_df.iloc[val_idx]
        X_val = X_train_all[val_idx]
        t_inf0 = time.perf_counter()
        val_probs = predict_positive_proba(
            fold_model, X_val, batch_size=config.inference_batch_size
        )
        inference_seconds = time.perf_counter() - t_inf0

        metrics = score_predictions(
            val_df, val_probs, model_name="mlp", trip_arcs_by_id=trip_arcs_by_id
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
                "epochs_run": fold_history["n_iter"],
                "final_training_loss": fold_history["final_training_loss"],
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
    final_model = build_mlp_pipeline(config)
    t_full0 = time.perf_counter()
    final_model.fit(X_final_fit, y_final_fit)
    full_fit_seconds = time.perf_counter() - t_full0
    final_history = extract_training_history(final_model)

    t_test0 = time.perf_counter()
    test_probs = predict_positive_proba(
        final_model, X_test_all, batch_size=config.inference_batch_size
    )
    test_inference_seconds = time.perf_counter() - t_test0

    student_test_metrics = score_predictions(
        test_df, test_probs, model_name="mlp", trip_arcs_by_id=trip_arcs_by_id
    )
    student_test_metrics.update(
        {
            "metrics_scope": "candidate_classifier_only_not_full_route_reconstruction",
            "cv_splits": config.n_splits,
            "train_trip_count": n_train_trips,
            "student_test_trip_count": int(test_df["trip_id"].nunique()),
            "fit_rows_final": int(len(X_final_fit)),
            "subsampled_final_train": final_subsampled,
            "fit_seconds_on_training": full_fit_seconds,
            "student_test_inference_seconds": test_inference_seconds,
            "epochs_run": final_history["n_iter"],
            "training_history": final_history,
            "feature_count": len(feature_columns),
            "feature_columns": list(feature_columns),
            "random_state": config.random_state,
            "sklearn_version": sklearn.__version__,
            "configuration": config.to_dict(),
        }
    )

    cv_summary: dict[str, object] = {
        "model": "mlp",
        "metrics_scope": "candidate_classifier_only_not_full_route_reconstruction",
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
        "epochs_run_mean": float(fold_df["epochs_run"].mean()),
        "route_continuity_mean": (
            float(fold_df["route_continuity"].mean())
            if fold_df["route_continuity"].notna().any()
            else None
        ),
    }

    return final_model, fold_df, cv_summary, student_test_metrics


def save_mlp_artifacts(
    model: object,
    fold_df: pd.DataFrame,
    cv_summary: dict[str, object],
    student_test_metrics: dict[str, object],
    output_dir: str | Path,
    model_output_path: str | Path | None = None,
    authorize_save: bool = False,
) -> dict[str, str]:
    """Persist trained MLP model and metrics ONLY when explicitly authorized."""
    if not authorize_save:
        raise PermissionError(
            "Saving MLP models or metrics is blocked unless authorize_save=True "
            "(CLI flag --authorize-save). Wait for the approved shared feature CSV."
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    folds_path = out_dir / "mlp_cv_folds.csv"
    summary_path = out_dir / "mlp_cv_summary.json"
    test_path = out_dir / "mlp_student_test_results.json"
    history_path = out_dir / "mlp_training_history.json"

    fold_df.to_csv(folds_path, index=False)
    summary_path.write_text(json.dumps(cv_summary, indent=2), encoding="utf-8")
    test_path.write_text(
        json.dumps(student_test_metrics, indent=2), encoding="utf-8"
    )
    history_payload = student_test_metrics.get("training_history", {})
    history_path.write_text(
        json.dumps(history_payload, indent=2), encoding="utf-8"
    )

    saved = {
        "cv_folds_csv": str(folds_path),
        "cv_summary_json": str(summary_path),
        "student_test_json": str(test_path),
        "training_history_json": str(history_path),
    }

    if model_output_path is not None:
        m_path = Path(model_output_path)
        m_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, m_path)
        saved["model_joblib"] = str(m_path)

    return saved


__all__ = [
    "benchmark_mlp_latency",
    "compute_route_continuity",
    "evaluate_mlp_experiment",
    "observation_scores",
    "save_mlp_artifacts",
    "score_predictions",
]
