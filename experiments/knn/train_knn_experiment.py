"""Full leakage-free KNN hyperparameter tuning, 5-fold GroupKFold CV, and student_test evaluation.

SRN: PES1UG24CS676 (Person 4)
Dataset: data/processed/person3_extended_candidates.csv.gz (19 approved route-blind features)

Anti-leakage guarantees:
- Uses ONLY `train` partition (80 trips) for hyperparameter tuning and 5-fold GroupKFold CV.
- Fits StandardScaler strictly inside each training fold.
- Never reads or uses `.route` files as model inputs.
- Evaluates `student_test` (10 trips) ONCE after final model selection.
"""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import time
import tracemalloc

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.knn.config import KNNConfig, ResourceBudget  # noqa: E402
from experiments.knn.data_contract import load_candidate_dataset  # noqa: E402
from experiments.knn.evaluate import score_predictions  # noqa: E402
from experiments.knn.model import (  # noqa: E402
    build_knn_pipeline,
    predict_positive_proba,
    select_training_subset,
)

DATASET_PATH = REPO_ROOT / "data" / "processed" / "person3_extended_candidates.csv.gz"
MODEL_DIR = REPO_ROOT / "artifacts" / "models" / "knn"
METRICS_DIR = REPO_ROOT / "artifacts" / "metrics"

K_VALUES = (5, 11, 21, 35, 51)
WEIGHTS_OPTIONS = ("uniform", "distance")
METRICS_OPTIONS = ("euclidean", "manhattan")
TUNING_TRAIN_CAP = 100_000
QUERY_BATCH_SIZE = 2_000
RANDOM_STATE = 42


def _batched_kneighbors(
    nn_index: NearestNeighbors,
    X_query_scaled: np.ndarray,
    n_neighbors: int,
    batch_size: int = QUERY_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    n_q = len(X_query_scaled)
    dists = np.empty((n_q, n_neighbors), dtype=np.float32)
    indices = np.empty((n_q, n_neighbors), dtype=np.int32)
    for start in range(0, n_q, batch_size):
        stop = min(start + batch_size, n_q)
        d_b, i_b = nn_index.kneighbors(X_query_scaled[start:stop], n_neighbors=n_neighbors)
        dists[start:stop] = d_b.astype(np.float32, copy=False)
        indices[start:stop] = i_b.astype(np.int32, copy=False)
    return dists, indices


def _knn_probs_from_neighbors(
    dists: np.ndarray,
    neighbor_labels: np.ndarray,
    k: int,
    weights: str,
) -> np.ndarray:
    """Exact sklearn-equivalent KNeighborsClassifier positive probability from k-nearest neighbors."""
    sub_y = neighbor_labels[:, :k]
    if weights == "uniform":
        return sub_y.mean(axis=1, dtype=np.float64)

    sub_d = dists[:, :k].astype(np.float64, copy=False)
    zero_mask = sub_d == 0.0
    has_zero = zero_mask.any(axis=1)

    probs = np.empty(len(sub_d), dtype=np.float64)
    if (~has_zero).any():
        inv_d = 1.0 / sub_d[~has_zero]
        probs[~has_zero] = (inv_d * sub_y[~has_zero]).sum(axis=1) / inv_d.sum(axis=1)
    if has_zero.any():
        zm = zero_mask[has_zero]
        probs[has_zero] = (zm * sub_y[has_zero]).sum(axis=1) / zm.sum(axis=1)
    return probs


def run_knn_training() -> dict[str, object]:
    print("=== Loading Person 3 Extended Candidate Dataset (compact float32) ===", flush=True)
    t_load0 = time.perf_counter()
    table, feature_columns, estimate = load_candidate_dataset(
        DATASET_PATH,
        budget=ResourceBudget(max_memory_mb=1024.0, chunk_rows=200_000),
        use_float32=True,
        enforce_memory_budget=True,
    )
    load_seconds = time.perf_counter() - t_load0
    print(
        f"Loaded {len(table):,} rows x {len(feature_columns)} features in {load_seconds:.2f}s "
        f"(df_mem={table.memory_usage(deep=True).sum() / 1024**2:.1f} MB)",
        flush=True,
    )

    train_mask = (table["partition"] == "train").to_numpy()
    test_mask = (table["partition"] == "student_test").to_numpy()

    train_meta = table.loc[
        train_mask, ["trip_id", "observation_id", "candidate_arc_id", "label"]
    ].reset_index(drop=True)
    test_meta = table.loc[
        test_mask, ["trip_id", "observation_id", "candidate_arc_id", "label"]
    ].reset_index(drop=True)

    X_train_all = table.loc[train_mask, feature_columns].to_numpy(dtype=np.float32, copy=True)
    y_train_all = train_meta["label"].to_numpy(dtype=np.int8, copy=True)
    groups_train = train_meta["trip_id"].astype(str).to_numpy()

    X_test_all = table.loc[test_mask, feature_columns].to_numpy(dtype=np.float32, copy=True)
    del table
    gc.collect()

    splitter = GroupKFold(n_splits=5)
    fold_splits = list(splitter.split(X_train_all, y_train_all, groups_train))

    # -------------------------------------------------------------------------
    # Stage 1: Hyperparameter Tuning on train ONLY (5-fold GroupKFold by trip)
    # -------------------------------------------------------------------------
    print("\n=== Stage 1: 5-Fold GroupKFold KNN Hyperparameter Tuning (train only) ===", flush=True)
    k_max = max(K_VALUES)
    tuning_fold_records: list[dict[str, object]] = []

    for metric in METRICS_OPTIONS:
        for fold_idx, (fit_idx, val_idx) in enumerate(fold_splits, start=1):
            X_fit_sub, y_fit_sub, _ = select_training_subset(
                X_train_all[fit_idx],
                y_train_all[fit_idx],
                max_rows=TUNING_TRAIN_CAP,
                random_state=RANDOM_STATE + fold_idx,
            )
            scaler = StandardScaler()
            t_fit0 = time.perf_counter()
            X_fit_scaled = scaler.fit_transform(X_fit_sub).astype(np.float32, copy=False)
            nn_index = NearestNeighbors(
                n_neighbors=k_max, algorithm="brute", metric=metric, n_jobs=-1
            )
            nn_index.fit(X_fit_scaled)
            fit_s = time.perf_counter() - t_fit0

            X_val_scaled = scaler.transform(X_train_all[val_idx]).astype(np.float32, copy=False)
            t_q0 = time.perf_counter()
            dists, idxs = _batched_kneighbors(nn_index, X_val_scaled, n_neighbors=k_max)
            query_s = time.perf_counter() - t_q0
            nbr_labels = y_fit_sub[idxs]
            del idxs, X_fit_scaled, X_val_scaled
            gc.collect()

            val_meta_fold = train_meta.iloc[val_idx]
            for k in K_VALUES:
                for weights in WEIGHTS_OPTIONS:
                    probs = _knn_probs_from_neighbors(dists, nbr_labels, k=k, weights=weights)
                    m = score_predictions(val_meta_fold, probs, model_name="knn")
                    tuning_fold_records.append(
                        {
                            "metric": metric,
                            "n_neighbors": k,
                            "weights": weights,
                            "algorithm": "brute",
                            "max_train_rows_per_fold": TUNING_TRAIN_CAP,
                            "fold": fold_idx,
                            "candidate_accuracy": m["candidate_accuracy"],
                            "observation_top1_accuracy": m["observation_top1_accuracy"],
                            "observation_macro_f1": m["observation_macro_f1"],
                            "candidate_macro_f1": m["candidate_macro_f1"],
                            "candidate_weighted_f1": m["candidate_weighted_f1"],
                            "candidate_precision": m["candidate_precision"],
                            "candidate_recall": m["candidate_recall"],
                            "candidate_f1": m["candidate_f1"],
                            "candidate_binary_log_loss": m["candidate_binary_log_loss"],
                            "roc_auc": m["roc_auc"],
                            "pr_auc": m["pr_auc"],
                            "fit_seconds": fit_s,
                            "inference_seconds": query_s,
                        }
                    )
            del dists, nbr_labels
            gc.collect()
            print(
                f"  Tuned metric={metric:<9} fold={fold_idx}/5 | fit={fit_s:.2f}s | query={query_s:.1f}s",
                flush=True,
            )

    tuning_folds_df = pd.DataFrame(tuning_fold_records)
    agg_cols = [
        "observation_top1_accuracy",
        "observation_macro_f1",
        "candidate_accuracy",
        "candidate_macro_f1",
        "candidate_weighted_f1",
        "candidate_precision",
        "candidate_recall",
        "candidate_binary_log_loss",
        "roc_auc",
        "pr_auc",
        "fit_seconds",
        "inference_seconds",
    ]
    tuning_summary = (
        tuning_folds_df.groupby(
            ["metric", "n_neighbors", "weights", "algorithm", "max_train_rows_per_fold"],
            as_index=False,
        )[agg_cols]
        .mean()
        .sort_values(
            by=["observation_top1_accuracy", "observation_macro_f1", "candidate_binary_log_loss"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )

    print("\nTop 8 KNN Configurations (5-Fold GroupKFold CV on train):", flush=True)
    print(
        tuning_summary[
            [
                "metric",
                "n_neighbors",
                "weights",
                "observation_top1_accuracy",
                "observation_macro_f1",
                "candidate_binary_log_loss",
                "inference_seconds",
            ]
        ]
        .head(8)
        .to_string(index=False),
        flush=True,
    )

    best_row = tuning_summary.iloc[0]
    best_metric = str(best_row["metric"])
    best_k = int(best_row["n_neighbors"])
    best_weights = str(best_row["weights"])

    # -------------------------------------------------------------------------
    # Stage 2: Reference-Set Size Verification (100k vs 200k) on train CV ONLY
    # -------------------------------------------------------------------------
    print(
        f"\n=== Stage 2: Verifying Winning KNN (K={best_k}, weights={best_weights}, metric={best_metric}) "
        "with 200k Stratified Training Reference Cap ===",
        flush=True,
    )
    final_train_cap = 200_000
    best_cfg = KNNConfig(
        n_neighbors=best_k,
        weights=best_weights,
        metric=best_metric,
        algorithm="brute",
        n_jobs=-1,
        n_splits=5,
        random_state=RANDOM_STATE,
        batch_size=QUERY_BATCH_SIZE,
        max_train_rows_per_fold=final_train_cap,
        use_float32=True,
    )

    cv_fold_rows: list[dict[str, object]] = []
    tracemalloc.start()
    for fold_idx, (fit_idx, val_idx) in enumerate(fold_splits, start=1):
        X_fit_sub, y_fit_sub, was_sub = select_training_subset(
            X_train_all[fit_idx],
            y_train_all[fit_idx],
            max_rows=final_train_cap,
            random_state=RANDOM_STATE + fold_idx,
        )
        fold_pipe = build_knn_pipeline(best_cfg)
        t0 = time.perf_counter()
        fold_pipe.fit(X_fit_sub, y_fit_sub)
        fit_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        val_probs = predict_positive_proba(
            fold_pipe, X_train_all[val_idx], batch_size=QUERY_BATCH_SIZE
        )
        inf_s = time.perf_counter() - t1

        m = score_predictions(train_meta.iloc[val_idx], val_probs, model_name="knn")
        m.update(
            {
                "fold": fold_idx,
                "fit_rows": int(len(X_fit_sub)),
                "subsampled_train_fold": was_sub,
                "validation_rows": int(len(val_idx)),
                "validation_trip_count": int(train_meta.iloc[val_idx]["trip_id"].nunique()),
                "fit_seconds": fit_s,
                "inference_seconds": inf_s,
            }
        )
        cv_fold_rows.append(m)
        print(
            f"  Final KNN CV Fold {fold_idx}/5 | Top-1={m['observation_top1_accuracy']:.4f} "
            f"| Obs Macro F1={m['observation_macro_f1']:.4f} "
            f"| LogLoss={m['candidate_binary_log_loss']:.4f} | inf={inf_s:.1f}s",
            flush=True,
        )
        del fold_pipe, val_probs
        gc.collect()

    _, peak_cv_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cv_folds_df = pd.DataFrame(cv_fold_rows)

    # Compare 200k vs 100k strictly on train CV Top-1 Accuracy
    cv_top1_200k = float(cv_folds_df["observation_top1_accuracy"].mean())
    cv_top1_100k = float(best_row["observation_top1_accuracy"])
    print(
        f"  CV Top-1 comparison on train: 100k={cv_top1_100k:.4f} vs 200k={cv_top1_200k:.4f}",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # Stage 3: Final Refit on train ONLY + One-Time Development Evaluation on student_test
    # -------------------------------------------------------------------------
    print("\n=== Stage 3: Final Refit on train & Development Evaluation on student_test ===", flush=True)
    tracemalloc.start()
    X_final_fit, y_final_fit, final_sub = select_training_subset(
        X_train_all, y_train_all, max_rows=final_train_cap, random_state=RANDOM_STATE
    )
    final_pipe = build_knn_pipeline(best_cfg)
    t_f0 = time.perf_counter()
    final_pipe.fit(X_final_fit, y_final_fit)
    full_fit_s = time.perf_counter() - t_f0

    t_i0 = time.perf_counter()
    test_probs = predict_positive_proba(
        final_pipe, X_test_all, batch_size=QUERY_BATCH_SIZE
    )
    test_inf_s = time.perf_counter() - t_i0
    _, peak_final_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = round(max(peak_cv_bytes, peak_final_bytes) / (1024.0 * 1024.0), 2)

    test_metrics = score_predictions(test_meta, test_probs, model_name="knn")

    # Save artifacts under artifacts/models/knn/ and artifacts/metrics/
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "model.joblib"
    scaler_path = MODEL_DIR / "scaler.joblib"
    joblib.dump(final_pipe, model_path, compress=3)
    joblib.dump(final_pipe.named_steps["scale"], scaler_path, compress=3)
    model_size_mb = round(model_path.stat().st_size / (1024.0 * 1024.0), 3)

    cv_summary = {
        "model": "knn",
        "partition": "train_5fold_group_kfold",
        "metrics_scope": "candidate_selection_only_not_full_route_reconstruction",
        "cv_splits": 5,
        "fit_rows_per_fold": final_train_cap,
        "candidate_accuracy_mean": float(cv_folds_df["candidate_accuracy"].mean()),
        "candidate_accuracy_std": float(cv_folds_df["candidate_accuracy"].std(ddof=1)),
        "observation_top1_accuracy_mean": float(cv_folds_df["observation_top1_accuracy"].mean()),
        "observation_top1_accuracy_std": float(cv_folds_df["observation_top1_accuracy"].std(ddof=1)),
        "observation_macro_f1_mean": float(cv_folds_df["observation_macro_f1"].mean()),
        "observation_macro_f1_std": float(cv_folds_df["observation_macro_f1"].std(ddof=1)),
        "candidate_macro_f1_mean": float(cv_folds_df["candidate_macro_f1"].mean()),
        "candidate_weighted_f1_mean": float(cv_folds_df["candidate_weighted_f1"].mean()),
        "candidate_precision_mean": float(cv_folds_df["candidate_precision"].mean()),
        "candidate_recall_mean": float(cv_folds_df["candidate_recall"].mean()),
        "candidate_f1_mean": float(cv_folds_df["candidate_f1"].mean()),
        "candidate_binary_log_loss_mean": float(cv_folds_df["candidate_binary_log_loss"].mean()),
        "candidate_binary_log_loss_std": float(cv_folds_df["candidate_binary_log_loss"].std(ddof=1)),
        "roc_auc_mean": float(cv_folds_df["roc_auc"].mean()),
        "pr_auc_mean": float(cv_folds_df["pr_auc"].mean()),
        "fit_seconds_mean": float(cv_folds_df["fit_seconds"].mean()),
        "inference_seconds_mean": float(cv_folds_df["inference_seconds"].mean()),
        "tuning_100k_best_cv_top1": cv_top1_100k,
        "route_continuity_mean": None,
    }

    test_metrics.update(
        {
            "partition": "student_test_development_evaluation",
            "metrics_scope": "candidate_selection_only_not_full_route_reconstruction",
            "train_rows_total": int(len(X_train_all)),
            "fit_rows_final": int(len(X_final_fit)),
            "subsampled_final_train": final_sub,
            "student_test_rows": int(len(X_test_all)),
            "student_test_trip_count": int(test_meta["trip_id"].nunique()),
            "training_time_s": round(full_fit_s, 4),
            "inference_time_s": round(test_inf_s, 4),
            "model_size_mb": model_size_mb,
            "peak_traced_memory_mb": peak_mem_mb,
            "sklearn_version": sklearn.__version__,
            "random_state": RANDOM_STATE,
            "configuration": best_cfg.to_dict(),
        }
    )

    metadata = {
        "model_key": "knn",
        "model_name": "K-Nearest Neighbors (KNN)",
        "algorithm_family": "Instance-based",
        "owner": "Person 4 (SRN: PES1UG24CS676)",
        "dataset": "data/processed/person3_extended_candidates.csv.gz",
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "route_blind_verified": True,
        "hmm_decoder_integrated": False,
        "metrics_scope": "candidate_selection_only_not_full_route_reconstruction",
        "sklearn_version": sklearn.__version__,
        "model_size_mb": model_size_mb,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    (MODEL_DIR / "feature_columns.json").write_text(
        json.dumps(feature_columns, indent=2), encoding="utf-8"
    )
    (MODEL_DIR / "hyperparameters.json").write_text(
        json.dumps(best_cfg.to_dict(), indent=2), encoding="utf-8"
    )
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    combined_metrics = {"cv_summary": cv_summary, "student_test": test_metrics}
    (MODEL_DIR / "metrics.json").write_text(
        json.dumps(combined_metrics, indent=2), encoding="utf-8"
    )

    tuning_summary.to_csv(METRICS_DIR / "knn_tuning_results.csv", index=False)
    cv_folds_df.to_csv(METRICS_DIR / "knn_cv_folds.csv", index=False)
    (METRICS_DIR / "knn_student_test_metrics.json").write_text(
        json.dumps(combined_metrics, indent=2), encoding="utf-8"
    )

    print("\n=== FINAL KNN STUDENT-TEST RESULTS (Development Evaluation) ===", flush=True)
    print(json.dumps(test_metrics, indent=2), flush=True)
    return combined_metrics


if __name__ == "__main__":
    run_knn_training()
