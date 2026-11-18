# PathSnap Person 4 — Reusable MLP Implementation Report

**Workspace Root:** `C:\Users\hp\Desktop\ML_hacthon`
**Package Location:** `experiments/mlp/`
**Training Status:** **NOT RUN on real dataset yet** (intentionally gated until Person 2's final candidate generator and Person 3's approved shared feature schema are delivered).
**Safety Compliance:** No teammates' files or existing KNN files modified, no GitHub pushes/commits, no saved real-data models or fabricated metrics, synthetic fixtures used strictly for unit testing.

---

## 1. Files Created

| File | Responsibility |
| :--- | :--- |
| `experiments/mlp/__init__.py` | Public package exports for MLP config, shared data contract, pipeline, evaluation, and gated artifact saver. |
| `experiments/mlp/config.py` | Frozen `MLPConfig` dataclass with full hyperparameter validation; re-exports `ResourceBudget` from `experiments.knn.config`. |
| `experiments/mlp/data_contract.py` | Direct reuse/re-export of `experiments.knn.data_contract` so KNN and MLP share one identical schema and memory-safety contract. |
| `experiments/mlp/model.py` | Leakage-safe `build_mlp_pipeline` (`StandardScaler` + `MLPClassifier`), batched `predict_positive_proba`, `predict_classes`, `extract_training_history`, and reused `select_training_subset`. |
| `experiments/mlp/evaluate.py` | Reuses `observation_scores`, `compute_route_continuity`, and `score_predictions` from `experiments.knn.evaluate`; adds `benchmark_mlp_latency`, `evaluate_mlp_experiment`, and authorization-gated `save_mlp_artifacts`. |
| `experiments/mlp/run_mlp.py` | CLI runner supporting `--benchmark-only` pre-flight resource/latency checks and explicit `--authorize-save` safety gate. |
| `experiments/mlp/test_mlp.py` | 7 synthetic-fixture unit tests covering contract validation, leakage/partition guards, memory budget, hyperparameters, reproducibility, grouped CV scaling isolation, metrics, and save gate. |
| `reports/person4/MLP_IMPLEMENTATION_REPORT.md` | This implementation report. |

---

## 2. Model Architecture & Configuration

The MLP model is implemented as a scikit-learn `Pipeline` (`experiments/mlp/model.py`):

```text
Input Features (dynamic numeric schema)
  └── StandardScaler()                    # Fit strictly inside each training fold
        └── MLPClassifier(...)            # Configurable feed-forward neural network
```

### Supported Configurable Hyperparameters (`MLPConfig`)
- **`hidden_layer_sizes`** (`tuple[int, ...]`, default `(64, 32)`): Validated non-empty tuple of positive integers.
- **`activation`** (default `"relu"`): `"identity"`, `"logistic"`, `"tanh"`, or `"relu"`.
- **`solver`** (default `"adam"`): `"adam"`, `"sgd"`, or `"lbfgs"`.
- **`alpha`** (default `1e-4`): L2 weight regularization strength ($\ge 0$).
- **`learning_rate`** (default `"constant"`): `"constant"`, `"invscaling"`, or `"adaptive"` (passed when `solver="sgd"`).
- **`learning_rate_init`** (default `1e-3`): Initial step size for `"adam"` / `"sgd"`.
- **`max_iter`** (default `50`): Maximum training epochs/iterations.
- **`early_stopping`** (default `True`): Internal early stopping on `"adam"` / `"sgd"` (explicitly rejected if paired with `solver="lbfgs"`), paired with `validation_fraction` (default `0.1`) and `n_iter_no_change` (default `10`).
- **`batch_size`** (default `1024`): Training mini-batch size (`"auto"` or positive int) for memory-conscious training on large candidate tables.
- **`inference_batch_size`** (default `50_000`): Chunk size for memory-bounded `predict_positive_proba`.
- **`random_state`** (default `42`): Deterministic weight initialization, mini-batch shuffling, and fold-only sampling.
- **`use_float32`** (default `True`) and **`max_train_rows_per_fold`** (default `None`).

### Genuine Optimization History Recording
`extract_training_history(pipeline)` reads real `loss_curve_` and `validation_scores_` attributes from fitted `MLPClassifier` instances (handling `None` safely when `early_stopping=False`). No artificial loss curves are ever constructed.

---

## 3. Data Contract Used

`experiments/mlp/data_contract.py` directly reuses `experiments/knn/data_contract.py` without modification, enforcing Person 1's shared candidate-feature CSV contract:

- **Required Metadata Columns:** `partition`, `trip_id`, `observation_id`, `candidate_arc_id`, `label`
- **Feature Columns:** Dynamically detected as all remaining numeric columns (never assuming the 9 temporary baseline features).
- **`trip_id` Preservation:** Always read with `dtype={"trip_id": str}` so leading zeros (`"00000001"`, `"00000090"`) are never stripped.
- **Memory-Safe Pre-Flight Check:** `estimate_csv_resources()` streams `partition` and `trip_id` in chunks plus a small sample before loading full datasets, raising `MemoryGuardError` if estimated working memory exceeds `ResourceBudget.max_memory_mb`.

---

## 4. Leakage Protections

1. **Partition Enforcement:** Only `"train"` and `"student_test"` partitions are permitted; any evaluator partition (`evaluator*`, `eval_test`, etc.) raises `ValueError` during streaming pre-flight checks.
2. **Ground-Truth Column Rejection:** Rejects forbidden leakage columns (`route`, `route_index`, `route_arc_id`, `arc_id`, `true_arc_id`, `ground_truth`, `ground_truth_arc_id`, `label_valid`, `dist_to_route_m`) from ever entering model features.
3. **Trip-Grouped Cross-Validation:** `GroupKFold(n_splits=5)` grouped by `trip_id` ensures no GPS observations from the same trip appear in both fit and validation splits.
4. **In-Fold Scaling Isolation:** `StandardScaler()` is constructed inside a fresh `Pipeline` per fold and fit strictly on `X_fit` — never touching validation folds or `student_test` (verified by outlier injection unit test).
5. **Artifact Save Gate:** `save_mlp_artifacts(..., authorize_save=False)` raises `PermissionError` unless explicit authorization (`authorize_save=True` / `--authorize-save`) is provided.

---

## 5. Evaluation Methodology

Reuses the exact evaluation primitives from `experiments/knn/evaluate.py` (aligned with Person 1's `scripts/run_person1_models.py`):

- **Observation-Level Top-1 Accuracy:** Highest-probability candidate per eligible `(trip_id, observation_id)` with exactly 1 positive candidate.
- **Observation-Level Macro F1 (`trip_id::candidate_arc_id`):** Trip-qualified arc class keys so local 0-based arc IDs from different trips are never conflated.
- **Candidate Binary Log Loss:** Evaluated across all candidate rows (with `1e-15` probability clipping safety).
- **Candidate Precision / Recall / F1** plus counts of eligible, zero-positive (candidate miss), and multi-positive observations.
- **Fit Time & Inference Time:** Measured per CV fold and on full `train` → `student_test`.
- **Optional Route Continuity:** Computed **only** when a valid road-graph arc map (`trip_arcs_by_id`) or decoder interface is provided; otherwise explicitly reported as `None`.

> **Important Scope Clarification:** All metrics produced by `experiments/mlp/` measure **candidate-classifier performance only** (explicitly tagged as `metrics_scope = "candidate_classifier_only_not_full_route_reconstruction"`). Classifier Top-1 Accuracy and Macro F1 are **not** full sequential route-reconstruction accuracy, which requires Person 3's HMM/Viterbi decoder applied on top of classifier emission probabilities.

---

## 6. Exact Tests Executed & Actual Results

### 6.1 MLP Synthetic Unit Tests
```powershell
.\backend\.venv\Scripts\python.exe -m pytest experiments\mlp\test_mlp.py -v
```
**Actual Output Summary:** `7 passed, 7 warnings in 3.16s` *(exit code `0`; warnings are expected scikit-learn `ConvergenceWarning`s from intentionally small `max_iter=10`/`20` fast synthetic fixtures)*
- `test_shared_data_contract_preserves_trip_ids_and_rejects_leakage` — **PASSED**
- `test_memory_guard_blocks_oversized_load` — **PASSED**
- `test_configurable_mlp_hyperparameters_and_validation` — **PASSED**
- `test_reproducibility_probability_bounds_and_batched_inference` — **PASSED**
- `test_group_kfold_isolation_and_scaler_no_cross_fold_leakage` — **PASSED**
- `test_macro_f1_trip_qualification_and_optional_route_continuity` — **PASSED**
- `test_end_to_end_synthetic_mlp_evaluation_and_save_authorization_gate` — **PASSED**

### 6.2 KNN Synthetic Unit Tests
```powershell
.\backend\.venv\Scripts\python.exe -m pytest experiments\knn\test_knn.py -v
```
**Actual Output Summary:** `8 passed in 2.02s` *(exit code `0`, 0 warnings)*

### 6.3 Full Workspace Regression Suite (Backend + KNN + MLP + Person 1)
```powershell
.\backend\.venv\Scripts\python.exe -m pytest -v
```
**Actual Output Summary:** `34 passed, 8 warnings in 3.54s` *(exit code `0`)*
- `backend/tests/test_api.py`: **12 passed**
- `experiments/knn/test_knn.py`: **8 passed**
- `experiments/mlp/test_mlp.py`: **7 passed**
- `tests/test_labels_and_metrics.py`: **7 passed**

### 6.4 CLI Verification
```powershell
.\backend\.venv\Scripts\python.exe experiments\mlp\run_mlp.py --help
```
**Actual Output Summary:** Exited with code `0`; verified all configurable MLP hyperparameters, `--benchmark-only`, and `--authorize-save` safety gate.

---

## 7. Known Limitations

1. **Internal Early-Stopping Split Is Row-Wise Inside Each Training Fold:** When `early_stopping=True`, scikit-learn's `MLPClassifier` holds out `validation_fraction` randomly from the fold's `X_fit` rows rather than by `trip_id`. While this **never** leaks held-out CV validation trips or `student_test` trips, internal stopping scores inside a training fold can be slightly optimistic compared to outer trip-grouped CV folds (which remain the true generalization estimate).
2. **Class Imbalance Handling:** scikit-learn's `MLPClassifier` does not accept `class_weight="balanced"` directly. Because positive candidates represent ~11–12% of candidate rows, binary probability calibration and Top-1 ranking should be monitored carefully during real-data experiments (comparing Log Loss and Top-1 Accuracy against Logistic Regression and KNN).
3. **CPU Training Speed on Multi-Million-Row Datasets:** On ~1.59M training rows, deep/wide MLPs with large `max_iter` can take several minutes per fold on CPU; `--benchmark-only` should always be executed first to size `batch_size`, `hidden_layer_sizes`, and `max_iter` appropriately.
4. **No Decoder Built In:** Raw MLP Top-1 predictions evaluate each GPS observation independently and can exhibit road-network discontinuities until integrated with Person 3's HMM/Viterbi decoder.

---

## 8. Dependencies Before Real-Data Training

1. **Person 2 (Sujeet):** Final approved candidate generation pipeline (replacing `src/pathsnap/candidates_baseline.py`).
2. **Person 3 (Monish):** Final approved shared candidate-feature CSV conforming to `partition, trip_id, observation_id, candidate_arc_id, label, <numeric features...>`, plus live feature-extraction function for Laptop 1's FastAPI inference pipeline.
3. **Person 3 (Monish):** Road-connectivity helper / HMM & Viterbi decoder interface for full sequential route-reconstruction evaluation and continuity comparison (classifier-only vs. classifier+HMM).
4. **Explicit Team Approval:** Only after the final shared feature CSV is approved should real-data training be executed with `--benchmark-only` first and `--authorize-save` second.
