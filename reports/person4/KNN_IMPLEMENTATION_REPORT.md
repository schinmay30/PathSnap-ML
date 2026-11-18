# PathSnap Person 4 — Reusable KNN Implementation Report

**Workspace Root:** `C:\Users\hp\Desktop\ML_hacthon`
**Package Location:** `experiments/knn/`
**Training Status:** **NOT RUN on real dataset yet** (intentionally gated until Person 2's final candidate generator and Person 3's approved shared feature schema are delivered).
**Safety Compliance:** No teammates' files modified, no GitHub pushes/commits, no saved models or fabricated metrics, synthetic fixtures used strictly for unit testing.

---

## 1. Architecture & Files Created (`experiments/knn/`)

| File | Responsibility |
| :--- | :--- |
| `experiments/knn/__init__.py` | Public exports for the reusable KNN module. |
| `experiments/knn/config.py` | Frozen dataclasses `KNNConfig` (`n_neighbors`, `weights`, `metric`, `algorithm`, `n_jobs`, `leaf_size`, `p`, `n_splits`, `batch_size`, `max_train_rows_per_fold`, `use_float32`) and `ResourceBudget` (`max_memory_mb`, `sample_rows`, `chunk_rows`). |
| `experiments/knn/data_contract.py` | Shared candidate-feature CSV validator, streaming pre-flight memory estimator (`estimate_csv_resources`), `MemoryGuardError`, dynamic feature detection (`detect_feature_columns`), and `load_candidate_dataset`. |
| `experiments/knn/model.py` | Leakage-safe `build_knn_pipeline` (`StandardScaler` + `KNeighborsClassifier`), deterministic fold-only stratified subsampling (`select_training_subset`), and chunked memory-bounded `predict_positive_proba`. |
| `experiments/knn/evaluate.py` | Exact `observation_scores` (Top-1 Accuracy + `trip_id::candidate_arc_id` Macro F1), optional `compute_route_continuity`, `score_predictions`, `benchmark_knn_latency`, `evaluate_knn_experiment`, and gated `save_knn_artifacts(..., authorize_save=False)`. |
| `experiments/knn/run_knn.py` | CLI supporting `--benchmark-only` (streaming RAM + probe latency check without training) and explicit `--authorize-save` gate before writing artifacts. |
| `experiments/knn/test_knn.py` | 8 unit tests using strictly small synthetic fixtures. |

---

## 2. Compliance with PathSnap & Person 1 Contracts

1. **Candidate-Feature CSV Contract:** Requires `partition, trip_id, observation_id, candidate_arc_id, label` plus arbitrary dynamically detected numeric feature columns. Does **not** assume or hardcode the 9 temporary baseline features.
2. **Zero-Padded `trip_id` Preservation:** Always loads CSVs with `dtype={"trip_id": str}` (verified in unit tests with `"00000001"`–`"00000090"`).
3. **Anti-Leakage Safeguards:**
   - Rejects any partition outside `{"train", "student_test"}` (blocks `evaluator*` data).
   - Rejects any accidental ground-truth/route column names (`route`, `route_index`, `route_arc_id`, `arc_id`, `true_arc_id`, `ground_truth`, `label_valid`, `dist_to_route_m`) from entering features.
   - Uses `GroupKFold(n_splits=5)` grouped by `trip_id` on `train` only, keeping `student_test` completely held out.
   - Encapsulates `StandardScaler()` inside a fresh `sklearn.pipeline.Pipeline` fit strictly inside each training fold.
4. **Large-Dataset Resource & Latency Management:**
   - `estimate_csv_resources()` streams only `partition` and `trip_id` in chunks plus a small sample (`sample_rows=5000`) to estimate DataFrame RAM and KNN index working memory **before** loading millions of rows; raises `MemoryGuardError` if `estimated_knn_working_mb > max_memory_mb`.
   - `benchmark_knn_latency()` measures real probe fit + query latency (`ms_per_1000_queries`) prior to full 5-fold CV.
   - Supports `float32` feature matrices, batched `predict_proba` (`batch_size=50_000`), and optional fold-only deterministic stratified capping (`max_train_rows_per_fold`, defaulting to `None`) while always evaluating 100% of validation and `student_test` observations.
5. **Exact Evaluation Metrics:**
   - Observation-level Top-1 Accuracy and Macro F1 with **`trip_id::candidate_arc_id`** class keys (matching `scripts/run_person1_models.py` and `tests/test_labels_and_metrics.py`).
   - Candidate Binary Log Loss (with safe epsilon clipping `1e-15` for KNN exact `0.0`/`1.0` probabilities), plus candidate Precision, Recall, F1, eligible/missing/multi-positive observation counts.
   - Training time (`fit_seconds`) and inference time (`inference_seconds`) per fold and on `student_test`.
   - `route_continuity`: explicitly returns `None` unless a valid road-graph arc connectivity map (`trip_arcs_by_id`) or decoder interface is provided.
6. **Save Authorization Gate:** `save_knn_artifacts()` raises `PermissionError` unless `authorize_save=True` (CLI `--authorize-save`). No `.joblib` models or official metrics were written.

---

## 3. Automated Tests Executed & Actual Results

### 3.1 KNN Synthetic Unit Tests
```powershell
.\backend\.venv\Scripts\python.exe -m pytest experiments\knn\test_knn.py -v
```
**Actual Result:** `8 passed in 4.69s` (0 errors, 0 warnings)
- `test_preserves_leading_zeros_in_trip_id_and_detects_custom_features` — **PASSED**
- `test_rejects_leakage_columns_and_evaluator_partition` — **PASSED**
- `test_memory_guard_blocks_oversized_load` — **PASSED**
- `test_configurable_knn_hyperparameters_and_batched_inference` — **PASSED**
- `test_macro_f1_qualifies_classes_by_trip_id` — **PASSED**
- `test_group_kfold_and_scaler_have_no_trip_leakage` — **PASSED**
- `test_route_continuity_none_without_graph_and_exact_with_graph` — **PASSED**
- `test_end_to_end_synthetic_evaluation_and_save_authorization_gate` — **PASSED**

### 3.2 Full Workspace Regression Suite (Backend + KNN + Person 1)
```powershell
.\backend\.venv\Scripts\python.exe -m pytest -v
```
**Actual Result:** `27 passed, 1 warning in 3.89s`
- `backend/tests/test_api.py`: **12 passed**
- `experiments/knn/test_knn.py`: **8 passed**
- `tests/test_labels_and_metrics.py`: **7 passed**

---

## 4. Planned Execution Commands (Once Approved Feature CSV Is Ready)

**Step 1 — Safe memory & latency pre-flight check (no training, no writes):**
```powershell
.\backend\.venv\Scripts\python.exe experiments\knn\run_knn.py --candidates <APPROVED_FEATURES_CSV> --benchmark-only
```

**Step 2 — Dry evaluation (no files saved):**
```powershell
.\backend\.venv\Scripts\python.exe experiments\knn\run_knn.py --candidates <APPROVED_FEATURES_CSV> --n-neighbors 15 --weights distance --metric euclidean --algorithm kd_tree
```

**Step 3 — Authorized final run (saves CV folds, CV summary, student_test JSON, and model `.joblib`):**
```powershell
.\backend\.venv\Scripts\python.exe experiments\knn\run_knn.py --candidates <APPROVED_FEATURES_CSV> --n-neighbors 15 --weights distance --metric euclidean --algorithm kd_tree --authorize-save
```

---

## 5. Remaining Dependencies Before Official KNN Training

1. **Person 2 (Sujeet):** Final approved candidate generation pipeline (replacing `src/pathsnap/candidates_baseline.py`).
2. **Person 3 (Monish):** Final approved shared candidate-feature CSV (and live feature-extraction module) following the shared contract (`partition, trip_id, observation_id, candidate_arc_id, label, <numeric features...>`).
3. **Person 3 (Monish) / Team:** Confirmation of whether classifier-only route continuity should be evaluated using `.arcs` connectivity (already supported via `trip_arcs_by_id`) and/or via Person 3's HMM/Viterbi decoder interface.
4. **Next Person 4 Task:** Implement the matching reusable **MLP** module inside `experiments/mlp/` (with genuine epoch-wise training & validation loss history logging), also gated until the approved feature CSV arrives.
