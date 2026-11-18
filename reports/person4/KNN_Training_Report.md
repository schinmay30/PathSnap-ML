# PathSnap Person 4 — KNN Training and Evaluation Report

- **Author:** Person 4 (SRN: `PES1UG24CS676`)
- **Model Family:** Instance-based (`sklearn.neighbors.KNeighborsClassifier`)
- **Dataset:** `data/processed/person3_extended_candidates.csv.gz` (`1,775,299` candidate rows, 19 approved route-blind numerical features)
- **Partitions:** `train` (`80` trips, `1,591,642` rows) for tuning & 5-fold `GroupKFold` CV; `student_test` (`10` trips, `183,657` rows) strictly for final development evaluation
- **Evaluation Scope:** Candidate-selection performance only — **NOT** full sequential route-reconstruction accuracy (which requires HMM/Viterbi decoding).

---

## 1. Dataset & Approved 19-Feature Schema

Loaded once in compact `float32` + categorical chunks (`147.3 MB` DataFrame RAM, `5.25 s` load time). Metadata columns (`partition`, `trip_id`, `observation_id`, `candidate_arc_id`, `label`) were excluded from model inputs, `trip_id` was preserved as zero-padded `str`, and `.route` files were never used as inputs.

**19 Approved Route-Blind Features:**
`dist_m`, `dist_rel_m`, `dist_rank`, `n_candidates`, `arc_length_m`, `along_fraction`, `heading_cos`, `moving`, `speed_ms`, `nearest_endpoint_m`, `previous_step_m`, `previous_step_available`, `previous_bearing_cos`, `previous_bearing_available`, `zero_length_arc`, `entry_in_degree`, `exit_out_degree`, `nearest_runner_up_gap_m`, `has_runner_up`.

---

## 2. Data Leakage Prevention & Scalability Design

1. **Trip-Grouped Splitting:** `GroupKFold(n_splits=5)` grouped by `trip_id` on the `train` partition (`80` trips); all candidates of a trip stay in the same fold.
2. **In-Fold Scaling:** `StandardScaler()` is encapsulated inside `sklearn.pipeline.Pipeline` and fit strictly on each training fold.
3. **Strict `student_test` Holdout:** `student_test` (`10` trips) was never used during algorithm benchmarking, hyperparameter tuning, or reference-size selection.
4. **19D Algorithm Benchmark (Why `algorithm="brute"`):** In 19 dimensions, tree-based spatial indices suffer from the curse of dimensionality. Direct probe benchmarking on `n_fit = 200,000` × `10,000` queries measured `brute` (`euclidean`: `2.76 s`, `manhattan`: `7.98 s`) vs. `kd_tree` (`37.84 s`, `13.7x` slower) and `ball_tree` (over `3x` slower than `kd_tree`).
5. **Memory & Latency Control:** Stratified training-fold subsampling (`100,000` rows during 20-config tuning; `200,000` rows for final 5-fold CV and final fit, fixed `random_state=42`) was applied strictly inside training splits while always evaluating **100%** of validation fold rows (`~318k`/fold) and **100%** of `student_test` (`183,657` rows) using batched queries (`batch_size=2,000`).

---

## 3. Hyperparameter Tuning Results (`train` 5-Fold `GroupKFold` CV Only)

Evaluated all `20` combinations of `K ∈ {5, 11, 21, 35, 51}`, `weights ∈ {uniform, distance}`, and `metric ∈ {euclidean, manhattan}` (`100k` stratified train reference per fold). Complete table saved in `artifacts/metrics/knn_tuning_results.csv`.

### Top 8 Configurations Ranked by 5-Fold CV Observation Top-1 Accuracy

| Rank | Metric | `K` | Weights | CV Obs Top-1 Acc | CV Obs Macro F1 | CV Cand Acc | CV Binary Log Loss | CV ROC-AUC |
| ---: | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | `manhattan` | 51 | `distance` | **0.8284** | **0.7048** | **0.9579** | **0.1531** | **0.9762** |
| 2 | `manhattan` | 51 | `uniform` | 0.8277 | 0.7040 | 0.9576 | 0.1538 | 0.9760 |
| 3 | `manhattan` | 35 | `distance` | 0.8271 | 0.7035 | 0.9578 | 0.1686 | 0.9748 |
| 4 | `manhattan` | 35 | `uniform` | 0.8271 | 0.7021 | 0.9576 | 0.1692 | 0.9746 |
| 5 | `manhattan` | 21 | `uniform` | 0.8251 | 0.6993 | 0.9572 | 0.2043 | 0.9713 |
| 6 | `manhattan` | 21 | `distance` | 0.8246 | 0.6990 | 0.9574 | 0.2038 | 0.9715 |
| 7 | `manhattan` | 11 | `uniform` | 0.8207 | 0.6903 | 0.9563 | 0.2837 | 0.9640 |
| 8 | `manhattan` | 11 | `distance` | 0.8195 | 0.6899 | 0.9563 | 0.2834 | 0.9642 |

**Selection Rationale:**
- **Metric (`manhattan` vs. `euclidean`):** L1 (`manhattan`) consistently outperformed L2 (`euclidean`, best `0.8173` Top-1) by `+1.1` percentage points across all $K$, because the 19-feature vector mixes continuous distances, binary flags (`moving`, `has_runner_up`, `zero_length_arc`), and discrete graph degrees where L1 is less sensitive to extreme outlier steps.
- **Neighborhood Size (`K=51`) & Weighting (`distance`):** Larger $K$ steadily improved probability calibration (Log Loss dropped from `0.491` at $K=5$ to `0.153` at $K=51$) and observation-level ranking, with inverse-distance weighting delivering the highest Top-1 Accuracy, Macro F1, ROC-AUC, and PR-AUC.

---

## 4. Final 5-Fold `GroupKFold` CV (`train`, `200,000` Stratified Fit Rows/Fold)

Increasing the per-fold reference set from `100,000` to `200,000` improved 5-fold CV Observation Top-1 Accuracy from `0.8284` to **`0.8291`** and Observation Macro F1 from `0.7048` to **`0.7066`** (`artifacts/metrics/knn_cv_folds.csv`).

| Fold | Val Trips | Eligible Obs | Cand Acc | Obs Top-1 Acc | Obs Macro F1 | Binary Log Loss | ROC-AUC | PR-AUC | Fit (s) | Inf (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 15 | 41,873 | 0.9504 | 0.8178 | 0.6505 | 0.2052 | 0.9697 | 0.8814 | 0.07 | 436.67 |
| 2 | 17 | 30,717 | 0.9673 | 0.8354 | 0.7354 | 0.1350 | 0.9779 | 0.9010 | 0.08 | 490.25 |
| 3 | 16 | 35,529 | 0.9612 | 0.8300 | 0.7259 | 0.1371 | 0.9790 | 0.9023 | 0.08 | 477.71 |
| 4 | 16 | 32,453 | 0.9649 | 0.8348 | 0.6908 | 0.1387 | 0.9789 | 0.8946 | 0.08 | 644.14 |
| 5 | 16 | 44,158 | 0.9499 | 0.8277 | 0.7302 | 0.1700 | 0.9752 | 0.8929 | 0.12 | 319.65 |
| **Mean** | **16** | **36,946** | **0.9588** | **0.8291** | **0.7066** | **0.1572** | **0.9761** | **0.8944** | **0.09** | **473.69** |

---

## 5. Final Development Evaluation on Held-Out `student_test`

> **Notice:** These results are **development evaluation** on the 10 `student_test` trips (`183,657` candidate rows; `19,131` eligible observations; `2` candidate misses), **not** reserved evaluator-partition results, and measure candidate selection only (prior to HMM/Viterbi decoding).

| Metric | Value (`student_test`) |
| :--- | ---: |
| **Observation Top-1 Accuracy** | **0.8403 (84.03%)** |
| **Observation Macro F1 (`trip_id::arc_id`)** | **0.6780** |
| **Candidate Accuracy** | **0.9654 (96.54%)** |
| **Candidate Macro F1** | **0.9024** |
| **Candidate Weighted F1** | **0.9645** |
| **Candidate Precision** | **0.8775** |
| **Candidate Recall** | **0.7765** |
| **Candidate Binary F1** | **0.8239** |
| **Candidate Binary Log Loss** | **0.1124** |
| **ROC-AUC** | **0.9832** |
| **PR-AUC** | **0.9124** |
| **Confusion Matrix `[[TN, FP], [FN, TP]]`** | `[[162452, 2074], [4276, 14855]]` |
| **Training Time (200k refit)** | `0.0792 s` |
| **Inference Time (183,657 rows)** | `230.2563 s` |
| **Serialized Model Size (`model.joblib`)** | `6.663 MB` |
| **Peak Traced Memory** | `134.17 MB` |

---

## 6. Saved Artifacts & Verification

- **Model & Preprocessing:** `artifacts/models/knn/model.joblib` (`6.663 MB`), `artifacts/models/knn/scaler.joblib`
- **Schema & Config:** `artifacts/models/knn/feature_columns.json`, `artifacts/models/knn/hyperparameters.json`, `artifacts/models/knn/metadata.json`, `artifacts/models/knn/metrics.json`
- **Machine-Readable Metrics:** `artifacts/metrics/knn_tuning_results.csv`, `artifacts/metrics/knn_cv_folds.csv`, `artifacts/metrics/knn_student_test_metrics.json`
- **Training Script & Tests:** `experiments/knn/train_knn_experiment.py`, `experiments/knn/test_knn.py`
- **Verification:** Reloaded `artifacts/models/knn/model.joblib` and verified finite `predict_proba` output shape `(2, 2)` on 19-feature input; all `experiments/knn/test_knn.py` unit tests pass.
