# PathSnap Person 4 — Latest GitHub Repository Integration Audit

**Audited Repository:** `https://github.com/schinmay30/PathSnap-ML.git`
**Branch / Commit Audited:** `main` @ `0ae4920f28017528f817f4a063c2725e3a431f92` (`2026-11-18 11:09:13 +0530`, Author: Chinmay Patel / Person 1)
**Local Workspace Audited:** `C:\Users\hp\Desktop\ML_hacthon`
**Audit Mode:** Strictly read-only inspection (no GitHub pushes/commits, no local overwrites, no model training, no dataset changes).

---

## 1. Executive Summary

The latest `main` branch of `PathSnap-ML` contains **two commits**, both authored by **Person 1 (Chinmay Patel)**. It provides a clean, well-tested foundation for raw data loading, trip validation, metric geometry, road-network analysis, monotone DP GPS-to-route reference labeling (`results/labels/observation_labels.csv.gz`), a stand-in baseline candidate generator (`src/pathsnap/candidates_baseline.py`), two baseline models (**Logistic Regression** and **Gaussian Naive Bayes**), EDA figures/tables, 7 passing unit tests, and the shared team `report/report_outline.md`.

### Current Implementation Matrix Across Team Responsibilities

| Component | Owner | Status on GitHub `main` | Status in Local `ML_hacthon` |
| :--- | :--- | :--- | :--- |
| Raw data loader, validation, geo, network stats | Person 1 | **Complete** (`src/pathsnap/{data_io,preprocess,geo,network}.py`) | Not pulled yet |
| Reference route alignment & label table | Person 1 | **Complete** (`src/pathsnap/labels.py`, `results/labels/observation_labels.csv.gz`) | Not pulled yet |
| Stand-in baseline candidate generation (9 features) | Person 1 | **Complete (stand-in only)** (`src/pathsnap/candidates_baseline.py`) | Not pulled yet |
| Logistic Regression & Gaussian Naive Bayes | Person 1 | **Complete** (`src/pathsnap/person1_models.py`, `experiments/person1/results/`) | Not pulled yet |
| Final candidate generation & Folium maps | Person 2 (Sujeet) | **Not present yet** | Not present |
| Tree / Ensemble models (DT, RF, Extra Trees) | Person 2 | **Not present yet** | Not present |
| Final feature engineering & Boosting (HistGB) | Person 3 (Monish) | **Not present yet** | Not present |
| Bayesian optimization & final model (`artifacts/final/`) | Person 3 | **Not present yet** | Not present |
| HMM / Viterbi decoder | Person 3 | **Not present yet** | Not present |
| **KNN model implementation** | **Person 4** | **Not present yet** | Scaffold only (`experiments/knn/`) |
| **MLP model implementation** | **Person 4** | **Not present yet** | Scaffold only (`experiments/mlp/`) |
| **FastAPI backend (Laptop 1)** | **Person 4** | **Not present yet** | **Foundation Complete & Tested** (12/12 pytest) |
| **React + Vite frontend (Laptop 2)** | **Person 4** | **Not present yet** | **Complete & Verified** (`lint` + `build` 0 warnings) |

> **Critical Governance Finding:** Person 1's `README.md` and `src/pathsnap/candidates_baseline.py` explicitly state that the current 9-feature baseline generator is a **temporary stand-in** to be replaced by **Person 2 (Sujeet)'s candidate generation** and **Person 3 (Monish)'s feature engineering**. Therefore, **do NOT run final official KNN/MLP training until the team's approved shared candidate-feature CSV is finalized** (or explicitly ratified by the team).

---

## 2. Deep Dive: Person 1's Pipeline & Data Contracts

### 2.1 Data Loader & Preprocessing
- **`src/pathsnap/data_io.py`**:
  - `read_track(path)` → `list[tuple[float, float, float]]` (`lon, lat, timestamp_sec`)
  - `read_nodes(path)` → `list[tuple[float, float]]` (`lon, lat`, 0-based `node_id`)
  - `read_arcs(path)` → `list[tuple[int, int]]` (`from_node_id, to_node_id`, 0-based `arc_id`)
  - `read_route(path)` → `list[int]` (chronological sequence of `arc_id`s)
  - `find_dataset_root(path)` → automatically handles the double-nested `Student folder/Student folder/` layout by locating the unique directory containing both `train/` and `student_test/`.
- **`src/pathsnap/preprocess.py`**:
  - `TripData(trip_id, track, nodes, arcs, route)` frozen dataclass.
  - `load_trip(trip_dir)` loads all four files and runs `validate_trip(trip)` (checks non-empty files, finite values, valid lon/lat ranges, strictly increasing timestamps, valid node/arc bounds).
  - **FastAPI Integration Warning:** `load_trip()` unconditionally loads `.route` and raises `FileNotFoundError` if `.route` is absent. For Laptop 1's live inference API, Person 4 must **not** pass `trip.route` into any feature generator, model, or API response — keep `.route` strictly separated for server-side evaluation only.
- **`src/pathsnap/geo.py`**:
  - `project_local(lon, lat, lat0)` — per-trip local equirectangular metric projection ($x = R \cdot \text{lon} \cdot \cos(\text{lat}_0)$, $y = R \cdot \text{lat}$, $R = 6{,}371{,}000\text{ m}$), accurate to $< 0.5\text{ m}$ over trip scale.
  - `haversine_m(...)`, `point_to_segments_m(...)`, `pairs_point_segment_m(...)`.

### 2.2 Reference Route Alignment & Labels (`src/pathsnap/labels.py`)
Because `.route` lists traversed arcs in travel order (much shorter than `.track`), Person 1 aligned each GPS observation to a route arc using **monotone dynamic programming** (`align_trip_to_route`):
- **States:** route position index $j \in \{0, \dots, m-1\}$.
- **Emission cost:** point-to-arc distance in meters, capped at `DEFAULT_COST_CAP_M = 100.0`.
- **Transition:** stay at $j$ or move forward $j' > j$ with penalty `DEFAULT_SKIP_PENALTY_M = 1.0` per skipped arc; backward jumps forbidden.
- **Validity threshold:** `label_valid = (dist_to_arc_m <= 50.0)`. Invalid rows (e.g., long stationary off-route dwells in `train` trips `00000054` and `00000033`) are **dropped from labelled data**, never labelled `0`.
- **Tracked Output (`results/labels/observation_labels.csv.gz`):**
  ```text
  partition,trip_id,observation_id,route_index,arc_id,dist_to_arc_m,label_valid
  ```
  - Valid observations: `185,277 / 208,417` in `train` (88.9%), `19,133 / 19,160` in `student_test` (99.86%).
- **Helper functions:**
  - `label_candidates(candidates, labels)` — inner-joins on `["trip_id", "observation_id"]` where `label_valid == True`, sets `label = int(candidate_arc_id == arc_id)`.
  - `candidate_miss_rate(labelled, labels=None)` — reports fraction of valid observations missing their reference arc among generated candidates.

### 2.3 Stand-In Candidate Generator (`src/pathsnap/candidates_baseline.py`)
- Parameters: `radius_m = 50.0`, `max_candidates = 20`, `spacing_m = 10.0` (KD-tree over densified arcs).
- Never reads `.route` (zero leakage).
- Summary from `results/candidate_generation_summary.csv`:
  - **`train`:** `185,277` valid observations → **`1,591,642` candidate rows** (mean `8.59` candidates/obs, miss rate `0.00295` / 0.30%).
  - **`student_test`:** `19,133` valid observations → **`183,657` candidate rows** (mean `9.60` candidates/obs, miss rate `0.00010` / 0.01%).
  - Output file `data/processed/baseline_candidates.csv.gz` is **gitignored** and must be generated locally via `scripts/build_baseline_candidates.py` if needed for smoke-testing.

---

## 3. Exact Shared Feature CSV Contract & ID Handling

From `README.md` and `scripts/run_person1_models.py`, the team's shared candidate-feature CSV contract is:

### Required Metadata Columns (5 columns)
| Column | Type | Semantics & Critical Rules |
| :--- | :--- | :--- |
| `partition` | `str` | Strictly `"train"` or `"student_test"` (evaluator data is **never** allowed). |
| `trip_id` | `str` | **Zero-padded 8-digit string** (e.g., `"00000090"`). **MUST always read CSVs with `dtype={"trip_id": str}`** so leading zeros are preserved. |
| `observation_id` | `int` | 0-based row index into that trip's `.track` file. Uniquely identifies an observation only together with `trip_id`. |
| `candidate_arc_id` | `int` | 0-based row index into that trip's `.arcs` file. **Local to one trip** — arc `57` in two different trips represents two completely different roads! |
| `label` | `int` | Binary `{0, 1}`: `1` if `candidate_arc_id` matches the aligned reference `arc_id` (and `label_valid == True`), else `0`. |

### Feature Columns
- **Dynamic Contract Rule:** `scripts/run_person1_models.py` treats **every column not in `METADATA_COLUMNS`** as a numeric feature, checks that all values are finite (`np.isfinite`), and fits `StandardScaler` inside a scikit-learn `Pipeline` on each training fold.
- **Current 9 Stand-In Baseline Features** (in `src/pathsnap/candidates_baseline.py` — **NOT final approved schema**, pending Person 2 & Person 3):
  1. `dist_m` — perpendicular/clamped distance from GPS point to candidate arc (m)
  2. `dist_rel_m` — `dist_m - min_dist_m` among candidates for that observation (m)
  3. `dist_rank` — 0-based rank by distance within observation's candidate set
  4. `n_candidates` — total number of candidates generated for that observation
  5. `arc_length_m` — length of the candidate arc (m)
  6. `along_fraction` — closest-point projection fraction $t \in [0, 1]$ along the arc
  7. `heading_cos` — cosine similarity between GPS motion vector and directed arc vector (`0.0` when stationary)
  8. `moving` — binary indicator (`1` if `speed_ms > 1.0`, else `0`)
  9. `speed_ms` — local speed estimated from neighboring GPS fix (with gap fallback at `dt > 5.0 s`)

### Official Evaluation Protocol & Metrics (`scripts/run_person1_models.py`)
Person 4's KNN and MLP evaluation **must match this exact protocol** so all 5+ models are apples-to-apples comparable in the Section 4.2 comparison table:
1. **CV Split:** `GroupKFold(n_splits=5)` grouped by `trip_id` across the 80 `train` trips (`random_state=42` where applicable).
2. **Preprocessing Isolation:** `StandardScaler()` (and any imputation/transforms) fit strictly inside each fold's training split via `sklearn.pipeline.Pipeline`.
3. **Observation-Level Top-1 Accuracy & Macro F1:**
   - Group rows by `(trip_id, observation_id)`.
   - Filter to eligible observations with **exactly 1 positive candidate** (`sum(label) == 1`); report `observations_without_positive_candidate_label` and `observations_with_multiple_positive_labels` separately.
   - Select candidate with maximum predicted positive probability `p = predict_proba(X)[:, class_1_idx]` per observation.
   - **Critical Class Qualification:** Construct class labels as `f"{trip_id}::{candidate_arc_id}"` so local arc IDs from different trips are never conflated!
4. **Candidate-Level Binary Log Loss:** `sklearn.metrics.log_loss(y_candidate, probabilities, labels=[0, 1])`.
5. **Additional Metrics Required by `Deliverables.pdf` & `report/report_outline.md`:**
   - Route Continuity (fraction of consecutive predicted arcs where `prev.to_node == next.from_node` — requires `.arcs` lookup per trip, not yet computed in `run_person1_models.py`!).
   - Candidate Precision, Recall, F1, Confusion Matrix (diagnostic).
   - Fit time (CV & full train) and inference time on `student_test`.
   - For **MLP only**: actual epoch-wise training and validation loss curves (never fabricate epoch curves for KNN).

### Person 1 Reference Baselines (Measured on Stand-In 9 Features)
Recorded in `experiments/person1/results/` (for context only; will be re-run once approved features land):
- **Logistic Regression:** 5-fold CV Top-1 `0.7876 ± 0.0158`, CV Macro F1 `0.6390 ± 0.0430`, Log Loss `0.2634 ± 0.0377`; `student_test` Top-1 `0.8061`, Macro F1 `0.6106`, Log Loss `0.2356`, fit `1.55 s`, inference `11.3 ms`.
- **Gaussian Naive Bayes:** 5-fold CV Top-1 `0.7771 ± 0.0155`, CV Macro F1 `0.6218 ± 0.0446`, Log Loss `0.5740 ± 0.1774`; `student_test` Top-1 `0.7908`, Macro F1 `0.5960`, Log Loss `0.5794`, fit `0.47 s`, inference `23.0 ms`.

---

## 4. Existing Tests, Dependencies & Report Structure

### 4.1 Automated Tests (`tests/test_labels_and_metrics.py`)
- Executed via `backend\.venv\Scripts\python.exe -m pytest` against the audited repo: **7 passed in 4.91 s**.
- Tests verify: local projection vs. Haversine agreement (`< 0.5 m`), point-to-segment clamping, monotone DP route alignment, off-route invalid flagging (`> 50 m`), road-network component/break counts, **`trip_id::arc_id` Macro F1 isolation**, and `label_candidates` + `candidate_miss_rate`.

### 4.2 Dependencies Comparison (`requirements.txt`)
| Package | GitHub `main` | Local `ML_hacthon` | Merge Action Needed |
| :--- | :---: | :---: | :--- |
| `numpy`, `pandas`, `scipy`, `scikit-learn`, `joblib`, `matplotlib`, `pytest` | Yes | Yes | Compatible (union versions cleanly) |
| `fastapi`, `uvicorn[standard]`, `pydantic`, `httpx`, `networkx` | No | Yes | Keep in merged `requirements.txt` for Laptop 1 API + graph/HMM |
| `folium` / `optuna` (or `scikit-optimize`) | No | No | Confirm with Persons 2 & 3 when merging |

### 4.3 Shared Technical Report Structure (`report/report_outline.md`)
Person 4's assigned ownership in the team's report outline:
- **Section 1 (Problem & System Overview):** Draft system architecture & two-laptop deployment diagram.
- **Section 4.2 & 4.3 (Model Comparison & Individual Subsections):** Supply real measured metrics, configurations, loss/objectives, overfitting/underfitting analysis, and plots for **KNN** and **MLP**.
- **Section 7 (Geographic Visualizations):** Optional UI route display screenshots complementing Person 2's mandatory Folium maps.
- **Section 8 (API, UI, and Deployment):** Primary owner — API endpoints, model loading, two-laptop deployment steps, screenshots of working React UI & Laptop 1 CMD API logs, and proof of server-side ground-truth isolation.
- **Section 9 (Limitations, Reproducibility, Conclusion):** Consolidate final section across all teammates.

---

## 5. Comparison: GitHub `main` vs. Local Workspace (`ML_hacthon`)

| Area | GitHub `main` | Local `ML_hacthon` | Conflict / Integration Assessment |
| :--- | :--- | :--- | :--- |
| `src/pathsnap/` & `scripts/` | Exists (Person 1) | Does not exist | **Zero conflict** — can be pulled cleanly when ready. |
| `backend/` & `frontend/` | Does not exist | **Exists & verified** (Person 4) | **Zero conflict** — unique to Person 4. |
| `experiments/` | `experiments/person1/results/` | `experiments/knn/`, `experiments/mlp/` | **Zero conflict** — cleanly separated subfolders. |
| `artifacts/` | `.gitignore` tracks ONLY `artifacts/final/*.joblib` (ignores `artifacts/models/` & `*.joblib`) | `artifacts/models/`, `artifacts/metrics/` | **Minor alignment needed:** Update `backend/app/core/config.py` & `backend/models/loader.py` to also search `artifacts/final/` so the final tracked model is auto-discovered! |
| `reports/` vs. `report/` | `report/report_outline.md` | `reports/person4/*.md` | **Zero conflict** (different folder names), align conventions before final submission. |
| `tests/` | `tests/test_labels_and_metrics.py` | `tests/.gitkeep` & `backend/tests/test_api.py` | **Zero conflict** — both pytest suites can run together (`pytest tests backend/tests`). |
| Root `README.md`, `requirements.txt`, `.gitignore` | Person 1 versions | Person 4 versions | **Manual merge required** on a feature branch (never overwrite Person 1's entries; combine both cleanly). |
| Git repo initialization | Remote `main` exists | Local `ML_hacthon` is not yet a git repo | **Important:** Do NOT overwrite local work or push directly to `main`. Connect via a clean feature branch (e.g., `person4/backend-frontend-models`) after backing up and merging root files carefully. |

---

## 6. Recommended Implementation Sequence for Person 4 (KNN, MLP & Integration)

### Step 1: Confirm Approved Feature Schema with Persons 1, 2 & 3 (Gate Before Final Training)
- Ask **Person 2 (Sujeet)** and **Person 3 (Monish)** for the ETA and path of the **approved shared candidate-feature CSV** (or confirm whether Person 1's 9-feature baseline will serve as v1 and be upgraded later).
- Confirm whether `route_continuity` will be computed via a shared helper in `src/pathsnap/` so all 8 models report identical continuity numbers.

### Step 2: Implement Schema-Agnostic KNN & MLP Code First (Ready for Any Approved CSV)
Design Person 4's model code so it reads **any** CSV conforming to the shared contract (`partition, trip_id, observation_id, candidate_arc_id, label, <numeric features...>`), importing or mirroring Person 1's exact validation and `observation_scores` (`trip_id::candidate_arc_id`) logic:
1. **KNN (`experiments/knn/` / `src/pathsnap/person4_models.py`):**
   - Pipeline: `StandardScaler()` → `KNeighborsClassifier` (evaluate e.g. `n_neighbors`, `weights="distance"` vs. `"uniform"`, `metric="euclidean"` / `"manhattan"`).
   - **Critical Scalability Engineering (1.59M training rows!):**
     - Standard KNN over 1.59M candidate rows × 183K test rows (and 5 CV folds of ~1.27M train / ~318K val rows) can be very slow if run naively.
     - Use `algorithm="kd_tree"` (if feature count $\le 15$–$20$) or `"ball_tree"`, `n_jobs=-1`, and benchmark single-fold timing first.
     - If exact full-set KNN exceeds hackathon time budgets, document a principled, reproducible, **trip-balanced / stratified training subsample** (fixed `random_state=42`, strictly inside each training fold, evaluated on **100% of validation and `student_test` rows**) and clearly report this design choice and inference-cost limitation in Section 4.3!
   - Do **not** generate epoch-wise training loss curves for KNN (prohibited by guidelines).
2. **MLP (`experiments/mlp/` / `src/pathsnap/person4_models.py`):**
   - Pipeline: `StandardScaler()` → MLP (`sklearn.neural_network.MLPClassifier` with `early_stopping=True`, L2 `alpha`, `batch_size`, `random_state=42` — or PyTorch CPU with explicit train + trip-grouped validation split to log **real epoch-wise train loss and validation loss curves**!).
   - Note: `MLPClassifier` in scikit-learn exposes `loss_curve_` (training loss per epoch) and `validation_scores_` (validation score per epoch), whereas a clean PyTorch MLP (PyTorch 2.15 CPU is available, or `MLPClassifier` with `partial_fit` epoch loop) records **both exact training binary log loss and validation binary log loss per epoch** — ideal for Section 3.3 / 4.3 diagnostic curves!
   - Handle class imbalance cleanly (positives are ~11.6% of candidates; log uncalibrated vs. class-weighted impact on both Top-1 Accuracy and binary Log Loss).
3. **Shared Evaluation & Artifacts:**
   - Compute identical metrics to Person 1 (`GroupKFold(5)` by `trip_id`, `observation_top1_accuracy`, `observation_macro_f1` with `trip_id::candidate_arc_id`, `candidate_binary_log_loss`, plus `route_continuity`, candidate precision/recall/F1, confusion matrix, fit time, inference time).
   - Save CSVs and plots under `experiments/person4/results/` (or `experiments/{knn,mlp}/results/`) and export comparison summary compatible with `artifacts/metrics/model_comparison.json` so the React Model Comparison page renders real results automatically.

### Step 3: Run Official KNN & MLP Training ONLY on the Approved Feature CSV
- Once Persons 2 & 3 push the approved shared candidate-feature CSV (or the team formally approves the feature schema), execute the full 5-fold grouped CV + `student_test` runs for KNN and MLP, record real logs/seeds/configs/plots, and verify reproducibility.

### Step 4: Align Backend & Frontend with Final Team Pipeline (Person 3 Integration)
1. Update `backend/app/core/config.py` and `backend/models/loader.py` to also discover `artifacts/final/*.joblib` (matching upstream `.gitignore`).
2. Wire the real live inference pipeline into `backend/app/services/prediction_service.py`:
   - Load `.track`, `.nodes`, `.arcs` ONLY → run Person 2's candidate generator → run Person 3's feature extractor → score with loaded model (`predict_proba`) → decode route with Person 3's HMM/Viterbi decoder → evaluate accuracy server-side against `.route` (never returning `.route` to React!).
3. Export consolidated team metrics into `artifacts/metrics/model_comparison.json` so the React **Model Comparison** page populates with real numbers for Logistic Regression, Gaussian NB, KNN, MLP, and teammates' models.
4. Sync via a clean Git feature branch (PR into `main`), preserving all teammates' commits and files.
5. Run full two-laptop rehearsal (Laptop 1 FastAPI + Laptop 2 React UI across all 10 trips) and capture genuine UI & CMD API log screenshots for Section 8 of the technical report.

---

## 7. Open Coordination Questions for Teammates

1. **Person 2 (Sujeet) & Person 3 (Monish):** When will the approved shared candidate-feature CSV (and reusable Python feature-extraction function for FastAPI live inference) be pushed to `main`?
2. **Person 1 (Chinmay) & Person 3 (Monish):** Should Person 4 also compute `route_continuity` using `.arcs` on classifier-only Top-1 predictions so the Section 4.2 table has complete continuity values?
3. **Team Git Workflow:** Confirm branch naming convention (e.g., `person4/knn-mlp-api-ui`) for opening a PR into `main` without overwriting Person 1's `README.md`, `requirements.txt`, or `.gitignore`.
4. **Person 3 (Monish):** Confirm exact function signature and artifact format for the final model (`artifacts/final/*.joblib`) and HMM/Viterbi decoder so `backend/app/services/prediction_service.py` integrates cleanly on the first pass.
