# PathSnap Hackathon — Person 4 Workspace Inspection Report

**Workspace Root:** `C:\Users\hp\Desktop\ML_hacthon` (note exact directory spelling: `ML_hacthon`)
**Role:** Person 4 (KNN & MLP Models, FastAPI Backend, React + Vite Frontend, System Integration, Final Demonstration & Screenshots)
**Inspection Scope:** Read-only audit of project guideline PDFs, raw dataset partitions, and local development environment. No implementation files or dataset modifications have been made.

---

## 1. Project Overview

**Course / Event:** PES University (Dept. of CSE, RR Campus) — `UE24CS352A: Machine Learning`
**Project Title:** **PathSnap: Distributed Map-Matching & Route Reconstruction**

PathSnap is an end-to-end distributed machine learning system that reconstructs realistic vehicle trajectories by matching noisy GPS observations (subject to urban multipath interference, GPS dropouts, and irregular sampling) to road-network segments (`.arcs` connecting `.nodes`).

### Required End-to-End Pipeline
1. **Raw Data Inspection & Preprocessing:** Parse per-trip GPS tracks, road-network nodes/arcs, and reference routes; handle geospatial calculations and GPS gaps.
2. **Candidate Generation & Feature Engineering:** Generate candidate road segments per GPS observation and extract meaningful spatial, geometric, and kinematic features.
3. **Model Experimentation (Minimum 5 Models):** Train, cross-validate (group-aware across the 80 training trips), and compare at least five distinct models across multiple algorithm families on the 10 `student_test` trips.
4. **Bayesian Hyperparameter Optimisation:** Perform Bayesian search (group-aware CV on the 80 training trips only) on the selected winning model, retain all trial logs, and compare tuned vs. untuned performance on `student_test`.
5. **HMM & Viterbi Route Reconstruction:** Combine classifier candidate probabilities (emissions) with road-network connectivity (transitions) via Viterbi decoding, handling GPS gaps, missing candidates, and invalid transitions. Compare classifier-only vs. classifier+HMM performance.
6. **Two-Laptop Distributed Deployment:**
   - **Laptop 1 (Backend / Model & API):** Runs a **FastAPI** server hosting the trained final classifier, preprocessing pipeline, HMM/Viterbi decoder, road-network data, and server-side evaluation logic.
   - **Laptop 2 (Frontend / UI):** Runs a **React + Vite** evaluator-facing web UI that lists the 10 evaluation trips, sends only the selected `trip_id` to Laptop 1's API, and renders the returned predicted route, map, and evaluation metrics.
7. **Deliverables (Exactly Two):**
   - **Deliverable 1:** Public GitHub repository containing all source code (data processing, feature engineering, all experimented models, evaluation, Bayesian optimization, HMM/Viterbi, FastAPI backend, React UI, Folium visualization scripts, deployment instructions/README, saved model & config artifacts, and experiment logs/CSVs).
   - **Deliverable 2:** Single consolidated Team Technical Report (PDF) following Sections 2–7 of `Deliverables.pdf`, including mandatory Folium maps, model comparison tables/plots, Bayesian optimization records, HMM analysis, and UI + backend CMD API-log screenshots.
   - *Note from `Deliverables.pdf` header:* **"There will be first phase evaluation before lunch break."**

### Inspected Guideline Documents (`PathSnap_student_materials/`)
All five PDFs in `PathSnap_student_materials/` (matching `PathSnap_student_materials.zip`) were read in full:
1. `Main Guidelines.pdf` (2 pages) — Challenge overview, two-laptop architecture, 12-stage workflow.
2. `Dataset Exploration.pdf` (4 pages) — File schemas, exploration checklist, geospatial considerations, reference-route caveats, 80/10/10 partitions, leakage rules.
3. `Models and Training Guidelines.pdf` (6 pages) — 5+ model requirement, algorithm families, evaluation metrics, Bayesian optimization, HMM/Viterbi requirements, final two-laptop live evaluation protocol, independent-work policy.
4. `Deliverables.pdf` (7 pages) — GitHub repo contents, required PDF report structure (Sections 2.2, 3.1–3.3, 4, 5, 6, 7, 8), compulsory per-model questions, Folium map requirements, UI/API screenshot requirements.
5. `Assessment and Plagiarism Policy.pdf` (3 pages) — Originality, cross-team similarity checks (code, weights, configs, results, repo history), AI-use disclosure/authenticity rules, viva expectations, leakage prohibitions.

---

## 2. Person 4 Responsibilities

Based on the team assignment and the official guidelines, Person 4 owns the following six areas:

1. **KNN Model Implementation (`Instance-based` family):**
   - Implement KNN (e.g., `KNeighborsClassifier` — uniform and/or distance-weighted) on the shared feature dataset produced by Persons 1–2.
   - Ensure proper feature scaling fit **strictly inside training folds** (no leakage across trips) since distance-based models are scale-sensitive.
   - Evaluate with group-aware CV on the 80 `train` trips and evaluate on the 10 `student_test` trips.
   - Record all required metrics: Accuracy, Precision, Recall, Macro F1-score, Confusion Matrix, Log Loss (if calibrated/probabilistic), Observation-level Top-1 Accuracy, Candidate Miss Rate, Route Continuity, Training Time, and Inference Time.
   - Save experiment logs, seeds, configurations, diagnostic plots, and answer all compulsory Section 3.2 questions (model understanding, hyperparameters, objective, student-test metrics, overfitting/underfitting evidence).
   - *Guideline note:* Do **not** generate artificial epoch-wise training curves for KNN (forbidden by `Deliverables.pdf` Section 3.3).

2. **MLP Model Implementation (`Neural networks` family):**
   - Implement an MLP classifier (via `scikit-learn` `MLPClassifier` and/or `PyTorch` CPU) with consistent feature scaling fit on training data only.
   - Log actual epoch-wise training and validation loss histories (allowed and valuable for Section 3.3 plots).
   - Record all required metrics (including Log Loss, Macro F1, Top-1 Accuracy, Route Continuity, Training Time, Inference Time), configurations, random seeds, and overfitting/underfitting evidence.
   - Answer all compulsory Section 3.2 questions for the technical report.
   - *(Conditional)* If either KNN or MLP is chosen as the team's winning model, coordinate Bayesian hyperparameter optimization and artifact export with Person 3.

3. **FastAPI Backend (Laptop 1):**
   - Build the production FastAPI service exposing endpoints to:
     - List available evaluation trip IDs (10 trips — supporting `student_test` during development and seamless switching to the 10 reserved evaluator trips on evaluation day).
     - Accept a selected `trip_id`, run the complete real-time inference pipeline (raw trip loading → preprocessing → candidate generation → feature extraction → final ML model scoring → HMM/Viterbi decoding), and compute accuracy/evaluation metrics **strictly server-side** using internal reference data (`.route`) without ever exposing ground-truth labels to the model or UI.
     - Return structured JSON containing the predicted route geometry/arc IDs, raw GPS points, road-network context (or downsampled viewport arcs), evaluation metrics (classifier-only vs. HMM/Viterbi if desired), and timing telemetry.
     - Emit clear, structured HTTP & inference request logs to stdout/CMD on Laptop 1 (mandatory: evaluators must see live API logs on Laptop 1 receiving inference requests).
     - Enable CORS and network binding (`0.0.0.0`) so Laptop 2 can communicate reliably over LAN/hotspot.

4. **React + Vite Frontend (Laptop 2):**
   - Build an evaluator-facing React + Vite web application that:
     - Connects to Laptop 1's FastAPI backend via configurable API base URL (e.g., `VITE_API_BASE_URL` / runtime settings).
     - Displays all 10 available evaluation trips cleanly for one-click selection.
     - Sends the selected `trip_id` to the FastAPI backend and shows loading/error states gracefully.
     - Renders an interactive geographical map displaying the raw GPS trajectory, relevant road-network segments, and the predicted reconstructed route with clear legends, color coding, and layer controls (e.g., Leaflet / React-Leaflet).
     - Displays returned evaluation metrics (accuracy, route continuity, inference time, etc.) clearly without ever receiving or displaying raw ground-truth labels from the client side.

5. **Integration with Person 3's Final ML Model & HMM/Viterbi Decoder:**
   - Integrate the frozen preprocessing artifacts, feature extractor, final tuned ML model, and Person 3's HMM/Viterbi decoder into the FastAPI inference pipeline.
   - Verify end-to-end execution across **all 10 `student_test` trips** (and verify that switching the data directory to any unseen 10-trip evaluator folder works out of the box with zero hardcoded trip IDs or answers).

6. **Final Demonstration, Two-Laptop Testing, & Screenshots:**
   - Test cross-laptop networking (Laptop 1 FastAPI ↔ Laptop 2 React UI) end-to-end across all 10 trips.
   - Capture genuine screenshots for Section 7 of the consolidated PDF report:
     - Working React UI showing trip selection, predicted route map, and evaluation results.
     - CMD/terminal output on Laptop 1 showing real incoming FastAPI inference requests and successful responses.
   - Prepare to explain KNN, MLP, FastAPI architecture, CORS/network flow, server-side ground-truth isolation, and UI implementation during the viva.

---

## 3. Dataset Structure

### Workspace File Layout
Currently present in `C:\Users\hp\Desktop\ML_hacthon`:
- `PathSnap_student_materials/` (5 guideline PDFs)
- `PathSnap_student_materials.zip` (762,978 bytes — same 5 PDFs)
- `Student folder.zip` (29,410,509 bytes — 453 zip entries)
- `Student folder/Student folder/` (**Important: double-nested directory from archive extraction!**)
  - `train/` — **80 trip folders**
  - `student_test/` — **10 trip folders**

### Trip Naming Convention & Partitions
- Trip IDs are **8-digit zero-padded integers** (`00000000` through `00000099`).
- Total universe of 100 trips split with **zero overlap**:
  - **`train` (80 trips):** Present locally. Used for training, group-aware CV, preprocessing fitting, and Bayesian optimization.
  - **`student_test` (10 trips):** Present locally — IDs: `00000000`, `00000007`, `00000009`, `00000040`, `00000052`, `00000056`, `00000065`, `00000076`, `00000077`, `00000091`. Used for development-stage model comparison and decoder tuning.
  - **Reserved Evaluator Test (10 trips):** Not included in student package (corresponds to missing IDs `00000015`, `00000017`, `00000023`, `00000030`, `00000049`, `00000059`, `00000067`, `00000073`, `00000087`, `00000094`). Provided only during final evaluation.

### Per-Trip File Formats (Verified Across All 90 Trips)
Every single one of the 90 inspected trip folders (`80 train + 10 student_test`) contains **exactly 4 files**, named `<trip_id>.<ext>`, **headerless** and **tab-separated (`\t`)**:

| File | Delimiter / Header | Columns & Types | Indexing / Meaning |
| :--- | :--- | :--- | :--- |
| `<id>.track` | `\t`, no header | `longitude` (float), `latitude` (float), `timestamp_sec` (float) | Chronological GPS observations. Nominally 1 Hz. |
| `<id>.nodes` | `\t`, no header | `longitude` (float), `latitude` (float) | Road-network vertices; **0-based line index = `node_id`**. |
| `<id>.arcs` | `\t`, no header | `from_node_id` (int), `to_node_id` (int) | Directed road segments; **0-based line index = `arc_id`**. |
| `<id>.route` | single col, no header | `arc_id` (int) | Chronological sequence of traversed road segments (reference path). |

### Critical Empirical Findings from Inspection
1. **`.route` is NOT 1-to-1 with `.track`:** Each `.route` file lists the ordered sequence of traversed road segments (far fewer rows than GPS points — e.g., trip `00000001` has 1,315 GPS points vs. 525 route arcs; `00000000` has 2,503 GPS points vs. 179 route arcs). Aligning GPS points to per-observation candidate labels requires careful preprocessing/projection (owned by Person 1/2, needed by Person 4).
2. **Global Geographic Distribution:** Trips are located in **different regions worldwide** (e.g., Europe, Russia, Central Asia, North America, South America, East Asia):
   - `train`: longitude `[-100.418538, 137.237029]`, latitude `[-34.640291, 61.267258]`
   - `student_test`: longitude `[-93.424337, 77.271115]`, latitude `[42.645529, 61.699590]`
   - *Implication:* Coordinate order is `(longitude, latitude)`. A single global UTM zone is invalid; per-trip local projection (e.g., local tangent plane / per-trip UTM) or geodesic/Haversine formulas are required.
3. **Summary Counts & Sampling Gaps:**
   - **`train` (80 trips):**
     - `.track`: 361 to 19,313 points/trip (median 1,529; total **208,417** GPS points)
     - `.nodes`: 299 to 155,003 nodes/trip (median 9,608)
     - `.arcs`: 649 to 307,392 arcs/trip (median 19,526)
     - `.route`: 59 to 2,266 arcs/trip (median 280; total 31,336)
     - Time step `dt`: min `1.0s`, median `1.0s`, **max `153,064.0s` (~42.5 hours!)**, `96.50%` exactly 1s, `0` non-positive steps.
     - Route continuity (`prev.to_node == next.from_node`): `31,243 / 31,256` (`99.96%`).
   - **`student_test` (10 trips):**
     - `.track`: 262 to 3,958 points/trip (median 1,420; total **19,160** GPS points)
     - `.nodes`: 1,770 to 26,635 nodes/trip (median 6,596)
     - `.arcs`: 3,577 to 56,027 arcs/trip (median 13,145)
     - `.route`: 95 to 1,323 arcs/trip (median 250; total 3,477)
     - Time step `dt`: min `1.0s`, median `1.0s`, **max `210.0s`** (trip `00000065` has a 210s gap; `00000076` has a 32s gap), `98.62%` exactly 1s, `0` non-positive steps.
     - Route continuity: `3,466 / 3,467` (`99.97%`).
4. **Data Integrity & Graph Nuances (All 90 Trips):**
   - `0` NaN/Inf values in `.track` or `.nodes`.
   - `0` out-of-bounds `node_id`s in `.arcs` or `arc_id`s in `.route`.
   - **`24` self-loop arcs** (`from_node == to_node`) and **`1,903` duplicate `(from_node, to_node)` directed arc pairs** exist across the 90 trips — code must key arcs by integer `arc_id` (0-based row index), not by `(from_node, to_node)` tuples!

### `student_test` Per-Trip Inventory

| Trip ID | Track Points | Nodes | Arcs | Route Arcs | Lon Range | Lat Range | Max `dt` (s) |
| :--- | ---: | ---: | ---: | ---: | :--- | :--- | ---: |
| `00000000` | 2,503 | 8,542 | 18,196 | 179 | `[54.248, 54.964]` | `[56.081, 56.255]` | 2.0 |
| `00000007` | 1,354 | 2,061 | 4,369 | 123 | `[77.122, 77.271]` | `[43.389, 43.496]` | 1.0 |
| `00000009` | 1,049 | 6,745 | 13,106 | 429 | `[13.632, 13.707]` | `[42.646, 42.662]` | 2.0 |
| `00000040` | 2,995 | 12,945 | 27,026 | 309 | `[61.069, 61.293]` | `[55.175, 55.416]` | 1.0 |
| `00000052` | 262 | 1,770 | 3,577 | 95 | `[35.858, 35.896]` | `[56.795, 56.827]` | 2.0 |
| `00000056` | 3,958 | 14,759 | 27,631 | 1,323 | `[-93.424, -93.349]` | `[44.836, 44.873]` | 3.0 |
| `00000065` | 768 | 6,448 | 13,184 | 145 | `[49.263, 49.330]` | `[53.513, 53.539]` | 210.0 |
| `00000076` | 1,486 | 26,635 | 56,027 | 374 | `[43.961, 44.034]` | `[56.240, 56.329]` | 32.0 |
| `00000077` | 3,911 | 4,424 | 9,622 | 268 | `[50.789, 50.841]` | `[61.640, 61.700]` | 1.0 |
| `00000091` | 874 | 6,360 | 12,303 | 232 | `[2.411, 2.444]` | `[47.053, 47.100]` | 1.0 |

*Performance note for API/UI:* Trips can contain up to 3,958 GPS points and 56,027 road-network arcs in `student_test` (and up to 19,313 points / 307,392 arcs in `train`). The API should **not** blindly serialize all 50,000–300,000 arcs to the browser; it should return only the relevant local road-network corridor (e.g., bounding box / candidate arcs around the trajectory) along with the GPS points and predicted route so the React map stays responsive.

---

## 4. Technology Requirements & Current Environment Audit

### Core Runtimes (Verified Installed)
| Tool | Status | Version / Location |
| :--- | :--- | :--- |
| Python | Installed | `Python 3.13.7` (`C:\Program Files\Python313\python.exe`) |
| pip | Installed | `25.2` |
| Node.js | Installed | `v22.19.0` (`C:\Program Files\nodejs\node.exe`) |
| npm / npx | Installed | `10.9.3` (`C:\Program Files\nodejs\npm.cmd`) |
| Git | Installed | `git version 2.55.0.windows.3` (workspace is **not** yet a git repository) |

### Relevant Python Packages — Installed vs. Missing
**Already installed in global Python 3.13.7:**
- ML / Neural Nets: `scikit-learn 1.9.0`, `torch 2.15.0+cpu`, `scipy 1.18.1`, `joblib 1.6.0`
- Data / Graph / HMM: `numpy 2.5.3`, `pandas 3.0.5`, `networkx 3.7`, `hmmlearn 0.3.3`
- Plotting / Notebooks: `matplotlib 3.11.1`, `seaborn 0.13.2`, `jupyterlab 4.6.6`, `ipykernel 7.4.0`
- Testing / HTTP: `pytest 9.2.0`, `httpx 0.28.1`, `requests 2.34.2`, `PyYAML 6.0.3`

**Missing (must be installed before backend/visualization work — recommended inside a project-local `.venv` inside `ML_hacthon`):**
- Backend API: `fastapi`, `uvicorn[standard]`, `pydantic` (v2), `python-multipart` (if needed)
- Geospatial / Maps (if needed by backend or report generation): `folium`, `pyproj` / `shapely` (confirm with Persons 1–3)
- Frontend (via `npm` inside frontend subfolder): `react`, `react-dom`, `vite`, `@vitejs/plugin-react`, `leaflet` / `react-leaflet` (or chosen map library)

---

## 5. Important Project Restrictions

From `Main Guidelines.pdf`, `Dataset Exploration.pdf`, `Models and Training Guidelines.pdf`, `Deliverables.pdf`, and `Assessment and Plagiarism Policy.pdf`:

1. **Workspace & File Safety:**
   - Perform all development inside `C:\Users\hp\Desktop\ML_hacthon`.
   - Never modify, rename, move, or delete original PDFs, zip archives, or dataset files in `Student folder/`.
   - Never overwrite teammates' work.
2. **Strict Anti-Leakage Rules:**
   - Never split observations from the same trip across train/validation/test partitions (must use **trip-level group-aware CV**, e.g., `GroupKFold` grouped by `trip_id`).
   - Fit scalers (`StandardScaler`, etc.) and learned preprocessors **strictly on training folds / training trips only**.
   - Never use evaluator-testing trips or labels for training, feature selection, hyperparameter tuning, or threshold selection.
3. **Server-Side Ground Truth Isolation (Critical for FastAPI & React):**
   - Ground-truth `.route` labels must remain **strictly server-side** on Laptop 1.
   - Ground-truth labels must **never** be passed into the ML prediction model, HMM decoder, or sent/exposed to the React UI on Laptop 2.
   - Hardcoded outputs or bypassing the API are strictly prohibited.
4. **Live Evaluation Protocol (Two Laptops):**
   - Laptop 1 runs the FastAPI backend; Laptop 2 runs the React UI.
   - System must work dynamically for **all 10 reserved evaluation trips** (not just a demo trip).
   - Evaluators will inspect **live API logs on Laptop 1's CMD/terminal** verifying real inference requests arriving at the endpoint.
5. **No Fabrication & Academic Integrity:**
   - Do not fabricate metrics, results, training curves, maps, or screenshots.
   - Do not generate artificial epoch-wise loss plots for non-iterative models like KNN.
   - All results reported in the PDF must be reproducible from the submitted public GitHub repository.
   - No sharing of code, weights, configs, derived datasets, or results across teams (cross-team similarity checks will be run on GitHub code, weights, configs, git history, and reports).
   - Do not commit secrets, passwords, or API keys to GitHub.
   - Every team member must be able to explain their code, design choices, and results in the viva.

---

## 6. Dependencies on Other Teammates (Persons 1, 2, and 3)

To ensure Person 4's KNN/MLP models are apples-to-apples comparable with the rest of the team's 5+ models, and to ensure seamless FastAPI/React integration, the following contracts and artifacts are needed:

### From Person 1 (Data Processing, EDA, Candidate Generation & Feature Engineering)
1. **Raw data loader & canonical path config:** Handling of `Student folder/Student folder/{train,student_test}` and a configurable `DATA_DIR` for evaluation-day trips.
2. **Candidate generation pipeline & function:** Given a trip's `.track`, `.nodes`, and `.arcs`, returns the candidate `arc_id` list per GPS observation (e.g., top-$K$ nearest arcs within radius $R$) and reports candidate miss rate.
3. **Label alignment specification:** Exact method mapping `.route` (traversed arc sequence) to observation-level ground-truth candidate labels (`y`) for training and evaluation.
4. **Feature engineering pipeline & schema:**
   - Exact ordered list of feature column names, data types, and missing-value rules.
   - Reusable Python module (not notebook-only code) that can transform an unseen raw trip into `(X_candidates, candidate_metadata)` at inference time inside FastAPI!
5. **Pre-generated train & `student_test` tabular datasets** (or deterministic script to generate them) with `trip_id` and `obs_idx` columns preserved for `GroupKFold` CV and observation-level Top-1 evaluation.

### From Person 2 (Baseline/Other Models & Shared Evaluation Harness)
1. **Shared Group-Aware CV split definition:** Exact `GroupKFold` folds (number of folds, `trip_id` fold assignments, random seed) so KNN and MLP CV Macro F1 scores are directly comparable in the Section 3.1 comparison table.
2. **Standardized evaluation function:** Shared script/formula computing:
   - Binary/candidate metrics: Accuracy, Precision, Recall, Macro F1-score, Confusion Matrix, Log Loss.
   - Map-matching metrics: Observation-level Top-1 Accuracy, Candidate Miss Rate, Route Continuity, Sequence-level metric (e.g., route F1 / arc-sequence accuracy / precision-recall against `.route`), and Inference Time.
3. **Result logging schema:** Agreed CSV/JSON format for Section 3.1 consolidated table and Section 3.3 comparison plots.

### From Person 3 (Model Selection, Bayesian Optimization & HMM/Viterbi Decoder)
1. **Final selected & Bayesian-tuned model artifacts:** Saved model file (e.g., `.joblib` / `.pkl` / `.pt`), fitted preprocessing/scaler object, feature column manifest, and inference wrapper exposing `predict_proba(X)`.
2. **HMM/Viterbi decoder module & configuration:**
   - Clean callable Python interface taking a trip's road graph (`.nodes`, `.arcs`), GPS points (`.track`), candidates, and classifier emission probabilities, and returning:
     - Predicted `arc_id` per GPS observation
     - Reconstructed chronological road segment sequence (`predicted_route_arcs`)
     - Decoder diagnostics (e.g., GPS gap handling count, fallback/disconnected transition count)
   - Saved HMM hyperparameter/config file (transition parameters, emission temperature/smoothing, gap thresholds).
3. **Server-side accuracy calculation contract:** Confirmation of the exact accuracy metric(s) Laptop 1 should compute against `.route` and return to the React UI (e.g., route-level F1 / overlap accuracy, observation-level Top-1 accuracy, route continuity, classifier-only vs. HMM comparison).

---

## 7. Proposed Implementation Plan (Phased)

### Phase 1: Team Contract Alignment & Environment Setup
- Confirm folder structure and GitHub repo workflow with Persons 1–3 (e.g., `src/models/knn.py`, `src/models/mlp.py`, `backend/`, `frontend/`, `artifacts/`, `reports/`).
- Create a workspace-local Python virtual environment (`.venv/`) and install `fastapi`, `uvicorn[standard]`, `pydantic`, plus any shared dependencies from `requirements.txt`.
- Define a frozen JSON API contract (`GET /api/health`, `GET /api/trips`, `POST /api/predict`) so frontend and backend development can proceed cleanly in parallel with Person 1–3's ML pipeline.

### Phase 2: KNN & MLP Model Implementation & Experimentation
- Wait for (or prototype against) Person 1's feature schema and Person 2's CV splits.
- Implement **KNN** (`src/models/train_knn.py`):
  - Pipeline with `StandardScaler` fit inside each CV fold + `KNeighborsClassifier` (testing appropriate `n_neighbors`, `weights='distance'` vs. `'uniform'`, distance metric).
  - Run group-aware CV on the 80 `train` trips, evaluate on the 10 `student_test` trips, log all metrics & timings, save confusion matrix and model results JSON/CSV.
- Implement **MLP** (`src/models/train_mlp.py`):
  - Pipeline with `StandardScaler` fit inside each CV fold + MLP architecture (PyTorch or sklearn `MLPClassifier` with early stopping, L2 regularization/dropout, class imbalance handling if candidate dataset is imbalanced).
  - Log actual epoch-wise training/validation loss curves, group-aware CV metrics, `student_test` metrics, timings, confusion matrix, and saved artifacts.
- Draft Person 4's Section 3.2 & 3.3 report inputs for KNN and MLP using strictly measured results.

### Phase 3: FastAPI Backend (Laptop 1)
- Build modular FastAPI application (`backend/`):
  - Configurable `EVAL_DATA_DIR` environment variable pointing to `student_test` during dev and the evaluator trip directory during final demo.
  - Startup loader (`lifespan`) loading frozen preprocessing, final tuned model, and HMM/Viterbi config once into memory (fast per-request inference).
  - Endpoints:
    - `GET /api/health` — confirms model, decoder, and dataset directory readiness.
    - `GET /api/trips` — dynamically scans `EVAL_DATA_DIR` and returns the 10 available trip IDs + basic non-ground-truth metadata (e.g., GPS point count, bounding box). Never reads or exposes `.route`.
    - `POST /api/predict` (payload: `{"trip_id": "00000000"}`) — loads `.track`, `.nodes`, `.arcs` for `trip_id`, runs full live inference (preprocessing → candidates → features → final classifier → HMM/Viterbi), then separately loads `.route` **server-side only** to compute evaluation metrics, logs request + timing + metrics clearly to CMD stdout, and returns predicted route coordinates/arc IDs, raw GPS points, local corridor road network, and metrics (no ground-truth route labels returned).
  - CORS middleware configured for Laptop 2's origin, bound to `0.0.0.0:8000`.
  - Unit & integration tests via `pytest` + `httpx`/`TestClient` across all 10 `student_test` trips.

### Phase 4: React + Vite Frontend (Laptop 2)
- Scaffold React + Vite app (`frontend/`) with Leaflet/React-Leaflet for interactive maps (note: verify whether Laptop 2 has internet access for OSM map tiles during evaluation, and provide a clean vector/canvas road-network fallback if offline!).
- Features:
  - Trip selector panel listing all 10 trips fetched live from `GET /api/trips`.
  - Interactive map showing raw GPS points, nearby road network segments, and the reconstructed predicted route with distinguishable colors, legend, zoom-to-fit, and layer toggles.
  - Metrics dashboard card showing server-returned evaluation accuracy, route continuity, classifier-only vs. HMM improvement (if returned), GPS point count, predicted arc count, and inference latency.
  - Clean connection status indicator showing Laptop 1 API URL and live health status.

### Phase 5: End-to-End Integration, Two-Laptop Verification & Screenshots
- Plug in Person 3's final tuned model artifacts and HMM/Viterbi decoder (replacing any temporary mock/prototype adapter).
- Verify all 10 `student_test` trips end-to-end across two separate laptops over the same Wi-Fi/hotspot (checking Windows Firewall port `8000`).
- Verify zero ground-truth leakage in browser DevTools Network tab (confirming `.route` IDs/coordinates are never in any API response).
- Capture authentic screenshots of the React UI and Laptop 1 CMD API logs for Deliverable 2 (Section 7) and verify README deployment instructions.

---

## 8. Questions & Missing Information

Before starting implementation, please clarify the following items with the team (Persons 1–3) or confirm your preferences:

1. **GitHub Repository & Folder Convention:** Has the team already created a shared GitHub repo and agreed on a folder layout (e.g., `src/`, `backend/`, `frontend/`, `artifacts/`, `reports/`), or should Person 4 propose one?
2. **Person 1 Feature Pipeline Readiness:** What is the ETA and exact format (Parquet/CSV + callable Python module) for Person 1's candidate generation and feature engineering pipeline? Are candidate datasets binary classification (`is_true_arc` per `(obs_idx, candidate_arc_id)`) or multi-class?
3. **Person 2 Evaluation & CV Standard:** How many folds (`k`) in `GroupKFold` is the team using, and what exact primary metric is used for model selection (e.g., Macro F1 vs. Observation-level Top-1 Accuracy) and for route-level evaluation against `.route`?
4. **KNN & MLP Framework Choice:** For MLP, do you prefer `scikit-learn`'s `MLPClassifier` (simpler serialization with `joblib`, fast on CPU) or `PyTorch` (more control over custom loss, batch normalization, dropout, and exact train+val epoch loss curves)?
5. **Evaluator-Day Folder Structure:** On evaluation day, will the organizers supply the 10 reserved trips in the same `<trip_id>/<trip_id>.{track,nodes,arcs,route}` layout inside a folder on Laptop 1? (Our FastAPI `EVAL_DATA_DIR` config will support this seamlessly, plus auto-detect nested folders if unzipped similarly to `Student folder/Student folder/`.)
6. **Offline vs. Online Map Tiles on Evaluation Day:** Will Laptop 2 have internet access during final evaluation to load OpenStreetMap/Carto basemap tiles in React-Leaflet, or should the UI also work cleanly without external tile servers (rendering road-network arcs + GPS + predicted route directly)?
7. **First-Phase Evaluation Before Lunch Break:** `Deliverables.pdf` mentions *"There will be first phase evaluation before lunch break."* What milestones does your team plan to have ready by that checkpoint (e.g., EDA + baseline models vs. API/UI skeleton)?
8. **UI Ground-Truth Display Restriction:** `Models and Training Guidelines.pdf` (Page 6) states: *"Ground-truth labels must remain server-side and must never be exposed to the prediction model or UI."* Therefore, the React UI must display **only** the raw GPS points, road network, **predicted route**, and **computed evaluation scores** — never the reference `.route` polyline. Confirm Person 1–3 are aligned that reference-vs-predicted Folium comparison maps belong strictly in offline EDA/report scripts on `train`/`student_test`, never in the live evaluation UI.
