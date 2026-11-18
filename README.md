# PathSnap ML — Distributed Map-Matching & Route Reconstruction

**Course:** UE24CS352A: Machine Learning (PES University, Dept. of CSE, RR Campus)
Student project for GPS-to-road map matching, sequential route reconstruction, and two-laptop live evaluation.

> **Data Integrity Notice:** Keep the provided dataset (`Student folder/`) and guideline PDFs (`PathSnap_student_materials/`) strictly unmodified and outside Git. All scripts accept `--data-root` and read only `train` and `student_test`; evaluator data is never used during development. `.gitignore` excludes raw `.track`, `.nodes`, `.arcs`, `.route`, evaluator partition folders, and `.zip` archives.

---

## Project Architecture (Two-Laptop Deployment)

| Machine | Component | Responsibility |
| :--- | :--- | :--- |
| **Laptop 1** | **FastAPI Backend (`backend/`)** | Hosts the trained classifier, preprocessing pipeline, HMM/Viterbi decoder, road-network data, and **server-side** evaluation logic. Logs every incoming inference request to CMD/stdout. |
| **Laptop 2** | **React + Vite Frontend (`frontend/`)** | Displays the 10 evaluation trips, sends only the selected `trip_id` to Laptop 1's API, and renders the predicted route and returned evaluation metrics. Never receives ground-truth `.route` labels. |

---

## Setup Instructions

### Prerequisites
- **Python:** 3.13+ (verified with `Python 3.13.7`)
- **Node.js:** 20+ / 22+ LTS (verified with `Node.js v22.19.0`)
- **npm:** 10+ (verified with `npm 10.9.3`)

### 1. Python Virtual Environment (PowerShell on Windows)

```powershell
python -m venv backend\.venv
.\backend\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Frontend Setup (React + Vite, JavaScript)

```powershell
cd frontend
npm install
cd ..
```

---

## Person 1: Data Processing, Reference Labels & Baseline Models

| Step | Command | Output |
| :--- | :--- | :--- |
| Inventory | `python scripts/inspect_dataset.py --data-root "PATH"` | `results/dataset_inventory.csv` |
| Exploration | `python scripts/explore_dataset.py --data-root "PATH"` | `results/trajectory_stats.csv`, `network_connectivity.csv`, `label_quality_per_trip.csv`, `results/figures/*.png` |
| Reference labels | `python scripts/build_labels.py --data-root "PATH"` | `results/labels/observation_labels.csv.gz`, `label_quality_per_trip.csv` |
| Baseline candidates | `python scripts/build_baseline_candidates.py --data-root "PATH"` | `data/processed/baseline_candidates.csv.gz` (gitignored), `results/candidate_generation_summary.csv` |
| Models (LR & GNB) | `python scripts/run_person1_models.py --candidates data/processed/baseline_candidates.csv.gz` | `experiments/person1/results/` |
| Plots and facts sheet | `python scripts/summarize_person1.py` | `results/figures/person1_*.png`, `results/person1_report_facts.md` |
| Unit tests | `python -m pytest tests -q` | Verifies geometry, labels, network stats, and `trip_id::arc_id` metrics |

Findings, assumptions, and caveats are documented in `results/data_quality_notes.md`.

### Shared Label Table Contract
`.route` is a list of arcs in travel order (not one entry per GPS row), and arc IDs are row indices inside one trip's road network. `scripts/build_labels.py` aligns each GPS row to a route arc using monotone dynamic programming (`src/pathsnap/labels.py`) and writes:

`partition, trip_id, observation_id, route_index, arc_id, dist_to_arc_m, label_valid`

- `trip_id` is a zero-padded string (`00000090`); always read CSVs with `dtype={"trip_id": str}`.
- `observation_id` is the 0-based row index in the trip's `.track` file.
- `label_valid` is `False` when the row is more than 50 m from its aligned arc (dropped, never labelled `0`).
- `pathsnap.labels.label_candidates(candidates, labels)` converts a candidate table into `0/1` labels, and `candidate_miss_rate(...)` reports how often the reference arc is missing.

### Baseline Candidates (Stand-In) & Shared Candidate-Feature CSV Contract
`src/pathsnap/candidates_baseline.py` generates stand-in candidates (arcs within 50 m, nearest 20) and 9 GPS/network-only features without touching `.route`. Person 2's candidate generation and Person 3's features replace it once finalized.

**Shared Candidate-Feature CSV Contract:**
`partition`, `trip_id`, `observation_id`, `candidate_arc_id`, `label`, followed by numeric feature columns.
- `partition` is `train` or `student_test` only.
- Evaluation uses 5-fold `GroupKFold` by `trip_id` on `train`, then evaluates on `student_test`.
- Reports observation-level Top-1 Accuracy and Macro F1 (classes qualified as `trip_id::arc_id`, since arc IDs are local to each trip) and candidate-level binary Log Loss.

---

## Person 4: FastAPI Backend & React + Vite Frontend

### Run FastAPI Backend (Laptop 1)

```powershell
# Local development (with auto-reload)
.\backend\.venv\Scripts\uvicorn.exe app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8000

# Two-laptop deployment (exposed on LAN)
.\backend\.venv\Scripts\uvicorn.exe app.main:app --app-dir backend --host 0.0.0.0 --port 8000
```

Interactive Swagger docs: `http://127.0.0.1:8000/docs`

### Run React + Vite Frontend (Laptop 2)

```powershell
cd frontend
npm run dev
# Optional: expose on LAN
npm run dev -- --host
```

### Verify Frontend Lint & Production Build

```powershell
cd frontend
npm run lint
npm run build
```

### Run Full Automated Test Suites (Person 1 + Person 4)

```powershell
# Person 1 unit tests
.\backend\.venv\Scripts\python.exe -m pytest tests -v

# Person 4 FastAPI backend tests
.\backend\.venv\Scripts\python.exe -m pytest backend\tests -v
```

---

## Repository Layout

- `src/pathsnap/`: shared core package (`data_io.py`, `preprocess.py`, `geo.py`, `labels.py`, `network.py`, `candidates_baseline.py`, `person1_models.py`, `person3_models.py`).
- `scripts/`: inventory, exploration, label building, baseline candidates, Person 1 experiments, and Person 3 tree-ensemble baselines (`run_person3_models.py`).
- `backend/`: Laptop 1 FastAPI backend (`app/{main,core,routes,schemas,services}`, `models/`, `tests/`).
- `frontend/`: Laptop 2 React + Vite (JavaScript) evaluator dashboard.
- `experiments/`: model experimentation directories (`person1/`, `knn/`, `mlp/`).
- `results/`: generated tables, figures, labels, and notes (no raw dataset files).
- `artifacts/`: `artifacts/final/` for tracked final model (`.joblib`), `artifacts/metrics/` for comparison outputs.
- `report/report_outline.md` & `reports/person4/`: shared technical report outline and Person 4 audit/implementation reports.
- `tests/`: shared core unit tests (`test_labels_and_metrics.py`, `test_person3_models.py`).
