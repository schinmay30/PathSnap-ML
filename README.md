# PathSnap ML

Student project for GPS-to-road map matching and route reconstruction.

## Data setup

Keep the provided dataset outside Git unless the course organisers explicitly permit publishing it. All scripts take the extracted dataset folder with `--data-root`. They read only `train` and `student_test`; evaluator data is never used. `.gitignore` excludes evaluator partition folders and raw `.track`, `.nodes`, `.arcs`, and `.route` files.

```powershell
pip install -r requirements.txt
```

## Person 1: data processing, labels and assigned models

| Step | Command | Output |
|---|---|---|
| Inventory | `python scripts/inspect_dataset.py --data-root "PATH"` | `results/dataset_inventory.csv` |
| Exploration | `python scripts/explore_dataset.py --data-root "PATH"` | `results/trajectory_stats.csv`, `network_connectivity.csv`, `label_quality_per_trip.csv`, `results/figures/*.png` |
| Reference labels | `python scripts/build_labels.py --data-root "PATH"` | `results/labels/observation_labels.csv.gz`, `label_quality_per_trip.csv` |
| Baseline candidates | `python scripts/build_baseline_candidates.py --data-root "PATH"` | `data/processed/baseline_candidates.csv.gz` (not tracked), `results/candidate_generation_summary.csv` |
| Models | `python scripts/run_person1_models.py --candidates data/processed/baseline_candidates.csv.gz` | `experiments/person1/results/` |
| Plots and facts sheet | `python scripts/summarize_person1.py` | `results/figures/person1_*.png`, `results/person1_report_facts.md` |
| Tests | `python -m pytest tests -q` | |

`inspect_dataset.py` opens the input files for reading and writes its inventory CSV only to the requested output path (default: `results/dataset_inventory.csv`). It does not edit, move, or delete original dataset files.

Findings, assumptions and caveats are in `results/data_quality_notes.md`.

### Shared label table (proposed contract)
`.route` is a list of arcs in travel order, not one entry per GPS row, and arc IDs are row indices inside one trip's road network. `scripts/build_labels.py` aligns each GPS row to a route arc (monotone dynamic programming, see `src/pathsnap/labels.py`) and writes one row per observation:

`partition, trip_id, observation_id, route_index, arc_id, dist_to_arc_m, label_valid`

- `trip_id` is a zero-padded string (`00000090`); read CSVs with `dtype={"trip_id": str}`.
- `observation_id` is the 0-based row index in the trip's `.track` file.
- `label_valid` is False when the row is more than 50 m from its aligned arc. Such rows are dropped, not labelled 0.
- `pathsnap.labels.label_candidates(candidates, labels)` turns a candidate table into 0/1 labels, and `candidate_miss_rate(...)` reports how often the reference arc is missing from the candidates.

### Baseline candidates (stand-in)
`src/pathsnap/candidates_baseline.py` generates candidates (all arcs within 50 m, nearest 20) and nine GPS/network-only features so Person 1's models can run now. The reference route is never used for features. Sujeet's candidate generation and Monish's features replace it once ready; the model script only needs the contract below.

### Candidate-feature CSV contract
`partition`, `trip_id`, `observation_id`, `candidate_arc_id`, `label`, followed by numeric feature columns. `partition` is `train` or `student_test` only. The experiment script uses GroupKFold by trip on `train`, then evaluates on `student_test`. It reports observation-level Top-1 Accuracy and Macro F1 (classes are `trip_id::arc_id`, because arc IDs are not shared across trips) and candidate-level binary Log Loss. Observations without exactly one positive label are counted separately.

## Repository layout

- `src/pathsnap/data_io.py`: raw file readers.
- `src/pathsnap/preprocess.py`: trip loader and validation.
- `src/pathsnap/geo.py`: local metric projection, haversine, point-to-segment distance.
- `src/pathsnap/labels.py`: GPS-to-route alignment, candidate labelling, candidate miss rate.
- `src/pathsnap/network.py`: road-network connectivity statistics.
- `src/pathsnap/candidates_baseline.py`: stand-in candidate generation and features.
- `src/pathsnap/person1_models.py`: Logistic Regression and Gaussian Naive Bayes pipelines.
- `scripts/`: inventory, exploration, label building, Person 1 experiments.
- `tests/`: unit tests.
- `results/`: generated tables, figures and notes. Raw data is not stored here.
- `artifacts/final/`: place for the final selected model (the only `.joblib` location Git tracks).
