# PathSnap ML

Student project for GPS-to-road map matching and route reconstruction.

## Data setup

Keep the provided dataset outside Git unless the course organizers explicitly permit publishing it. The data-processing scripts accept the extracted dataset folder as a command-line path. They inspect only `train` and `student_test`; evaluator data is intentionally excluded.

## Person 1: data processing and assigned models

Run:

```powershell
python scripts/inspect_dataset.py --data-root "PATH_TO_EXTRACTED_STUDENT_FOLDER" --output results/dataset_inventory.csv
```

`--data-root` should point to a folder containing `train` and `student_test`. If extraction created an extra nested `Student folder` directory, pass the inner folder. The script can also locate that nested layout automatically.

The inventory records file presence and size, row counts, coordinate bounds, timestamp order and gaps, and basic node/arc/route consistency. It does not assume one GPS observation equals one route entry.

Person 1's raw readers are in `src/pathsnap/data_io.py`; `src/pathsnap/preprocess.py` loads and validates a trip while preserving observation order and keeping the reference route separate. Once Person 2 and Person 3 provide the shared candidate-feature CSV, run Person 1's Logistic Regression and Gaussian Naive Bayes experiments with:

```powershell
pip install -r requirements.txt
python scripts/run_person1_models.py --candidates "PATH_TO_candidate_features.csv"
```

The candidate-feature CSV contract is: `partition`, `trip_id`, `observation_id`, `candidate_arc_id`, `label`, followed by numeric feature columns. `partition` must contain `train` and `student_test` only; `label` is 1 for the mapped reference candidate and 0 for other candidates. The label must come from the team's verified route-to-observation alignment; do not assign route rows to GPS rows by position. The script uses GroupKFold by trip on `train`, then evaluates `student_test`. It records observation-level Top-1 Accuracy and Macro F1 plus candidate-level binary Log Loss, and reports observations without exactly one positive candidate label separately.

## Repository layout

- `src/pathsnap/data_io.py`: readers for the raw whitespace-separated files.
- `src/pathsnap/preprocess.py`: trip loader and validation checks.
- `src/pathsnap/person1_models.py`: Logistic Regression and Gaussian Naive Bayes pipelines.
- `scripts/inspect_dataset.py`: dataset inventory and data-quality report.
- `scripts/run_person1_models.py`: Person 1 model experiments on the shared candidate table.
- `results/`: generated summaries; raw data is not stored here.

