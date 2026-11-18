"""Train Person 1's classifiers on a shared candidate-feature CSV.

Expected metadata columns: partition, trip_id, observation_id,
candidate_arc_id, label. Every other column must be a numeric feature.
`label` is 1 for the reference arc candidate and 0 otherwise. The script
performs group-aware cross-validation by trip on the training partition,
then evaluates the final fit on student_test. It never uses evaluator data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pathsnap.person1_models import build_models  # noqa: E402


METADATA_COLUMNS = {
    "partition", "trip_id", "observation_id", "candidate_arc_id", "label"
}
PARTITIONS = {"train", "student_test"}


def read_candidate_table(path: Path) -> tuple[pd.DataFrame, list[str]]:
    table = pd.read_csv(path)
    missing = METADATA_COLUMNS.difference(table.columns)
    if missing:
        raise ValueError(f"Candidate CSV is missing columns: {sorted(missing)}")
    partitions = set(table["partition"].dropna().astype(str).unique())
    unexpected = partitions.difference(PARTITIONS)
    if unexpected:
        raise ValueError(
            "Only train and student_test rows are allowed in this experiment; "
            f"unexpected partition(s): {sorted(unexpected)}"
        )
    table = table[table["partition"].isin(PARTITIONS)].copy()
    if table.empty:
        raise ValueError("Candidate-feature table has no development rows")
    table["label"] = pd.to_numeric(table["label"], errors="raise").astype(int)
    if not set(table["label"].unique()).issubset({0, 1}):
        raise ValueError("`label` must contain only 0 and 1")
    if table[list(METADATA_COLUMNS)].isna().any().any():
        raise ValueError("Candidate metadata contains missing values")
    feature_columns = [column for column in table.columns if column not in METADATA_COLUMNS]
    if not feature_columns:
        raise ValueError("No numeric feature columns found")
    table[feature_columns] = table[feature_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(table[feature_columns].to_numpy(dtype=float)).all():
        raise ValueError("Feature columns contain missing or infinite values")
    if set(table["partition"].unique()) != PARTITIONS:
        raise ValueError("Input must include both train and student_test partitions")
    return table, feature_columns


def observation_scores(
    frame: pd.DataFrame, probabilities: np.ndarray
) -> tuple[float, float, int, int, int]:
    """Compute Top-1 and Macro F1 for groups with exactly one positive candidate.

    Groups with zero positive candidates are not silently counted as model
    errors: they indicate a candidate-label coverage issue and are reported.
    Groups with multiple positive candidates are also reported separately.
    """
    scored = frame[["trip_id", "observation_id", "candidate_arc_id", "label"]].copy()
    scored["positive_probability"] = probabilities
    scored["observation_key"] = (
        scored["trip_id"].astype(str) + "::" + scored["observation_id"].astype(str)
    )
    y_true: list[str] = []
    y_pred: list[str] = []
    no_positive = 0
    multiple_positive = 0
    for _, group in scored.groupby("observation_key", sort=False):
        positives = group[group["label"] == 1]
        if len(positives) == 0:
            no_positive += 1
            continue
        if len(positives) != 1:
            multiple_positive += 1
            continue
        chosen = group.loc[group["positive_probability"].idxmax()]
        y_true.append(str(positives.iloc[0]["candidate_arc_id"]))
        y_pred.append(str(chosen["candidate_arc_id"]))
    if not y_true:
        return float("nan"), float("nan"), 0, no_positive, multiple_positive
    return (
        float(accuracy_score(y_true, y_pred)),
        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        len(y_true),
        no_positive,
        multiple_positive,
    )


def score_predictions(
    frame: pd.DataFrame, probabilities: np.ndarray, model_name: str
) -> dict[str, object]:
    y_candidate = frame["label"].to_numpy(dtype=int)
    obs_accuracy, obs_macro_f1, eligible, no_positive, multi_positive = (
        observation_scores(frame, probabilities)
    )
    return {
        "model": model_name,
        "candidate_binary_log_loss": float(
            log_loss(y_candidate, probabilities, labels=[0, 1])
        ),
        "observation_top1_accuracy": obs_accuracy,
        "observation_macro_f1": obs_macro_f1,
        "eligible_observations": eligible,
        "observations_without_positive_candidate_label": no_positive,
        "observations_with_multiple_positive_labels": multi_positive,
    }


def positive_probability(model, features: pd.DataFrame) -> np.ndarray:
    classifier = model.named_steps["classifier"]
    classes = list(classifier.classes_)
    if 1 not in classes:
        return np.zeros(len(features), dtype=float)
    return model.predict_proba(features)[:, classes.index(1)]


def run_experiments(table: pd.DataFrame, feature_columns: list[str], output_dir: Path) -> None:
    train = table[table["partition"] == "train"].copy()
    student_test = table[table["partition"] == "student_test"].copy()
    if train["trip_id"].nunique() < 2:
        raise ValueError("At least two training trips are needed for grouped CV")
    n_splits = min(5, train["trip_id"].nunique())
    splitter = GroupKFold(n_splits=n_splits)
    X_train = train[feature_columns]
    y_train = train["label"].to_numpy(dtype=int)
    groups = train["trip_id"].astype(str).to_numpy()

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    for model_name, model in build_models().items():
        for fold_index, (fit_index, validation_index) in enumerate(
            splitter.split(X_train, y_train, groups), start=1
        ):
            fold_model = build_models()[model_name]
            fold_model.fit(X_train.iloc[fit_index], y_train[fit_index])
            valid = train.iloc[validation_index]
            valid_probabilities = positive_probability(
                fold_model, valid[feature_columns]
            )
            metrics = score_predictions(valid, valid_probabilities, model_name)
            metrics["fold"] = fold_index
            metrics["validation_trip_count"] = valid["trip_id"].nunique()
            fold_rows.append(metrics)

        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        test_probabilities = positive_probability(model, student_test[feature_columns])
        inference_seconds = time.perf_counter() - start
        final_metrics = score_predictions(
            student_test, test_probabilities, model_name
        )
        final_metrics.update(
            {
                "student_test_trip_count": student_test["trip_id"].nunique(),
                "fit_seconds_on_all_training_candidates": fit_seconds,
                "student_test_inference_seconds": inference_seconds,
                "configuration": json.dumps(
                    model.named_steps["classifier"].get_params(), default=str
                ),
            }
        )
        final_rows.append(final_metrics)
        joblib.dump(model, output_dir / f"{model_name}.joblib")

    pd.DataFrame(fold_rows).to_csv(output_dir / "person1_cv_folds.csv", index=False)
    pd.DataFrame(final_rows).to_csv(
        output_dir / "person1_student_test_results.csv", index=False
    )
    print(f"Saved Person 1 model results and model artifacts in {output_dir.resolve()}")
    print(pd.DataFrame(final_rows)[
        ["model", "observation_top1_accuracy", "observation_macro_f1", "candidate_binary_log_loss"]
    ].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "experiments" / "person1" / "results"
    )
    args = parser.parse_args()
    table, feature_columns = read_candidate_table(args.candidates)
    run_experiments(table, feature_columns, args.output_dir)


if __name__ == "__main__":
    main()
