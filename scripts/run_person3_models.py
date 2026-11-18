"""Evaluate fixed Person 3 baselines on trip-grouped folds and student_test."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.candidates_baseline import FEATURE_COLUMNS
from pathsnap.person3_models import build_models
from run_person1_models import METADATA_COLUMNS, positive_probability, score_predictions

RESULTS = ROOT / "experiments/person3/results"


def read_candidates(path):
    table = pd.read_csv(path, usecols=sorted(METADATA_COLUMNS) + FEATURE_COLUMNS,
                        dtype={"trip_id": str})
    if table[list(METADATA_COLUMNS)].isna().any().any():
        raise ValueError("Missing candidate metadata")
    if set(table.partition) != {"train", "student_test"}:
        raise ValueError("Expected train and student_test partitions")
    if not table.label.isin([0, 1]).all():
        raise ValueError("Expected binary labels")
    if not np.isfinite(table[FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Expected finite baseline features")
    return table


def training_folds(train):
    if not train.partition.eq("train").all():
        raise ValueError("Only training trips may enter cross-validation")
    splitter = GroupKFold(n_splits=min(5, train.trip_id.nunique()))
    return splitter.split(train, train.label, groups=train.trip_id)


def fit_and_score(model, name, fit, evaluation):
    features = fit[FEATURE_COLUMNS]
    start = perf_counter()
    model.fit(features, fit.label)
    fit_seconds = perf_counter() - start
    features = evaluation[FEATURE_COLUMNS]
    start = perf_counter()
    probabilities = positive_probability(model, features)
    inference_seconds = perf_counter() - start
    metrics = score_predictions(evaluation, probabilities, name)
    metrics.update(fit_seconds=fit_seconds, inference_seconds=inference_seconds,
                   fit_trips=fit.trip_id.nunique(), evaluation_trips=evaluation.trip_id.nunique(),
                   sklearn_version=sklearn.__version__,
                   configuration=json.dumps(model.named_steps["classifier"].get_params(), sort_keys=True))
    return metrics


def run_experiments(table, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    models = build_models()
    filenames = ["person3_cv_folds.csv", "person3_cv_summary.csv", "person3_student_test_results.csv"]
    for filename in filenames + [f"{name}.joblib" for name in models]:
        if (output_dir / filename).exists():
            raise FileExistsError(output_dir / filename)
    train = table.loc[table.partition == "train"]
    folds = []
    for fold, (fit_idx, valid_idx) in enumerate(training_folds(train), start=1):
        fit, valid = train.iloc[fit_idx], train.iloc[valid_idx]
        for name, model in models.items():
            metrics = fit_and_score(clone(model), name, fit, valid)
            metrics.update(fold=fold, validation_trip_ids=json.dumps(sorted(valid.trip_id.unique())))
            folds.append(metrics)
            print(f"Completed fold {fold}: {name}", flush=True)
    folds = pd.DataFrame(folds)
    folds.to_csv(output_dir / filenames[0], index=False)
    scores = ["observation_top1_accuracy", "observation_macro_f1", "candidate_binary_log_loss",
              "fit_seconds", "inference_seconds"]
    summary = folds.groupby("model")[scores].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.to_csv(output_dir / filenames[1])
    student = table.loc[table.partition == "student_test"]
    final = []
    for name, model in models.items():
        final.append(fit_and_score(model, name, train, student))
        joblib.dump(model, output_dir / f"{name}.joblib")
    pd.DataFrame(final).to_csv(output_dir / filenames[2], index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=ROOT / "data/processed/baseline_candidates.csv.gz")
    args = parser.parse_args()
    run_experiments(read_candidates(args.candidates), RESULTS)


if __name__ == "__main__":
    main()
