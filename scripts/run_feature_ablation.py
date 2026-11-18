"""Paired train-only ablation of the two fixed Person 3 tree baselines."""

import argparse
import json
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn

from build_extended_candidates import EXTENDED
from run_person3_models import (FEATURE_COLUMNS, ROOT, build_models, positive_probability,
                                read_candidates, score_predictions, training_folds)
from pathsnap.candidate_features import ADDED_FEATURES


def run(baseline_path, extended_path, output):
    baseline = read_candidates(baseline_path)
    extended = pd.read_csv(extended_path, dtype={"trip_id": str})
    pd.testing.assert_frame_equal(baseline, extended[baseline.columns], check_exact=True)
    del baseline
    if not np.isfinite(extended[ADDED_FEATURES].to_numpy()).all():
        raise ValueError("Non-finite extended features")
    train = extended.loc[extended.partition == "train"]
    del extended
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for fold, (fit_idx, valid_idx) in enumerate(training_folds(train), 1):
        fit, valid = train.iloc[fit_idx], train.iloc[valid_idx]
        for variant, features in [("baseline", FEATURE_COLUMNS),
                                  ("extended", FEATURE_COLUMNS + ADDED_FEATURES)]:
            for name, model in build_models().items():
                x_fit, x_valid = fit[features], valid[features]
                start = perf_counter()
                model.fit(x_fit, fit.label)
                fit_seconds = perf_counter() - start
                start = perf_counter()
                probabilities = positive_probability(model, x_valid)
                inference_seconds = perf_counter() - start
                row = score_predictions(valid, probabilities, name)
                row.update(fold=fold, variant=variant, fit_seconds=fit_seconds,
                           inference_seconds=inference_seconds, validation_rows=len(valid),
                           validation_trips=json.dumps(sorted(valid.trip_id.unique())),
                           sklearn_version=sklearn.__version__)
                rows.append(row)
                print(f"Completed fold {fold}: {variant} {name}", flush=True)
        pd.DataFrame(rows).to_csv(output / "folds.csv", index=False)
    metrics = ["observation_top1_accuracy", "observation_macro_f1", "candidate_binary_log_loss",
               "fit_seconds", "inference_seconds"]
    pd.DataFrame(rows).groupby(["model", "variant"])[metrics].mean().to_csv(output / "means.csv")


if __name__ == "__main__":
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "data/processed/baseline_candidates.csv.gz")
    parser.add_argument("--extended", type=Path, default=EXTENDED)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/person3/feature_ablation")
    args = parser.parse_args()
    run(args.baseline, args.extended, args.output)
