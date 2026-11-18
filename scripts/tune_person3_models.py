"""Train-only nested GroupKFold tuning. Four sequential trials per study:
fixed baseline, one seeded random proposal, then two TPE proposals.
Two outer folds assess selection; a separate two-fold study selects final
parameters on all training trips. Its best objective is a selection score.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import optuna
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold

from run_person3_models import (ROOT, METADATA_COLUMNS, FEATURE_COLUMNS, build_models,
                                positive_probability, score_predictions)
from pathsnap.candidate_features import ADDED_FEATURES

FEATURES = FEATURE_COLUMNS + ADDED_FEATURES
SEED = 42
BUDGET = 4
SPACE = {
    "extra_trees": {"n_estimators": [24, 40, 64], "max_depth": [12, 18, 24],
                    "min_samples_leaf": [2, 5, 12], "max_features": ["sqrt", 0.7]},
    "hist_gradient_boosting": {"max_iter": [60, 100, 150], "max_leaf_nodes": [15, 31, 63],
                               "min_samples_leaf": [10, 20, 50], "l2_regularization": [0., 1., 5.]},
}


def load_train(path):
    chunks = pd.read_csv(path, usecols=sorted(METADATA_COLUMNS) + FEATURES,
                         dtype={"trip_id": str}, chunksize=100_000)
    train = pd.concat([c.loc[c.partition.eq("train")] for c in chunks], ignore_index=True)
    if train.trip_id.nunique() < 4 or train[list(METADATA_COLUMNS)].isna().any().any():
        raise ValueError("Nested evaluation requires complete metadata and at least four trips")
    if not train.label.isin([0, 1]).all() or not np.isfinite(train[FEATURES]).all().all():
        raise ValueError("Invalid training labels or features")
    return train


def splits(frame):
    if not frame.partition.eq("train").all():
        raise ValueError("Only training trips may enter a split")
    return GroupKFold(n_splits=2).split(frame, groups=frame.trip_id)


def evaluate(name, params, fit, valid):
    model = build_models(SEED)[name]
    model.set_params(**{f"classifier__{k}": v for k, v in params.items()})
    # All 19 inputs are finite numerical features. Trees need no scaling.
    x_fit, x_valid = fit[FEATURES], valid[FEATURES]
    start = perf_counter()
    model.fit(x_fit, fit.label)
    fit_seconds = perf_counter() - start
    start = perf_counter()
    probabilities = positive_probability(model, x_valid)
    inference_seconds = perf_counter() - start
    metrics = score_predictions(valid, probabilities, name)
    metrics.update(fit_seconds=fit_seconds, inference_seconds=inference_seconds)
    return metrics


def record(output, filename, row):
    path = output / filename
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def search(name, frame, stage, output):
    folds = list(splits(frame))
    fixed = build_models(SEED)[name].named_steps["classifier"].get_params()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(
        seed=SEED, n_startup_trials=2))
    study.enqueue_trial({k: fixed[k] for k in SPACE[name]})

    def objective(trial):
        start = perf_counter()
        params = {key: trial.suggest_categorical(key, values) for key, values in SPACE[name].items()}
        scores = []
        for fold, (fit, valid) in enumerate(folds, 1):
            validation = frame.iloc[valid]
            metrics = evaluate(name, params, frame.iloc[fit], validation)
            scores.append(metrics["observation_top1_accuracy"])
            record(output, "inner_folds.csv", dict(stage=stage, trial=trial.number, fold=fold,
                   validation_trips=json.dumps(sorted(validation.trip_id.unique())), **metrics))
        value = float(np.mean(scores))
        record(output, "trials.csv", dict(stage=stage, model=name, trial=trial.number,
               params=json.dumps(params), mean_inner_top1=value, seconds=perf_counter()-start))
        return value

    study.optimize(objective, n_trials=BUDGET, n_jobs=1)
    return dict(parameters=study.best_params, best_trial=study.best_trial.number,
                mean_inner_top1=study.best_value)


def run(train, output):
    output.mkdir(parents=True, exist_ok=False)
    config = dict(seed=SEED, trials_per_study=BUDGET, startup_trials=2, outer_folds=2,
                  inner_folds=2, sampler="TPESampler", features=FEATURES, search_space=SPACE,
                  training_rows=len(train), training_trips=train.trip_id.nunique(),
                  sklearn_version=sklearn.__version__, optuna_version=optuna.__version__)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    best = {}
    outer_rows = []
    for fold, (fit, valid) in enumerate(splits(train), 1):
        development, validation = train.iloc[fit], train.iloc[valid]
        stage = f"outer_{fold}"
        best[stage] = {}
        for name in SPACE:
            winner = search(name, development, stage, output)
            best[stage][name] = winner
            for variant, params in [("fixed", {}), ("tuned", winner["parameters"])]:
                metrics = evaluate(name, params, development, validation)
                row = dict(fold=fold, variant=variant, validation_rows=len(valid),
                           validation_trips=json.dumps(sorted(validation.trip_id.unique())), **metrics)
                outer_rows.append(row)
                record(output, "outer_validation.csv", row)
    cols = ["observation_top1_accuracy", "observation_macro_f1", "candidate_binary_log_loss",
            "fit_seconds", "inference_seconds"]
    pd.DataFrame(outer_rows).groupby(["model", "variant"])[cols].mean().to_csv(output / "outer_means.csv")
    best["all_train"] = {name: search(name, train, "all_train", output) for name in SPACE}
    (output / "best_parameters.json").write_text(json.dumps(best, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=ROOT / "data/processed/person3_extended_candidates.csv.gz")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/person3/tuning")
    args = parser.parse_args()
    run(load_train(args.candidates), args.output)
