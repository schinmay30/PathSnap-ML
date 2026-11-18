"""Bayesian Hyperparameter Optimization for Person 2 Tree Models (Decision Tree & Random Forest).

Uses Gaussian Process surrogate regression with Expected Improvement (EI)
over 5-fold GroupKFold cross-validation on the 80 training trips (zero test leakage).
After optimization, retrains the best parameter configuration on all training trips,
evaluates out-of-sample on the 10 student-test trips, and exports trial logs and comparisons.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pathsnap.bayesian_opt import (  # noqa: E402
    GaussianProcessBayesianOptimizer,
    Hyperparameter,
)
from pathsnap.person2_models import build_models, build_pipeline  # noqa: E402
from run_person2_models import (  # noqa: E402
    positive_probability,
    read_candidate_table,
    score_predictions,
)


def evaluate_params_cv(
    model_type: str,
    params: dict[str, Any],
    train_df: pd.DataFrame,
    feature_columns: list[str],
    splitter: GroupKFold,
    groups: np.ndarray,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Run 5-fold GroupKFold CV on training trips and return (macro_f1, top1_acc, log_loss)."""
    X_train = train_df[feature_columns]
    y_train = train_df["label"].to_numpy(dtype=int)

    f1_list, acc_list, loss_list = [], [], []

    for fit_idx, val_idx in splitter.split(X_train, y_train, groups):
        pipe = build_pipeline(model_type, params, random_state=random_state)
        # Preprocessing (StandardScaler) is fit strictly on training split
        pipe.fit(X_train.iloc[fit_idx], y_train[fit_idx])

        valid = train_df.iloc[val_idx]
        probs = positive_probability(pipe, valid[feature_columns])
        metrics = score_predictions(valid, probs, model_type)

        f1_list.append(metrics["observation_macro_f1"])
        acc_list.append(metrics["observation_top1_accuracy"])
        loss_list.append(metrics["candidate_binary_log_loss"])

    return float(np.mean(f1_list)), float(np.mean(acc_list)), float(np.mean(loss_list))


def run_bayesian_optimization_for_model(
    model_type: str,
    param_space: list[Hyperparameter],
    n_trials: int,
    train_df: pd.DataFrame,
    feature_columns: list[str],
    splitter: GroupKFold,
    groups: np.ndarray,
    output_dir: Path,
    random_state: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame]:
    print(f"\n==================================================================")
    print(f"BAYESIAN OPTIMIZATION: {model_type.upper()} ({n_trials} trials)")
    print(f"==================================================================")
    opt = GaussianProcessBayesianOptimizer(param_space, random_state=random_state)

    for trial_idx in range(1, n_trials + 1):
        t0 = time.perf_counter()
        params, gp_mean, gp_std, ei = opt.suggest_next_params(n_candidates=1500)

        # Evaluate on training folds
        f1_score, top1_acc, b_loss = evaluate_params_cv(
            model_type=model_type,
            params=params,
            train_df=train_df,
            feature_columns=feature_columns,
            splitter=splitter,
            groups=groups,
            random_state=random_state,
        )
        duration = time.perf_counter() - t0

        opt.register_trial(
            trial_idx=trial_idx,
            params=params,
            score=f1_score,
            top1_accuracy=top1_acc,
            log_loss=b_loss,
            gp_mean=gp_mean,
            gp_std=gp_std,
            ei=ei,
            duration=duration,
        )

        print(
            f"  Trial {trial_idx:2d}/{n_trials:2d} | CV Macro F1: {f1_score:.4f} | "
            f"Top-1: {top1_acc*100:.2f}% | LogLoss: {b_loss:.4f} | "
            f"GP(mu={gp_mean:.3f}, sigma={gp_std:.3f}, EI={ei:.4f}) | "
            f"params={params} | {duration:.2f}s"
        )

    best_trial = opt.get_best_trial()
    print(f"\n>>> Best {model_type} Configuration (Trial #{best_trial.trial}):")
    print(f"    CV Macro F1: {best_trial.score:.4f} | CV Top-1: {best_trial.top1_accuracy*100:.2f}%")
    print(f"    Parameters: {best_trial.params}")

    df_trials = opt.export_trials_dataframe()
    df_trials.to_csv(output_dir / f"trials_{model_type}.csv", index=False)

    best_json_path = output_dir / f"best_params_{model_type}.json"
    with best_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_type": model_type,
                "best_trial_number": best_trial.trial,
                "cv_macro_f1": best_trial.score,
                "cv_top1_accuracy": best_trial.top1_accuracy,
                "cv_log_loss": best_trial.log_loss,
                "parameters": best_trial.params,
            },
            f,
            indent=2,
        )

    return best_trial.params, df_trials


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=PROJECT_ROOT / "results" / "candidates" / "candidate_features.csv.gz",
        help="Shared candidate-feature CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "person2" / "bayesian_opt",
        help="Directory to save Bayesian optimization trials and tuned models",
    )
    parser.add_argument(
        "--dt-trials",
        type=int,
        default=12,
        help="Number of Bayesian optimization trials for Decision Tree",
    )
    parser.add_argument(
        "--rf-trials",
        type=int,
        default=8,
        help="Number of Bayesian optimization trials for Random Forest",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    table, feature_columns = read_candidate_table(args.candidates)
    train_df = table[table["partition"] == "train"].copy()
    test_df = table[table["partition"] == "student_test"].copy()

    groups = train_df["trip_id"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=min(5, train_df["trip_id"].nunique()))

    # 1. Search spaces
    dt_space = [
        Hyperparameter("max_depth", "int", low=5, high=18),
        Hyperparameter("min_samples_leaf", "int", low=2, high=25),
        Hyperparameter("min_samples_split", "int", low=2, high=25),
        Hyperparameter("criterion", "choice", choices=["gini", "entropy", "log_loss"]),
    ]

    rf_space = [
        Hyperparameter("n_estimators", "int", low=40, high=130),
        Hyperparameter("max_depth", "int", low=8, high=18),
        Hyperparameter("min_samples_leaf", "int", low=2, high=15),
        Hyperparameter("max_features", "choice", choices=["sqrt", "log2"]),
    ]

    # 2. Run Bayesian Optimization
    best_dt_params, _ = run_bayesian_optimization_for_model(
        model_type="decision_tree",
        param_space=dt_space,
        n_trials=args.dt_trials,
        train_df=train_df,
        feature_columns=feature_columns,
        splitter=splitter,
        groups=groups,
        output_dir=output_dir,
    )

    best_rf_params, _ = run_bayesian_optimization_for_model(
        model_type="random_forest",
        param_space=rf_space,
        n_trials=args.rf_trials,
        train_df=train_df,
        feature_columns=feature_columns,
        splitter=splitter,
        groups=groups,
        output_dir=output_dir,
    )

    # 3. Retrain tuned models on full train set and evaluate on student_test
    print("\n==================================================================")
    print("OUT-OF-SAMPLE EVALUATION: TUNED VS UNTUNED (STUDENT_TEST)")
    print("==================================================================")

    comparison_records = []
    baseline_models = build_models(random_state=42)

    tuned_configs = [
        ("decision_tree", best_dt_params),
        ("random_forest", best_rf_params),
    ]

    for model_name, tuned_params in tuned_configs:
        # Untuned Baseline
        base_pipe = baseline_models[model_name]
        t0 = time.perf_counter()
        base_pipe.fit(train_df[feature_columns], train_df["label"])
        base_fit_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        base_test_probs = positive_probability(base_pipe, test_df[feature_columns])
        base_inf_s = time.perf_counter() - t0
        base_metrics = score_predictions(test_df, base_test_probs, model_name)

        comparison_records.append(
            {
                "model": model_name,
                "variant": "untuned_baseline",
                "student_test_top1_accuracy": base_metrics["observation_top1_accuracy"],
                "student_test_macro_f1": base_metrics["observation_macro_f1"],
                "student_test_log_loss": base_metrics["candidate_binary_log_loss"],
                "fit_time_seconds": base_fit_s,
                "inference_time_seconds": base_inf_s,
                "parameters": json.dumps(base_pipe.named_steps["classifier"].get_params(), default=str),
            }
        )

        # Tuned Pipeline
        tuned_pipe = build_pipeline(model_name, tuned_params, random_state=42)
        t0 = time.perf_counter()
        tuned_pipe.fit(train_df[feature_columns], train_df["label"])
        tuned_fit_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        tuned_test_probs = positive_probability(tuned_pipe, test_df[feature_columns])
        tuned_inf_s = time.perf_counter() - t0
        tuned_metrics = score_predictions(test_df, tuned_test_probs, model_name)

        comparison_records.append(
            {
                "model": model_name,
                "variant": "bayesian_tuned",
                "student_test_top1_accuracy": tuned_metrics["observation_top1_accuracy"],
                "student_test_macro_f1": tuned_metrics["observation_macro_f1"],
                "student_test_log_loss": tuned_metrics["candidate_binary_log_loss"],
                "fit_time_seconds": tuned_fit_s,
                "inference_time_seconds": tuned_inf_s,
                "parameters": json.dumps(tuned_pipe.named_steps["classifier"].get_params(), default=str),
            }
        )

        joblib.dump(tuned_pipe, output_dir / f"{model_name}_tuned.joblib")
        print(f"Saved tuned model: {output_dir / f'{model_name}_tuned.joblib'}")

    df_comp = pd.DataFrame(comparison_records)
    comp_csv = output_dir / "tuned_vs_untuned_comparison.csv"
    df_comp.to_csv(comp_csv, index=False)

    print("\n" + df_comp[
        [
            "model",
            "variant",
            "student_test_top1_accuracy",
            "student_test_macro_f1",
            "student_test_log_loss",
            "fit_time_seconds",
            "inference_time_seconds",
        ]
    ].to_string(index=False))
    print(f"\nSaved comparison summary: {comp_csv}")


if __name__ == "__main__":
    main()
