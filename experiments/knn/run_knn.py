"""CLI runner for Person 4's PathSnap KNN resource benchmarking and training.

Safety gates:
- Use `--benchmark-only` to inspect memory & latency without full training.
- Full training will NOT save models or metrics unless `--authorize-save` is
  explicitly supplied on the command line.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.knn.config import KNNConfig, ResourceBudget  # noqa: E402
from experiments.knn.data_contract import (  # noqa: E402
    estimate_csv_resources,
    load_candidate_dataset,
)
from experiments.knn.evaluate import (  # noqa: E402
    benchmark_knn_latency,
    evaluate_knn_experiment,
    save_knn_artifacts,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="Path to approved shared candidate-feature CSV (or .csv.gz).",
    )
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument(
        "--weights", choices=["uniform", "distance"], default="distance"
    )
    parser.add_argument("--metric", type=str, default="euclidean")
    parser.add_argument(
        "--algorithm",
        choices=["auto", "ball_tree", "kd_tree", "brute"],
        default="kd_tree",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-memory-mb", type=float, default=2048.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Only estimate dataset RAM and run a small latency probe; do not train.",
    )
    parser.add_argument(
        "--authorize-save",
        action="store_true",
        help="Explicitly authorize writing trained model and metric artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "experiments" / "knn" / "results",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=REPO_ROOT / "artifacts" / "models" / "knn_model.joblib",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = KNNConfig(
        n_neighbors=args.n_neighbors,
        weights=args.weights,
        metric=args.metric,
        algorithm=args.algorithm,
        n_jobs=args.n_jobs,
        n_splits=args.n_splits,
        batch_size=args.batch_size,
        max_train_rows_per_fold=args.max_train_rows,
        random_state=args.random_state,
    )
    budget = ResourceBudget(max_memory_mb=args.max_memory_mb)

    estimate = estimate_csv_resources(
        args.candidates, budget=budget, use_float32=config.use_float32
    )
    print("=== KNN Resource Pre-Flight Estimate ===")
    print(json.dumps(estimate.to_dict(), indent=2))

    if args.benchmark_only:
        sample = pd.read_csv(
            args.candidates, nrows=budget.sample_rows, dtype={"trip_id": str}
        )
        feat_cols = list(estimate.feature_columns)
        X_s = sample[feat_cols].to_numpy(
            dtype=np.float32 if config.use_float32 else np.float64
        )
        y_s = pd.to_numeric(sample["label"], errors="raise").to_numpy(dtype=int)
        probe = benchmark_knn_latency(X_s, y_s, config)
        print("=== KNN Probe Latency Benchmark ===")
        print(json.dumps(probe, indent=2))
        return 0

    table, feature_columns, _ = load_candidate_dataset(
        args.candidates,
        budget=budget,
        use_float32=config.use_float32,
        enforce_memory_budget=True,
    )
    model, fold_df, cv_summary, test_metrics = evaluate_knn_experiment(
        table, feature_columns, config
    )

    print("=== KNN 5-Fold GroupKFold CV Summary ===")
    print(json.dumps(cv_summary, indent=2))
    print("=== KNN Student-Test Summary ===")
    print(json.dumps(test_metrics, indent=2))

    if args.authorize_save:
        saved = save_knn_artifacts(
            model=model,
            fold_df=fold_df,
            cv_summary=cv_summary,
            student_test_metrics=test_metrics,
            output_dir=args.output_dir,
            model_output_path=args.model_out,
            authorize_save=True,
        )
        print("=== Saved Authorized Artifacts ===")
        print(json.dumps(saved, indent=2))
    else:
        print(
            "Dry evaluation complete. No model or metric files were written "
            "(pass --authorize-save once final feature schema is approved)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
