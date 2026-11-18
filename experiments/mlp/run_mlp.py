"""CLI runner for Person 4's PathSnap MLP resource benchmarking and training.

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

from experiments.mlp.config import MLPConfig, ResourceBudget  # noqa: E402
from experiments.mlp.data_contract import (  # noqa: E402
    estimate_csv_resources,
    load_candidate_dataset,
)
from experiments.mlp.evaluate import (  # noqa: E402
    benchmark_mlp_latency,
    evaluate_mlp_experiment,
    save_mlp_artifacts,
)


def _parse_hidden_layers(spec: str) -> tuple[int, ...]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("hidden-layers must be comma-separated positive integers")
    try:
        layers = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("hidden-layers must contain integers only") from exc
    if any(n < 1 for n in layers):
        raise argparse.ArgumentTypeError("hidden-layers values must be >= 1")
    return layers


def _parse_batch_size(spec: str) -> int | str:
    if spec.strip().lower() == "auto":
        return "auto"
    try:
        value = int(spec)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch-size must be 'auto' or a positive integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("batch-size must be >= 1")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="Path to approved shared candidate-feature CSV (or .csv.gz).",
    )
    parser.add_argument(
        "--hidden-layers",
        type=_parse_hidden_layers,
        default=(64, 32),
        help="Comma-separated hidden layer sizes, e.g. '64,32' or '128,64'.",
    )
    parser.add_argument(
        "--activation",
        choices=["identity", "logistic", "tanh", "relu"],
        default="relu",
    )
    parser.add_argument(
        "--solver",
        choices=["lbfgs", "sgd", "adam"],
        default="adam",
    )
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument(
        "--learning-rate",
        choices=["constant", "invscaling", "adaptive"],
        default="constant",
    )
    parser.add_argument("--learning-rate-init", type=float, default=1e-3)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument(
        "--early-stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable internal early stopping (for adam/sgd).",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--n-iter-no-change", type=int, default=10)
    parser.add_argument("--batch-size", type=_parse_batch_size, default=1024)
    parser.add_argument("--inference-batch-size", type=int, default=50_000)
    parser.add_argument("--n-splits", type=int, default=5)
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
        default=REPO_ROOT / "experiments" / "mlp" / "results",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=REPO_ROOT / "artifacts" / "models" / "mlp_model.joblib",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = MLPConfig(
        hidden_layer_sizes=args.hidden_layers,
        activation=args.activation,
        solver=args.solver,
        alpha=args.alpha,
        learning_rate=args.learning_rate,
        learning_rate_init=args.learning_rate_init,
        max_iter=args.max_iter,
        early_stopping=args.early_stopping,
        validation_fraction=args.validation_fraction,
        n_iter_no_change=args.n_iter_no_change,
        batch_size=args.batch_size,
        inference_batch_size=args.inference_batch_size,
        n_splits=args.n_splits,
        max_train_rows_per_fold=args.max_train_rows,
        random_state=args.random_state,
    )
    budget = ResourceBudget(max_memory_mb=args.max_memory_mb)

    estimate = estimate_csv_resources(
        args.candidates, budget=budget, use_float32=config.use_float32
    )
    print("=== MLP Resource Pre-Flight Estimate ===")
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
        probe = benchmark_mlp_latency(X_s, y_s, config)
        print("=== MLP Probe Latency Benchmark ===")
        print(json.dumps(probe, indent=2))
        return 0

    table, feature_columns, _ = load_candidate_dataset(
        args.candidates,
        budget=budget,
        use_float32=config.use_float32,
        enforce_memory_budget=True,
    )
    model, fold_df, cv_summary, test_metrics = evaluate_mlp_experiment(
        table, feature_columns, config
    )

    print("=== MLP 5-Fold GroupKFold CV Summary ===")
    print(json.dumps(cv_summary, indent=2))
    print("=== MLP Student-Test Summary ===")
    print(json.dumps(test_metrics, indent=2))

    if args.authorize_save:
        saved = save_mlp_artifacts(
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
