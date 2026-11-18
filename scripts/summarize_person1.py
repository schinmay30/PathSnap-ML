"""Plots and a facts sheet for Person 1's models, generated from the saved results.

Reads experiments/person1/results/*.csv and the saved joblib models; writes
figures to results/figures/ and results/person1_report_facts.md. Every number
in the facts sheet comes from those files, so it can be verified by re-running.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLORS = {"logistic_regression": "#d95f02", "gaussian_naive_bayes": "#1b7837"}
LABELS = {"logistic_regression": "Logistic Regression", "gaussian_naive_bayes": "Gaussian Naive Bayes"}


def figures(folds, final, fig_dir):
    fig_dir.mkdir(parents=True, exist_ok=True)
    models = list(COLORS)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, column, title in zip(
        axes,
        ["observation_macro_f1", "observation_top1_accuracy", "candidate_binary_log_loss"],
        ["CV Macro F1 by fold", "CV Top-1 accuracy by fold", "CV candidate log loss by fold"],
    ):
        for x, model in enumerate(models):
            sub = folds[folds.model == model]
            ax.scatter([x] * len(sub), sub[column], color=COLORS[model], s=40, zorder=3)
            ax.hlines(sub[column].mean(), x - 0.25, x + 0.25, color="#222222", lw=2, zorder=4)
        ax.set_xticks(range(len(models)), [LABELS[m] for m in models], rotation=10)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Group 5-fold CV on the 80 training trips (black bar = mean)")
    fig.tight_layout(); fig.savefig(fig_dir / "person1_cv_by_fold.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    width = 0.35
    for ax, (suffix_train, test_col, title) in zip(
        axes,
        [
            ("train_observation_top1_accuracy", "observation_top1_accuracy", "Top-1 accuracy"),
            ("train_observation_macro_f1", "observation_macro_f1", "Macro F1"),
            ("train_candidate_binary_log_loss", "candidate_binary_log_loss", "Candidate log loss"),
        ],
    ):
        cv_col = test_col
        for x, model in enumerate(models):
            row = final[final.model == model].iloc[0]
            cv_mean = folds[folds.model == model][cv_col].mean()
            vals = [row[suffix_train], cv_mean, row[test_col]]
            for k, (v, hatch) in enumerate(zip(vals, ["", "//", "xx"])):
                ax.bar(x + (k - 1) * 0.25, v, 0.25, color=COLORS[model], hatch=hatch, edgecolor="#222222")
        ax.set_xticks(range(len(models)), [LABELS[m] for m in models], rotation=10)
        ax.set_title(title); ax.grid(axis="y", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, fc="white", ec="#222222", hatch=h) for h in ["", "//", "xx"]]
    fig.legend(handles, ["train (refit)", "CV mean", "student test"], loc="lower center", ncol=3)
    fig.suptitle("Train vs cross-validation vs student-test")
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(fig_dir / "person1_train_cv_test.png", dpi=130); plt.close(fig)


def facts(folds, cv, final, results_dir, out_path):
    lines = ["# Person 1 model facts (generated; do not edit by hand)", "",
             "Source: `experiments/person1/results/`. Partitions: CV on the 80 `train` trips (GroupKFold by trip, 5 folds); "
             "final comparison on the 10 `student_test` trips. Re-generate with `python scripts/summarize_person1.py`.", ""]
    for model in COLORS:
        f = final[final.model == model].iloc[0]
        c = cv[cv.model == model].iloc[0]
        cfg = json.loads(f["configuration"])
        keep = ["C", "penalty", "l1_ratio", "solver", "max_iter", "class_weight", "random_state", "var_smoothing", "priors"]
        cfg = {k: v for k, v in cfg.items() if k in keep}
        lines += [
            f"## {LABELS[model]}", "",
            f"- Configuration (classifier step; features are standardised first): `{cfg}`",
            f"- Random state {int(f['random_state'])}; scikit-learn {f['sklearn_version']}",
            f"- CV (5 folds): Top-1 {c.observation_top1_accuracy_mean:.4f} ± {c.observation_top1_accuracy_std:.4f}, "
            f"Macro F1 {c.observation_macro_f1_mean:.4f} ± {c.observation_macro_f1_std:.4f}, "
            f"log loss {c.candidate_binary_log_loss_mean:.4f} ± {c.candidate_binary_log_loss_std:.4f}, "
            f"fit time {c.fit_seconds_mean:.2f} s per fold",
            f"- Student test: Top-1 {f.observation_top1_accuracy:.4f}, Macro F1 {f.observation_macro_f1:.4f}, "
            f"log loss {f.candidate_binary_log_loss:.4f} ({int(f.eligible_observations)} eligible observations; "
            f"{int(f.observations_without_positive_candidate_label)} without the reference arc among candidates)",
            f"- Train (refit on all 80 trips, evaluated on the same trips): Top-1 {f.train_observation_top1_accuracy:.4f}, "
            f"Macro F1 {f.train_observation_macro_f1:.4f}, log loss {f.train_candidate_binary_log_loss:.4f}",
            f"- Fit time on all training candidates {f.fit_seconds_on_all_training_candidates:.2f} s; "
            f"inference on student test {f.student_test_inference_seconds * 1000:.1f} ms",
            "",
        ]
    lr = joblib.load(results_dir / "logistic_regression.joblib")
    gnb = joblib.load(results_dir / "gaussian_naive_bayes.joblib")
    features = list(lr.feature_names_in_)
    coef = pd.Series(lr.named_steps["classifier"].coef_[0], index=features).sort_values(key=abs, ascending=False)
    lines += ["## Learned parameters", "", "Logistic Regression coefficients on standardised features (largest magnitude first):", ""]
    lines += [f"- `{k}`: {v:+.3f}" for k, v in coef.items()]
    clf = gnb.named_steps["classifier"]
    pos = list(clf.classes_).index(1)
    gap = pd.Series(clf.theta_[pos] - clf.theta_[1 - pos], index=features).sort_values(key=abs, ascending=False)
    lines += ["", "Gaussian NB: positive-class mean minus negative-class mean per standardised feature:", ""]
    lines += [f"- `{k}`: {v:+.3f}" for k, v in gap.items()]
    lines += ["", "## Questions the report must answer for each model (your own analysis goes here)", "",
              "A1. Why is this algorithm suitable for PathSnap?  ", "A2. Strengths and limitations for this task (tie them to the numbers above)?  ",
              "B5. Important hyperparameters and initial values (see configuration).  ", "B6. Loss / objective used.  ",
              "C7-8. Student-test Top-1, Macro F1, log loss (see above).  ", "C9. Overfitting or underfitting, and the evidence (compare train, CV and student-test).  "]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "experiments/person1/results")
    args = parser.parse_args()
    folds = pd.read_csv(args.results_dir / "person1_cv_folds.csv")
    cv = pd.read_csv(args.results_dir / "person1_cv_summary.csv")
    final = pd.read_csv(args.results_dir / "person1_student_test_results.csv")
    figures(folds, final, PROJECT_ROOT / "results/figures")
    facts(folds, cv, final, args.results_dir, PROJECT_ROOT / "results/person1_report_facts.md")
    print("Wrote figures and results/person1_report_facts.md")


if __name__ == "__main__":
    main()
