"""Build v1 from recorded training-only selections; refuse existing release names."""
import json
import os
import platform
from importlib.metadata import version
from inspect import signature
from tempfile import TemporaryDirectory
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from run_person3_models import ROOT, build_models
from pathsnap.deployment import FEATURES, load_package, sha256
from pathsnap.viterbi import decode

NAME = "hist_gradient_boosting"
TABLE = "data/processed/person3_extended_candidates.csv.gz"
LABELS = "results/labels/observation_labels.csv.gz"
CHOICE = "experiments/person3/tuning/best_parameters.json"
CALIBRATION = "experiments/person3/decoder_ablation/manifest.json"


def training_rows(path):
    chunks = pd.read_csv(ROOT / path, dtype={"trip_id": str}, chunksize=100000)
    return pd.concat([c.loc[c.partition.eq("train")] for c in chunks], ignore_index=True)


def export():
    directory = ROOT / "artifacts/final"
    directory.mkdir(parents=True, exist_ok=True)
    model_name = "final_person3_hgb_v1.joblib"
    manifest_name = "viterbi_person3_hgb_v1.json"
    if any((directory / name).exists() for name in (model_name, manifest_name)):
        raise FileExistsError("Version 1 already exists")
    selection = json.loads((ROOT / CHOICE).read_text())["all_train"][NAME]
    calibration = json.loads((ROOT / CALIBRATION).read_text())
    policy, = {fold["selected"][NAME] for fold in calibration["folds"]}
    decoder = {p.name: p.default for p in signature(decode).parameters.values()
               if p.kind == p.KEYWORD_ONLY}
    decoder.update(calibration["policies"][policy])
    candidates, labels = training_rows(TABLE), training_rows(LABELS)
    key = ["trip_id", "observation_id"]
    roster = sorted(training_rows("results/dataset_inventory.csv").trip_id)
    valid = labels.loc[labels.label_valid, key + ["arc_id"]]
    joined = candidates.merge(valid, on=key, how="left", validate="many_to_one")
    n_expected = int(training_rows("results/candidate_generation_summary.csv").candidate_rows.iloc[0])
    if (len(roster) != 80 or sorted(candidates.trip_id.unique()) != roster
            or len(joined) != n_expected or joined.arc_id.isna().any()
            or candidates.duplicated(key + ["candidate_arc_id"]).any()
            or not joined.label.eq(joined.arc_id.eq(joined.candidate_arc_id)).all()
            or not np.isfinite(candidates[FEATURES].to_numpy()).all()):
        raise ValueError("Training roster, labels, keys, count or feature checks failed")
    model = build_models()[NAME]
    model.set_params(**{"classifier__"+k: v for k,v in selection["parameters"].items()})
    model.fit(candidates[FEATURES], candidates.label)
    with TemporaryDirectory(dir=directory) as temp:
        binary = Path(temp) / model_name
        metadata = Path(temp) / manifest_name
        joblib.dump(model, binary, compress=3)
        manifest = dict(format_version=1, model_version="person3-hgb-v1", model_file=model_name,
            model_sha256=sha256(binary), model_bytes=binary.stat().st_size, classes=[0, 1],
            feature_names=FEATURES, classifier_parameters=model.named_steps["classifier"].get_params(),
            selection=selection, selection_record=CHOICE, selection_stage="all_train",
            decoder_policy=policy, decoder=decoder, decoder_selection_record=CALIBRATION,
            partition="train", training_trips=len(roster), trip_ids=roster,
            training_rows=len(candidates), positive_rows=int(candidates.label.sum()),
            label_alignment="Monotone DP on chronological trip route positions; distance cost capped at 100 m, 1 m per skipped arc, valid within 50 m; candidate positive iff arc ID matches valid alignment.",
            sources_sha256={p:sha256(ROOT/p) for p in (TABLE,LABELS,CHOICE,CALIBRATION)},
            versions={"python":platform.python_version(), **{p:version(p) for p in
                      ("scikit-learn","numpy","pandas","scipy","joblib")}})
        metadata.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        released = load_package(binary, metadata)
        probe = candidates[FEATURES].iloc[::997]
        np.testing.assert_array_equal(model.predict_proba(probe), released.predict_proba(probe))
        os.link(binary, directory / model_name)
        try:
            os.link(metadata, directory / manifest_name)
        except OSError:
            (directory / model_name).unlink()
            raise
    print(f"Verified and exported {len(candidates)} training rows from {len(roster)} trips")


if __name__ == "__main__":
    export()
