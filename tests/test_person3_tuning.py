import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import tune_person3_models as tuning


def test_nested_tuning_and_artifacts(tmp_path):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(120, 19)), columns=tuning.FEATURES)
    frame["trip_id"] = np.repeat(["01", "02", "03", "04", "test"], 24)
    frame["partition"] = np.where(frame.trip_id == "test", "student_test", "train")
    frame["observation_id"] = np.tile(np.repeat(np.arange(12), 2), 5)
    frame["candidate_arc_id"] = np.arange(120) % 2
    frame["label"] = frame.candidate_arc_id
    frame["reference_arc_id"] = frame.label
    path = tmp_path / "candidates.csv"
    frame.to_csv(path, index=False)
    train = tuning.load_train(path)
    assert len(train) == 96 and "reference_arc_id" not in train
    with pytest.raises(ValueError, match="Only training"):
        tuning.splits(frame)
    output = tmp_path / "tuning"
    tuning.run(train, output)
    outer = pd.read_csv(output / "outer_validation.csv")
    inner = pd.read_csv(output / "inner_folds.csv")
    for row in outer.itertuples():
        forbidden = set(json.loads(row.validation_trips))
        selected = inner.loc[inner.stage == f"outer_{row.fold}"]
        assert all(forbidden.isdisjoint(json.loads(ids)) for ids in selected.validation_trips)
    assert len(outer) == 8 and len(inner) == 48
    trials = pd.read_csv(output / "trials.csv")
    assert trials.mean_inner_top1.between(0, 1).all()
    best = json.loads((output / "best_parameters.json").read_text())
    for name, selection in best["all_train"].items():
        subset = trials.loc[(trials.stage == "all_train") & (trials.model == name)]
        assert selection["mean_inner_top1"] == pytest.approx(subset.mean_inner_top1.max())
    with pytest.raises(FileExistsError):
        tuning.run(train, output)
