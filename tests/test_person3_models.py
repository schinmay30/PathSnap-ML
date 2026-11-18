import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_person3_models import FEATURE_COLUMNS, build_models, positive_probability, training_folds


@pytest.mark.parametrize("name", ["extra_trees", "hist_gradient_boosting"])
def test_seeded_probabilities(name):
    features = pd.DataFrame(np.random.default_rng(7).normal(size=(120, 9)), columns=FEATURE_COLUMNS)
    labels = (features.dist_m > 0).astype(int)
    model = build_models()[name]
    classifier = model.named_steps["classifier"]
    assert not hasattr(classifier, "classes_")
    if name == "hist_gradient_boosting":
        assert classifier.early_stopping is False
    model.fit(features, labels)
    probabilities = model.predict_proba(features)
    np.testing.assert_array_equal(classifier.classes_, [0, 1])
    assert probabilities.shape == (120, 2)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    np.testing.assert_allclose(positive_probability(model, features), probabilities[:, 1])
    repeated = build_models()[name].fit(features, labels)
    np.testing.assert_allclose(repeated.predict_proba(features), probabilities, rtol=0, atol=1e-14)


def test_positive_probability_respects_reversed_classes():
    model = SimpleNamespace(named_steps={"classifier": SimpleNamespace(classes_=[1, 0])},
                            predict_proba=lambda x: np.array([[0.8, 0.2]]))
    np.testing.assert_array_equal(positive_probability(model, pd.DataFrame(index=[0])), [0.8])


def test_disjoint_trip_folds():
    table = pd.DataFrame({"trip_id": list("aabbccddeeff"), "label": [0, 1] * 6,
                          "partition": "train"})
    seen = []
    for fit, valid in training_folds(table):
        assert set(table.iloc[fit].trip_id).isdisjoint(table.iloc[valid].trip_id)
        seen.extend(valid)
    assert sorted(seen) == list(table.index)
    table.loc[0, "partition"] = "student_test"
    with pytest.raises(ValueError, match="Only training trips"):
        training_folds(table)
