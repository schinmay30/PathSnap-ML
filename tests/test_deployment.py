import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "backend"))
from models.loader import find_hmm_artifact, find_model_artifact
from pathsnap.deployment import FEATURES, load_package
from pathsnap.route_inference import RoadTrip


def release_paths():
    dirs = [ROOT / "artifacts/final"]
    return find_model_artifact("final", dirs), find_hmm_artifact(dirs)


def test_registry_discovery_schema_and_probabilities():
    package = load_package(*release_paths())
    frame = pd.DataFrame(np.zeros((2, 19)), columns=FEATURES)
    p = package.predict_proba(frame)
    np.testing.assert_array_equal(package.classes_, [0, 1])
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1), 1)
    for bad in [frame.iloc[:, ::-1], frame.iloc[:, :-1], frame.assign(label=1)]:
        with pytest.raises(ValueError, match="19 ordered"):
            package.predict_proba(bad)
    with pytest.raises(ValueError, match="Non-finite"):
        package.predict_proba(frame.assign(dist_m=np.inf))


def test_reject_mismatched_release(tmp_path):
    binary, path = release_paths()
    manifest = json.loads(path.read_text())
    manifest["model_sha256"] = "invalid"
    bad = tmp_path / "manifest.json"
    bad.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="do not match"):
        load_package(binary, bad)


def test_route_free_trip_preserves_empty_fixes():
    package = load_package(*release_paths())
    trip = RoadTrip("test", [(0.,0.,0.), (1.,1.,1.)], [(0.,0.), (.001,0.)], [(0,1)])
    result = package.predict(trip)
    assert result == [dict(observation_id=0, arc_id=0, status="start"),
                      dict(observation_id=1, arc_id=None, status="no_candidates")]
