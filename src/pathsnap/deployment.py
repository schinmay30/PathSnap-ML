"""Load a trusted local PathSnap HGB/Viterbi release for full-track inference."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from pathsnap.route_inference import FEATURES, RoadTrip, DirectedRoadGraph, feature_table, predict_trip


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class RoutePackage:
    def __init__(self, model, manifest):
        if (manifest["feature_names"] != FEATURES or list(model.feature_names_in_) != FEATURES
                or list(model.classes_) != [0, 1]
                or not isinstance(model.named_steps["classifier"], HistGradientBoostingClassifier)):
            raise ValueError("Incompatible classifier, feature schema or class order")
        self.model = model
        self.manifest = manifest
        self.feature_names_in_ = model.feature_names_in_
        self.classes_ = model.classes_

    def predict_proba(self, frame):
        if list(frame.columns) != FEATURES or not all(np.issubdtype(t, np.number) for t in frame.dtypes):
            raise ValueError("Expected the 19 ordered numeric features")
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError("Non-finite input")
        p = self.model.predict_proba(frame)
        if (p.shape != (len(frame), 2) or not np.isfinite(p).all()
                or np.any((p < 0) | (p > 1)) or not np.allclose(p.sum(axis=1), 1)):
            raise ValueError("Invalid classifier probabilities")
        return p

    def predict(self, trip: RoadTrip):
        """One (observation_id, nullable arc_id, status) dictionary per original GPS fix."""
        if not trip.track:
            return []
        rows, _ = predict_trip(trip, DirectedRoadGraph(trip.nodes, trip.arcs),
                               feature_table(trip), self, policy=self.manifest["decoder"])
        return [dict(observation_id=int(r.observation_id),
                     arc_id=None if pd.isna(r.hmm) else int(r.hmm), status=r.status)
                for r in rows.itertuples()]


def load_package(model_path, manifest_path):
    """Accept the two paths returned by backend.models.loader; load trusted joblib only."""
    model_path = Path(model_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if (manifest["format_version"] != 1 or manifest["model_file"] != model_path.name
            or manifest["model_sha256"] != sha256(model_path)):
        raise ValueError("Model and manifest do not match")
    if manifest["versions"]["scikit-learn"] != sklearn.__version__:
        raise ValueError("Install the scikit-learn version recorded in the release manifest")
    return RoutePackage(joblib.load(model_path), manifest)
