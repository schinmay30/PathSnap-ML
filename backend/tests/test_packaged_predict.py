import json
import os
from pathlib import Path
import shutil
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

from app.core.config import Settings
from app.main import create_app
from app.services import model_registry_service
from app.services.prediction_service import route_geometry
from pathsnap.route_inference import RoadTrip

RELEASE = Path(__file__).resolve().parents[2] / "artifacts/final"


def configure(tmp_path, monkeypatch):
    dest = tmp_path / "artifacts/final"
    shutil.copytree(RELEASE, dest)
    data = tmp_path / "development"
    data.mkdir()
    monkeypatch.setenv("PATHSNAP_TRIP_DATA_DIR", str(data))
    return dest, data


def test_corrupt_release_and_other_models(tmp_path, monkeypatch):
    dest, data = configure(tmp_path, monkeypatch)
    folder = data / "demo"
    folder.mkdir()
    for suffix in ("track", "arcs", "nodes"):
        (folder / f"demo.{suffix}").touch()
    manifest = next(dest.glob("viterbi*.json"))
    info = json.loads(manifest.read_text())
    info["feature_names"].reverse()
    manifest.write_text(json.dumps(info))
    (dest / "knn-placeholder.joblib").touch()
    with TestClient(create_app(Settings(workspace_root=tmp_path))) as client:
        assert client.get("/api/health").json()["models_ready"] is False
        for model in ("final", "knn", "mlp"):
            resp = client.post("/api/predict", json={"trip_id":"demo", "model":model})
            assert resp.status_code == 503 and "not ready" in resp.json()["detail"]


@pytest.mark.parametrize("boundary", ["soft_jump", "gap_reset", "transition_reset", "displacement_reset"])
def test_geometry_respects_status_and_missing_arcs(boundary):
    trip = RoadTrip("x", [], [(0,0), (1,1), (2,2), (3,3)], [(0,1), (1,2), (2,3)])
    pairs = [(0,"start"), (1,boundary), (None,"no_candidates"), (2,"restart_after_missing")]
    assert route_geometry([dict(arc_id=a,status=s) for a,s in pairs], trip) == [
        [[0,0],[1,1]], [[1,1],[2,2]], [[2,2],[3,3]]]


def test_real_trip_prediction(tmp_path, monkeypatch):
    source = os.environ.get("PATHSNAP_SMOKE_TRIP_DIR")
    if not source:
        pytest.skip("Set PATHSNAP_SMOKE_TRIP_DIR to an extracted real development trip")
    source = Path(source)
    _, data = configure(tmp_path, monkeypatch)
    folder = data / source.name
    folder.mkdir()
    for suffix in ("track", "nodes", "arcs"):
        filename = f"{source.name}.{suffix}"
        shutil.copyfile(source / filename, folder / filename)
    trip = RoadTrip.read(folder)
    loader = Mock(wraps=model_registry_service.load_package)
    monkeypatch.setattr(model_registry_service, "load_package", loader)
    with TestClient(create_app(Settings(workspace_root=tmp_path))) as client:
        assert client.get("/api/health").json()["hmm_decoder_loaded"]
        for endpoint in ("/api/predict", "/predict"):
            response = client.post(endpoint, json={"trip_id":source.name, "model":"final"})
            assert response.status_code == 200, response.text
        loader.assert_called_once()
    body = response.json()
    json.dumps(body, allow_nan=False)
    assert "ground_truth" not in response.text.lower() and ".route" not in response.text.lower()
    assert body["metrics"]["accuracy"] is None and body["hmm_applied"]
    matches = body["matches"]
    assert [m["observation_id"] for m in matches] == list(range(len(trip.track)))
    assert all(set(m) == {"arc_id", "status", "observation_id"} for m in matches)
    assert body["predicted_arc_ids"] == [m["arc_id"] for m in matches if m["arc_id"] is not None]
    # Reconstruct components independently: every edge must be the observed arc,
    # and only topologically adjoining arcs within a continuous run may share a line.
    components = []
    prev = None
    for match in matches:
        cur = match["arc_id"]
        if cur is not None:
            a, b = trip.arcs[cur]
            points = [list(trip.nodes[n][::-1]) for n in (a, b)]
            if prev is None or match["status"] != "matched":
                components.append(points)
            elif prev != cur:
                if trip.arcs[prev][1] == a:
                    components[-1].append(points[1])
                else:
                    components.append(points)
        prev = cur
    assert body["predicted_route"] == components
