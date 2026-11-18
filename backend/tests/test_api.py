from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint_api_and_root():
    for path in ("/api/health", "/health"):
        resp = client.get(path)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "PathSnap FastAPI Backend"
        assert data["dataset_dir_available"] is True
        assert data["active_partition"] == "student_test"
        assert data["partition_role"] == "student_development_test"
        assert data["available_trip_count"] == 10
        assert data["models_ready"] is False
        assert data["hmm_decoder_loaded"] is False
        assert "timestamp" in data


def test_trips_endpoint_discovers_real_student_test_without_leaking_ground_truth():
    for path in ("/api/trips", "/trips"):
        resp = client.get(path)
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 10
        assert data["partition"] == "student_test"
        assert data["partition_role"] == "student_development_test"
        assert len(data["trips"]) == 10

        first = data["trips"][0]
        assert set(first.keys()) == {"trip_id", "track_points", "num_nodes", "num_arcs"}
        assert first["track_points"] > 0
        assert first["num_nodes"] > 0
        assert first["num_arcs"] > 0

        # Ensure ground-truth fields are never exposed
        serialized = resp.text.lower()
        assert ".route" not in serialized
        assert "ground_truth" not in serialized
        assert "route_arcs" not in serialized


def test_models_endpoint_reports_not_ready_honestly():
    for path in ("/api/models", "/models"):
        resp = client.get(path)
        assert resp.status_code == 200
        data = resp.json()
        assert data["models_ready"] is False
        assert data["hmm_decoder_loaded"] is False

        keys = {m["key"] for m in data["models"]}
        assert {"knn", "mlp", "final"}.issubset(keys)
        for m in data["models"]:
            assert m["loaded"] is False
            assert m["status"] == "not_ready"
            assert m["hmm_enabled"] is False
            assert m["artifact_path"] is None


def test_model_comparison_endpoint_returns_empty_until_experiments_run():
    for path in ("/api/models/comparison", "/models/comparison"):
        resp = client.get(path)
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is False
        assert data["models"] == []


def test_predict_valid_trip_returns_503_when_models_not_ready():
    trips_resp = client.get("/api/trips")
    valid_trip_id = trips_resp.json()["trips"][0]["trip_id"]

    for path in ("/api/predict", "/predict"):
        resp = client.post(path, json={"trip_id": valid_trip_id, "model": "knn"})
        assert resp.status_code == 503
        data = resp.json()
        assert "not ready" in data["detail"].lower()


def test_predict_nonexistent_trip_returns_404():
    resp = client.post("/api/predict", json={"trip_id": "99999999", "model": "knn"})
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_predict_unsupported_model_returns_400():
    trips_resp = client.get("/api/trips")
    valid_trip_id = trips_resp.json()["trips"][0]["trip_id"]

    resp = client.post(
        "/api/predict", json={"trip_id": valid_trip_id, "model": "unknown_model"}
    )
    assert resp.status_code == 400
    assert "unsupported model" in resp.json()["detail"].lower()


def test_predict_invalid_trip_id_characters_returns_422():
    resp = client.post("/api/predict", json={"trip_id": "../train/00000001", "model": "knn"})
    assert resp.status_code == 422
    assert resp.json()["code"] == "validation_error"


def test_predict_missing_trip_id_returns_422():
    resp = client.post("/api/predict", json={"model": "mlp"})
    assert resp.status_code == 422


def test_cors_preflight_allows_react_vite_dev_origin():
    resp = client.options(
        "/api/predict",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "POST" in resp.headers.get("access-control-allow-methods", "")


def test_cors_allows_lan_ip_for_two_laptop_deployment():
    resp = client.get(
        "/api/health",
        headers={"Origin": "http://192.168.1.42:5173"},
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://192.168.1.42:5173"


def test_swagger_openapi_docs_available():
    docs_resp = client.get("/docs")
    assert docs_resp.status_code == 200
    openapi_resp = client.get("/openapi.json")
    assert openapi_resp.status_code == 200
    paths = openapi_resp.json()["paths"]
    assert "/api/health" in paths
    assert "/api/trips" in paths
    assert "/api/models" in paths
    assert "/api/models/comparison" in paths
    assert "/api/predict" in paths
