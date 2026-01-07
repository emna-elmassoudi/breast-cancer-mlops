from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["n_features_in_"] == 30
    assert data["classes_"] == [0, 1]

def test_predict_ok():
    payload = {"features": [0.0] * 30}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data

def test_predict_bad_input():
    payload = {"features": [0.0] * 10}
    response = client.post("/predict", json=payload)
    assert response.status_code in (400, 422)
    