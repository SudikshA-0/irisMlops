import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_home(client):
    r = client.get("/")
    assert r.status_code == 200


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["model"] == "RandomForestClassifier"
    assert body["accuracy"] >= 0.9


def test_predict_setosa(client):
    r = client.post(
        "/predict",
        json={"sepal_length": 5.1, "sepal_width": 3.5,
              "petal_length": 1.4, "petal_width": 0.2},
    )
    assert r.status_code == 200
    assert r.json()["flower_name"] == "setosa"


def test_predict_validation_error(client):
    r = client.post("/predict", json={"sepal_length": -1})
    assert r.status_code == 422
