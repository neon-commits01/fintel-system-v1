from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_model_info():
    response = client.get("/model_info")
    assert response.status_code == 200
    data = response.json()

    assert data["model_name"] == "TF-IDF + Linear SVM"
    assert data["model_version"] == "v1-baseline"
    assert set(data["classes"]) == {"negative", "neutral", "positive"}


def test_predict():
    response = client.post(
        "/predict",
        json={
            "text": "Operating profit fell compared with the previous year."
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] in ["positive", "neutral", "negative"]
    assert data["model_name"] == "TF-IDF + Linear SVM"
    assert data["model_version"] == "v1-baseline"

def test_predict_batch():
    response = client.post(
        "/predict_batch",
        json = {
            "texts":[
                "Operating profit fell compared with the previous year.",
                "The company signed a new contract worth EUR 10 million.",
                "Financial terms were not disclosed."

            ]
        }
    )
    assert response.status_code == 200
    data = response.json()

    # check the length of the predictions
    assert len(data["predictions"]) == 3
    assert all (
        prediction in ["positive", "negative", "neutral"]
        for prediction in data["predictions"]
    )
    assert data["model_name"] == "TF-IDF + Linear SVM"
    assert data["model_version"] == "v1-baseline"



def test_predict_empty_text():
    response = client.post(
        "/predict",
        json={
            "text": " "
        }
    )

    assert response.status_code == 400

def test_predict_batch_empty_text():
    response = client.post(
        "/predict_batch",
        json = {
            "texts":[
                "Operating profit fell compared with the previous year.",
                "   ",
                "Financial terms were not disclosed."
             
            ]
        }
    )
    assert response.status_code == 400