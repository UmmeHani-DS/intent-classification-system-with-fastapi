import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from api.main import app

@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c

class TestHealthEndpoint:
    def test_health_endpoint_success(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert isinstance(data["models_loaded"], bool)
        assert isinstance(data["nlp_loaded"], bool)

class TestModelInfoEndpoint:
    def test_model_info_success(self, client):
        response = client.get("/api/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert isinstance(data["classes"], list)
        assert len(data["classes"]) > 0

    def test_model_info_when_model_not_loaded(self, client):
        with patch('api.main.model_state.model_info', {}):
            response = client.get("/api/model/info")
            assert response.status_code == 503
            assert "Model not loaded" in response.json()["detail"]

class TestSingleClassification:
    def test_single_classification_success(self, client):
        response = client.post("/api/classify", json={"text": "Send an email to John about the meeting"})
        assert response.status_code == 200
        data = response.json()
        assert "intent" in data and "confidence" in data

    def test_single_classification_with_keywords(self, client):
        response = client.post("/api/classify", json={"text": "How can I schedule a meeting and find the right email address?"})
        assert response.status_code == 200
        assert "intent" in response.json()

    def test_single_classification_empty_text(self, client):
        response = client.post("/api/classify", json={"text": ""})
        assert response.status_code == 422

    def test_single_classification_whitespace_only(self, client):
        response = client.post("/api/classify", json={"text": "   "})
        assert response.status_code == 422

    def test_single_classification_very_long_text(self, client):
        response = client.post("/api/classify", json={"text": "word " * 500})
        assert response.status_code == 422

    def test_single_classification_missing_text_field(self, client):
        response = client.post("/api/classify", json={})
        assert response.status_code == 422

    def test_single_classification_invalid_text_type(self, client):
        response = client.post("/api/classify", json={"text": 123})
        assert response.status_code == 422

    def test_single_classification_no_keywords(self, client):
        response = client.post("/api/classify", json={"text": "I like pizza and movies"})
        assert response.status_code == 200
        assert "intent" in response.json()

class TestBatchClassification:
    def test_batch_classification_success(self, client):
        response = client.post("/api/classify/batch", json={
            "texts": [
                "Send an email to John",
                "Schedule a meeting tomorrow",
                "What's the weather like?"
            ]
        })
        assert response.status_code == 200
        assert len(response.json()["results"]) == 3

    def test_batch_classification_single_item(self, client):
        response = client.post("/api/classify/batch", json={"texts": ["Find the nearest restaurant"]})
        assert response.status_code == 200
        assert len(response.json()["results"]) == 1

    def test_batch_classification_empty_list(self, client):
        response = client.post("/api/classify/batch", json={"texts": []})
        assert response.status_code == 422

    def test_batch_classification_too_many_items(self, client):
        texts = [f"Text {i}" for i in range(101)]
        response = client.post("/api/classify/batch", json={"texts": texts})
        assert response.status_code == 422

    def test_batch_classification_with_empty_text(self, client):
        response = client.post("/api/classify/batch", json={"texts": ["Valid", "", "Another"]})
        assert response.status_code == 422

    def test_batch_classification_mixed_content(self, client):
        response = client.post("/api/classify/batch", json={
            "texts": [
                "Find a restaurant",
                "Tell me a joke",
                "How does gravity work?",
                "Schedule a meeting for 3 PM"
            ]
        })
        assert response.status_code == 200
        assert len(response.json()["results"]) == 4

class TestErrorHandling:
    def test_invalid_endpoint(self, client):
        response = client.get("/api/does_not_exist")
        assert response.status_code == 404

    def test_invalid_http_method(self, client):
        response = client.get("/api/classify")
        assert response.status_code == 405

    def test_malformed_json(self, client):
        response = client.post(
            "/api/classify",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_classification_with_model_not_loaded(self, client):
        with patch('api.main.model_state.model', None):
            response = client.post("/api/classify", json={"text": "Test text"})
            assert response.status_code == 503

class TestInputValidation:
    def test_text_trimming(self, client):
        response = client.post("/api/classify", json={"text": "  Send an email  "})
        assert response.status_code == 200
        assert response.json()["text"] == "Send an email"

    def test_special_characters(self, client):
        response = client.post("/api/classify", json={"text": "Send an email with symbols: @#$%^&*()"})
        assert response.status_code == 200

    def test_unicode_characters(self, client):
        response = client.post("/api/classify", json={"text": "Enviar un correo electrónico 📧"})
        assert response.status_code == 200

    def test_numbers_in_text(self, client):
        response = client.post("/api/classify", json={"text": "Schedule a meeting at 3:30 PM on March 15th, 2024"})
        assert response.status_code == 200

class TestPerformance:
    def test_large_batch_processing(self, client):
        texts = [f"Send email number {i}" for i in range(50)]
        response = client.post("/api/classify/batch", json={"texts": texts})
        assert response.status_code == 200
        assert len(response.json()["results"]) == 50

    def test_concurrent_requests(self, client):
        import threading
        results = []

        def make_request():
            r = client.post("/api/classify", json={"text": "Test concurrent request"})
            results.append(r.status_code)

        threads = [threading.Thread(target=make_request) for _ in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert all(status == 200 for status in results)
