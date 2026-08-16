from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Sales AI CI/CD is running"}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_analyze_opportunity_medium_risk():
    response = client.post(
        "/analyze-opportunity",
        json={
            "account_name": "ABC Trading",
            "opportunity_name": "Digital Signage Rollout",
            "stage": "Proposal",
            "value": 50000,
            "probability": 60,
        },
    )

    assert response.status_code == 200
    assert response.json()["risk_level"] == "medium"
    assert "decision-maker follow-up" in response.json()["recommended_action"]


def test_rag_answer_medium_probability():
    response = client.post(
        "/rag-answer",
        json={
            "question": "What should I do for a 60 percent opportunity?",
        },
    )

    assert response.status_code == 200
    assert "Medium probability" in response.json()["context"]
    assert "active follow-up" in response.json()["answer"]