from fastapi.testclient import TestClient

from src.api import main


def test_analysis_endpoint_returns_real_pipeline_response(monkeypatch):
    monkeypatch.setattr(
        main,
        "validate_public_url",
        lambda _url: ("example.com", "93.184.216.34", 443),
    )
    monkeypatch.setattr(
        main.model_registry,
        "predict",
        lambda _config, _features: {
            "score": 0.2,
            "model_family": "test_model",
            "top_factors": [],
            "explanation_method": "test",
        },
    )
    monkeypatch.setattr(
        main.monitoring,
        "record_analysis",
        lambda data: {"analysis_id": 1, **data},
    )

    response = TestClient(main.app).post(
        "/api/analyze",
        json={"url": "https://example.com/", "browser_enabled": False},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] in {"safe", "suspicious", "phishing"}
    assert body["model_family"] == "test_model"
    assert body["browser_ok"] is False
    assert "evidence_summary" in body


def test_health_endpoint_reports_model_readiness():
    response = TestClient(main.app).get("/api/health")

    assert response.status_code == 200
    assert response.json()["service"] == "phishing-url-analyzer"
