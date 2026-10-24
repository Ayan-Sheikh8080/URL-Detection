"""API tests: contract, validation and monitoring endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.agent.agent import PhishingAgent
from src.agent.reasoning import RuleBasedReasoner
from src.config import Settings
from src.serving.api import create_app
from tests.conftest import StubRegistry, StubSandbox, phishing_evidence


@pytest.fixture
def client(tmp_path):
    settings = Settings(
        model_dir=tmp_path / "models",
        events_path=tmp_path / "events.jsonl",
        phishing_threshold=0.65,
        suspicious_threshold=0.35,
    )
    agent = PhishingAgent(
        settings,
        registry=StubRegistry(url_score=0.88),
        sandbox=StubSandbox(phishing_evidence()),
        reasoner=RuleBasedReasoner(),
    )
    app = create_app(agent=agent, settings=settings)
    with TestClient(app) as test_client:
        yield test_client


def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["reasoner"] == "rule_based"


def test_analyze_returns_verdict_with_reasons_and_factors(client):
    response = client.post("/analyze", json={"url": "https://paypal-secure-login.com/signin"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["verdict"] == "phishing"
    assert payload["confidence"] > 0.5
    assert payload["reasons"]
    assert payload["top_factors"]
    assert payload["evidence_summary"]["final_url"]
    assert payload["browser"]["ok"] is True


def test_analyze_rejects_non_http_schemes(client):
    for url in ["javascript:alert(1)", "file:///etc/passwd", "ftp://x.com/a"]:
        response = client.post("/analyze", json={"url": url})
        assert response.status_code == 422


def test_analyze_rejects_empty_and_hostless_urls(client):
    assert client.post("/analyze", json={"url": ""}).status_code == 422
    assert client.post("/analyze", json={"url": "https://"}).status_code == 422


def test_analyze_url_only_mode(client):
    response = client.post(
        "/analyze", json={"url": "https://example.com", "browse": False}
    )
    assert response.status_code == 200
    assert response.json()["evidence_summary"]["browser_error"]


def test_monitoring_stats_counts_analyses(client):
    client.post("/analyze", json={"url": "https://paypal-secure-login.com/signin"})
    client.post("/analyze", json={"url": "https://example.com", "browse": False})

    response = client.get("/monitoring/stats")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total_analyses"] == 2
    assert sum(payload["verdicts"].values()) == 2
    assert payload["verdicts"].get("phishing", 0) >= 1
    assert "p50" in payload["latency_ms"]
    assert "failure_rate" in payload["browser"]


def test_monitoring_drift_reports_unavailable_without_events(tmp_path):
    settings = Settings(model_dir=tmp_path / "models", events_path=tmp_path / "events.jsonl")
    agent = PhishingAgent(
        settings,
        registry=StubRegistry(url_score=0.1),
        sandbox=StubSandbox(phishing_evidence()),
        reasoner=RuleBasedReasoner(),
    )
    app = create_app(agent=agent, settings=settings)
    with TestClient(app) as test_client:
        payload = test_client.get("/monitoring/drift").json()
    assert payload["available"] is False


def test_analyze_response_schema_is_json_serializable(client):
    import json

    payload = client.post("/analyze", json={"url": "https://x.top/login"}).json()
    assert json.loads(json.dumps(payload))
