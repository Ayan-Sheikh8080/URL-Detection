"""End-to-end agent tests using a stubbed sandbox and model registry."""

from __future__ import annotations

from src.agent.agent import PhishingAgent
from src.agent.reasoning import RuleBasedReasoner
from tests.conftest import StubRegistry, StubSandbox, benign_evidence, phishing_evidence


def _agent(evidence: dict, *, url_score: float, combined_score: float | None = None, settings=None):
    return PhishingAgent(
        settings,
        registry=StubRegistry(url_score=url_score, combined_score=combined_score),
        sandbox=StubSandbox(evidence),
        reasoner=RuleBasedReasoner(),
    )


def test_agent_flags_phishing_link(settings):
    agent = _agent(phishing_evidence(), url_score=0.88, settings=settings)
    result = agent.analyze("https://paypal-secure-login.com/signin")
    assert result.verdict == "phishing"
    assert result.confidence >= 0.6
    assert result.reasons
    assert result.model_score == 0.88
    assert result.browser["ok"] is True


def test_agent_clears_benign_link(settings):
    agent = _agent(benign_evidence(), url_score=0.03, settings=settings)
    result = agent.analyze("https://example.com/news")
    assert result.verdict == "safe"
    assert result.top_factors


def test_agent_uses_combined_score_when_available(settings):
    agent = _agent(phishing_evidence(), url_score=0.2, combined_score=0.95, settings=settings)
    result = agent.analyze("https://paypal-secure-login.com/signin")
    assert result.combined_score == 0.95
    assert result.verdict == "phishing"


def test_agent_survives_browser_failure(settings):
    failure = {"requested_url": "https://example.com", "error": "TimeoutError: navigation timeout"}
    agent = _agent(failure, url_score=0.05, settings=settings)
    result = agent.analyze("https://example.com")
    assert result.browser["ok"] is False
    assert result.verdict in {"safe", "suspicious", "phishing"}
    assert result.evidence_summary["browser_error"]


def test_url_only_mode_skips_browsing(settings):
    sandbox = StubSandbox(phishing_evidence())
    agent = PhishingAgent(
        settings,
        registry=StubRegistry(url_score=0.9),
        sandbox=sandbox,
        reasoner=RuleBasedReasoner(),
    )
    result = agent.analyze("https://paypal-secure-login.com/signin", browse=False)
    assert sandbox.calls == []
    assert result.evidence_summary["browser_error"]


def test_agent_populates_domain_checks_and_timings(settings):
    agent = _agent(phishing_evidence(), url_score=0.8, settings=settings)
    result = agent.analyze("https://paypal-secure-login.com/signin")
    assert result.domain_checks["final_base"] == "account-secure-verify.top"
    assert result.timings_ms["total_ms"] > 0
    assert "url_model_ms" in result.timings_ms


def test_agent_works_without_a_model(settings):
    """No promoted model must not crash analysis."""
    from tests.conftest import StubRegistry

    class EmptyRegistry(StubRegistry):
        def load(self, config: str = "url_only"):
            return None

        def load_best(self):
            return None

    agent = PhishingAgent(
        settings,
        registry=EmptyRegistry(),
        sandbox=StubSandbox(benign_evidence()),
        reasoner=RuleBasedReasoner(),
    )
    result = agent.analyze("https://example.com", browse=False)
    assert result.model_score is None
    assert result.top_factors == []
    assert result.verdict in {"safe", "suspicious", "phishing"}


def test_agent_confidence_increases_with_evidence(settings):
    weak = _agent(benign_evidence(), url_score=0.4, settings=settings).analyze("https://example.com")
    strong = _agent(phishing_evidence(), url_score=0.95, settings=settings).analyze(
        "https://paypal-secure-login.com/signin"
    )
    assert strong.confidence >= weak.confidence
