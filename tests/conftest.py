"""Shared pytest fixtures: stub sandbox, stub reasoner and stub model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.evidence import Evidence, FormEvidence  # noqa: E402
from src.agent.reasoning import RuleBasedReasoner  # noqa: E402
from src.agent.sandbox_client import SandboxClient  # noqa: E402
from src.config import Settings  # noqa: E402


class StubSandbox(SandboxClient):
    """Sandbox client returning fixed evidence (no browser involved)."""

    def __init__(self, evidence: dict | None = None):
        super().__init__("")
        self.evidence = evidence or benign_evidence()
        self.calls: list[str] = []

    def visit(self, url: str, *, screenshot: bool = True) -> dict:
        self.calls.append(url)
        payload = dict(self.evidence)
        payload.setdefault("requested_url", url)
        return payload


class StubModel:
    """Minimal registry model double with linear-style attribution."""

    def __init__(self, score: float = 0.1, feature_columns: list[str] | None = None):
        self.score = score
        self.name = "stub"
        self.config = "url_only"
        self.feature_columns = feature_columns or ["url_length", "has_https", "suspicious_keyword_count"]
        self.metrics: dict[str, float] = {"f1": 0.9}

    def predict_proba(self, features: dict) -> float:
        return float(self.score)

    def top_factors(self, features: dict, k: int = 5) -> list[dict]:
        return [
            {
                "feature": col,
                "value": float(features.get(col, 0.0)),
                "contribution": round(float(features.get(col, 0.0)) * 0.01, 4),
                "direction": "increases_risk",
            }
            for col in self.feature_columns[:k]
        ]


class StubRegistry:
    def __init__(self, url_score: float = 0.1, combined_score: float | None = None):
        self.url_model = StubModel(score=url_score)
        self.combined_model = StubModel(score=combined_score) if combined_score is not None else None

    def load(self, config: str = "url_only"):
        if config == "combined":
            return self.combined_model
        return self.url_model

    def load_best(self):
        return self.url_model

    def available_configs(self) -> list[str]:
        return ["url_only"] + (["combined"] if self.combined_model else [])


def benign_evidence(**overrides) -> dict:
    """Evidence for a clean, trusted page."""
    evidence = Evidence(
        requested_url="https://example.com/news",
        final_url="https://example.com/news",
        redirect_chain=["https://example.com/news"],
        status_code=200,
        title="Example News",
        text_sample="Today's headlines and analysis from the Example newsroom.",
        forms=[FormEvidence(action="https://example.com/search", method="get", input_names=["q"])],
        iframes=[],
        has_password_field=False,
        external_script_count=2,
        num_hidden_inputs=0,
        cert={
            "issuer": "Let's Encrypt",
            "subject_cn": "example.com",
            "days_to_expiry": 90,
            "self_signed": False,
            "host_matches": True,
        },
        domain_checks={"same_base_domain": True, "brand_mismatch": False, "homoglyph": False},
        error=None,
    ).to_dict()
    evidence.update(overrides)
    return evidence


def phishing_evidence(**overrides) -> dict:
    """Evidence for a clear credential-harvesting page."""
    evidence = Evidence(
        requested_url="https://paypal-secure-login.com/signin",
        final_url="https://account-secure-verify.top/login",
        redirect_chain=[
            "https://paypal-secure-login.com/signin",
            "https://cdn-redirect.info/go",
            "https://account-secure-verify.top/login",
        ],
        status_code=200,
        title="PayPal - Verify Your Account",
        text_sample="Your account is limited. Confirm your password now.",
        forms=[
            FormEvidence(
                action="https://collect-data-server.icu/post.php",
                method="post",
                has_password=True,
                input_names=["email", "password"],
                submits_cross_domain=True,
            )
        ],
        iframes=["https://ads.tracker-js.ru/frame"],
        has_password_field=True,
        external_script_count=9,
        num_hidden_inputs=4,
        cert={
            "issuer": "R3",
            "subject_cn": "account-secure-verify.top",
            "days_to_expiry": 5,
            "self_signed": False,
            "host_matches": True,
        },
        domain_checks={
            "same_base_domain": False,
            "final_base": "account-secure-verify.top",
            "brand_shown": "paypal",
            "brand_mismatch": True,
            "homoglyph": False,
            "suspicious_tld": True,
        },
        error=None,
    ).to_dict()
    evidence.update(overrides)
    return evidence


@pytest.fixture
def settings() -> Settings:
    return Settings(phishing_threshold=0.65, suspicious_threshold=0.35)


@pytest.fixture
def rule_reasoner() -> RuleBasedReasoner:
    return RuleBasedReasoner()


@pytest.fixture
def benign() -> dict:
    return benign_evidence()


@pytest.fixture
def phishing() -> dict:
    return phishing_evidence()


@pytest.fixture
def stubbed_agent(settings):
    from src.agent.agent import PhishingAgent

    sandbox = StubSandbox(phishing_evidence())
    return PhishingAgent(
        settings,
        registry=StubRegistry(url_score=0.8),
        sandbox=sandbox,
        reasoner=RuleBasedReasoner(),
    ), sandbox


def random_matrix(n: int = 20, features: int = 4) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((n, features))
