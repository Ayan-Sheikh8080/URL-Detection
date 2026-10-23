"""The phishing analysis agent: orchestrates tools and produces a verdict."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from src.agent.checks import cert_details, domain_checks
from src.agent.evidence import summarize
from src.agent.reasoning import get_reasoner
from src.agent.sandbox_client import SandboxClient
from src.config import Settings
from src.features.featurize import FEATURE_COLUMNS_BY_CONFIG
from src.features.page_features import extract_page_features
from src.features.url_features import extract_url_features
from src.ml.model_registry import ModelRegistry


@dataclass
class AnalysisResult:
    url: str
    verdict: str
    confidence: float
    reasons: list[str]
    model_score: float | None = None
    combined_score: float | None = None
    top_factors: list[dict[str, Any]] = field(default_factory=list)
    evidence_summary: dict[str, Any] = field(default_factory=dict)
    domain_checks: dict[str, Any] = field(default_factory=dict)
    browser: dict[str, Any] = field(default_factory=dict)
    reasoner: str = "rule_based"
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PhishingAgent:
    """Analyze one link end-to-end using the four proposal tools."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        registry: ModelRegistry | None = None,
        sandbox: SandboxClient | None = None,
        reasoner: Any | None = None,
    ):
        self.settings = settings or Settings.from_env()
        self.registry = registry if registry is not None else ModelRegistry(self.settings.model_dir)
        self.sandbox = sandbox or SandboxClient(
            self.settings.sandbox_url, timeout_seconds=self.settings.sandbox_timeout_seconds + 5
        )
        self.reasoner = reasoner or get_reasoner(self.settings)

    # ------------------------------------------------------------------ #
    def analyze(self, url: str, *, browse: bool = True) -> AnalysisResult:
        timings: dict[str, float] = {}

        # Tool 1: URL model
        t0 = time.monotonic()
        url_features = extract_url_features(url)
        url_model = self.registry.load("url_only") or self.registry.load_best()
        url_score = url_model.predict_proba(url_features) if url_model else None
        top_factors = url_model.top_factors(url_features) if url_model else []
        timings["url_model_ms"] = _ms(t0)

        # Tool 2: sandboxed browser visit
        t0 = time.monotonic()
        evidence = self.sandbox.visit(url) if browse else {
            "requested_url": url,
            "error": "browsing skipped (requested url-only analysis)",
        }
        timings["browser_ms"] = _ms(t0)

        # Tool 3: certificate + domain/brand checks
        t0 = time.monotonic()
        final_url = evidence.get("final_url") or url
        host = _host(final_url)
        if evidence.get("error") is None:
            evidence["cert"] = cert_details(host)
        else:
            evidence.setdefault("cert", {})
        domain_info = domain_checks(url, final_url, evidence.get("title") or "")
        evidence["domain_checks"] = domain_info
        timings["checks_ms"] = _ms(t0)

        # Combined model (content + URL) when page evidence is available
        combined_score = None
        combined_model = self.registry.load("combined")
        if combined_model is not None and combined_model is not url_model and evidence.get("error") is None:
            features = dict(url_features)
            features.update(extract_page_features(evidence))
            combined_score = combined_model.predict_proba(features)
            if not top_factors:
                top_factors = combined_model.top_factors(features)

        # Tool 4: reasoning layer
        t0 = time.monotonic()
        verdict = self.reasoner.reason(
            url_score=url_score,
            combined_score=combined_score,
            evidence=evidence,
            domain_info=domain_info,
            phishing_threshold=self.settings.phishing_threshold,
            suspicious_threshold=self.settings.suspicious_threshold,
        )
        timings["reasoning_ms"] = _ms(t0)
        timings["total_ms"] = round(sum(timings.values()), 2)

        return AnalysisResult(
            url=url,
            verdict=verdict.verdict,
            confidence=verdict.confidence,
            reasons=list(verdict.reasons),
            model_score=url_score,
            combined_score=combined_score,
            top_factors=top_factors,
            evidence_summary=summarize(evidence),
            domain_checks=domain_info,
            browser={
                "ok": evidence.get("error") is None,
                "error": evidence.get("error"),
                "load_ms": evidence.get("load_ms", 0.0),
            },
            reasoner=verdict.reasoner,
            timings_ms=timings,
        )


def _ms(start: float) -> float:
    return round((time.monotonic() - start) * 1000, 2)


def _host(url: str) -> str:
    from urllib.parse import urlparse

    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


__all__ = [
    "PhishingAgent",
    "AnalysisResult",
    "FEATURE_COLUMNS_BY_CONFIG",
]
