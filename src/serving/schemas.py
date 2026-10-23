"""Request/response schemas for the analysis API."""

from __future__ import annotations

from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator


class AnalyzeRequest(BaseModel):
    url: str = Field(..., description="Link to analyze", max_length=2048)
    browse: bool = Field(
        default=True,
        description="Visit the link in the sandboxed browser (set false for URL-only scoring)",
    )

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError("url must not be empty")
        try:
            parsed = urlparse(value)
        except ValueError as exc:
            raise ValueError(f"unparseable url: {exc}") from exc
        if parsed.scheme.lower() not in {"http", "https"}:
            raise ValueError("only http and https URLs are supported")
        if not parsed.hostname:
            raise ValueError("url must include a hostname")
        return value


class VerdictFactors(BaseModel):
    feature: str
    value: float
    contribution: float
    direction: str


class AnalyzeResponse(BaseModel):
    url: str
    verdict: str
    confidence: float
    reasons: list[str]
    model_score: float | None = None
    combined_score: float | None = None
    top_factors: list[VerdictFactors] = []
    evidence_summary: dict = {}
    domain_checks: dict = {}
    browser: dict = {}
    reasoner: str
    timings_ms: dict = {}


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    configs: list[str]
    reasoner: str
    sandbox: str
