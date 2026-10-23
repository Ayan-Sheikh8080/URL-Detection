"""Structured evidence collected from a sandboxed page visit."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FormEvidence:
    action: str = ""
    method: str = "get"
    has_password: bool = False
    input_names: list[str] = field(default_factory=list)
    submits_cross_domain: bool = False


@dataclass
class Evidence:
    """Everything the sandboxed browser observed about a link."""

    requested_url: str = ""
    final_url: str = ""
    redirect_chain: list[str] = field(default_factory=list)
    status_code: int | None = None
    title: str = ""
    text_sample: str = ""
    forms: list[FormEvidence] = field(default_factory=list)
    iframes: list[str] = field(default_factory=list)
    has_password_field: bool = False
    external_script_count: int = 0
    num_hidden_inputs: int = 0
    screenshot_path: str = ""
    load_ms: float = 0.0
    cert: dict[str, Any] = field(default_factory=dict)
    domain_checks: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Evidence:
        forms = [FormEvidence(**f) for f in data.get("forms") or []]
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        known["forms"] = forms
        return cls(**known)

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.final_url)


def summarize(evidence: Evidence | dict) -> dict[str, Any]:
    """Compact evidence summary embedded in API responses and LLM prompts."""
    data = evidence.to_dict() if isinstance(evidence, Evidence) else dict(evidence)
    forms = data.get("forms") or []
    return {
        "final_url": data.get("final_url", ""),
        "redirect_count": max(len(data.get("redirect_chain") or []) - 1, 0),
        "status_code": data.get("status_code"),
        "title": (data.get("title") or "")[:200],
        "form_count": len(forms),
        "has_password_field": bool(data.get("has_password_field")),
        "cross_domain_form": any(f.get("submits_cross_domain") for f in forms),
        "iframe_count": len(data.get("iframes") or []),
        "cert_days_to_expiry": (data.get("cert") or {}).get("days_to_expiry"),
        "browser_error": data.get("error"),
    }
