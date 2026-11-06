from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

@dataclass
class FormElement:
    """Represents a form element found on a page."""
    action: str = ""
    method: str = "GET"
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    action_domain: str = ""  # Domain form submits to

@dataclass
class IframeElement:
    """Represents an iframe element found on a page."""
    src: str = ""
    name: str = ""
    sandbox: str = ""

@dataclass
class CertificateInfo:
    """TLS certificate information."""
    is_valid: bool = False
    issuer: str = ""
    subject: str = ""
    fingerprint: str = ""
    expiration_date: str = ""

@dataclass
class EvidenceSummary:
    """Collected evidence from browser sandbox."""
    redirect_chain: List[str] = field(default_factory=list)
    final_domain: str = ""
    forms: List[FormElement] = field(default_factory=list)
    iframes: List[IframeElement] = field(default_factory=list)
    page_text: str = ""
    title: str = ""
    certificate: CertificateInfo = field(default_factory=CertificateInfo)
    forms_submitting_to_different_domain: int = 0
    password_fields: int = 0
    external_submission_forms: int = 0

@dataclass 
class AnalysisResult:
    """Final analysis result with verdict and reasons."""
    verdict: str = "safe"  # safe, suspicious, phishing
    confidence: float = 0.0
    reasons: List[str] = field(default_factory=list)
    model_score: float = 0.0
    combined_score: float = 0.0
    top_factors: List[Dict[str, Any]] = field(default_factory=list)
    evidence_summary: EvidenceSummary = field(default_factory=EvidenceSummary)
    browser_ok: bool = True
    load_ms: float = 0.0
    reasoner: str = "rule_based"