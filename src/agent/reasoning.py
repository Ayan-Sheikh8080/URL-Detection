from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class Verdict(Enum):
    SAFE = "safe"
    SUSPICIOUS = "suspicious" 
    PHISHING = "phishing"

@dataclass
class ReasoningInput:
    """Input to the reasoning layer."""
    url: str
    url_model_score: float  # 0.0 to 1.0, where 1 is most likely phishing
    evidence: Any  # EvidenceSummary dataclass
    page_text: str = ""
    certificate_info: Dict = None

@dataclass
class ReasoningOutput:
    """Output from the reasoning layer."""
    verdict: Verdict
    confidence: float  # 0.0 to 1.0
    reasons: List[str]
    top_factors: List[Dict[str, str]]  # List of {feature, contribution, direction}
    model_score: float
    combined_score: float
    reasoner: str  # "rule_based" or "llm_based"

class RuleBasedReasoner:
    """Deterministic rule-based reasoning layer."""
    
    def __init__(self, url_weight: float = 0.4, evidence_weight: float = 0.6):
        self.url_weight = url_weight
        self.evidence_weight = evidence_weight
    
    def reason(self, input_data: ReasoningInput) -> ReasoningOutput:
        """Produce verdict and reasons using rule-based logic."""
        
        # Start with URL model score
        url_score = input_data.url_model_score
        combined_score = self.url_weight * url_score
        
        reasons = []
        top_factors = []
        
        # Check evidence
        evidence = input_data.evidence
        if evidence is None:
            evidence = type('Evidence', (), {'forms': [], 'iframes': [], 'final_domain': '', 
                                           'page_text': '', 'certificate': type('Cert', (), {'is_valid': True})()})()
        
        # Rule 1: Forms with password fields submitting to different domain
        if evidence.forms:
            has_password = any(
                'password' in inp.get('type', '').lower() 
                for form in evidence.forms 
                for inp in form.get('inputs', [])
            )
            if has_password:
                combined_score += self.evidence_weight * 0.3
                reasons.append("Form with password field detected")
                top_factors.append({"feature": "password_form", "contribution": "0.3", "direction": "increases_risk"})
        
        # Rule 2: Domain mismatch between original and final
        if evidence.domain_mismatch if hasattr(evidence, 'domain_mismatch') else False:
            combined_score += self.evidence_weight * 0.25
            reasons.append("Final domain differs from original link")
            top_factors.append({"feature": "domain_mismatch", "contribution": "0.25", "direction": "increases_risk"})
        
        # Rule 3: External form submission (form submits to different domain)
        if evidence.get('forms_submitting_to_different_domain', 0) > 0:
            combined_score += self.evidence_weight * 0.2
            reasons.append("Form submits to different domain")
            top_factors.append({"feature": "external_form_submit", "contribution": "0.2", "direction": "increases_risk"})
        
        # Rule 4: Suspicious TLD
        if evidence.final_domain:
            from src.agent.checks import extract_domain_features
            domain_features = extract_domain_features(evidence.final_domain, input_data.url)
            if domain_features.get('tld_suspicious', False):
                combined_score += self.evidence_weight * 0.15
                reasons.append("Suspicious top-level domain")
                top_factors.append({"feature": "suspicious_tld", "contribution": "0.15", "direction": "increases_risk"})
        
        # Rule 5: Certificate invalid
        cert = getattr(evidence, 'certificate', None)
        if cert and not cert.is_valid:
            combined_score += self.evidence_weight * 0.25
            reasons.append("Invalid TLS certificate")
            top_factors.append({"feature": "invalid_certificate", "contribution": "0.25", "direction": "increases_risk"})
        
        # Rule 6: Page text has suspicious patterns (prompt injection check)
        page_text = input_data.page_text
        if page_text:
            from src.agent.checks import check_for_prompt_injection
            injection_features = check_for_prompt_injection(page_text)
            if injection_features.get('prompt_injection_detected', False):
                combined_score += self.evidence_weight * 0.1
                reasons.append("Suspicious page content patterns detected")
        
        # Ensure combined score stays in [0, 1]
        combined_score = max(0.0, min(1.0, combined_score))
        
        # Determine verdict based on threshold
        # Thresholds: < 0.3 = safe, 0.3-0.7 = suspicious, > 0.7 = phishing
        if combined_score >= 0.7:
            verdict = Verdict.PHISHING
        elif combined_score >= 0.3:
            verdict = Verdict.SUSPICIOUS
        else:
            verdict = Verdict.SAFE
        
        # Generate confidence based on score magnitude
        confidence = combined_score
        
        # If URL model strongly indicates one way, respect that
        if url_score > 0.8 and combined_score < 0.5:
            # URL says phishing but evidence says safe - trust URL model more
            verdict = Verdict.PHISHING
            confidence = url_score
        elif url_score < 0.2 and combined_score > 0.5:
            # URL says safe but evidence says phishing - trust evidence
            verdict = Verdict.PHISHING
            confidence = combined_score
        
        return ReasoningOutput(
            verdict=verdict,
            confidence=confidence,
            reasons=reasons,
            top_factors=top_factors,
            model_score=url_score,
            combined_score=combined_score,
            reasoner="rule_based"
        )


class LLMBasedReasoner:
    """LLM-based reasoning layer (Amazon Bedrock or similar)."""
    
    def __init__(self, llm_provider: str = "bedrock", model_id: str = "claude-3-haiku-20240307"):
        self.llm_provider = llm_provider
        self.model_id = model_id
        # In production, would initialize LLM client here
        self.initialized = False
    
    def reason(self, input_data: ReasoningInput) -> ReasoningOutput:
        """Produce verdict using LLM reasoning."""
        # Placeholder - in production would call LLM
        # For now, fall back to rule-based
        from .checks import check_for_prompt_injection
        
        # At minimum check for prompt injection
        if input_data.page_text:
            injection_features = check_for_prompt_injection(input_data.page_text)
            if injection_features.get('prompt_injection_detected', False):
                return ReasoningOutput(
                    verdict=Verdict.PHISHING,
                    confidence=0.8,
                    reasons=["Prompt injection detected in page content - verdict forced to phishing"],
                    top_factors=[{"feature": "prompt_injection", "contribution": "0.8", "direction": "increases_risk"}],
                    model_score=input_data.url_model_score,
                    combined_score=0.8,
                    reasoner="llm_based_fallback"
                )
        
        # Would call LLM in production
        # Return rule-based as fallback
        reasoner = RuleBasedReasoner()
        return reasoner.reason(input_data)


# Factory function to get the appropriate reasoner
def get_reasoner(provider: str = "rule_based") -> object:
    """Get a reasoning instance based on the configured provider."""
    if provider in ("rule_based", "rules", "default"):
        from .reasoning import RuleBasedReasoner
        return RuleBasedReasoner()
    elif provider in ("bedrock", "llm", "amazon"):
        from .reasoning import LLMBasedReasoner
        return LLMBasedReasoner(llm_provider="bedrock")
    else:
        from .reasoning import RuleBasedReasoner
        return RuleBasedReasoner()