"""Page-evidence feature extraction (feature set 2 of the proposal).

Turns the structured evidence collected by the sandboxed browser (redirect
chain, forms, iframes, text, certificate, domain checks) into numeric
features for the content-only and combined models.
"""

from __future__ import annotations

from src.features.url_features import registrable_domain

PAGE_FEATURE_COLUMNS = [
    "redirect_count",
    "cross_domain_redirect",
    "final_host_matches_original",
    "has_password_field",
    "form_count",
    "cross_domain_form",
    "iframe_count",
    "external_script_count",
    "num_hidden_inputs",
    "text_length",
    "title_length",
    "title_login_keyword",
    "brand_mismatch",
    "final_uses_http",
    "cert_self_signed",
    "cert_host_mismatch",
    "cert_days_to_expiry",
    "browser_ok",
]

DEFAULT_CERT_DAYS = 397.0  # neutral fill when the certificate could not be read

_TITLE_KEYWORDS = ("login", "sign in", "verify", "account", "password", "update", "confirm")


def extract_page_features(evidence: dict) -> dict[str, float]:
    """Extract numeric features from an evidence dict produced by the sandbox.

    Missing or failed browser evidence degrades gracefully: features fall back
    to safe defaults and `browser_ok` records the failure so the model can
    treat the row accordingly.
    """
    error = evidence.get("error")
    browser_ok = 0.0 if error else 1.0

    requested = evidence.get("requested_url", "") or ""
    final = evidence.get("final_url", "") or ""
    final_host = _host(final)
    requested_host = _host(requested)

    redirect_chain = evidence.get("redirect_chain") or []
    forms = evidence.get("forms") or []
    iframes = evidence.get("iframes") or []
    cert = evidence.get("cert") or {}
    domain_info = evidence.get("domain_checks") or {}
    title = evidence.get("title") or ""
    text = evidence.get("text_sample") or ""

    cross_domain_form = 0.0
    for form in forms:
        action_host = _host(form.get("action", ""))
        if action_host and final_host and action_host != final_host:
            cross_domain_form = 1.0
            break

    cert_days = cert.get("days_to_expiry")
    if not isinstance(cert_days, (int, float)):
        cert_days = DEFAULT_CERT_DAYS
    cert_days = float(min(max(cert_days, -365), DEFAULT_CERT_DAYS))

    features: dict[str, float] = {
        "redirect_count": float(max(len(redirect_chain) - 1, 0)),
        "cross_domain_redirect": 1.0
        if redirect_chain and _chain_crosses_domains(redirect_chain)
        else 0.0,
        "final_host_matches_original": 1.0
        if final_host and requested_host
        and registrable_domain(final_host) == registrable_domain(requested_host)
        else 0.0,
        "has_password_field": 1.0 if evidence.get("has_password_field") else 0.0,
        "form_count": float(len(forms)),
        "cross_domain_form": cross_domain_form,
        "iframe_count": float(len(iframes)),
        "external_script_count": float(evidence.get("external_script_count") or 0),
        "num_hidden_inputs": float(evidence.get("num_hidden_inputs") or 0),
        "text_length": float(len(text)),
        "title_length": float(len(title)),
        "title_login_keyword": 1.0
        if any(k in title.lower() for k in _TITLE_KEYWORDS)
        else 0.0,
        "brand_mismatch": 1.0 if domain_info.get("brand_mismatch") else 0.0,
        "final_uses_http": 1.0
        if final and not final.lower().startswith("https://")
        else 0.0,
        "cert_self_signed": 1.0 if cert.get("self_signed") else 0.0,
        "cert_host_mismatch": 1.0 if cert.get("host_matches") is False else 0.0,
        "cert_days_to_expiry": cert_days,
        "browser_ok": browser_ok,
    }
    return features


def _host(url: str) -> str:
    from urllib.parse import urlparse

    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def _chain_crosses_domains(chain: list[str]) -> bool:
    domains = {registrable_domain(_host(u)) for u in chain if u}
    domains.discard("")
    return len(domains) > 1
