"""Convert browser evidence into stable numeric model features."""

import re
from urllib.parse import urljoin, urlsplit

from src.agent.checks import check_for_prompt_injection
from src.ml.features import BRANDS, parse_http_url


CONTENT_FEATURE_NAMES = (
    "forms_count",
    "password_fields_count",
    "external_form_count",
    "iframes_count",
    "external_iframe_count",
    "redirect_count",
    "page_text_length",
    "title_length",
    "certificate_checked",
    "certificate_valid",
    "is_https",
    "brand_mismatch",
    "prompt_injection_detected",
)
CONTENT_MODEL_FEATURE_NAMES = (
    "has_password_field",
    "has_external_form",
    "iframes_count",
    "redirect_count",
    "title_length",
    "is_https",
)
MAX_PAGE_TITLE_LENGTH = 300
MAX_IFRAME_COUNT = 100


def extract_content_features(url: str, evidence: dict) -> dict[str, float]:
    parsed, original_host = parse_http_url(url)
    final_url = evidence.get("final_url") or url
    try:
        _, final_host = parse_http_url(final_url)
    except ValueError:
        final_host = original_host

    forms = evidence.get("forms") or []
    password_fields = 0
    external_forms = 0
    for form in forms:
        for field in form.get("inputs") or []:
            if str(field.get("type", "")).lower() == "password":
                password_fields += 1
        action = form.get("action") or ""
        if action:
            action_url = urljoin(final_url, action)
            try:
                _, action_host = parse_http_url(action_url)
            except ValueError:
                continue
            if action_host != final_host:
                external_forms += 1

    frames = (evidence.get("iframes") or [])[:MAX_IFRAME_COUNT]
    external_frames = 0
    for frame in frames:
        source = frame.get("src") or ""
        if not source:
            continue
        frame_url = urljoin(final_url, source)
        try:
            _, frame_host = parse_http_url(frame_url)
        except ValueError:
            continue
        if frame_host != final_host:
            external_frames += 1

    text = str(evidence.get("page_text") or "")[:5000]
    title = str(evidence.get("title") or "")[:MAX_PAGE_TITLE_LENGTH]
    text_words = set(re.findall(r"[a-z0-9]+", f"{title} {text}".lower()))
    claimed_brands = text_words.intersection(BRANDS)
    has_brand_mismatch = any(
        brand in claimed_brands and brand not in final_host.split(".")
        for brand in BRANDS
    )
    certificate = evidence.get("certificate") or {}
    redirects = evidence.get("redirect_chain") or []
    injection = check_for_prompt_injection(text)

    return {
        "forms_count": float(len(forms)),
        "password_fields_count": float(password_fields),
        "has_password_field": float(password_fields > 0),
        "external_form_count": float(external_forms),
        "has_external_form": float(external_forms > 0),
        "iframes_count": float(len(frames)),
        "external_iframe_count": float(external_frames),
        "redirect_count": float(max(0, len(redirects) - 1)),
        "page_text_length": float(len(text)),
        "title_length": float(len(title)),
        "certificate_checked": float("is_valid" in certificate),
        "certificate_valid": float(bool(certificate.get("is_valid", False))),
        "is_https": float(parsed.scheme.lower() == "https"),
        "brand_mismatch": float(has_brand_mismatch),
        "prompt_injection_detected": float(injection["prompt_injection_detected"]),
    }
