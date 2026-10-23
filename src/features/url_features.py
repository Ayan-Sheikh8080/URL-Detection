"""URL-based feature extraction (feature set 1 of the proposal).

Extracts lexical features from a URL string only: length, symbols, HTTPS
usage and domain characteristics. No network access is performed here.
"""

from __future__ import annotations

import math
import re
from urllib.parse import urlparse

IPV4_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")

SUSPICIOUS_TLDS = {
    "xyz", "top", "icu", "click", "link", "live", "support", "info", "rest", "cfd", "monster", "quest", "skin", "loan", "trade", "win",
}

SHORTENER_HOSTS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "cutt.ly", "rebrand.ly", "shorturl.at", "rb.gy", "s.id", "tiny.cc",
}

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "sign-in", "verify", "verification", "update",
    "secure", "security", "account", "banking", "password", "confirm",
    "suspend", "limited", "unlock", "wallet", "invoice", "paypal",
    "recover", "session", "authorize", "authenticate",
]

URL_FEATURE_COLUMNS = [
    "url_length",
    "host_length",
    "path_length",
    "query_length",
    "count_dots",
    "count_hyphens",
    "count_underscores",
    "count_at",
    "count_question",
    "count_equals",
    "count_ampersand",
    "count_percent",
    "count_slash",
    "count_digits",
    "digit_ratio",
    "has_ip_host",
    "has_https",
    "has_double_slash_redirect",
    "num_subdomains",
    "tld_length",
    "is_shortened",
    "suspicious_keyword_count",
    "has_suspicious_tld",
    "host_entropy",
    "has_userinfo",
    "has_port",
    "is_punycode",
]


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    freq: dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


def registrable_domain(host: str) -> str:
    """Approximate registrable (base) domain without a PSL dependency.

    Handles a small set of common two-part public suffixes (co.uk, com.au...).
    """
    host = (host or "").strip().lower().rstrip(".")
    if not host:
        return ""
    if IPV4_RE.match(host):
        return host
    labels = [label for label in host.split(".") if label]
    if len(labels) <= 2:
        return host
    two_part_suffixes = {
        "co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au", "org.au",
        "co.in", "com.br", "com.cn", "com.pk", "com.tr", "co.jp", "co.kr",
    }
    last_two = ".".join(labels[-2:])
    if last_two in two_part_suffixes and len(labels) >= 3:
        return ".".join(labels[-3:])
    return last_two


def extract_url_features(url: str) -> dict[str, float]:
    """Extract lexical features from a single URL string."""
    url = (url or "").strip()
    parsed = urlparse(url)
    host = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    try:
        port = parsed.port
    except ValueError:
        port = None

    host_labels = [label for label in host.split(".") if label]
    tld = host_labels[-1] if host_labels else ""
    digits = sum(c.isdigit() for c in url)
    lowered = url.lower()

    features: dict[str, float] = {
        "url_length": float(len(url)),
        "host_length": float(len(host)),
        "path_length": float(len(path)),
        "query_length": float(len(query)),
        "count_dots": float(url.count(".")),
        "count_hyphens": float(url.count("-")),
        "count_underscores": float(url.count("_")),
        "count_at": float(url.count("@")),
        "count_question": float(url.count("?")),
        "count_equals": float(url.count("=")),
        "count_ampersand": float(url.count("&")),
        "count_percent": float(url.count("%")),
        "count_slash": float(url.count("/")),
        "count_digits": float(digits),
        "digit_ratio": round(digits / len(url), 4) if url else 0.0,
        "has_ip_host": 1.0 if IPV4_RE.match(host) else 0.0,
        "has_https": 1.0 if parsed.scheme.lower() == "https" else 0.0,
        "has_double_slash_redirect": 1.0 if "//" in path else 0.0,
        "num_subdomains": float(max(len(host_labels) - 2, 0)),
        "tld_length": float(len(tld)),
        "is_shortened": 1.0 if host in SHORTENER_HOSTS else 0.0,
        "suspicious_keyword_count": float(sum(k in lowered for k in SUSPICIOUS_KEYWORDS)),
        "has_suspicious_tld": 1.0 if tld in SUSPICIOUS_TLDS else 0.0,
        "host_entropy": round(_entropy(host), 4),
        "has_userinfo": 1.0 if "@" in (parsed.netloc or "").split("@")[0] and "@" in (parsed.netloc or "") else 0.0,
        "has_port": 1.0 if port not in (None, 80, 443) else 0.0,
        "is_punycode": 1.0 if "xn--" in host else 0.0,
    }
    return features


def features_to_vector(features: dict[str, float]) -> list[float]:
    """Order a feature dict into the canonical URL feature vector."""
    return [float(features.get(col, 0.0)) for col in URL_FEATURE_COLUMNS]
