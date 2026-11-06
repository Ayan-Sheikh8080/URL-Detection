"""Deterministic URL features shared by training and online inference."""

import ipaddress
import re
from urllib.parse import urlsplit


SUSPICIOUS_TLDS = {"click", "country", "gq", "link", "ml", "tk", "top", "work", "xyz"}
BRANDS = {
    "amazon", "apple", "bank", "facebook", "google", "microsoft",
    "netflix", "paypal",
}

URL_FEATURE_NAMES = (
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "symbol_count",
    "digit_count",
    "https",
    "has_ip_address",
    "subdomain_count",
    "path_depth",
    "query_parameter_count",
    "has_at_symbol",
    "has_double_slash_in_path",
    "has_punycode",
    "has_suspicious_tld",
    "has_brand_keyword",
    "has_brand_in_subdomain",
    "uses_shortener",
)

_SHORTENERS = {"bit.ly", "goo.gl", "tinyurl.com", "t.co", "short.link"}


def parse_http_url(url: str, *, allow_credentials: bool = False):
    """Parse an HTTP(S) URL, raising ValueError for malformed input."""
    if not isinstance(url, str) or not url.strip() or url != url.strip():
        raise ValueError("URL must be a non-empty string without surrounding whitespace")
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        if parsed.scheme.lower() not in {"http", "https"} or not hostname:
            raise ValueError
        if any(character.isspace() or ord(character) < 33 for character in hostname):
            raise ValueError
        port = parsed.port  # Force validation of malformed ports.
        if port == 0:
            raise ValueError
        hostname = hostname.encode("idna").decode("ascii").lower().rstrip(".")
    except (UnicodeError, ValueError):
        raise ValueError("URL must be a valid absolute HTTP or HTTPS URL") from None
    if not allow_credentials and (parsed.username is not None or parsed.password is not None):
        raise ValueError("URLs containing embedded credentials are not allowed")
    return parsed, hostname


def extract_url_features(url: str, *, allow_credentials: bool = False) -> dict[str, float]:
    """Extract the stable numeric feature schema used by every URL model."""
    parsed, hostname = parse_http_url(url, allow_credentials=allow_credentials)
    labels = hostname.split(".")
    try:
        ipaddress.ip_address(hostname)
        is_ip = True
    except ValueError:
        is_ip = False

    domain_labels = labels[-2:] if len(labels) >= 2 else labels
    registered_domain = ".".join(domain_labels)
    words = set(re.findall(r"[a-z0-9]+", f"{hostname} {parsed.path}".lower()))
    branded_domain = any(brand in words for brand in BRANDS)
    brand_subdomain = any(
        label in BRANDS and registered_domain.split(".")[0] != label
        for label in labels[:-2]
    )
    query_parameters = [item for item in parsed.query.split("&") if item]

    return {
        "url_length": float(len(url)),
        "hostname_length": float(len(hostname)),
        "path_length": float(len(parsed.path)),
        "query_length": float(len(parsed.query)),
        "symbol_count": float(sum(not char.isalnum() for char in url)),
        "digit_count": float(sum(char.isdigit() for char in url)),
        "https": float(parsed.scheme.lower() == "https"),
        "has_ip_address": float(is_ip),
        "subdomain_count": float(max(0, len(labels) - len(domain_labels))),
        "path_depth": float(len([part for part in parsed.path.split("/") if part])),
        "query_parameter_count": float(len(query_parameters)),
        "has_at_symbol": float("@" in url),
        "has_double_slash_in_path": float("//" in parsed.path),
        "has_punycode": float(any(label.startswith("xn--") for label in labels)),
        "has_suspicious_tld": float(labels[-1] in SUSPICIOUS_TLDS),
        "has_brand_keyword": float(branded_domain),
        "has_brand_in_subdomain": float(brand_subdomain),
        "uses_shortener": float(hostname in _SHORTENERS),
    }
