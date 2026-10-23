"""Certificate and domain/brand checks (tool 3 of the agent design)."""

from __future__ import annotations

import datetime as dt
import fnmatch
import re
import socket
import ssl
from urllib.parse import urlparse

from src.features.url_features import SUSPICIOUS_TLDS, registrable_domain

# Brands frequently impersonated in phishing pages. If a brand is shown in the
# page title but absent from the final domain, that is a lookalike signal.
KNOWN_BRANDS = [
    "paypal", "google", "facebook", "instagram", "microsoft", "apple",
    "amazon", "netflix", "whatsapp", "outlook", "chase", "bankofamerica",
    "coinbase", "dropbox", "linkedin", "github", "steam",
]

IPV4_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")


def cert_details(hostname: str, port: int = 443, timeout: float = 5.0) -> dict:
    """Fetch and summarize the TLS certificate presented by `hostname`.

    Never raises: failures are reported in the returned dict so the agent can
    treat "no certificate evidence" as missing data rather than a crash.
    """
    result: dict = {
        "issuer": None,
        "subject_cn": None,
        "days_to_expiry": None,
        "self_signed": False,
        "host_matches": None,
        "error": None,
    }
    if not hostname or IPV4_RE.match(hostname) is None and "." not in hostname:
        # Nothing sensible to check for empty or single-label hosts.
        if not hostname or "." not in hostname:
            result["error"] = "unqualified hostname"
            return result
    try:
        context = ssl.create_default_context()
        with socket.create_connection((hostname, port), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
    except Exception as exc:  # noqa: BLE001 - any failure is missing evidence
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    if not cert:
        result["error"] = "no peer certificate"
        return result

    subject_cn = _first_cn(cert.get("subject"))
    issuer_cn = _first_cn(cert.get("issuer"))
    san_names = [v for k, v in cert.get("subjectAltName", ()) if k == "DNS"]

    not_after_raw = cert.get("notAfter")
    days_to_expiry = None
    if not_after_raw:
        try:
            not_after = dt.datetime.strptime(not_after_raw, "%b %d %H:%M:%S %Y %Z")
            days_to_expiry = (not_after - dt.datetime.utcnow()).days
        except ValueError:
            days_to_expiry = None

    result.update(
        issuer=issuer_cn,
        subject_cn=subject_cn,
        days_to_expiry=days_to_expiry,
        self_signed=bool(subject_cn and issuer_cn and subject_cn == issuer_cn),
        host_matches=_hostname_matches(hostname, san_names),
    )
    return result


def _first_cn(subject_entries) -> str | None:
    for rdn in subject_entries or ():
        for key, value in rdn:
            if key == "commonName":
                return value
    return None


def _hostname_matches(hostname: str, san_names: list[str]) -> bool:
    hostname = hostname.lower().rstrip(".")
    for name in san_names:
        name = name.lower().rstrip(".")
        if name.startswith("*."):
            # Wildcards only cover one label level (www.example.com, not a.b.c).
            if fnmatch.fnmatchcase(hostname, name.replace("*.", "*.")):
                suffix = name[1:]  # ".example.com"
                if hostname.endswith(suffix) and "." not in hostname[: -len(suffix)]:
                    return True
        elif name == hostname:
            return True
    return False


def domain_checks(original_url: str, final_url: str, page_title: str = "") -> dict:
    """Compare the final landing domain with the original link and page brand."""
    original_host = (urlparse(original_url).hostname or "").lower()
    final_host = (urlparse(final_url or original_url).hostname or "").lower()

    original_base = registrable_domain(original_host)
    final_base = registrable_domain(final_host)
    same_base = bool(original_base) and original_base == final_base

    title_lower = (page_title or "").lower()
    brand_shown = next((b for b in KNOWN_BRANDS if b in title_lower), None)
    brand_mismatch = bool(brand_shown) and brand_shown not in final_host.replace("-", "")

    homoglyph = "xn--" in final_host or any(ord(ch) > 127 for ch in final_host)
    final_tld = final_host.rsplit(".", 1)[-1] if "." in final_host else ""
    suspicious_tld = final_tld in SUSPICIOUS_TLDS

    reasons: list[str] = []
    if not same_base:
        reasons.append(
            f"The link redirected from {original_base or original_url} to a "
            f"different domain ({final_base or 'unknown'})."
        )
    if brand_mismatch:
        reasons.append(
            f'The page presents itself as "{brand_shown}" but the final domain '
            f"{final_base} does not belong to that brand (lookalike page)."
        )
    if homoglyph:
        reasons.append("The final domain uses internationalized/punycode characters (possible homoglyph spoof).")
    if suspicious_tld:
        reasons.append(f"The final domain uses a TLD commonly abused for phishing (.{final_tld}).")

    return {
        "original_host": original_host,
        "final_host": final_host,
        "original_base": original_base,
        "final_base": final_base,
        "same_base_domain": same_base,
        "brand_shown": brand_shown,
        "brand_mismatch": brand_mismatch,
        "homoglyph": homoglyph,
        "suspicious_tld": suspicious_tld,
        "reasons": reasons,
    }
