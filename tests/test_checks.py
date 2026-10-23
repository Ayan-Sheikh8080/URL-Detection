"""Tests for certificate/domain checks and the safety policy."""

from __future__ import annotations

import pytest

from src.agent.browser_sandbox import is_private_host, is_visitable_url
from src.agent.checks import domain_checks


# --------------------------------------------------------------------------- #
# Domain / brand checks
# --------------------------------------------------------------------------- #
def test_same_domain_is_not_flagged():
    info = domain_checks("https://example.com/a", "https://example.com/b", "Example News")
    assert info["same_base_domain"] is True
    assert info["brand_mismatch"] is False
    assert info["reasons"] == []


def test_cross_domain_redirect_flagged():
    info = domain_checks("https://paypal.com/login", "https://evil-collect.top/login", "")
    assert info["same_base_domain"] is False
    assert any("different domain" in r for r in info["reasons"])


def test_brand_mismatch_flagged():
    info = domain_checks(
        "https://x.top/login", "https://x.top/login", "PayPal - Verify Your Account"
    )
    assert info["brand_mismatch"] is True
    assert any("paypal" in r.lower() for r in info["reasons"])


def test_subdomain_of_brand_is_not_mismatch():
    info = domain_checks(
        "https://www.paypal.com/signin", "https://www.paypal.com/signin", "Log in to PayPal"
    )
    assert info["brand_mismatch"] is False


def test_suspicious_tld_flagged():
    info = domain_checks("https://secure-login.xyz/a", "https://secure-login.xyz/a", "")
    assert info["suspicious_tld"] is True


def test_homoglyph_punycode_flagged():
    info = domain_checks("https://xn--pypal-4ve.com/login", "https://xn--pypal-4ve.com/login", "")
    assert info["homoglyph"] is True


def test_domain_checks_handles_empty_final_url():
    info = domain_checks("https://example.com", "", "")
    assert info["original_base"] == "example.com"


# --------------------------------------------------------------------------- #
# Sandbox safety policy
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "host",
    ["localhost", "127.0.0.1", "10.0.0.5", "192.168.1.10", "172.16.4.4", "169.254.169.254", "::1", "foo.local"],
)
def test_private_hosts_are_blocked(host):
    assert is_private_host(host) is True


@pytest.mark.parametrize("host", ["example.com", "8.8.8.8", "1.1.1.1"])
def test_public_hosts_allowed(host):
    assert is_private_host(host) is False


@pytest.mark.parametrize(
    "url,reason_fragment",
    [
        ("file:///etc/passwd", "scheme"),
        ("javascript:alert(1)", "scheme"),
        ("data:text/html,<h1>x</h1>", "scheme"),
        ("http://localhost:8080/admin", "private"),
        ("http://192.168.0.1/router", "private"),
        ("http://169.254.169.254/latest/meta-data/", "private"),
    ],
)
def test_unsafe_urls_rejected(url, reason_fragment):
    allowed, reason = is_visitable_url(url)
    assert allowed is False
    assert reason_fragment in reason


@pytest.mark.parametrize("url", ["https://example.com", "http://example.com/path?q=1"])
def test_safe_urls_allowed(url):
    allowed, _ = is_visitable_url(url)
    assert allowed is True


def test_private_blocking_can_be_relaxed_for_tests():
    allowed, _ = is_visitable_url("http://127.0.0.1:8000", block_private=False)
    assert allowed is True


def test_cert_details_on_unqualified_host_reports_error():
    from src.agent.checks import cert_details

    result = cert_details("localhost")
    assert result["error"] is not None
    assert result["days_to_expiry"] is None


def test_cert_details_never_raises_for_unresolvable_host():
    from src.agent.checks import cert_details

    result = cert_details("definitely-not-a-real-domain-xyz-12345.example")
    assert result["error"] is not None
