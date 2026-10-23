"""Tests for URL feature extraction."""

from __future__ import annotations

import pytest

from src.features.url_features import (
    URL_FEATURE_COLUMNS,
    extract_url_features,
    features_to_vector,
    registrable_domain,
)


def test_feature_columns_are_complete_and_numeric():
    features = extract_url_features("https://www.example.com/path?a=1")
    assert set(features) == set(URL_FEATURE_COLUMNS)
    assert all(isinstance(v, float) for v in features.values())


def test_vector_order_is_stable():
    features = extract_url_features("https://example.com")
    vector = features_to_vector(features)
    assert len(vector) == len(URL_FEATURE_COLUMNS)
    assert vector[URL_FEATURE_COLUMNS.index("has_https")] == 1.0


def test_https_and_http_are_distinguished():
    assert extract_url_features("https://example.com")["has_https"] == 1.0
    assert extract_url_features("http://example.com")["has_https"] == 0.0


def test_ip_host_and_port_detected():
    features = extract_url_features("http://192.168.0.10:8443/login")
    assert features["has_ip_host"] == 1.0
    assert features["has_port"] == 1.0


def test_shortener_and_suspicious_tld_flags():
    assert extract_url_features("https://bit.ly/abc")["is_shortened"] == 1.0
    assert extract_url_features("https://secure-verify.top/login")["has_suspicious_tld"] == 1.0


def test_suspicious_keyword_counting():
    features = extract_url_features("https://x.com/login/verify-account/secure/update")
    assert features["suspicious_keyword_count"] >= 3
    clean = extract_url_features("https://news.example.com/sports")
    assert clean["suspicious_keyword_count"] == 0


def test_punycode_detected():
    assert extract_url_features("https://xn--pypal-4ve.com/login")["is_punycode"] == 1.0


def test_long_phishing_url_scores_higher_than_short_clean_url():
    phishing = extract_url_features(
        "https://paypal.com.secure-login-verify.account-update.top/"
        "sessionid/9981237/login.php?token=abc&user=91&verify=1"
    )
    clean = extract_url_features("https://www.wikipedia.org/")
    assert phishing["url_length"] > clean["url_length"]
    assert phishing["suspicious_keyword_count"] > clean["suspicious_keyword_count"]


@pytest.mark.parametrize(
    "host,expected",
    [
        ("www.example.com", "example.com"),
        ("a.b.example.co.uk", "example.co.uk"),
        ("example.com", "example.com"),
        ("192.168.0.1", "192.168.0.1"),
    ],
)
def test_registrable_domain(host, expected):
    assert registrable_domain(host) == expected


def test_malformed_url_does_not_raise():
    features = extract_url_features("not a url")
    assert set(features) == set(URL_FEATURE_COLUMNS)
    assert features["has_https"] == 0.0
