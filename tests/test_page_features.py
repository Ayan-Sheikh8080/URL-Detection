"""Tests for page-evidence feature extraction and graceful degradation."""

from __future__ import annotations

import pandas as pd

from src.features.featurize import combined_feature_frame
from src.features.page_features import PAGE_FEATURE_COLUMNS, extract_page_features


def test_feature_columns_complete(benign):
    features = extract_page_features(benign)
    assert set(features) == set(PAGE_FEATURE_COLUMNS)
    assert features["browser_ok"] == 1.0
    assert features["has_password_field"] == 0.0


def test_phishing_evidence_raises_signals(phishing):
    features = extract_page_features(phishing)
    assert features["has_password_field"] == 1.0
    assert features["cross_domain_form"] == 1.0
    assert features["cross_domain_redirect"] == 1.0
    assert features["final_host_matches_original"] == 0.0
    assert features["brand_mismatch"] == 1.0
    assert features["redirect_count"] == 2.0


def test_browser_failure_degrades_gracefully():
    features = extract_page_features({"requested_url": "https://x.com", "error": "TimeoutError: 30000ms"})
    assert features["browser_ok"] == 0.0
    assert features["redirect_count"] == 0.0
    assert features["cert_days_to_expiry"] > 0


def test_missing_certificate_uses_neutral_default():
    features = extract_page_features({"final_url": "http://x.com", "cert": {}})
    assert features["cert_days_to_expiry"] > 0
    assert features["cert_host_mismatch"] == 0.0


def test_expired_certificate_flagged():
    evidence = {"final_url": "https://x.com", "cert": {"days_to_expiry": -3}}
    assert extract_page_features(evidence)["cert_days_to_expiry"] < 0


def test_http_final_url_flagged():
    assert extract_page_features({"final_url": "http://x.com/login"})["final_uses_http"] == 1.0
    assert extract_page_features({"final_url": "https://x.com/login"})["final_uses_http"] == 0.0


def test_combined_frame_has_both_feature_groups(benign):
    urls = pd.Series(["https://example.com/news"])
    frame = combined_feature_frame(urls, [benign])
    assert "url_length" in frame.columns
    assert "browser_ok" in frame.columns
    assert frame.shape[0] == 1


def test_combined_frame_length_mismatch_raises(benign):
    import pytest

    with pytest.raises(ValueError):
        combined_feature_frame(pd.Series(["https://a.com", "https://b.com"]), [benign])
