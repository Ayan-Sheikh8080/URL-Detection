"""Tests for dataset validation (missing, duplicate, invalid data)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.validate import (
    DataValidationError,
    assert_valid,
    validate_evidence_records,
    validate_urls,
)


def _frame(rows):
    return pd.DataFrame(rows, columns=["url", "label"])


def test_clean_dataset_passes():
    df = _frame(
        [
            ("https://a.com", 0),
            ("https://b.com/login", 1),
            ("http://c.org/x", 0),
        ]
    )
    report = validate_urls(df)
    assert report.ok
    assert report.stats["rows"] == 3


def test_empty_dataset_is_error():
    report = validate_urls(_frame([]))
    assert not report.ok
    assert "empty" in report.errors[0]


def test_missing_column_is_error():
    df = pd.DataFrame({"url": ["https://a.com"]})
    report = validate_urls(df)
    assert not report.ok
    assert any("missing required columns" in e for e in report.errors)


def test_missing_values_are_errors():
    df = _frame([("https://a.com", 0), (None, 1), ("https://c.com", None)])
    report = validate_urls(df)
    assert not report.ok
    assert sum("missing" in e for e in report.errors) == 2


def test_invalid_urls_are_errors():
    df = _frame([("https://a.com", 0), ("javascript:alert(1)", 1), ("ftp://x.com", 1)])
    report = validate_urls(df)
    assert not report.ok
    assert any("invalid URL" in e for e in report.errors)


def test_duplicates_are_warnings_not_errors():
    df = _frame([("https://a.com", 0), ("https://a.com", 0), ("https://b.com", 1)])
    report = validate_urls(df)
    assert report.ok
    assert any("duplicate" in w for w in report.warnings)


def test_invalid_labels_are_errors():
    df = _frame([("https://a.com", 2), ("https://b.com", 1)])
    report = validate_urls(df)
    assert not report.ok
    assert any("labels outside" in e for e in report.errors)


def test_single_class_is_error():
    df = _frame([("https://a.com", 1), ("https://b.com", 1)])
    report = validate_urls(df)
    assert not report.ok
    assert any("one class" in e for e in report.errors)


def test_class_imbalance_warns():
    rows = [("https://a.com", 0)] * 9 + [("https://b.com", 1)]
    report = validate_urls(_frame(rows))
    assert report.ok
    assert any("imbalance" in w for w in report.warnings)


def test_balanced_dataset_does_not_warn_about_imbalance():
    rows = [("https://a.com", 0), ("https://b.com", 0), ("https://c.com", 1), ("https://d.com", 1)]
    report = validate_urls(_frame(rows))
    assert report.ok
    assert not any("imbalance" in w for w in report.warnings)


def test_assert_valid_raises_with_report():
    report = validate_urls(_frame([]))
    with pytest.raises(DataValidationError):
        assert_valid(report)


def test_evidence_records_validation():
    good = [
        {"url": "https://a.com", "label": 0, "evidence": {"error": None}},
        {"url": "https://b.com", "label": 1, "evidence": {"error": None}},
    ]
    report = validate_evidence_records(good)
    assert report.ok

    bad = [{"url": "https://a.com", "label": 5}]
    report = validate_evidence_records(bad)
    assert not report.ok


def test_sample_dataset_in_repo_is_valid():
    from src.config import PROJECT_ROOT

    df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "sample_urls.csv")
    report = validate_urls(df)
    assert report.ok, report.summary()
