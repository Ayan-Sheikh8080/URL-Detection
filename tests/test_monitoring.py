"""Tests for monitoring metrics and drift detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import drift_report, population_stability_index
from src.monitoring.metrics import MetricsStore


def test_psi_is_zero_for_identical_distributions():
    values = list(range(50))
    assert population_stability_index(values, values) == 0.0


def test_psi_detects_large_shift():
    reference = [float(i) for i in range(100)]
    shifted = [float(i + 500) for i in range(100)]
    assert population_stability_index(reference, shifted) > 0.2


def test_psi_handles_empty_inputs():
    assert population_stability_index([], [1, 2, 3]) == 0.0
    assert population_stability_index([1, 2, 3], []) == 0.0


def test_psi_handles_constant_feature():
    assert population_stability_index([1, 1, 1], [1, 1, 1]) == 0.0


# --------------------------------------------------------------------------- #
# Metrics store
# --------------------------------------------------------------------------- #
def test_metrics_store_records_and_summarizes(tmp_path):
    store = MetricsStore(tmp_path / "events.jsonl")
    assert store.summary()["total_analyses"] == 0

    for verdict, latency, ok in [("phishing", 900, True), ("safe", 500, True), ("safe", 400, False)]:
        store.record(
            url="https://example.com",
            verdict=verdict,
            confidence=0.8,
            model_score=0.5,
            browser_ok=ok,
            latency_ms=latency,
            features={"url_length": 20.0},
        )

    summary = store.summary()
    assert summary["total_analyses"] == 3
    assert summary["verdicts"] == {"phishing": 1, "safe": 2}
    assert summary["latency_ms"]["max"] == 900
    assert summary["browser"]["failures"] == 1
    assert summary["browser"]["failure_rate"] == pytest.approx(1 / 3, abs=1e-3)


def test_metrics_store_ignores_corrupt_lines(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"url": "x", "verdict": "safe", "latency_ms": 1}\nnot json\n', encoding="utf-8")
    store = MetricsStore(path)
    assert len(store.events()) == 1


# --------------------------------------------------------------------------- #
# Drift report
# --------------------------------------------------------------------------- #
def test_drift_report_unavailable_without_reference(tmp_path):
    report = drift_report(
        reference_path=tmp_path / "missing.csv", events_path=tmp_path / "events.jsonl"
    )
    assert report["available"] is False


def test_drift_report_flags_shifted_feature(tmp_path):
    reference_path = tmp_path / "reference.csv"
    pd.DataFrame({"url_length": [float(i) for i in range(100)]}).to_csv(reference_path, index=False)

    events_path = tmp_path / "events.jsonl"
    store = MetricsStore(events_path)
    for i in range(40):
        store.record(
            url="https://example.com",
            verdict="safe",
            confidence=0.9,
            model_score=0.1,
            browser_ok=True,
            latency_ms=10.0,
            features={"url_length": float(i) + 900},
        )

    report = drift_report(reference_path=reference_path, events_path=events_path)
    assert report["available"] is True
    assert report["dataset_drift"] is True
    assert "url_length" in report["drifted_features"]


def test_drift_report_stable_when_distributions_match(tmp_path):
    reference_path = tmp_path / "reference.csv"
    values = [float(i % 50) for i in range(200)]
    pd.DataFrame({"url_length": values}).to_csv(reference_path, index=False)

    events_path = tmp_path / "events.jsonl"
    store = MetricsStore(events_path)
    # Draw a representative sample of the same distribution (not a biased
    # prefix), so PSI should see no drift.
    rng = np.random.default_rng(0)
    sample = rng.choice(values, size=120, replace=False)
    for value in sample:
        store.record(
            url="https://example.com",
            verdict="safe",
            confidence=0.9,
            model_score=0.1,
            browser_ok=True,
            latency_ms=10.0,
            features={"url_length": float(value)},
        )

    report = drift_report(reference_path=reference_path, events_path=events_path)
    assert report["available"] is True
    assert report["dataset_drift"] is False
