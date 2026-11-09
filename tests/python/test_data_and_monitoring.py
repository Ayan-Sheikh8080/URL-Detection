import csv

import pytest

from src.data import prepare
from src.data.snapshot import snapshot_dataset
from src.agent.features import CONTENT_MODEL_FEATURE_NAMES, extract_content_features
from src.monitoring.drift import check_feature_drift
from src.monitoring.metrics import MonitoringMetrics
from src.ml.features import URL_FEATURE_NAMES
from src.training.train import _model_input


def test_dvc_raw_snapshot_preserves_content_and_records_hash(tmp_path):
    source = tmp_path / "source.csv"
    snapshot = tmp_path / "versioned" / "snapshot.csv"
    source.write_text("url,label\nhttps://example.com,0\n", encoding="utf-8")

    metadata = snapshot_dataset(source, snapshot)

    assert snapshot.read_bytes() == source.read_bytes()
    assert metadata["size_bytes"] == source.stat().st_size
    assert len(metadata["sha256"]) == 64


def test_prepare_normalizes_deduplicates_and_reports_invalid_rows(tmp_path, monkeypatch):
    raw = tmp_path / "input.csv"
    with raw.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["url", "label", "source"])
        writer.writeheader()
        writer.writerows([
            {"url": "https://safe.example/", "label": 0, "source": "test"},
            {"url": "bad.example/path", "label": 1, "source": "test"},
            {"url": "https://bad.example/path", "label": 1, "source": "test"},
            {"url": "invalid url", "label": 1, "source": "test"},
        ])
    processed = tmp_path / "processed.csv"
    metadata = tmp_path / "metadata.json"
    monkeypatch.setattr(prepare, "PROCESSED_PATH", processed)
    monkeypatch.setattr(prepare, "METADATA_PATH", metadata)

    report = prepare.prepare_dataset(raw)

    assert report["input_rows"] == 4
    assert report["processed_rows"] == 2
    assert len(report["issues"]) == 2
    with processed.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[1]["url"] == "https://bad.example/path"


def test_prepare_rejects_conflicting_labels_for_same_normalized_url(tmp_path):
    raw = tmp_path / "input.csv"
    raw.write_text(
        "url,label,source\nhttps://same.example,0,test\nsame.example,1,test\n",
        encoding="utf-8",
    )

    with pytest.raises(prepare.DatasetValidationError, match="Conflicting labels"):
        prepare.prepare_dataset(raw)


def test_prepare_phiusiil_schema_maps_labels_and_shared_page_features(tmp_path, monkeypatch):
    raw = tmp_path / "phiusiil.csv"
    columns = ["URL", "label", *prepare.PHIUSIIL_PAGE_COLUMNS.values()]
    with raw.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows([
            {
                "URL": "https://paypal@evil.example/login",
                "label": 0,
                "HasPasswordField": 1,
                "HasExternalFormSubmit": 1,
                "NoOfiFrame": 200,
                "NoOfURLRedirect": 1,
                "Title": "A" * 350,
                "IsHTTPS": 1,
            },
            {
                "URL": "https://legitimate.example/",
                "label": 1,
                "HasPasswordField": 0,
                "HasExternalFormSubmit": 0,
                "NoOfiFrame": 0,
                "NoOfURLRedirect": 0,
                "Title": "Welcome",
                "IsHTTPS": 1,
            },
            {
                "URL": "https://paypal@evil.example/login",
                "label": 0,
                "HasPasswordField": 1,
                "HasExternalFormSubmit": 1,
                "NoOfiFrame": 200,
                "NoOfURLRedirect": 1,
                "Title": "A" * 350,
                "IsHTTPS": 1,
            },
        ])
    processed = tmp_path / "processed.csv"
    metadata = tmp_path / "metadata.json"
    monkeypatch.setattr(prepare, "PROCESSED_PATH", processed)
    monkeypatch.setattr(prepare, "METADATA_PATH", metadata)

    report = prepare.prepare_dataset(raw)

    assert report["processed_rows"] == 2
    assert report["duplicate_rows"] == 1
    assert report["issues"] == []
    assert report["label_distribution"] == {"1": 1, "0": 1}
    with processed.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    phishing = next(row for row in rows if row["label"] == "1")
    assert phishing["has_password_field"] == "1.0"
    assert phishing["has_external_form"] == "1.0"
    assert float(phishing["iframes_count"]) == 100
    assert phishing["title_length"] == "300.0"


def test_live_content_features_use_the_shared_phiusiil_schema():
    features = extract_content_features(
        "https://bank.example/login",
        {
            "final_url": "https://bank.example/login",
            "title": "L" * 400,
            "forms": [{
                "action": "https://other.example/submit",
                "inputs": [{"type": "password"}],
            }],
            "iframes": [{"src": ""}] * 101,
            "redirect_chain": ["https://old.example/", "https://bank.example/login"],
        },
    )

    assert features["has_password_field"] == 1
    assert features["has_external_form"] == 1
    assert features["iframes_count"] == 100
    assert features["redirect_count"] == 1
    assert features["title_length"] == 300
    assert features["is_https"] == 1


def test_combined_model_uses_only_runtime_available_content_features():
    row = {
        "url": "https://example.com/",
        "label": 0,
        "has_password_field": 0,
        "has_external_form": 0,
        "iframes_count": 0,
        "redirect_count": 0,
        "title_length": 0,
        "is_https": 1,
    }

    matrix, labels, feature_order = _model_input([row], "combined")

    expected_order = [f"url_{name}" for name in URL_FEATURE_NAMES] + [
        f"page_{name}" for name in CONTENT_MODEL_FEATURE_NAMES
    ]
    assert feature_order == expected_order
    assert list(matrix.columns) == expected_order
    assert len(labels) == 1


def test_drift_detects_distribution_change():
    reference = [{"feature": 0.0} for _ in range(20)]
    recent = [{"feature": 1.0} for _ in range(20)]

    result = check_feature_drift(reference, recent)

    assert result["drift_detected"] is True
    assert result["psi_by_feature"]["feature"] > result["psi_threshold"]


def test_drift_keeps_features_when_some_historical_rows_lack_them():
    reference = [{"shared": 0.0, "new_feature": 0.0} for _ in range(20)]
    recent = [{"shared": 1.0}] + [
        {"shared": 1.0, "new_feature": 1.0} for _ in range(19)
    ]

    result = check_feature_drift(reference, recent)

    assert result["features_analyzed"] == 2
    assert result["psi_by_feature"]["new_feature"] > result["psi_threshold"]


def test_monitoring_persists_stats_and_recent_features(tmp_path):
    metrics = MonitoringMetrics(tmp_path / "events.jsonl")
    metrics.record_analysis({
        "verdict": "phishing",
        "model_score": 0.91,
        "load_ms": 125,
        "browser_ok": False,
        "features": {"url_length": 50},
    })

    restored = MonitoringMetrics(tmp_path / "events.jsonl")

    assert restored.get_stats()["total_analyses"] == 1
    assert restored.get_stats()["verdict_distribution"]["phishing"] == 1
    assert restored.get_stats()["browser_failure_rate"] == 100
    assert restored.recent_features() == [{"url_length": 50.0}]
