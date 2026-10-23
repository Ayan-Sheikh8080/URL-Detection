"""Tests for the training/evaluation glue and the prepare stage."""

from __future__ import annotations

import pandas as pd
import pytest

from src.training.common import (
    binary_metrics,
    build_matrix,
    config_available,
    make_models,
    summarize_matrix,
)


# --------------------------------------------------------------------------- #
# Model factory
# --------------------------------------------------------------------------- #
def test_make_models_includes_core_families():
    models = make_models({"models": {"logistic_regression": {"max_iter": 100}, "random_forest": {"n_estimators": 5}}})
    assert "logistic_regression" in models
    assert "random_forest" in models
    for pipeline in models.values():
        assert "scaler" in pipeline.named_steps


def test_make_models_uses_params():
    models = make_models({"models": {"logistic_regression": {"C": 0.25, "max_iter": 77}}})
    clf = models["logistic_regression"].named_steps["clf"]
    assert clf.C == 0.25
    assert clf.max_iter == 77


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_binary_metrics_perfect_classifier():
    y = [0, 0, 1, 1]
    metrics = binary_metrics(y, y, [0.1, 0.2, 0.8, 0.9])
    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_binary_metrics_counts_misses():
    metrics = binary_metrics([0, 1, 1], [0, 0, 1])
    assert metrics["recall"] == 0.5
    assert metrics["precision"] == 1.0
    assert metrics["confusion_matrix"] == [[1, 0], [1, 1]]


def test_binary_metrics_handles_single_class_labels():
    metrics = binary_metrics([1, 1], [1, 1], [0.9, 0.8])
    assert metrics["roc_auc"] is None


# --------------------------------------------------------------------------- #
# Feature matrices
# --------------------------------------------------------------------------- #
def test_build_matrix_url_only():
    frame = build_matrix(pd.Series(["https://a.com", "https://b.com"]), None, "url_only")
    assert frame.shape[0] == 2
    assert "url_length" in frame.columns


def test_build_matrix_content_only_requires_evidence():
    with pytest.raises(ValueError):
        build_matrix(pd.Series(["https://a.com"]), None, "content_only")


def test_build_matrix_combined(benign):
    frame = build_matrix(pd.Series(["https://a.com"]), [benign], "combined")
    assert "url_length" in frame.columns
    assert "browser_ok" in frame.columns


def test_build_matrix_unknown_config_raises():
    with pytest.raises(ValueError):
        build_matrix(pd.Series(["https://a.com"]), None, "nope")


def test_summarize_matrix_fills_and_reorders():
    frame = pd.DataFrame({"b": [1.0], "a": [float("inf")]})
    result = summarize_matrix(frame, ["a", "b"])
    assert list(result.columns) == ["a", "b"]
    assert result.iloc[0]["a"] == 0.0


def test_summarize_matrix_missing_column_raises():
    with pytest.raises(ValueError):
        summarize_matrix(pd.DataFrame({"a": [1.0]}), ["a", "b"])


def test_config_availability_requires_enough_crawled_rows():
    assert config_available("url_only", []) is True
    assert config_available("content_only", []) is False
    assert config_available("combined", [{"i": 1}] * 5) is False
    assert config_available("combined", [{"i": 1}] * 10) is True


# --------------------------------------------------------------------------- #
# Prepare stage
# --------------------------------------------------------------------------- #
def test_load_raw_dataset_maps_columns(tmp_path):
    from src.data.prepare import load_raw_dataset

    path = tmp_path / "raw.csv"
    pd.DataFrame({"URL": ["https://a.com", "https://b.com"], "label": [1, 0]}).to_csv(path, index=False)
    df = load_raw_dataset(path, positive_is="phishing")
    assert list(df.columns) == ["url", "label"]
    assert df.iloc[0]["label"] == 1


def test_load_raw_dataset_flips_legitimate_positive(tmp_path):
    from src.data.prepare import load_raw_dataset

    path = tmp_path / "phiusiil.csv"
    pd.DataFrame({"URL": ["https://a.com", "https://b.com"], "label": [1, 0]}).to_csv(path, index=False)
    df = load_raw_dataset(path, positive_is="legitimate")
    assert df.iloc[0]["label"] == 0  # was legitimate -> not phishing
    assert df.iloc[1]["label"] == 1


def test_load_raw_dataset_rejects_unknown_label_convention(tmp_path):
    from src.data.prepare import load_raw_dataset

    path = tmp_path / "raw.csv"
    pd.DataFrame({"url": ["https://a.com"], "label": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_raw_dataset(path, positive_is="maybe")


def test_load_raw_dataset_missing_file():
    from src.data.prepare import load_raw_dataset

    with pytest.raises(FileNotFoundError):
        load_raw_dataset("does/not/exist.csv")


def test_load_raw_dataset_missing_columns(tmp_path):
    from src.data.prepare import load_raw_dataset

    path = tmp_path / "raw.csv"
    pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_raw_dataset(path)


def test_prepare_writes_split_files(tmp_path):
    from src.data.prepare import prepare

    raw = tmp_path / "raw.csv"
    rows = [{"url": f"https://site{i}.com/page", "label": i % 2} for i in range(40)]
    pd.DataFrame(rows).to_csv(raw, index=False)

    checks: dict = {}
    real_dir = None
    import src.data.prepare as prepare_module

    real_dir = prepare_module.PROCESSED_DIR
    prepare_module.PROCESSED_DIR = tmp_path / "processed"
    try:
        meta = prepare(
            raw_path=raw,
            positive_is="phishing",
            test_size=0.25,
            random_state=0,
            crawled_path=tmp_path / "none.jsonl",
        )
        checks = meta
        assert (prepare_module.PROCESSED_DIR / "train.csv").exists()
        assert (prepare_module.PROCESSED_DIR / "test.csv").exists()
    finally:
        prepare_module.PROCESSED_DIR = real_dir

    assert checks["raw_rows"] == 40
    assert checks["train_rows"] + checks["test_rows"] == 40
    assert checks["crawled"]["present"] is False


def test_prepare_dedupes_before_splitting(tmp_path):
    from src.data.prepare import prepare
    import src.data.prepare as prepare_module

    raw = tmp_path / "raw.csv"
    rows = [{"url": "https://dup.com/x", "label": 0}] * 10 + [
        {"url": f"https://u{i}.com", "label": 1} for i in range(10)
    ]
    pd.DataFrame(rows).to_csv(raw, index=False)

    real_dir = prepare_module.PROCESSED_DIR
    prepare_module.PROCESSED_DIR = tmp_path / "processed"
    try:
        meta = prepare(
            raw_path=raw,
            test_size=0.2,
            random_state=0,
            crawled_path=tmp_path / "none.jsonl",
        )
    finally:
        prepare_module.PROCESSED_DIR = real_dir

    assert meta["raw_rows"] == 20
    assert meta["rows_after_dedupe"] == 11
    assert meta["sha256"]
