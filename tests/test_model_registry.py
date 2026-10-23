"""Tests for the model registry: promotion, scoring and factor attribution."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.model_registry import LoadedModel, ModelRegistry


def _linear_model(columns, weights):
    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    rng = np.random.default_rng(0)
    X = rng.random((40, len(columns)))
    y = (X @ np.array(weights) > np.mean(X @ np.array(weights))).astype(int)
    pipeline.fit(X, y)
    return pipeline


def _write_index(path, artifact, columns, name="logistic_regression", config="url_only"):
    payload = {
        "models": {
            config: {
                "model_name": name,
                "artifact": artifact,
                "feature_columns": columns,
                "metrics": {"f1": 0.9},
            }
        },
        "best": {"run": f"{config}__{name}", "metric": "f1", "value": 0.9},
    }
    (path / "index.json").write_text(json.dumps(payload), encoding="utf-8")


def test_load_returns_none_when_index_missing(tmp_path):
    registry = ModelRegistry(tmp_path)
    assert registry.load("url_only") is None
    assert registry.load_best() is None
    assert registry.available_configs() == []


def test_load_returns_none_when_artifact_missing(tmp_path):
    _write_index(tmp_path, "missing.joblib", ["a", "b"])
    assert ModelRegistry(tmp_path).load("url_only") is None


def test_promoted_model_scores_and_reports_columns(tmp_path):
    columns = ["url_length", "has_https"]
    joblib.dump(_linear_model(columns, [1.0, -1.0]), tmp_path / "m.joblib")
    _write_index(tmp_path, "m.joblib", columns)

    registry = ModelRegistry(tmp_path)
    model = registry.load("url_only")
    assert model is not None
    score = model.predict_proba({"url_length": 150.0, "has_https": 0.0})
    assert 0.0 <= score <= 1.0


def test_missing_features_default_to_zero(tmp_path):
    columns = ["url_length", "has_https"]
    joblib.dump(_linear_model(columns, [1.0, 1.0]), tmp_path / "m.joblib")
    _write_index(tmp_path, "m.joblib", columns)
    model = ModelRegistry(tmp_path).load("url_only")
    assert 0.0 <= model.predict_proba({}) <= 1.0


def test_load_is_cached(tmp_path):
    columns = ["a", "b"]
    joblib.dump(_linear_model(columns, [1.0, 1.0]), tmp_path / "m.joblib")
    _write_index(tmp_path, "m.joblib", columns)
    registry = ModelRegistry(tmp_path)
    assert registry.load("url_only") is registry.load("url_only")


def test_load_best_prefers_url_only(tmp_path):
    for config in ("url_only", "combined"):
        columns = ["a", "b"]
        joblib.dump(_linear_model(columns, [1.0, 1.0]), tmp_path / f"{config}.joblib")
    index = {
        "models": {
            "combined": {
                "model_name": "rf",
                "artifact": "combined.joblib",
                "feature_columns": ["a", "b"],
                "metrics": {"f1": 0.99},
            },
            "url_only": {
                "model_name": "lr",
                "artifact": "url_only.joblib",
                "feature_columns": ["a", "b"],
                "metrics": {"f1": 0.8},
            },
        }
    }
    (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")
    best = ModelRegistry(tmp_path).load_best()
    assert best is not None
    assert best.config == "url_only"


def test_top_factors_are_ranked_by_absolute_contribution(tmp_path):
    columns = ["strong", "weak", "unused"]
    joblib.dump(_linear_model(columns, [5.0, 0.1, 0.0]), tmp_path / "m.joblib")
    _write_index(tmp_path, "m.joblib", columns)
    model = ModelRegistry(tmp_path).load("url_only")

    factors = model.top_factors({"strong": 10.0, "weak": 1.0, "unused": 0.0}, k=3)
    assert len(factors) == 3
    assert factors[0]["feature"] == "strong"
    assert factors[0]["direction"] in {"increases_risk", "decreases_risk"}
    assert {"feature", "value", "contribution", "direction"} == set(factors[0])


def test_top_factors_support_tree_importances():
    from sklearn.ensemble import RandomForestClassifier

    class TreeWrapper:
        def __init__(self):
            self.feature_columns = ["a", "b", "c"]
            rng = np.random.default_rng(1)
            X = rng.random((30, 3))
            y = (X[:, 0] > 0.5).astype(int)
            self.model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
            self.name = "rf"
            self.config = "url_only"
            self.metrics = {}

        predict_proba = LoadedModel.predict_proba
        top_factors = LoadedModel.top_factors

    wrapper = TreeWrapper()
    factors = wrapper.top_factors({"a": 1.0, "b": 0.0, "c": 0.0}, k=2)
    assert len(factors) == 2
    assert factors[0]["feature"] == "a"


def test_reference_features_round_trip(tmp_path):
    import pandas as pd

    registry = ModelRegistry(tmp_path)
    assert registry.reference_features() is None
    pd.DataFrame({"url_length": [10.0, 20.0]}).to_csv(
        tmp_path / "reference_features.csv", index=False
    )
    reference = ModelRegistry(tmp_path).reference_features()
    assert reference is not None
    assert list(reference["url_length"]) == [10.0, 20.0]


def test_corrupt_index_does_not_crash(tmp_path):
    (tmp_path / "index.json").write_text("{not json", encoding="utf-8")
    registry = ModelRegistry(tmp_path)
    with pytest.raises(json.JSONDecodeError):
        registry.load("url_only")
