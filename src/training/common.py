"""Shared helpers for the training and evaluation stages."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PROJECT_ROOT
from src.data.prepare import load_crawled_evidence
from src.features.featurize import (
    FEATURE_COLUMNS_BY_CONFIG,
    combined_feature_frame,
    page_feature_frame,
    url_feature_frame,
)


def make_models(training_params: dict) -> dict[str, Pipeline]:
    """Instantiate the three model families from params.yaml."""
    params = (training_params or {}).get("models", {})
    lr_params = dict(params.get("logistic_regression", {}))
    rf_params = dict(params.get("random_forest", {}))
    xgb_params = dict(params.get("xgboost", {}))

    models: dict[str, Pipeline] = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(**lr_params)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(**rf_params)),
            ]
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", XGBClassifier(eval_metric="logloss", **xgb_params)),
            ]
        )
    except ImportError:  # optional dependency
        print("[train] xgboost not installed - skipping that model family")

    return models


def binary_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Precision / recall / F1 / accuracy / confusion matrix for the phishing class."""
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if y_proba is not None:
        try:
            auc = float(roc_auc_score(y_true, y_proba))
            # roc_auc_score returns nan (without raising) when y_true has one class.
            metrics["roc_auc"] = None if math.isnan(auc) else round(auc, 4)
        except ValueError:
            metrics["roc_auc"] = None
    return metrics


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
    test_path = PROJECT_ROOT / "data" / "processed" / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "processed dataset missing - run `python -m src.data.prepare` (dvc repro prepare) first"
        )
    return pd.read_csv(train_path), pd.read_csv(test_path)


def build_matrix(urls: pd.Series, evidence_records: list[dict] | None, config: str) -> pd.DataFrame:
    """Build the feature matrix for a configuration."""
    if config == "url_only":
        return url_feature_frame(urls)
    if config == "content_only":
        if evidence_records is None:
            raise ValueError("content_only requires evidence records")
        return page_feature_frame(evidence_records)
    if config == "combined":
        if evidence_records is None:
            raise ValueError("combined requires evidence records")
        return combined_feature_frame(urls, evidence_records)
    raise ValueError(f"unknown config: {config}")


def load_crawled() -> list[dict]:
    return load_crawled_evidence()


def config_available(config: str, crawled: list[dict]) -> bool:
    """URL-only always works; content/combined need crawled page evidence."""
    if config == "url_only":
        return True
    return bool(crawled) and len(crawled) >= 10


def summarize_matrix(matrix: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    missing = [c for c in expected if c not in matrix.columns]
    if missing:
        raise ValueError(f"feature matrix is missing columns: {missing}")
    matrix = matrix[expected].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return matrix.astype(float)


__all__ = [
    "FEATURE_COLUMNS_BY_CONFIG",
    "binary_metrics",
    "build_matrix",
    "config_available",
    "load_crawled",
    "load_processed",
    "make_models",
    "summarize_matrix",
]
