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

    # Small datasets benefit from balanced class weights.  An explicit value in
    # params.yaml always wins.
    lr_params.setdefault("class_weight", "balanced")
    rf_params.setdefault("class_weight", "balanced_subsample")

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

        xgb_params.setdefault("eval_metric", "logloss")
        models["xgboost"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", XGBClassifier(**xgb_params)),
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


def load_crawled(split: str | None = None) -> list[dict]:
    """Load crawled evidence from the prepared train/test split (or raw)."""
    return load_crawled_evidence(split=split)


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


def cross_validate(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    folds: int = 5,
) -> dict:
    """Stratified cross-validation metrics used for model selection.

    The final test split is never touched here; ``evaluate.py`` owns it.  This
    keeps model promotion and the reported agent-vs-URL-only comparison honest.
    """
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold

    y = pd.Series(y).astype(int).reset_index(drop=True)
    X = X.reset_index(drop=True)
    counts = y.value_counts()
    if y.nunique() < 2 or int(counts.min()) < 2:
        raise ValueError("cross-validation requires at least two rows of each class")

    n_splits = max(2, min(folds, int(counts.min())))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics: list[dict] = []
    confusion = np.zeros((2, 2), dtype=int)

    for train_idx, valid_idx in splitter.split(X, y):
        model = clone(pipeline)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        predictions = model.predict(X.iloc[valid_idx])
        probabilities = _positive_proba(model, X.iloc[valid_idx])
        metrics = binary_metrics(y.iloc[valid_idx], predictions, probabilities)
        fold_metrics.append(metrics)
        confusion += np.asarray(metrics["confusion_matrix"], dtype=int)

    summary: dict = {}
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        values = [m[key] for m in fold_metrics if isinstance(m.get(key), (int, float))]
        if not values:
            summary[key] = None
            continue
        mean = float(np.mean(values))
        summary[key] = round(mean, 4)
        summary[f"{key}_std"] = round(float(np.std(values)), 4)
    summary["confusion_matrix"] = confusion.tolist()
    summary["n_folds"] = n_splits
    return summary


def _positive_proba(model, X) -> list[float]:
    if not hasattr(model, "predict_proba"):
        return []
    return model.predict_proba(X)[:, 1].tolist()


__all__ = [
    "FEATURE_COLUMNS_BY_CONFIG",
    "binary_metrics",
    "build_matrix",
    "config_available",
    "cross_validate",
    "load_crawled",
    "load_processed",
    "make_models",
    "summarize_matrix",
]
