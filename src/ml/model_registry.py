"""Model registry: loads trained models and produces scores + top factors.

Models are stored as joblib artifacts next to an ``index.json`` written by the
training stage. MLflow holds the experiment history; the registry only needs the
promoted artifacts, which keeps serving fast and offline-capable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

CONFIGS = ("url_only", "content_only", "combined")


class ModelNotFoundError(RuntimeError):
    """Raised when no promoted model artifact is available."""


@dataclass
class LoadedModel:
    """A promoted model plus everything needed to score a raw feature dict."""

    name: str
    config: str
    feature_columns: list[str]
    model: Any
    metrics: dict[str, float] = field(default_factory=dict)

    def predict_proba(self, features: dict[str, float]) -> float:
        row = pd.DataFrame(
            [{col: float(features.get(col, 0.0)) for col in self.feature_columns}],
            columns=self.feature_columns,
        )
        proba = self.model.predict_proba(row)
        return float(proba[0][1])

    def top_factors(self, features: dict[str, float], k: int = 5) -> list[dict[str, Any]]:
        """Approximate per-feature contributions toward the phishing score.

        Linear models use coefficient x value; tree ensembles use impurity
        importance x value. Both are signed so the direction is interpretable.
        """
        values = np.array([float(features.get(c, 0.0)) for c in self.feature_columns])
        estimator = self.model
        # Unwrap sklearn Pipeline(StandardScaler -> estimator)
        if hasattr(estimator, "named_steps"):
            estimator = list(estimator.named_steps.values())[-1]
            scaler = getattr(self.model, "named_steps", {}).get("scaler")
            if scaler is not None and hasattr(scaler, "mean_"):
                scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
                values = (values - scaler.mean_) / scale

        if hasattr(estimator, "coef_"):
            weights = np.ravel(estimator.coef_)
        elif hasattr(estimator, "feature_importances_"):
            importances = np.asarray(estimator.feature_importances_, dtype=float)
            weights = importances * np.sign(values)
        else:
            return []

        contributions = weights * values
        order = np.argsort(-np.abs(contributions))[:k]
        return [
            {
                "feature": self.feature_columns[i],
                "value": round(float(values[i]), 4),
                "contribution": round(float(contributions[i]), 4),
                "direction": "increases_risk" if contributions[i] > 0 else "decreases_risk",
            }
            for i in order
        ]


class ModelRegistry:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self._index: dict[str, Any] = {}
        self._cache: dict[str, LoadedModel | None] = {}

    # ------------------------------------------------------------------ #
    def index(self) -> dict[str, Any]:
        if not self._index:
            path = self.model_dir / "index.json"
            if path.exists():
                self._index = json.loads(path.read_text(encoding="utf-8"))
        return self._index

    def available_configs(self) -> list[str]:
        return sorted(self.index().get("models", {}).keys())

    def load(self, config: str = "url_only") -> LoadedModel | None:
        """Load a promoted model for a config, or None when unavailable."""
        if config in self._cache:
            return self._cache[config]

        entry = self.index().get("models", {}).get(config)
        if not entry:
            self._cache[config] = None
            return None

        artifact = self.model_dir / entry["artifact"]
        if not artifact.exists():
            self._cache[config] = None
            return None

        loaded = LoadedModel(
            name=entry.get("model_name", "unknown"),
            config=config,
            feature_columns=list(entry["feature_columns"]),
            model=joblib.load(artifact),
            metrics=dict(entry.get("metrics", {})),
        )
        self._cache[config] = loaded
        return loaded

    def load_best(self) -> LoadedModel | None:
        """Load the overall best model, preferring the URL model for scoring.

        The URL model has no dependency on a live browser visit, so the agent
        always has at least one usable signal.
        """
        for config in ("url_only", "combined", "content_only"):
            model = self.load(config)
            if model is not None:
                return model
        return None

    def reference_features(self) -> pd.DataFrame | None:
        """Reference feature distribution captured at training time (for drift)."""
        path = self.model_dir / "reference_features.csv"
        if not path.exists():
            return None
        try:
            return pd.read_csv(path)
        except Exception:  # noqa: BLE001
            return None
