"""Model registry: loads trained models and produces scores + top factors.

Models are stored as joblib artifacts next to an ``index.json`` written by the
training stage. MLflow holds the experiment history; the registry only needs the
promoted artifacts, which keeps serving fast and offline-capable.

Local explanations use SHAP for tree ensembles and exact coefficient x value
contributions for linear models.  If SHAP cannot explain an estimator, the
registry falls back to impurity importance x value and labels the method
accordingly.
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
    _explainer: Any = field(default=None, repr=False, compare=False)

    def predict_proba(self, features: dict[str, float]) -> float:
        row = pd.DataFrame(
            [{col: float(features.get(col, 0.0)) for col in self.feature_columns}],
            columns=self.feature_columns,
        )
        # NumPy input keeps serving independent of whether the pipeline was fit
        # on a DataFrame or an array (and avoids feature-name warnings).
        proba = self.model.predict_proba(row.to_numpy(dtype=float))
        return float(proba[0][1])

    def top_factors(self, features: dict[str, float], k: int = 5) -> list[dict[str, Any]]:
        """Per-feature contributions toward the phishing score.

        For linear models this is exact (coefficient x scaled value).  For tree
        ensembles SHAP's TreeExplainer is used when available; otherwise a signed
        impurity-importance fallback keeps the API available offline.
        """
        original = np.array(
            [float(features.get(c, 0.0)) for c in self.feature_columns], dtype=float
        )
        estimator = self.model
        scaled = original.copy()

        # Unwrap sklearn Pipeline(StandardScaler -> estimator)
        if hasattr(estimator, "named_steps"):
            scaler = getattr(self.model, "named_steps", {}).get("scaler")
            if scaler is not None and hasattr(scaler, "mean_"):
                scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
                scaled = (original - scaler.mean_) / scale
            estimator = list(estimator.named_steps.values())[-1]

        contributions: np.ndarray | None = None
        method = "importance"

        if hasattr(estimator, "coef_"):
            weights = np.ravel(estimator.coef_)
            contributions = weights * scaled
            method = "linear"

        if contributions is None:
            contributions = self._shap_contributions(estimator, scaled)
            if contributions is not None:
                method = "shap"

        if contributions is None:
            if hasattr(estimator, "feature_importances_"):
                importances = np.asarray(estimator.feature_importances_, dtype=float)
                contributions = importances * np.sign(scaled)
            else:
                return []

        order = np.argsort(-np.abs(contributions))[:k]
        factors = []
        for i in order:
            contribution = float(contributions[i])
            factors.append(
                {
                    "feature": self.feature_columns[i],
                    "value": round(float(original[i]), 4),
                    "contribution": round(contribution, 6),
                    "direction": (
                        "increases_risk"
                        if contribution > 0
                        else "decreases_risk"
                        if contribution < 0
                        else "neutral"
                    ),
                    "method": method,
                }
            )
        return factors

    def _shap_contributions(self, estimator, scaled: np.ndarray) -> np.ndarray | None:
        try:
            import shap

            if self._explainer is None:
                self._explainer = shap.TreeExplainer(estimator)
            raw = self._explainer.shap_values(scaled.reshape(1, -1))
        except Exception:  # noqa: BLE001 - explanation must never break serving
            return None

        # Different SHAP/Sklearn versions return (n, f), (n, f, classes), or
        # [class0, class1] lists.  Normalise to one contribution per feature.
        if isinstance(raw, list):
            if len(raw) > 1:
                raw = raw[-1]
            else:
                raw = raw[0]
        array = np.asarray(raw, dtype=float)
        if array.ndim == 3:
            array = array[0, :, -1] if array.shape[0] == 1 else array[:, :, -1].mean(axis=0)
        elif array.ndim == 2:
            array = array[0] if array.shape[0] == 1 else array.mean(axis=0)
        if array.ndim != 1 or len(array) != len(self.feature_columns):
            return None
        return array


class ModelRegistry:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self._index: dict[str, Any] = {}
        self._index_loaded = False
        self._cache: dict[str, LoadedModel | None] = {}

    # ------------------------------------------------------------------ #
    def index(self) -> dict[str, Any]:
        if not self._index_loaded:
            path = self.model_dir / "index.json"
            if path.exists():
                try:
                    self._index = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    self._index = {}
            self._index_loaded = True
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

        try:
            loaded = LoadedModel(
                name=entry.get("model_name", "unknown"),
                config=config,
                feature_columns=list(entry["feature_columns"]),
                model=joblib.load(artifact),
                metrics=dict(entry.get("metrics", {})),
            )
        except Exception:  # noqa: BLE001 - a corrupt artifact is missing evidence
            loaded = None
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
