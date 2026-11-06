"""Load promoted model artifacts and provide typed inference results."""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "model_bundle.joblib"
logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.model_path = models_dir / MODEL_PATH.name
        self._bundle: dict[str, dict[str, Any]] | None = None
        self._explainers: dict[int, Any] = {}

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {self.model_path}; run the data and training pipeline"
            )
        self._bundle = joblib.load(self.model_path)
        if not isinstance(self._bundle, dict) or not self._bundle:
            raise ValueError(f"Invalid model bundle at {self.model_path}")

    def predict(self, config: str, features: dict[str, float]) -> dict[str, Any]:
        if self._bundle is None:
            self.load()
        if config not in self._bundle:
            raise ValueError(f"Unknown model configuration: {config}")
        entry = self._bundle[config]
        model = entry["model"]
        order = entry["feature_order"]
        values = [float(features.get(name, 0.0)) for name in order]
        score = float(model.predict_proba([values])[0][list(model.classes_).index(1)])
        top_factors, explanation_method = self._top_factors(model, order, values)
        return {
            "score": score,
            "model_family": entry["model_family"],
            "top_factors": top_factors,
            "explanation_method": explanation_method,
        }

    def _top_factors(
        self, model: Any, order: list[str], values: list[float]
    ) -> tuple[list[dict[str, Any]], str]:
        contributions: list[float] | None = None
        method = "model_weights"
        try:
            import numpy as np
            import shap

            explainer = self._explainers.get(id(model))
            if explainer is None:
                explainer = shap.Explainer(model)
                self._explainers[id(model)] = explainer
            explanation = explainer(np.asarray([values], dtype=float))
            shap_values = np.asarray(explanation.values)
            if shap_values.ndim == 3:
                shap_values = shap_values[0, :, -1]
            elif shap_values.ndim == 2:
                shap_values = shap_values[0]
            if shap_values.ndim == 1 and len(shap_values) == len(order):
                contributions = [float(value) for value in shap_values]
                method = "shap"
        except ImportError:
            pass
        except Exception:
            logger.exception("SHAP explanation failed; using model weights")

        if contributions is None:
            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
                contributions = [
                    float(weight) * value for weight, value in zip(importance, values)
                ]
            elif hasattr(model, "coef_"):
                coefficients = model.coef_[0]
                contributions = [
                    float(weight) * value for weight, value in zip(coefficients, values)
                ]
            else:
                return [], method
        ranked = sorted(
            zip(order, contributions),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        factors = []
        for name, value in ranked[:5]:
            if value == 0:
                continue
            factor = {
                "feature": name,
                "contribution": round(value, 4),
            }
            if method == "shap" or hasattr(model, "coef_"):
                factor["direction"] = "increases_risk" if value >= 0 else "reduces_risk"
            else:
                factor["direction"] = "model_associated"
            factors.append(factor)
        return factors, method

    def reference_features(self) -> list[dict[str, Any]]:
        path = self.models_dir / "reference_features.csv"
        if not path.exists():
            return []
        return pd.read_csv(path).to_dict(orient="records")

    def index(self) -> dict[str, Any]:
        path = self.models_dir / "index.json"
        if not path.exists():
            return {"trained": False, "configurations": {}}
        return json.loads(path.read_text(encoding="utf-8"))
