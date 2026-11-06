import os
import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import mlflow

MODELS_DIR = Path("models")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


class ModelRegistry:
    """Manages model versioning, promotion, and scoring."""
    
    def __init__(self, models_dir: Path = MODELS_DIR, tracking_uri: str = MLFLOW_TRACKING_URI):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(tracking_uri)
        self.promoted_models: Dict[str, Any] = {}
        self.reference_features: Optional[np.ndarray] = None
        self.feature_schema: Optional[Dict] = None
        
    def load_promoted_model(self, config: str = "combined") -> Optional[Any]:
        """Load the promoted model for a given configuration."""
        model_path = self.models_dir / f"{config}_model.pkl"
        index_path = self.models_dir / "index.json"
        
        if not model_path.exists():
            print(f"Model not found at {model_path}")
            return None
        
        # Load model
        model = joblib.load(model_path)
        
        # Load index if exists
        if index_path.exists():
            with open(index_path, 'r') as f:
                index = json.load(f)
            print(f"Loaded promoted model: {config} - {index.get('model_type', 'unknown')}")
            self.promoted_models[config] = model
            self.feature_schema = index.get('feature_schema', {})
            return model
        
        self.promoted_models[config] = model
        return model
    
    def promote_model(self, model: Any, config: str, feature_schema: Dict, 
                      metrics: Dict[str, float]) -> None:
        """Promote a model to the models directory."""
        model_path = self.models_dir / f"{config}_model.pkl"
        index_path = self.models_dir / "index.json"
        
        # Save model
        joblib.dump(model, model_path)
        
        # Create index
        index = {
            "version": "1.0.0",
            "config": config,
            "model_type": type(model).__name__,
            "feature_schema": feature_schema,
            "metrics": metrics,
            "promoted_at": json.loads(os.popen('date').read()) if hasattr(os.popen('date').read(), '__iter__') else str(__import__('datetime').datetime.utcnow())
        }
        
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
        
        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_param("config", config)
            mlflow.log_param("model_type", type(model).__name__)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
        
        self.promoted_models[config] = model
        self.feature_schema = feature_schema
        print(f"Model promoted: {config} model saved to {model_path}")
    
    def score(self, model_name: str, features: Dict[str, float]) -> float:
        """Get a prediction score from a model."""
        model = self.promoted_models.get(model_name)
        if model is None:
            print(f"Model {model_name} not found in promoted models")
            return 0.5  # Default neutral score
        
        try:
            # Convert features to array
            feature_array = self._features_to_array(features)
            prediction = model.predict_proba([feature_array])[0]
            # Return probability of phishing class (assuming binary classification)
            if len(prediction) >= 2:
                return float(prediction[1])  # Probability of positive class (phishing)
            return float(prediction[0])
        except Exception as e:
            print(f"Error scoring model: {e}")
            return 0.5
    
    def _features_to_array(self, features: Dict[str, float]) -> List[float]:
        """Convert feature dict to array based on schema."""
        if self.feature_schema is None:
            # Default: return values in alphabetical order of keys
            return [features.get(k, 0.0) for k in sorted(features.keys())]
        
        # Use schema to order features
        array = []
        for feature_name in self.feature_schema.get('feature_order', sorted(features.keys())):
            array.append(features.get(feature_name, 0.0))
        
        # Pad or trim to expected length
        expected_len = self.feature_schema.get('expected_length', len(array))
        if len(array) < expected_len:
            array.extend([0.0] * (expected_len - len(array)))
        elif len(array) > expected_len:
            array = array[:expected_len]
        
        return array
    
    def reference_features_available(self) -> bool:
        """Check if reference features for drift monitoring are available."""
        return self.reference_features is not None
    
    def set_reference_features(self, features_df: np.ndarray) -> None:
        """Set reference features for drift detection."""
        self.reference_features = features_df
    
    def get_reference_features(self) -> Optional[np.ndarray]:
        """Get reference features."""
        return self.reference_features


def load_best_model(config: str = "combined") -> Any:
    """Convenience function to load the best model for a configuration."""
    registry = ModelRegistry()
    return registry.load_promoted_model(config)


def score_url(url: str, model_name: str = "combined") -> Dict[str, Any]:
    """Score a URL using the model registry."""
    from src.models.url_features import extractUrlFeatures, predictUrl
    
    # Extract features
    features = extractUrlFeatures(url)
    if features is None:
        return {"score": 0.5, "verdict": "unknown", "error": "Feature extraction failed"}
    
    # Get prediction from Python model
    prediction = predictUrl(features)
    
    # Also try registry if available
    registry = ModelRegistry()
    registry_score = registry.score(model_name, features)
    
    # Combine scores (weighted average)
    combined_score = 0.7 * prediction.get("score", 0.5) + 0.3 * registry_score
    
    # Determine verdict
    score = combined_score
    if score >= 0.7:
        verdict = "phishing"
    elif score >= 0.3:
        verdict = "suspicious"
    else:
        verdict = "safe"
    
    return {
        "score": round(combined_score, 4),
        "verdict": verdict,
        "model_score": prediction.get("score", 0.5),
        "registry_score": round(registry_score, 4)
    }