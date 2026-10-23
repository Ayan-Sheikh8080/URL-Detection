"""Train stage: compare model families across three configurations.

Run with ``python -m src.training.train`` (DVC stage ``train``).

Configurations (proposal feature 6):
  * url_only      - lexical URL features
  * content_only  - page evidence features (needs crawled evidence)
  * combined      - URL + page evidence (the agent configuration)

Every run is logged to MLflow. The best model per configuration, the overall
best model, a model index and a reference feature sample for drift monitoring
are promoted into ``models/``.
"""

from __future__ import annotations

import json

import joblib
import mlflow
import pandas as pd

from src.config import PROJECT_ROOT, load_params
from src.training.common import (
    FEATURE_COLUMNS_BY_CONFIG,
    binary_metrics,
    build_matrix,
    config_available,
    load_crawled,
    load_processed,
    make_models,
    summarize_matrix,
)

EXPERIMENT_NAME = "phishing-link-detection"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def main() -> dict:
    params = load_params()
    training_params = params.get("training", {})
    metric_name = training_params.get("metric", "f1")

    train_df, test_df = load_processed()
    crawled = load_crawled()
    crawled_df = pd.DataFrame(crawled) if crawled else None

    models = make_models(training_params)
    if not models:
        raise RuntimeError("no model families available to train")

    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    index: dict[str, dict] = {"models": {}, "experiment": EXPERIMENT_NAME}

    for config in ("url_only", "content_only", "combined"):
        if not config_available(config, crawled):
            print(f"[train] skipping config '{config}' - no crawled page evidence available")
            continue

        columns = FEATURE_COLUMNS_BY_CONFIG[config]

        if config == "url_only":
            X_train = summarize_matrix(build_matrix(train_df["url"], None, config), columns)
            y_train = train_df["label"].astype(int)
            X_test = summarize_matrix(build_matrix(test_df["url"], None, config), columns)
            y_test = test_df["label"].astype(int)
        else:
            full_evidence = [row["evidence"] for row in crawled]
            crawled_urls = crawled_df["url"]
            crawled_labels = crawled_df["label"].astype(int)
            X_all = summarize_matrix(build_matrix(crawled_urls, full_evidence, config), columns)
            split = max(int(len(X_all) * 0.7), 1)
            if split >= len(X_all):
                split = max(len(X_all) - 1, 1)
            X_train, y_train = X_all.iloc[:split], crawled_labels.iloc[:split]
            X_test, y_test = X_all.iloc[split:], crawled_labels.iloc[split:]

        for model_name, pipeline in models.items():
            run_name = f"{config}__{model_name}"
            with mlflow.start_run(run_name=run_name):
                pipeline.fit(X_train, y_train)
                predictions = pipeline.predict(X_test)
                probabilities = _positive_proba(pipeline, X_test)
                metrics = binary_metrics(y_test, predictions, probabilities)

                mlflow.log_params(
                    {
                        "config": config,
                        "model_family": model_name,
                        "n_features": len(columns),
                        "n_train": len(X_train),
                        "n_test": len(X_test),
                        **{f"param__{k}": v for k, v in _nested(training_params).items()},
                    }
                )
                mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
                mlflow.log_dict({"features": columns, "metrics": metrics}, "schema.json")

                print(f"[train] {run_name}: {metric_name}={metrics[metric_name]}")

                results[run_name] = {"config": config, "model_name": model_name, **metrics}

                artifact = f"{config}__{model_name}.joblib"
                joblib.dump(pipeline, MODELS_DIR / artifact)
                mlflow.log_artifact(str(MODELS_DIR / artifact), artifact_path="model")

                entry = {
                    "model_name": model_name,
                    "artifact": artifact,
                    "feature_columns": columns,
                    "metrics": metrics,
                }
                _promote(index, config, entry, metric_name)

    if not results:
        raise RuntimeError("no configuration could be trained")

    # Reference feature distribution for drift monitoring (from the URL model).
    url_entry = index["models"].get("url_only")
    if url_entry:
        X_ref = summarize_matrix(
            build_matrix(train_df["url"], None, "url_only"), FEATURE_COLUMNS_BY_CONFIG["url_only"]
        )
        sample = X_ref.sample(n=min(len(X_ref), 500), random_state=42)
        sample.to_csv(MODELS_DIR / "reference_features.csv", index=False)

    best_run = max(results.items(), key=lambda kv: kv[1][metric_name])
    index["best"] = {"run": best_run[0], "metric": metric_name, "value": best_run[1][metric_name]}
    (MODELS_DIR / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    report = {
        "experiment": EXPERIMENT_NAME,
        "metric": metric_name,
        "best": index["best"],
        "results": results,
        "promoted": {cfg: meta["model_name"] for cfg, meta in index["models"].items()},
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[train] best run: {best_run[0]} ({metric_name}={best_run[1][metric_name]})")
    return report


def _promote(index: dict, config: str, entry: dict, metric_name: str) -> None:
    """Keep the best model per configuration."""
    current = index["models"].get(config)
    if current is None or entry["metrics"][metric_name] > current["metrics"][metric_name]:
        index["models"][config] = entry
        # Promoted names consumed by the model registry.
        joblib.dump(
            joblib.load(MODELS_DIR / entry["artifact"]),
            MODELS_DIR / f"{config}_model.pkl",
        )
    if "best" not in index or entry["metrics"][metric_name] > index["best"].get("value", -1):
        index["best"] = {"run": f"{config}__{entry['model_name']}", "metric": metric_name,
                         "value": entry["metrics"][metric_name]}


def _positive_proba(pipeline, X) -> list[float] | None:
    if not hasattr(pipeline, "predict_proba"):
        return None
    return pipeline.predict_proba(X)[:, 1].tolist()


def _nested(params: dict) -> dict:
    """Flatten nested training params for MLflow logging."""
    flat: dict = {}
    for key, value in (params or {}).items():
        if key == "models" or not isinstance(value, dict):
            continue
        for inner_key, inner_value in value.items():
            flat[f"{key}.{inner_key}"] = inner_value
    return flat


def _tracking_uri() -> str:
    """Default to a local SQLite tracking backend (MLflow's recommended store)."""
    import os

    configured = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if configured:
        return configured
    return f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"


if __name__ == "__main__":
    main()
