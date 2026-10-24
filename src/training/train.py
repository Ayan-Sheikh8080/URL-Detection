"""Train stage: compare model families across three configurations.

Run with ``python -m src.training.train`` (DVC stage ``train``).

Configurations (proposal feature 6):
  * url_only      - lexical URL features
  * content_only  - page evidence features (needs crawled evidence)
  * combined      - URL + page evidence (the agent configuration)

Every run is logged to MLflow.  Model selection uses stratified cross-validation
on the training split only; the held-out test split is reserved for
``evaluate.py``.  The best model per configuration, an overall best model, a
model index and a reference feature sample for drift monitoring are promoted into
``models/``.
"""

from __future__ import annotations

import json
import shutil

import joblib
import mlflow
import pandas as pd

from src.config import PROJECT_ROOT, load_params
from src.training.common import (
    FEATURE_COLUMNS_BY_CONFIG,
    build_matrix,
    config_available,
    cross_validate,
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

    train_df, _ = load_processed()
    crawled_train = load_crawled(split="train")

    models = make_models(training_params)
    if not models:
        raise RuntimeError("no model families available to train")

    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _clean_previous_artifacts()

    results: dict[str, dict] = {}
    index: dict[str, dict] = {"models": {}, "experiment": EXPERIMENT_NAME}

    for config in ("url_only", "content_only", "combined"):
        if not config_available(config, crawled_train):
            print(f"[train] skipping config '{config}' - no crawled page evidence available")
            continue

        columns = FEATURE_COLUMNS_BY_CONFIG[config]

        if config == "url_only":
            X_train = summarize_matrix(build_matrix(train_df["url"], None, config), columns)
            y_train = train_df["label"].astype(int)
        else:
            X_train = summarize_matrix(
                build_matrix(
                    pd.Series([row["url"] for row in crawled_train]),
                    [row["evidence"] for row in crawled_train],
                    config,
                ),
                columns,
            )
            y_train = pd.Series([row["label"] for row in crawled_train]).astype(int)

        if y_train.nunique() < 2 or int(y_train.value_counts().min()) < 2:
            print(f"[train] skipping config '{config}' - not enough rows per class")
            continue

        for model_name, pipeline in models.items():
            run_name = f"{config}__{model_name}"
            with mlflow.start_run(run_name=run_name):
                validation = cross_validate(pipeline, X_train, y_train)
                pipeline.fit(X_train, y_train)

                mlflow.log_params(
                    {
                        "config": config,
                        "model_family": model_name,
                        "n_features": len(columns),
                        "n_train": len(X_train),
                        **_flatten_params(training_params),
                    }
                )
                mlflow.log_metrics(
                    {f"cv_{k}": v for k, v in validation.items() if isinstance(v, (int, float))}
                )
                mlflow.log_dict(
                    {"features": columns, "metrics": validation}, "schema.json"
                )

                print(
                    f"[train] {run_name}: cv_{metric_name}="
                    f"{validation.get(metric_name)} (n={len(X_train)})"
                )

                result = {
                    "config": config,
                    "model_name": model_name,
                    "n_train": len(X_train),
                    "n_features": len(columns),
                    **validation,
                }
                results[run_name] = result

                artifact = f"{config}__{model_name}.joblib"
                joblib.dump(pipeline, MODELS_DIR / artifact)
                mlflow.log_artifact(str(MODELS_DIR / artifact), artifact_path="model")

                entry = {
                    "model_name": model_name,
                    "artifact": artifact,
                    "feature_columns": columns,
                    "metrics": validation,
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

    best_run = max(results.items(), key=lambda kv: kv[1][metric_name] or 0)
    index["best"] = {
        "run": best_run[0],
        "metric": metric_name,
        "value": best_run[1][metric_name],
    }
    (MODELS_DIR / "index.json").write_text(json.dumps(index, indent=2, allow_nan=False), encoding="utf-8")

    report = {
        "experiment": EXPERIMENT_NAME,
        "metric": metric_name,
        "selection": "stratified cross-validation on the training split",
        "best": index["best"],
        "results": results,
        "promoted": {cfg: meta["model_name"] for cfg, meta in index["models"].items()},
    }
    (REPORTS_DIR / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"[train] best run: {best_run[0]} ({metric_name}={best_run[1][metric_name]})")
    return report


def _promote(index: dict, config: str, entry: dict, metric_name: str) -> None:
    """Keep the best model per configuration."""
    current = index["models"].get(config)
    if current is None or (entry["metrics"].get(metric_name) or 0) > (
        current["metrics"].get(metric_name) or 0
    ):
        index["models"][config] = entry
        # Promoted name consumed by the model registry.
        shutil.copyfile(MODELS_DIR / entry["artifact"], MODELS_DIR / f"{config}_model.pkl")


def _clean_previous_artifacts() -> None:
    for pattern in ("*_model.pkl", "*__*.joblib"):
        for path in MODELS_DIR.glob(pattern):
            path.unlink()
    for name in ("index.json", "reference_features.csv"):
        path = MODELS_DIR / name
        if path.exists():
            path.unlink()


def _flatten_params(params: dict) -> dict:
    """Flatten nested training params for MLflow (models and families included)."""
    flat: dict = {}

    def walk(prefix: str, value) -> None:
        if isinstance(value, dict):
            for key, inner in value.items():
                walk(f"{prefix}.{key}" if prefix else str(key), inner)
        else:
            flat[f"param__{prefix}"] = value

    walk("", params or {})
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
