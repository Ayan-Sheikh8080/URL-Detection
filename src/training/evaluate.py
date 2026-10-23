"""Evaluate stage: score every promoted model and summarize the comparison.

Run with ``python -m src.training.evaluate`` (DVC stage ``evaluate``).

Produces ``reports/evaluation.json`` with precision, recall, F1, confusion
matrices and inference latency per configuration, which is what the proposal
uses to quantify how much page evidence adds over URL-only detection.
"""

from __future__ import annotations

import json
import time

import mlflow
import pandas as pd

from src.config import PROJECT_ROOT, load_params
from src.training.common import (
    FEATURE_COLUMNS_BY_CONFIG,
    binary_metrics,
    build_matrix,
    load_crawled,
    load_processed,
    summarize_matrix,
)

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def main() -> dict:
    params = load_params()
    metric_name = params.get("training", {}).get("metric", "f1")

    index_path = MODELS_DIR / "index.json"
    if not index_path.exists():
        raise FileNotFoundError("models/index.json missing - run `python -m src.training.train` first")
    index = json.loads(index_path.read_text(encoding="utf-8"))

    _, test_df = load_processed()
    crawled = load_crawled()
    crawled_df = pd.DataFrame(crawled) if crawled else None

    report: dict = {"metric": metric_name, "configs": {}, "comparison": {}}

    for config, entry in index.get("models", {}).items():
        import joblib

        artifact = MODELS_DIR / entry["artifact"]
        if not artifact.exists():
            continue
        pipeline = joblib.load(artifact)
        columns = FEATURE_COLUMNS_BY_CONFIG[config]

        if config == "url_only":
            X = summarize_matrix(build_matrix(test_df["url"], None, config), columns)
            y = test_df["label"].astype(int)
        else:
            if not crawled:
                continue
            X = summarize_matrix(
                build_matrix(crawled_df["url"], [r["evidence"] for r in crawled], config), columns
            )
            y = crawled_df["label"].astype(int)

        started = time.perf_counter()
        predictions = pipeline.predict(X)
        latency_ms = (time.perf_counter() - started) * 1000 / max(len(X), 1)

        probabilities = pipeline.predict_proba(X)[:, 1].tolist() if hasattr(pipeline, "predict_proba") else None
        metrics = binary_metrics(y, predictions, probabilities)
        metrics["latency_ms_per_sample"] = round(latency_ms, 4)
        metrics["n_samples"] = int(len(X))

        report["configs"][config] = {
            "model_name": entry["model_name"],
            "metrics": metrics,
        }

    # Comparison table: relative improvement of the agent configs over URL-only.
    url_f1 = report["configs"].get("url_only", {}).get("metrics", {}).get("f1")
    for config, payload in report["configs"].items():
        f1 = payload["metrics"]["f1"]
        improvement = None
        if url_f1:
            improvement = round((f1 - url_f1) / url_f1 * 100, 2)
        report["comparison"][config] = {
            "f1": f1,
            "precision": payload["metrics"]["precision"],
            "recall": payload["metrics"]["recall"],
            "f1_improvement_vs_url_only_pct": improvement,
            "latency_ms_per_sample": payload["metrics"]["latency_ms_per_sample"],
        }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    try:
        import os

        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "").strip()
            or f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
        )
        mlflow.set_experiment("phishing-link-detection")
        with mlflow.start_run(run_name="evaluation"):
            for config, payload in report["comparison"].items():
                for key, value in payload.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"{config}__{key}", value)
            mlflow.log_artifact(str(REPORTS_DIR / "evaluation.json"))
    except Exception as exc:  # noqa: BLE001 - tracking must not break evaluation
        print(f"[evaluate] MLflow logging skipped: {exc}")

    print(json.dumps(report["comparison"], indent=2))
    return report


if __name__ == "__main__":
    main()
