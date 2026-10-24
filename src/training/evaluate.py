"""Evaluate stage: score every promoted model and summarize the comparison.

Run with ``python -m src.training.evaluate`` (DVC stage ``evaluate``).

The final test split is touched only here.  ``url_only`` is scored on the URL
test split, while ``content_only`` and ``combined`` are scored on the held-out
crawled evidence split.  A *shared benchmark* then scores all three
configurations on exactly the same crawled test rows, which is the apples-to-
apples comparison the proposal asks for (how much page evidence adds over
URL-only detection).

Produces ``reports/evaluation.json`` with precision, recall, F1, confusion
matrices, inference latency and estimated cost per analysis.
"""

from __future__ import annotations

import json
import os
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
    eval_params = params.get("evaluation", {})

    index_path = MODELS_DIR / "index.json"
    if not index_path.exists():
        raise FileNotFoundError("models/index.json missing - run `python -m src.training.train` first")
    index = json.loads(index_path.read_text(encoding="utf-8"))

    _, test_df = load_processed()
    crawled_test = load_crawled(split="test")
    crawled_test_df = pd.DataFrame(crawled_test) if crawled_test else None

    report: dict = {
        "metric": metric_name,
        "evaluation_protocol": {
            "url_only": "held-out URL test split (data/processed/test.csv)",
            "content_only": "held-out crawled evidence split (data/processed/crawled_test.jsonl)",
            "combined": "held-out crawled evidence split (data/processed/crawled_test.jsonl)",
            "shared_benchmark": "all configurations scored on the same crawled test rows",
        },
        "configs": {},
        "comparison": {},
        "shared_benchmark": {},
    }

    for config, entry in index.get("models", {}).items():
        import joblib

        artifact = MODELS_DIR / entry["artifact"]
        if not artifact.exists():
            continue
        pipeline = joblib.load(artifact)
        columns = FEATURE_COLUMNS_BY_CONFIG[config]

        if config == "url_only":
            X, y = _url_matrix(test_df, columns)
            evaluation_set = "url_test"
        else:
            if not crawled_test:
                continue
            X, y = _crawled_matrix(crawled_test_df, columns, config)
            evaluation_set = "crawled_test"

        metrics = _score(pipeline, X, y)
        metrics["latency_ms_per_sample"] = _latency(pipeline, X)
        metrics["n_samples"] = int(len(X))
        metrics["estimated_cost_usd_per_analysis"] = _cost(
            config, metrics["latency_ms_per_sample"], eval_params
        )
        report["configs"][config] = {
            "model_name": entry["model_name"],
            "evaluation_set": evaluation_set,
            "metrics": metrics,
        }

    # Shared benchmark: same crawled test rows, every configuration -------------
    if crawled_test and report["configs"]:
        benchmark_rows: dict[str, dict] = {}
        for config, entry in index.get("models", {}).items():
            artifact = MODELS_DIR / entry["artifact"]
            if not artifact.exists():
                continue
            import joblib

            pipeline = joblib.load(artifact)
            columns = FEATURE_COLUMNS_BY_CONFIG[config]
            try:
                X, y = _crawled_matrix(crawled_test_df, columns, config)
            except ValueError:
                continue
            metrics = _score(pipeline, X, y)
            metrics["latency_ms_per_sample"] = _latency(pipeline, X)
            metrics["n_samples"] = int(len(X))
            metrics["estimated_cost_usd_per_analysis"] = _cost(
                config, metrics["latency_ms_per_sample"], eval_params
            )
            benchmark_rows[config] = {"model_name": entry["model_name"], "metrics": metrics}
        report["shared_benchmark"] = {
            "n_samples": len(crawled_test),
            "configs": benchmark_rows,
        }

    comparison_source = (
        report["shared_benchmark"]["configs"]
        if report["shared_benchmark"]
        else {config: payload for config, payload in report["configs"].items()}
    )
    url_f1 = comparison_source.get("url_only", {}).get("metrics", {}).get("f1")
    for config, payload in comparison_source.items():
        metrics = payload["metrics"]
        f1 = metrics["f1"]
        improvement = None
        if url_f1 is not None and url_f1 > 0:
            improvement = round((f1 - url_f1) / url_f1 * 100, 2)
        elif url_f1 == 0 and f1 > 0 and config != "url_only":
            improvement = None  # undefined / infinite relative improvement
        elif url_f1 == 0 and f1 == 0:
            improvement = 0.0
        report["comparison"][config] = {
            "f1": f1,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_improvement_vs_url_only_pct": improvement,
            "latency_ms_per_sample": metrics["latency_ms_per_sample"],
            "estimated_cost_usd_per_analysis": metrics["estimated_cost_usd_per_analysis"],
        }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "evaluation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )

    _log_to_mlflow(report)

    print(json.dumps(report["comparison"], indent=2))
    return report


def _url_matrix(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    return (
        summarize_matrix(build_matrix(df["url"], None, "url_only"), columns),
        df["label"].astype(int),
    )


def _crawled_matrix(
    df: pd.DataFrame, columns: list[str], config: str
) -> tuple[pd.DataFrame, pd.Series]:
    return (
        summarize_matrix(
            build_matrix(df["url"], [row["evidence"] for row in df.to_dict("records")], config),
            columns,
        ),
        df["label"].astype(int),
    )


def _score(pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    predictions = pipeline.predict(X)
    probabilities = (
        pipeline.predict_proba(X)[:, 1].tolist() if hasattr(pipeline, "predict_proba") else None
    )
    return binary_metrics(y, predictions, probabilities)


def _latency(pipeline, X: pd.DataFrame) -> float:
    started = time.perf_counter()
    pipeline.predict(X)
    return round((time.perf_counter() - started) * 1000 / max(len(X), 1), 4)


def _cost(config: str, model_latency_ms: float, eval_params: dict) -> float:
    """Rough cost per analysis, with clearly configurable assumptions.

    Defaults assume a small always-on CPU task (USD 0.06/hour), a 2.5 s sandbox
    visit and a 1.2 s Bedrock call when the LLM reasoner is enabled.  Adjust the
    values in params.yaml to match the actual deployment.
    """
    compute_per_ms = float(eval_params.get("compute_cost_usd_per_ms", 0.000000017))
    browser = float(eval_params.get("browser_cost_usd_per_analysis", 0.00004))
    llm = float(eval_params.get("llm_cost_usd_per_analysis", 0.00025))
    cost = model_latency_ms * compute_per_ms
    if config in {"content_only", "combined"}:
        cost += browser
        if os.getenv("LLM_PROVIDER", "rule_based").strip().lower() == "bedrock":
            cost += llm
    return round(cost, 8)


def _log_to_mlflow(report: dict) -> None:
    try:
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
            if report["shared_benchmark"]:
                mlflow.log_metric(
                    "shared_benchmark__n_samples", report["shared_benchmark"]["n_samples"]
                )
            mlflow.log_artifact(str(REPORTS_DIR / "evaluation.json"))
    except Exception as exc:  # noqa: BLE001 - tracking must not break evaluation
        print(f"[evaluate] MLflow logging skipped: {exc}")


if __name__ == "__main__":
    main()
