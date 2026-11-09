"""Train and compare URL-only, content-only, and combined phishing models."""

import json
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

try:
    import mlflow
except ImportError:
    mlflow = None

from src.agent.features import (
    CONTENT_FEATURE_NAMES,
    CONTENT_MODEL_FEATURE_NAMES,
    extract_content_features,
)
from src.data.prepare import METADATA_PATH, PROCESSED_PATH, ROOT, normalize_url
from src.ml.features import URL_FEATURE_NAMES, extract_url_features

RAW_EVIDENCE_PATH = ROOT / "data" / "raw" / "crawled_evidence.jsonl"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODEL_PATH = MODELS_DIR / "model_bundle.joblib"
EVALUATION_PATH = REPORTS_DIR / "evaluation.json"
REGISTRY_DIR = MODELS_DIR / "registry"
REGISTRY_MANIFEST_PATH = REGISTRY_DIR / "manifest.json"
MODEL_FAMILIES = {
    "logistic_regression": lambda: LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=100, class_weight="balanced", max_depth=6, random_state=42
    ),
    "xgboost": lambda: XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        n_jobs=1,
        random_state=42,
    ),
}
logger = logging.getLogger(__name__)


def _read_training_data() -> tuple[list[dict], list[dict]]:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Prepared data not found at {PROCESSED_PATH}; run python -m src.data.prepare first"
        )
    frame = pd.read_csv(PROCESSED_PATH)
    urls = frame[["url", "label", "source"]].to_dict(orient="records")
    if set(CONTENT_MODEL_FEATURE_NAMES).issubset(frame.columns):
        return urls, frame.to_dict(orient="records")

    labels = {row["url"]: int(row["label"]) for row in urls}
    evidence_rows = []
    with RAW_EVIDENCE_PATH.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                evidence = json.loads(line)
                normalized = normalize_url(evidence["url"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid evidence record on line {line_number}: {error}") from error
            if normalized in labels:
                evidence_rows.append({
                    "url": normalized,
                    "label": labels[normalized],
                    "evidence": evidence,
                })
    if not evidence_rows:
        raise ValueError("No crawled evidence records match the validated URL dataset")
    if len({record["label"] for record in evidence_rows}) != 2:
        raise ValueError("Crawled evidence must represent both phishing and legitimate URLs")
    return urls, evidence_rows


def _model_input(rows: list[dict], config: str) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    if config == "url_only":
        features = [
            extract_url_features(row["url"], allow_credentials=True)
            for row in rows
        ]
        labels = [int(row["label"]) for row in rows]
        order = list(URL_FEATURE_NAMES)
    else:
        features = []
        labels = []
        for row in rows:
            url_features = extract_url_features(
                row["url"], allow_credentials=True
            )
            content_features = (
                {name: float(row[name]) for name in CONTENT_MODEL_FEATURE_NAMES}
                if all(name in row for name in CONTENT_MODEL_FEATURE_NAMES)
                else extract_content_features(row["url"], row["evidence"])
            )
            features.append({
                **(
                    {f"url_{key}": value for key, value in url_features.items()}
                    if config == "combined"
                    else {}
                ),
                **{
                    f"page_{key}": content_features[key]
                    for key in CONTENT_MODEL_FEATURE_NAMES
                },
            })
            labels.append(int(row["label"]))
        order = (
            [f"url_{key}" for key in URL_FEATURE_NAMES]
            + [f"page_{key}" for key in CONTENT_MODEL_FEATURE_NAMES]
            if config == "combined"
            else [f"page_{key}" for key in CONTENT_MODEL_FEATURE_NAMES]
        )
    matrix = pd.DataFrame(features, columns=order).fillna(0.0)
    return matrix, np.asarray(labels, dtype=int), order


def _log_run(
    config: str,
    family: str,
    metrics: dict,
    feature_count: int,
    model,
    dataset_sha256: str,
) -> str | None:
    if mlflow is None:
        return None
    with mlflow.start_run(run_name=f"{config}-{family}") as run:
        mlflow.log_params({
            "configuration": config,
            "model_family": family,
            "feature_count": feature_count,
            "train_test_split": 0.2,
            "random_state": 42,
            "dataset_sha256": dataset_sha256,
        })
        mlflow.log_metrics({
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "training_seconds": metrics["training_seconds"],
            "inference_latency_ms_per_url": metrics["inference_latency_ms_per_url"],
            "estimated_cloud_cost_usd_per_analysis": metrics["estimated_cloud_cost_usd_per_analysis"],
        })
        mlflow.set_tags({
            "dataset_sha256": dataset_sha256,
            "model_family": family,
            "pipeline": "dvc",
        })
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_path = Path(temporary_directory) / "candidate.joblib"
            joblib.dump(model, artifact_path)
            mlflow.log_artifact(str(artifact_path), artifact_path="model")
        return run.info.run_id


def _register_local_version(
    dataset_sha256: str,
    best_model_runs: dict[str, str | None],
    configurations: dict,
) -> str:
    timestamp = datetime.now(timezone.utc)
    version = f"{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}-{dataset_sha256[:8]}"
    version_dir = REGISTRY_DIR / "versions" / version
    version_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(MODEL_PATH, version_dir / MODEL_PATH.name)

    manifest = {"schema_version": 1, "name": "phishing-url-analyzer", "versions": []}
    if REGISTRY_MANIFEST_PATH.exists():
        manifest = json.loads(REGISTRY_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["latest_version"] = version
    manifest["versions"].append({
        "version": version,
        "created_at": timestamp.isoformat(),
        "artifact": (version_dir / MODEL_PATH.name).relative_to(ROOT).as_posix(),
        "dataset_sha256": dataset_sha256,
        "tags": {
            "dataset": "PhiUSIIL",
            "pipeline": "dvc",
            "holdout_type": "random_stratified",
        },
        "best_model_runs": best_model_runs,
        "configurations": configurations,
    })
    REGISTRY_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return version


def train_models() -> dict:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError("Run python -m src.data.prepare before training")
    url_rows, evidence_rows = _read_training_data()
    datasets = {
        "url_only": url_rows,
        "content_only": evidence_rows,
        "combined": evidence_rows,
    }

    if mlflow is None:
        logger.warning(
            "MLflow is not installed; evaluation metrics will be written locally only"
        )
    else:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or (
            f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
        )
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("phishing-url-analysis")
    dataset_metadata = (
        json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        if METADATA_PATH.exists()
        else {}
    )
    dataset_sha256 = str(dataset_metadata.get("source_sha256", "unknown"))
    best_model_runs: dict[str, str | None] = {}

    bundle = {}
    evaluation = {
        "dataset_info": {
            "url_training_rows": len(url_rows),
            "page_feature_rows": len(evidence_rows),
            "url_model_features": list(URL_FEATURE_NAMES),
            "page_runtime_features": list(CONTENT_FEATURE_NAMES),
            "page_model_features": list(CONTENT_MODEL_FEATURE_NAMES),
        },
        "experiment_tracking": "mlflow" if mlflow is not None else "disabled",
        "configurations": {},
        "headline": (
            "Random stratified holdout precision, recall, and F1; "
            "validate on a separate time-based dataset before deployment"
        ),
    }

    for config, rows in datasets.items():
        matrix, labels, feature_order = _model_input(rows, config)
        if min(np.bincount(labels, minlength=2)) < 2:
            raise ValueError(f"{config} data needs at least two records per label")
        x_train, x_test, y_train, y_test = train_test_split(
            matrix,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels,
        )
        results = {}
        candidates = []
        for family, build_model in MODEL_FAMILIES.items():
            model = build_model()
            started = time.perf_counter()
            model.fit(x_train, y_train)
            training_seconds = time.perf_counter() - started
            inference_started = time.perf_counter()
            predictions = model.predict(x_test)
            inference_latency_ms = (
                (time.perf_counter() - inference_started) * 1000 / len(x_test)
            )
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, predictions, average="binary", zero_division=0
            )
            metrics = {
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "training_seconds": round(training_seconds, 4),
                "inference_latency_ms_per_url": round(inference_latency_ms, 4),
                "estimated_cloud_cost_usd_per_analysis": 0.0,
                "confusion_matrix": confusion_matrix(
                    y_test, predictions, labels=[0, 1]
                ).tolist(),
                "test_rows": int(len(y_test)),
            }
            run_id = _log_run(
                config,
                family,
                metrics,
                len(feature_order),
                model,
                dataset_sha256,
            )
            results[family] = metrics
            candidates.append((float(f1), family, model, run_id))

        best_score, best_family, best_model, best_run_id = max(
            candidates, key=lambda result: (result[0], result[1] == "logistic_regression")
        )
        best_model_runs[config] = best_run_id
        bundle[config] = {
            "model": best_model,
            "model_family": best_family,
            "feature_order": feature_order,
            "metrics": results[best_family],
        }
        evaluation["configurations"][config] = {
            "sample_count": len(rows),
            "best_model": best_family,
            "best_holdout_f1": round(best_score, 4),
            "models": results,
        }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    model_version = _register_local_version(
        dataset_sha256,
        best_model_runs,
        {
            name: {
                "model_family": item["model_family"],
                "holdout_f1": item["metrics"]["f1"],
            }
            for name, item in bundle.items()
        },
    )

    reference, _, reference_order = _model_input(evidence_rows, "combined")
    reference.to_csv(MODELS_DIR / "reference_features.csv", index=False)
    index = {
        "schema_version": 1,
        "model_version": model_version,
        "dataset_sha256": dataset_sha256,
        "model_artifact": str(MODEL_PATH.relative_to(ROOT)),
        "feature_order": reference_order,
        "configurations": {
            name: {
                "model_family": item["model_family"],
                "metrics": item["metrics"],
            }
            for name, item in bundle.items()
        },
    }
    (MODELS_DIR / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    EVALUATION_PATH.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(json.dumps(evaluation, indent=2))
    print(f"Saved trained model bundle to {MODEL_PATH}")
    return evaluation


if __name__ == "__main__":
    train_models()
