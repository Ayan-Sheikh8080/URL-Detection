
"""Evaluate URL-only models on OpenPhish + Tranco external benchmarks.

Labels:
    1 = phishing
    0 = legitimate

This script uses lexical URL features only. It does not visit URLs,
resolve DNS, or make network requests.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.features.url_features import extract_url_features

ROOT = Path(__file__).resolve().parents[1]
PHISH_FILE = ROOT / "data/external/openphish_community.txt"
LEGIT_FILE = ROOT / "data/external/tranco_presumed_benign.csv"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "reports/external"

MODEL_FILES = {
    "logistic_regression": "url_only__logistic_regression.joblib",
    "random_forest": "url_only__random_forest.joblib",
    "xgboost": "url_only__xgboost.joblib",
}

THRESHOLD = 0.5


def valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return (
            parsed.scheme.lower() in {"http", "https"}
            and bool(parsed.hostname)
        )
    except ValueError:
        return False


def load_benchmark() -> tuple[pd.DataFrame, dict]:
    if not PHISH_FILE.exists():
        raise FileNotFoundError(PHISH_FILE)

    if not LEGIT_FILE.exists():
        raise FileNotFoundError(LEGIT_FILE)

    phish_lines = PHISH_FILE.read_text(
        encoding="utf-8", errors="ignore"
    ).splitlines()

    phish_urls = list(dict.fromkeys(
        line.strip() for line in phish_lines if line.strip()
    ))

    legit_df = pd.read_csv(LEGIT_FILE)

    required = {"url", "label"}
    if not required.issubset(legit_df.columns):
        raise ValueError(
            f"Tranco CSV must contain {required}; "
            f"found {list(legit_df.columns)}"
        )

    legit_df = legit_df[["url", "label"]].copy()
    legit_df["url"] = legit_df["url"].astype(str).str.strip()
    legit_df["label"] = pd.to_numeric(
        legit_df["label"], errors="coerce"
    )

    if not legit_df["label"].dropna().isin([0]).all():
        raise ValueError(
            "Expected Tranco presumed-benign labels to be 0."
        )

    legit_urls = legit_df["url"].dropna().unique().tolist()

    phish_valid = [u for u in phish_urls if valid_url(u)]
    legit_valid = [u for u in legit_urls if valid_url(u)]

    # Do not silently assign conflicting labels to overlapping URLs.
    overlap = set(phish_valid) & set(legit_valid)

    if overlap:
        print(
            f"[warning] Removing {len(overlap)} URLs "
            "present in both classes."
        )

    phish_valid = [u for u in phish_valid if u not in overlap]
    legit_valid = [u for u in legit_valid if u not in overlap]

    records = (
        [{"url": u, "label": 1, "source": "OpenPhish"}
         for u in phish_valid]
        +
        [{"url": u, "label": 0, "source": "Tranco"}
         for u in legit_valid]
    )

    benchmark = pd.DataFrame(records)

    if benchmark.empty or benchmark["label"].nunique() != 2:
        raise ValueError("External benchmark must contain both classes.")

    benchmark = benchmark.drop_duplicates(
        subset=["url"]
    ).reset_index(drop=True)

    metadata = {
        "openphish_unique_urls": len(phish_urls),
        "openphish_valid_urls": len(phish_valid),
        "tranco_unique_urls": len(legit_urls),
        "tranco_valid_urls": len(legit_valid),
        "cross_class_overlap_removed": len(overlap),
        "total_samples": len(benchmark),
        "phishing_samples": int((benchmark["label"] == 1).sum()),
        "legitimate_samples": int((benchmark["label"] == 0).sum()),
    }

    return benchmark, metadata


def main() -> None:
    benchmark, dataset_metadata = load_benchmark()

    index_path = MODEL_DIR / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            "models/index.json missing. Train models first."
        )

    index = json.loads(index_path.read_text(encoding="utf-8"))
    feature_columns = index["models"]["url_only"]["feature_columns"]

    feature_rows = []
    for url in benchmark["url"]:
        features = extract_url_features(url)
        feature_rows.append([
            float(features.get(col, 0.0))
            for col in feature_columns
        ])

    X = pd.DataFrame(
        feature_rows,
        columns=feature_columns,
        dtype=float,
    )
    y = benchmark["label"].astype(int).to_numpy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "benchmark": "OpenPhish + Tranco external benchmark",
        "benchmark_type": "mixed_phishing_and_legitimate",
        "label_semantics": {
            "1": "phishing",
            "0": "legitimate",
        },
        "threshold": THRESHOLD,
        "evaluated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": dataset_metadata,
        "feature_count": len(feature_columns),
        "network_requests_made": 0,
        "models": {},
        "limitations": [
            "Tranco domains are presumed legitimate, not independently verified.",
            "OpenPhish labels are inherited from the feed.",
            "This benchmark measures lexical URL detection only.",
            "No URLs were opened or network requests made.",
        ],
    }

    prediction_rows = []

    for model_name, artifact_name in MODEL_FILES.items():
        artifact = MODEL_DIR / artifact_name

        if not artifact.exists():
            print(f"[warning] Missing artifact: {artifact_name}")
            continue

        model = joblib.load(artifact)

        classes = list(model.classes_)
        if 1 not in classes:
            raise ValueError(
                f"{model_name} does not contain phishing class 1."
            )

        positive_index = classes.index(1)
        probabilities = model.predict_proba(X)[:, positive_index]
        predictions = (probabilities >= THRESHOLD).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y, predictions, labels=[0, 1]
        ).ravel()

        metrics = {
            "accuracy": round(float(accuracy_score(y, predictions)), 4),
            "precision": round(float(
                precision_score(y, predictions, zero_division=0)
            ), 4),
            "recall": round(float(
                recall_score(y, predictions, zero_division=0)
            ), 4),
            "f1": round(float(
                f1_score(y, predictions, zero_division=0)
            ), 4),
            "roc_auc": round(float(
                roc_auc_score(y, probabilities)
            ), 4),
            "specificity": round(
                float(tn / (tn + fp)) if (tn + fp) else 0.0, 4
            ),
            "phishing_detection_rate": round(
                float(tp / (tp + fn)) if (tp + fn) else 0.0, 4
            ),
            "false_positive_rate": round(
                float(fp / (fp + tn)) if (fp + tn) else 0.0, 4
            ),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
            "sample_count": len(y),
            "predicted_phishing": int(predictions.sum()),
            "predicted_legitimate": int(len(predictions) - predictions.sum()),
            "mean_phishing_probability": round(
                float(probabilities.mean()), 6
            ),
        }

        report["models"][model_name] = {
            "artifact": artifact_name,
            "threshold": THRESHOLD,
            **metrics,
        }

        print(
            f"{model_name}: "
            f"ROC-AUC={metrics['roc_auc']:.4f}, "
            f"F1={metrics['f1']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"specificity={metrics['specificity']:.4f}"
        )

        for i, row in benchmark.iterrows():
            url = row["url"]
            prediction_rows.append({
                "url": url,
                "label": int(y[i]),
                "source": row["source"],
                "url_sha256": hashlib.sha256(
                    url.encode("utf-8")
                ).hexdigest(),
                f"{model_name}_phishing_probability": round(
                    float(probabilities[i]), 6
                ),
                f"{model_name}_prediction": (
                    "phishing" if predictions[i] == 1
                    else "legitimate"
                ),
            })

    if not report["models"]:
        raise RuntimeError("No URL-only model artifacts found.")

    # Merge individual model predictions into one CSV.
    predictions_df = benchmark.copy()
    predictions_df["url_sha256"] = predictions_df["url"].apply(
        lambda u: hashlib.sha256(u.encode("utf-8")).hexdigest()
    )

    for model_name, artifact_name in MODEL_FILES.items():
        if model_name not in report["models"]:
            continue

        model = joblib.load(MODEL_DIR / artifact_name)
        probabilities = model.predict_proba(X)[:, list(model.classes_).index(1)]
        predictions = (probabilities >= THRESHOLD).astype(int)

        predictions_df[f"{model_name}_phishing_probability"] = probabilities
        predictions_df[f"{model_name}_prediction"] = np.where(
            predictions == 1, "phishing", "legitimate"
        )

    predictions_df.to_csv(
        OUT_DIR / "openphish_predictions.csv",
        index=False,
    )

    (OUT_DIR / "openphish_evaluation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\nExternal benchmark summary:")
    print(json.dumps(dataset_metadata, indent=2))
    print(f"\nReport: reports/external/openphish_evaluation.json")
    print(f"Predictions: reports/external/openphish_predictions.csv")
    print("Network requests made: 0")


if __name__ == "__main__":
    main()
