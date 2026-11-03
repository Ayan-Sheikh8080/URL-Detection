"""Evaluate existing URL-only models on a local OpenPhish snapshot.

This script extracts lexical features only. It never visits the URLs.
The phishing-only feed cannot establish overall accuracy or specificity.
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

from src.features.url_features import extract_url_features

ROOT = Path(__file__).resolve().parents[1]
FEED = ROOT / "data" / "external" / "openphish_community.txt"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "reports" / "external"

MODEL_FILES = {
    "logistic_regression": "url_only__logistic_regression.joblib",
    "random_forest": "url_only__random_forest.joblib",
    "xgboost": "url_only__xgboost.joblib",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    if not FEED.exists():
        raise FileNotFoundError(f"Feed snapshot not found: {FEED}")

    raw_lines = FEED.read_text(encoding="utf-8").splitlines()
    urls = list(dict.fromkeys(line.strip() for line in raw_lines if line.strip()))

    # Validate URL syntax locally; do not resolve DNS or make network requests.
    valid_urls = []
    invalid_count = 0
    for url in urls:
        parsed = urlparse(url)
        if parsed.scheme.lower() in {"http", "https"} and parsed.hostname:
            valid_urls.append(url)
        else:
            invalid_count += 1

    if not valid_urls:
        raise ValueError("The feed contains no syntactically valid HTTP(S) URLs.")

    index_path = MODEL_DIR / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    feature_columns = index["models"]["url_only"]["feature_columns"]

    rows = []
    for url in valid_urls:
        features = extract_url_features(url)
        rows.append([float(features.get(col, 0.0)) for col in feature_columns])

    X = pd.DataFrame(rows, columns=feature_columns, dtype=float)
    predictions_csv = OUT_DIR / "openphish_predictions.csv"
    report_json = OUT_DIR / "openphish_evaluation.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    result = {
        "benchmark": "OpenPhish Community Feed",
        "benchmark_type": "phishing_only",
        "label_semantics": (
            "Feed entries are treated as phishing-positive examples. "
            "This evaluation does not include benign examples."
        ),
        "evaluated_at_utc": utc_now(),
        "feed_file": str(FEED.relative_to(ROOT)).replace("\\", "/"),
        "feed_file_mtime_utc": datetime.fromtimestamp(
            FEED.stat().st_mtime, timezone.utc
        ).isoformat(),
        "feed_sha256": hashlib.sha256(FEED.read_bytes()).hexdigest(),
        "raw_nonempty_lines": len([x for x in raw_lines if x.strip()]),
        "unique_feed_urls": len(urls),
        "valid_http_urls": len(valid_urls),
        "invalid_lines_skipped": invalid_count,
        "url_feature_count": len(feature_columns),
        "network_requests_made": 0,
        "models": {},
        "limitations": [
            "Phishing-only dataset: overall accuracy, specificity, and false-positive rate cannot be calculated.",
            "Feed labels are inherited from the source; they are not independently re-verified here.",
            "A phishing detection rate on this snapshot does not establish general real-world performance.",
            "Predictions use the existing trained artifacts without retraining.",
        ],
    }

    csv_rows = []
    for model_name, artifact_name in MODEL_FILES.items():
        artifact = MODEL_DIR / artifact_name
        if not artifact.exists():
            print(f"Skipping missing model artifact: {artifact_name}")
            continue

        model = joblib.load(artifact)
        probabilities = model.predict_proba(X)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        detected = int(predictions.sum())
        missed = int(len(predictions) - detected)

        result["models"][model_name] = {
            "artifact": artifact_name,
            "threshold": 0.5,
            "sample_count": len(valid_urls),
            "predicted_phishing": detected,
            "predicted_benign": missed,
            "phishing_detection_rate": round(detected / len(valid_urls), 4),
            "mean_phishing_probability": round(float(probabilities.mean()), 6),
            "median_phishing_probability": round(float(np.median(probabilities)), 6),
            "p95_phishing_probability": round(float(np.percentile(probabilities, 95)), 6),
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
        }

        print(
            f"{model_name}: flagged {detected}/{len(valid_urls)} "
            f"({detected / len(valid_urls):.1%}); "
            f"mean phishing probability={probabilities.mean():.4f}"
        )

        for i, url in enumerate(valid_urls):
            row = next(
                (r for r in csv_rows if r["url"] == url),
                {
                    "url": url,
                    "url_sha256": hashlib.sha256(url.encode("utf-8")).hexdigest(),
                },
            )
            row[f"{model_name}_phishing_probability"] = round(float(probabilities[i]), 6)
            row[f"{model_name}_prediction"] = (
                "phishing" if predictions[i] == 1 else "benign"
            )
            if row not in csv_rows:
                csv_rows.append(row)

    if not result["models"]:
        raise FileNotFoundError("No URL-only model artifacts were found.")

    with predictions_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["url", "url_sha256"]
        for name in MODEL_FILES:
            fieldnames.extend([f"{name}_phishing_probability", f"{name}_prediction"])
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)

    report_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\nReport: {report_json.relative_to(ROOT)}")
    print(f"Per-URL predictions: {predictions_csv.relative_to(ROOT)}")
    print("No URLs were opened and no network requests were made.")


if __name__ == "__main__":
    main()
