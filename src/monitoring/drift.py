"""Population Stability Index for numeric or categorical feature drift."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PSI_THRESHOLD = 0.25


def _psi_from_counts(expected: np.ndarray, actual: np.ndarray) -> float:
    epsilon = 1e-6
    expected_share = (expected + epsilon) / (expected.sum() + epsilon * len(expected))
    actual_share = (actual + epsilon) / (actual.sum() + epsilon * len(actual))
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def _numeric_psi(reference: list[float], recent: list[float]) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(recent, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if not len(ref) or not len(cur):
        raise ValueError("Both distributions must contain finite values")
    if np.ptp(ref) == 0:
        value = float(ref[0])
        delta = max(abs(value) * 1e-6, 1e-6)
        boundaries = np.array([
            value - 2 * delta,
            value - delta / 2,
            value + delta / 2,
            value + 2 * delta,
        ])
    else:
        boundaries = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))
        if len(boundaries) < 2:
            boundaries = np.array([float(ref.min()), float(ref.max())])
        delta = max(float(np.ptp(ref)) * 1e-9, 1e-9)
        boundaries[0] -= delta
        boundaries[-1] += delta
    expected_bins = np.clip(np.digitize(ref, boundaries[1:-1]), 0, len(boundaries) - 2)
    actual_bins = np.clip(np.digitize(cur, boundaries[1:-1]), 0, len(boundaries) - 2)
    bins = max(1, len(boundaries) - 1)
    return _psi_from_counts(
        np.bincount(expected_bins, minlength=bins),
        np.bincount(actual_bins, minlength=bins),
    )


def _feature_psi(reference: list[Any], recent: list[Any]) -> float:
    try:
        ref = [float(value) for value in reference if value is not None]
        cur = [float(value) for value in recent if value is not None]
        if ref and cur:
            return _numeric_psi(ref, cur)
    except (TypeError, ValueError):
        pass
    categories = sorted({str(value) for value in reference + recent if value is not None})
    if not categories:
        raise ValueError("Feature has no values")
    expected = np.array([sum(str(value) == category for value in reference) for category in categories])
    actual = np.array([sum(str(value) == category for value in recent) for category in categories])
    return _psi_from_counts(expected, actual)


def check_feature_drift(
    reference_rows: list[dict[str, Any]],
    recent_rows: list[dict[str, Any]],
    threshold: float = PSI_THRESHOLD,
) -> dict[str, Any]:
    result = {
        "drift_detected": False,
        "psi_threshold": threshold,
        "psi_score": 0.0,
        "features_analyzed": 0,
        "drifted_features": [],
        "psi_by_feature": {},
        "message": "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if not reference_rows or not recent_rows:
        result["message"] = "Reference or recent feature data is unavailable"
        return result
    reference_features = set().union(*(row.keys() for row in reference_rows))
    recent_features = set().union(*(row.keys() for row in recent_rows))
    common = reference_features.intersection(recent_features)
    for feature in sorted(common):
        try:
            psi = _feature_psi(
                [row[feature] for row in reference_rows if feature in row],
                [row[feature] for row in recent_rows if feature in row],
            )
        except (TypeError, ValueError):
            continue
        result["psi_by_feature"][feature] = round(psi, 4)
        if psi > threshold:
            result["drifted_features"].append({"feature": feature, "psi": round(psi, 4)})
    result["features_analyzed"] = len(result["psi_by_feature"])
    if result["psi_by_feature"]:
        result["psi_score"] = round(
            float(np.mean(list(result["psi_by_feature"].values()))), 4
        )
    result["drift_detected"] = result["psi_score"] > threshold
    result["message"] = (
        "Feature drift detected" if result["drift_detected"] else "No significant feature drift detected"
    )
    result["drifted_features"].sort(key=lambda row: row["psi"], reverse=True)
    return result


def check_drift(
    reference_features_path: str | Path = ROOT / "models" / "reference_features.csv",
    recent_features_path: str | Path = ROOT / "data" / "processed" / "recent_features.csv",
) -> dict[str, Any]:
    reference_path = Path(reference_features_path)
    recent_path = Path(recent_features_path)
    if not reference_path.exists() or not recent_path.exists():
        return check_feature_drift([], [])
    reference = pd.read_csv(reference_path).to_dict(orient="records")
    recent = pd.read_csv(recent_path).to_dict(orient="records")
    return check_feature_drift(reference, recent)


def generate_drift_report(output_path: str | Path = ROOT / "reports" / "drift_report.json") -> dict:
    report = check_drift()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(generate_drift_report(), indent=2))
