"""Data drift monitoring for the URL feature distribution.

Reference distribution is captured at training time
(``models/reference_features.csv``); current distribution comes from the live
analysis events. Uses Evidently AI when installed, and otherwise a built-in
PSI implementation so monitoring always works.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, Settings
from src.monitoring.metrics import MetricsStore


def population_stability_index(
    reference: list[float] | pd.Series,
    current: list[float] | pd.Series,
    bins: int = 10,
) -> float:
    """PSI between two samples of one feature. 0 = identical, >0.2 = drifted.

    Bin edges are taken from the reference quantiles. To stay robust when the
    reference has few distinct values (or many ties, as with count-like
    features), equal-width edges are used whenever the reference has fewer
    distinct edges than requested bins.
    """
    ref = pd.Series(reference).dropna().astype(float)
    cur = pd.Series(current).dropna().astype(float)
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    quantile_edges = sorted(set(float(ref.quantile(i / bins)) for i in range(bins + 1)))
    if len(quantile_edges) <= 2:
        # Degenerate reference (constant or nearly so): any change is drift.
        return 0.0 if _same_distribution(ref, cur) else 1.0

    if len(quantile_edges) - 1 < bins:
        edges = _equal_width_edges(ref, bins)
    else:
        edges = quantile_edges

    psi = 0.0
    epsilon = 1e-6
    for r, c in zip(_histogram(ref, edges), _histogram(cur, edges), strict=False):
        psi += (max(c, epsilon) - max(r, epsilon)) * math.log(max(c, epsilon) / max(r, epsilon))
    return round(psi, 4)


def drift_report(
    *,
    reference_path: str | Path | None = None,
    events_path: str | Path | None = None,
    threshold: float | None = None,
    min_events: int = 30,
) -> dict:
    """Compare the live feature distribution against the training reference."""
    settings = Settings()
    reference_path = Path(reference_path or settings.model_dir / "reference_features.csv")
    threshold = settings.psi_threshold if threshold is None else threshold

    if not reference_path.exists():
        return {"available": False, "reason": f"reference features missing: {reference_path}"}

    reference = pd.read_csv(reference_path)
    events = MetricsStore(events_path).events()
    feature_rows = [e["features"] for e in events if e.get("features")]

    if len(feature_rows) < min_events:
        return {
            "available": False,
            "reason": f"only {len(feature_rows)} events with features (need {min_events})",
            "events_with_features": len(feature_rows),
        }

    current = pd.DataFrame(feature_rows)
    shared = [c for c in reference.columns if c in current.columns]

    per_feature: dict[str, dict] = {}
    drifted: list[str] = []
    for column in shared:
        psi = population_stability_index(reference[column], current[column])
        per_feature[column] = {
            "psi": psi,
            "drifted": psi > threshold,
            "reference_mean": round(float(reference[column].mean()), 4),
            "current_mean": round(float(current[column].mean()), 4),
        }
        if psi > threshold:
            drifted.append(column)

    report = {
        "available": True,
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "threshold": threshold,
        "drifted_features": drifted,
        "drift_share": round(len(drifted) / max(len(shared), 1), 4),
        "dataset_drift": bool(drifted),
        "per_feature": per_feature,
        "evidently": _evidently_report(reference[shared], current[shared]),
    }

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "drift_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _evidently_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Optional Evidently AI summary; report availability without failing."""
    try:
        from evidently import ColumnMapping  # noqa: F401
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        result = report.as_dict()
        drifted = result["metrics"][0]["result"].get("number_of_drifted_columns")
        return {"available": True, "drifted_columns": drifted}
    except Exception as exc:  # noqa: BLE001 - Evidently is optional
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def _equal_width_edges(reference: pd.Series, bins: int) -> list[float]:
    low = float(reference.min())
    high = float(reference.max())
    if high <= low:
        return [low - 1e-6, high + 1e-6]
    step = (high - low) / bins
    return [low + i * step for i in range(bins + 1)]


def _same_distribution(ref: pd.Series, cur: pd.Series) -> bool:
    epsilon = 1e-9
    scale = max(abs(float(ref.mean())), 1.0)
    return abs(float(ref.mean()) - float(cur.mean())) <= epsilon * scale


def _histogram(values: pd.Series, edges: list[float]) -> list[float]:
    total = len(values)
    counts = [0] * (len(edges) - 1)
    for value in values:
        placed = False
        for i in range(len(edges) - 1):
            lower, upper = edges[i], edges[i + 1]
            if i == len(edges) - 2:
                inside = lower <= value <= upper
            else:
                inside = lower <= value < upper
            if inside:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    return [c / total for c in counts]
