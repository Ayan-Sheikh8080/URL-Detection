"""Custom monitoring metrics (proposal feature 10).

Every analysis appends one JSON line to ``events.jsonl``. The API exposes
aggregated verdict statistics, latency percentiles, browser timeout/error rates
and mean scores, which is exactly what the proposal asks to monitor.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT

_TIMEOUT_MARKERS = ("timeout", "timed out")


class MetricsStore:
    def __init__(self, events_path: str | Path | None = None):
        self.events_path = Path(
            events_path or PROJECT_ROOT / "data" / "monitoring" / "events.jsonl"
        )

    # ------------------------------------------------------------------ #
    def record(
        self,
        *,
        url: str,
        verdict: str,
        confidence: float,
        model_score: float | None,
        browser_ok: bool,
        latency_ms: float,
        features: dict | None = None,
        reasons: list[str] | None = None,
        browser_error: str | None = None,
    ) -> None:
        error = str(browser_error)[:300] if browser_error else None
        event = {
            "ts": time.time(),
            "url": url,
            "verdict": verdict,
            "confidence": _finite(confidence),
            "model_score": _finite(model_score),
            "browser_ok": bool(browser_ok),
            "browser_error": error,
            "browser_timed_out": bool(error and _is_timeout(error)),
            "latency_ms": _finite(latency_ms) or 0.0,
            "features": _finite_mapping(features or {}),
            "reasons": [str(item)[:200] for item in (reasons or [])[:3]],
        }
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.events_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, allow_nan=False) + "\n")

    # ------------------------------------------------------------------ #
    def events(self, limit: int | None = None) -> list[dict]:
        if not self.events_path.exists():
            return []
        rows: list[dict] = []
        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        if limit is not None:
            rows = rows[-max(int(limit), 1):]
        return rows

    def summary(self, window: int = 1000) -> dict:
        window = max(int(window), 1)
        events = self.events(limit=window)
        if not events:
            return {
                "total_analyses": 0,
                "window": window,
                "verdicts": {},
                "verdict_share": {},
                "latency_ms": {},
                "browser": {},
                "confidence_mean": None,
                "model_score_mean": None,
            }

        verdicts = Counter(str(e.get("verdict", "unknown")) for e in events)
        latencies = sorted(
            value
            for value in (_finite(e.get("latency_ms")) for e in events)
            if value is not None
        )
        failures = [e for e in events if not e.get("browser_ok")]
        timeouts = [e for e in failures if e.get("browser_timed_out")]
        errors = [e for e in failures if not e.get("browser_timed_out")]
        model_scores = [
            value
            for value in (_finite(e.get("model_score")) for e in events)
            if value is not None
        ]
        confidences = [
            value
            for value in (_finite(e.get("confidence")) for e in events)
            if value is not None
        ]
        error_reasons = Counter(
            str(e.get("browser_error"))[:160]
            for e in failures
            if e.get("browser_error")
        )

        return {
            "total_analyses": len(events),
            "window": window,
            "verdicts": dict(verdicts),
            "verdict_share": {k: round(v / len(events), 4) for k, v in verdicts.items()},
            "latency_ms": _latency_stats(latencies),
            "browser": {
                "failures": len(failures),
                "failure_rate": round(len(failures) / len(events), 4),
                "timeouts": len(timeouts),
                "timeout_rate": round(len(timeouts) / len(events), 4),
                "errors": len(errors),
                "error_rate": round(len(errors) / len(events), 4),
                "top_error_reasons": dict(error_reasons.most_common(5)),
            },
            "confidence_mean": _mean(confidences),
            "model_score_mean": _mean(model_scores),
        }


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_mapping(values: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in values.items():
        number = _finite(value)
        if number is not None:
            out[str(key)] = number
    return out


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _is_timeout(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _TIMEOUT_MARKERS)


def _latency_stats(sorted_values: list[float]) -> dict:
    if not sorted_values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "mean": round(sum(sorted_values) / len(sorted_values), 2),
        "p50": _percentile(sorted_values, 50),
        "p95": _percentile(sorted_values, 95),
        "p99": _percentile(sorted_values, 99),
        "max": round(sorted_values[-1], 2),
    }


def _percentile(sorted_values: list[float], pct: int) -> float:
    if not sorted_values:
        return 0.0
    index = min(int(round((pct / 100) * (len(sorted_values) - 1))), len(sorted_values) - 1)
    return round(sorted_values[index], 2)
