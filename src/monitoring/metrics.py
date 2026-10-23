"""Custom monitoring metrics (proposal feature 10).

Every analysis appends one JSON line to ``events.jsonl``. The API exposes
aggregated verdict statistics, latency percentiles and browser failure rates,
which is exactly what the proposal asks to monitor.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from src.config import PROJECT_ROOT


class MetricsStore:
    def __init__(self, events_path: str | Path | None = None):
        self.events_path = Path(events_path or PROJECT_ROOT / "data" / "monitoring" / "events.jsonl")

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
    ) -> None:
        event = {
            "ts": time.time(),
            "url": url,
            "verdict": verdict,
            "confidence": confidence,
            "model_score": model_score,
            "browser_ok": browser_ok,
            "latency_ms": latency_ms,
            "features": features or {},
            "reasons": (reasons or [])[:3],
        }
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.events_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")

    # ------------------------------------------------------------------ #
    def events(self, limit: int | None = None) -> list[dict]:
        if not self.events_path.exists():
            return []
        rows = []
        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if limit is not None:
            rows = rows[-limit:]
        return rows

    def summary(self, window: int = 1000) -> dict:
        events = self.events(limit=window)
        if not events:
            return {"total_analyses": 0, "verdicts": {}, "latency_ms": {}, "browser": {}}

        verdicts = Counter(e.get("verdict", "unknown") for e in events)
        latencies = sorted(float(e.get("latency_ms", 0.0)) for e in events)
        browser_failures = sum(1 for e in events if not e.get("browser_ok"))
        model_scores = [e["model_score"] for e in events if isinstance(e.get("model_score"), (int, float))]

        return {
            "total_analyses": len(events),
            "window": window,
            "verdicts": dict(verdicts),
            "verdict_share": {k: round(v / len(events), 4) for k, v in verdicts.items()},
            "latency_ms": {
                "mean": round(sum(latencies) / len(latencies), 2),
                "p50": _percentile(latencies, 50),
                "p95": _percentile(latencies, 95),
                "max": round(latencies[-1], 2),
            },
            "browser": {
                "failures": browser_failures,
                "failure_rate": round(browser_failures / len(events), 4),
            },
            "model_score_mean": round(sum(model_scores) / len(model_scores), 4) if model_scores else None,
        }


def _percentile(sorted_values: list[float], pct: int) -> float:
    if not sorted_values:
        return 0.0
    index = min(int(round((pct / 100) * (len(sorted_values) - 1))), len(sorted_values) - 1)
    return round(sorted_values[index], 2)
