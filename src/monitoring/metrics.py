"""Durable, process-local analysis counters and append-only event logging."""

import json
import logging
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MONITORING_EVENTS_PATH = ROOT / "data" / "monitoring" / "events.jsonl"
logger = logging.getLogger(__name__)


class MonitoringMetrics:
    def __init__(self, events_path: Path = MONITORING_EVENTS_PATH):
        self.events_path = events_path
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.total_analyses = 0
        self.verdict_distribution = {"safe": 0, "suspicious": 0, "phishing": 0}
        self.latency_samples: list[float] = []
        self.model_score_sum = 0.0
        self.known_cost_sum = 0.0
        self.known_cost_count = 0
        self.browser_failures = 0
        self.start_time = time.time()
        self._load_events()

    def _load_events(self) -> None:
        if not self.events_path.exists():
            return
        for line_number, line in enumerate(
            self.events_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                event = json.loads(line)
                self._apply(event)
            except (json.JSONDecodeError, TypeError, ValueError):
                logger.warning("Ignoring malformed monitoring event at line %s", line_number)

    def _apply(self, event: dict[str, Any]) -> None:
        self.total_analyses += 1
        verdict = event.get("verdict")
        if verdict in self.verdict_distribution:
            self.verdict_distribution[verdict] += 1
        latency = float(event.get("load_ms", 0))
        if math.isfinite(latency) and latency >= 0:
            self.latency_samples.append(latency)
            self.latency_samples = self.latency_samples[-1000:]
        self.model_score_sum += float(event.get("model_score", 0))
        estimated_cost = event.get("estimated_cloud_cost_usd")
        if estimated_cost is not None:
            self.known_cost_sum += float(estimated_cost)
            self.known_cost_count += 1
        if not event.get("browser_ok", True):
            self.browser_failures += 1

    def record_analysis(self, analysis_data: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            event = {
                **analysis_data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "analysis_id": self.total_analyses + 1,
            }
            serialized = json.dumps(event, allow_nan=False)
            with self.events_path.open("a", encoding="utf-8") as stream:
                stream.write(serialized + "\n")
            self._apply(event)
            return event

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            sorted_latencies = sorted(self.latency_samples)
            count = len(sorted_latencies)
            return {
                "total_analyses": self.total_analyses,
                "verdict_distribution": dict(self.verdict_distribution),
                "average_latency_ms": round(sum(sorted_latencies) / count, 2) if count else 0,
                "p50_latency_ms": round(sorted_latencies[math.ceil(0.5 * count) - 1], 2)
                if count
                else 0,
                "p95_latency_ms": round(sorted_latencies[math.ceil(0.95 * count) - 1], 2)
                if count
                else 0,
                "max_latency_ms": round(max(sorted_latencies), 2) if count else 0,
                "average_model_score": round(self.model_score_sum / self.total_analyses, 4)
                if self.total_analyses
                else 0,
                "average_estimated_cloud_cost_usd": round(
                    self.known_cost_sum / self.known_cost_count, 8
                )
                if self.known_cost_count
                else None,
                "cost_measurements": self.known_cost_count,
                "browser_failure_rate": round(
                    self.browser_failures / self.total_analyses * 100, 2
                )
                if self.total_analyses
                else 0,
                "uptime_seconds": round(time.time() - self.start_time, 2),
            }

    def recent_features(self, limit: int = 1000) -> list[dict[str, float]]:
        if not self.events_path.exists():
            return []
        rows: list[dict[str, float]] = []
        for line in self.events_path.read_text(encoding="utf-8").splitlines()[-limit:]:
            try:
                features = json.loads(line).get("features")
                if isinstance(features, dict):
                    rows.append({key: float(value) for key, value in features.items()})
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        return rows
