import json
import time
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

# Path for monitoring events
MONITORING_EVENTS_PATH = Path("data/monitoring/events.jsonl")

class MonitoringMetrics:
    """Collects and tracks monitoring metrics for the phishing URL analyzer."""
    
    def __init__(self, events_path: Path = MONITORING_EVENTS_PATH):
        self.events_path = events_path
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory counters
        self.total_analyses = 0
        self.verdict_distribution = {"safe": 0, "suspicious": 0, "phishing": 0}
        self.latency_samples = []
        self.model_scores = []
        self.browser_failures = 0
        self.start_time = time.time()
        
        # Load existing events if any
        self._load_events()
    
    def _load_events(self):
        """Load existing monitoring events from file."""
        if self.events_path.exists():
            try:
                with open(self.events_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                event = json.loads(line)
                                self.total_analyses = max(self.total_analyses, event.get('analysis_id', 0))
                                if 'verdict' in event:
                                    v = event['verdict']
                                    if v in self.verdict_distribution:
                                        self.verdict_distribution[v] += 1
                                if 'latency_ms' in event:
                                    self.latency_samples.append(event['latency_ms'])
                                if 'model_score' in event:
                                    self.model_scores.append(event['model_score'])
                                if event.get('browser_ok', True) == False:
                                    self.browser_failures += 1
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                print(f"Error loading events: {e}")
    
    def record_analysis(self, analysis_data: Dict[str, Any]):
        """Record a completed analysis event."""
        try:
            # Add timestamp
            analysis_data['timestamp'] = datetime.utcnow().isoformat() + "Z"
            analysis_data['analysis_id'] = self.total_analyses + 1
            
            # Append to events file
            with open(self.events_path, 'a') as f:
                f.write(json.dumps(analysis_data) + "\n")
            
            # Update in-memory counters
            self.total_analyses += 1
            
            # Update verdict distribution
            verdict = analysis_data.get('verdict', 'safe')
            if verdict in self.verdict_distribution:
                self.verdict_distribution[verdict] += 1
            
            # Update latency
            latency = analysis_data.get('load_ms', 0)
            self.latency_samples.append(latency)
            
            # Keep only last 1000 samples
            if len(self.latency_samples) > 1000:
                self.latency_samples = self.latency_samples[-1000:]
            
            # Update model score
            model_score = analysis_data.get('model_score', 0) or 0
            self.model_scores.append(model_score)
            
            # Update browser failures
            browser_ok = analysis_data.get('browser_ok', True)
            if browser_ok == False:
                self.browser_failures += 1
        
        except Exception as e:
            print(f"Error recording analysis: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current monitoring statistics."""
        # Calculate percentiles
        sorted_latencies = sorted(self.latency_samples)
        n = len(sorted_latencies)
        
        if n == 0:
            return {
                "total_analyses": 0,
                "verdict_distribution": {"safe": 0, "suspicious": 0, "phishing": 0},
                "average_latency_ms": 0,
                "p50_latency_ms": 0,
                "p95_latency_ms": 0,
                "max_latency_ms": 0,
                "average_model_score": 0,
                "browser_failure_rate": 0,
                "uptime_seconds": round(time.time() - self.start_time, 2)
            }
        
        p50_idx = int(n * 0.5)
        p95_idx = int(n * 0.95)
        
        avg_latency = sum(sorted_latencies) / n
        
        # Calculate average model score
        avg_model_score = sum(self.model_scores) / len(self.model_scores) if self.model_scores else 0
        
        # Browser failure rate
        browser_failure_rate = (self.browser_failures / self.total_analyses * 100) if self.total_analyses > 0 else 0
        
        return {
            "total_analyses": self.total_analyses,
            "verdict_distribution": dict(self.verdict_distribution),
            "average_latency_ms": round(avg_latency, 2),
            "p50_latency_ms": round(float(sorted_latencies[p50_idx]) if p50_idx < n else 0, 2),
            "p95_latency_ms": round(float(sorted_latencies[p95_idx]) if p95_idx < n else 0, 2),
            "max_latency_ms": round(max(sorted_latencies), 2),
            "average_model_score": round(avg_model_score, 4),
            "browser_failure_rate": round(browser_failure_rate, 2),
            "uptime_seconds": round(time.time() - self.start_time, 2)
        }
    
    def record_event(self, event_type: str, data: Dict[str, Any]):
        """Record a generic monitoring event."""
        try:
            event = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "event_type": event_type,
                **data
            }
            self.record_analysis(event)
        except Exception as e:
            print(f"Error recording event: {e}")


def get_monitoring_stats() -> Dict[str, Any]:
    """Convenience function to get current monitoring statistics."""
    metrics = MonitoringMetrics()
    return metrics.get_stats()


def record_analysis_event(analysis_data: Dict[str, Any]):
    """Convenience function to record an analysis event."""
    metrics = MonitoringMetrics()
    metrics.record_analysis(analysis_data)