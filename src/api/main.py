"""FastAPI application for end-to-end URL phishing analysis."""

import asyncio
import logging
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, generate_latest
import httpx

from src.agent.browser import SCREENSHOT_DIR, UnsafeTarget, analyze_url, validate_public_url
from src.agent.features import extract_content_features
from src.agent.reasoning import ReasoningInput, get_reasoner
from src.ml.features import extract_url_features
from src.ml.model_registry import ModelRegistry
from src.monitoring.drift import check_feature_drift
from src.monitoring.metrics import MonitoringMetrics

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Phishing URL Analyzer",
    version="1.0.0",
    description="URL and sandboxed-browser evidence analysis for phishing detection.",
)
model_registry = ModelRegistry()
monitoring = MonitoringMetrics()
prometheus_registry = CollectorRegistry()
analysis_counter = Counter(
    "url_analyzer_analyses_total",
    "Number of completed URL analyses",
    ["verdict"],
    registry=prometheus_registry,
)
analysis_latency = Gauge(
    "url_analyzer_last_analysis_latency_ms",
    "Latency of the latest completed URL analysis",
    registry=prometheus_registry,
)
browser_failures = Counter(
    "url_analyzer_browser_failures_total",
    "Number of analyses where browser evidence could not be collected",
    registry=prometheus_registry,
)


class AnalysisRequest(BaseModel):
    url: str = Field(min_length=1, max_length=8192)
    model_config_name: Literal["url_only", "content_only", "combined"] = Field(
        default="combined", alias="model_config"
    )
    use_llm: bool = False
    browser_enabled: bool = True


async def _browser_analysis(url: str) -> dict:
    sandbox_url = os.getenv("BROWSER_SANDBOX_URL")
    if not sandbox_url:
        return await asyncio.to_thread(
            analyze_url,
            url,
            max(1000, min(int(os.getenv("BROWSER_TIMEOUT_MS", "8000")), 15000)),
        )
    try:
        async with httpx.AsyncClient(timeout=18) as client:
            response = await client.post(f"{sandbox_url.rstrip('/')}/analyze", json={"url": url})
            response.raise_for_status()
            return response.json()
    except (httpx.HTTPError, ValueError) as error:
        logger.error("Browser sandbox request failed: %s", error)
        return {
            "browser_ok": False,
            "evidence": {
                "redirect_chain": [],
                "final_url": url,
                "final_domain": "",
                "title": "",
                "page_text": "",
                "forms": [],
                "iframes": [],
                "certificate": None,
                "screenshot_path": None,
            },
            "error": f"Browser sandbox unavailable: {error}",
        }


@app.get("/")
def root() -> dict:
    return {
        "message": "Phishing URL Analyzer API",
        "docs": "/docs",
        "endpoints": [
            "/api/analyze",
            "/api/health",
            "/api/monitoring/stats",
            "/api/monitoring/drift",
            "/metrics",
        ],
    }


@app.get("/api/health")
@app.get("/api/healthz")
def health() -> dict:
    index = model_registry.index()
    return {
        "status": "ok",
        "service": "phishing-url-analyzer",
        "version": "1.0.0",
        "model_ready": bool(index.get("configurations")),
    }


@app.post("/api/analyze")
async def analyze(request: AnalysisRequest) -> dict:
    started = asyncio.get_running_loop().time()
    try:
        await asyncio.to_thread(validate_public_url, request.url)
    except UnsafeTarget as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    url_features = extract_url_features(request.url)
    if request.browser_enabled:
        browser_result = await _browser_analysis(request.url)
    else:
        browser_result = {
            "browser_ok": False,
            "evidence": {
                "redirect_chain": [],
                "final_url": request.url,
                "final_domain": validate_public_url(request.url)[0],
                "title": "",
                "page_text": "",
                "forms": [],
                "iframes": [],
                "certificate": None,
                "screenshot_path": None,
            },
            "error": "Browser analysis was disabled for this request",
        }

    evidence = browser_result["evidence"]
    content_features = extract_content_features(request.url, evidence)
    features = {
        **url_features,
        **{f"url_{key}": value for key, value in url_features.items()},
        **{f"page_{key}": value for key, value in content_features.items()},
    }
    try:
        prediction = model_registry.predict(request.model_config_name, features)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail="No trained model is available. Run `python -m src.data.prepare` and `python -m src.training.train`.",
        ) from error
    except (ValueError, KeyError) as error:
        raise HTTPException(status_code=503, detail=f"Model inference failed: {error}") from error

    reasoner = get_reasoner("bedrock" if request.use_llm else None)
    reasoning = reasoner.reason(
        ReasoningInput(
            url=request.url,
            url_model_score=prediction["score"],
            evidence=evidence,
            page_text=evidence.get("page_text", ""),
        )
    )
    if not browser_result["browser_ok"]:
        reasoning.reasons.append(
            "Browser evidence was unavailable; interpret this result as URL-model-only evidence"
        )
    elapsed_ms = (asyncio.get_running_loop().time() - started) * 1000
    estimated_cost = None if reasoning.reasoner != "rule_based" else 0.0
    drift_features = {
        **{f"url_{key}": value for key, value in url_features.items()},
        **{f"page_{key}": value for key, value in content_features.items()},
    }
    event = monitoring.record_analysis({
        "verdict": reasoning.verdict,
        "model_score": prediction["score"],
        "combined_score": reasoning.combined_score,
        "load_ms": elapsed_ms,
        "browser_ok": browser_result["browser_ok"],
        "features": drift_features,
        "model_config": request.model_config_name,
        "reasoner": reasoning.reasoner,
        "estimated_cloud_cost_usd": estimated_cost,
    })
    analysis_counter.labels(reasoning.verdict).inc()
    analysis_latency.set(elapsed_ms)
    if not browser_result["browser_ok"]:
        browser_failures.inc()

    screenshot_path = evidence.get("screenshot_path")
    screenshot_url = (
        f"/api/screenshots/{Path(screenshot_path).name}" if screenshot_path else None
    )
    public_evidence = {key: value for key, value in evidence.items() if key != "screenshot_path"}
    return {
        "analysis_id": event["analysis_id"],
        "url": request.url,
        "verdict": reasoning.verdict,
        "confidence": reasoning.confidence,
        "reasons": reasoning.reasons,
        "model_score": prediction["score"],
        "combined_score": reasoning.combined_score,
        "model_config": request.model_config_name,
        "model_family": prediction["model_family"],
        "top_factors": prediction["top_factors"] + reasoning.top_factors,
        "top_factor_method": prediction["explanation_method"],
        "evidence_summary": public_evidence,
        "screenshot_url": screenshot_url,
        "browser_ok": browser_result["browser_ok"],
        "browser_error": browser_result["error"],
        "load_ms": round(elapsed_ms, 2),
        "estimated_cloud_cost_usd": estimated_cost,
        "reasoner": reasoning.reasoner,
    }


@app.get("/api/screenshots/{filename}")
def screenshot(filename: str):
    if Path(filename).name != filename or not filename.endswith(".png"):
        raise HTTPException(status_code=404, detail="Screenshot not found")
    path = (SCREENSHOT_DIR / filename).resolve()
    if path.parent != SCREENSHOT_DIR.resolve() or not path.is_file():
        raise HTTPException(status_code=404, detail="Screenshot not found")
    return FileResponse(path, media_type="image/png")


@app.get("/api/monitoring/stats")
def monitoring_stats() -> dict:
    return monitoring.get_stats()


@app.get("/api/monitoring/drift")
def monitoring_drift() -> dict:
    reference = model_registry.reference_features()
    recent = monitoring.recent_features()
    return check_feature_drift(reference, recent)


@app.get("/metrics")
def prometheus_metrics() -> PlainTextResponse:
    return PlainTextResponse(
        generate_latest(prometheus_registry).decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )
