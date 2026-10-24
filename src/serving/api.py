"""FastAPI analysis API (proposal feature 7).

Endpoints
  * ``POST /analyze``            - analyze one URL, return verdict + evidence + factors
  * ``GET  /healthz``            - liveness plus model/sandbox readiness
  * ``GET  /monitoring/stats``   - verdict/latency/browser-failure statistics
  * ``GET  /monitoring/drift``   - feature drift against the training reference

``create_app(agent=...)`` allows injecting a stub agent in tests.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from src.agent.agent import PhishingAgent
from src.config import Settings
from src.features.url_features import extract_url_features
from src.monitoring.drift import drift_report
from src.monitoring.metrics import MetricsStore
from src.serving.schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse


def create_app(agent: PhishingAgent | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    metrics = MetricsStore(settings.events_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.agent is None:
            app.state.agent = PhishingAgent(settings)
        yield

    app = FastAPI(
        title="Phishing Link Analysis API",
        description=(
            "Sandboxed-browser phishing analysis agent: URL model score + live page "
            "evidence + reasoning layer, with verdict, confidence and reasons."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.agent = agent
    app.state.settings = settings
    app.state.metrics = metrics

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        current: PhishingAgent | None = getattr(app.state, "agent", None)
        configs: list[str] = []
        model_loaded = False
        reasoner = "unknown"
        if current is not None:
            registry = getattr(current, "registry", None)
            if registry is not None:
                configs = registry.available_configs()
                model_loaded = registry.load_best() is not None
            reasoner = getattr(current.reasoner, "reasoner_name", type(current.reasoner).__name__)
        return HealthResponse(
            status="ok",
            model_loaded=model_loaded,
            configs=configs,
            reasoner=reasoner,
            sandbox=settings.sandbox_url or "local-playwright",
        )

    @app.post("/analyze", response_model=AnalyzeResponse)
    def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
        current: PhishingAgent | None = getattr(app.state, "agent", None)
        if current is None:
            raise HTTPException(status_code=503, detail="agent not initialized")

        try:
            result = current.analyze(request.url, browse=request.browse)
        except Exception:  # noqa: BLE001 - never leak internals to the client
            raise HTTPException(status_code=500, detail="analysis failed") from None

        if settings.monitoring_enabled:
            metrics.record(
                url=result.url,
                verdict=result.verdict,
                confidence=result.confidence,
                model_score=result.model_score,
                browser_ok=bool(result.browser.get("ok")),
                browser_error=result.browser.get("error"),
                latency_ms=result.timings_ms.get("total_ms", 0.0),
                features=extract_url_features(result.url),
                reasons=result.reasons,
            )
        return AnalyzeResponse(**result.to_dict())

    @app.get("/monitoring/stats")
    def monitoring_stats(
        window: int = Query(1000, ge=1, le=100_000, description="Number of recent analyses"),
    ) -> dict:
        return metrics.summary(window=window)

    @app.get("/monitoring/drift")
    def monitoring_drift() -> dict:
        return drift_report(
            reference_path=settings.model_dir / "reference_features.csv",
            events_path=settings.events_path,
            threshold=settings.psi_threshold,
            reports_dir=settings.reports_dir,
        )

    return app


app = create_app()
