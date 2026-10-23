"""HTTP wrapper around the sandboxed browser visit.

Runs in its own container (see ``Dockerfile.sandbox``) with no credentials and
no cloud role, on an internal network. The API container never launches a
browser, which keeps the blast radius of a hostile page inside this service.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from starlette.concurrency import run_in_threadpool

from src.agent.browser_sandbox import is_visitable_url, visit_url

app = FastAPI(
    title="phishing-browser-sandbox",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)

_SCREENSHOT_DIR = Path(os.getenv("SCREENSHOT_DIR", "/tmp/screenshots"))
_TIMEOUT = float(os.getenv("SANDBOX_TIMEOUT_SECONDS", "20"))
_BLOCK_PRIVATE = os.getenv("BLOCK_PRIVATE_NETWORKS", "true").lower() in {"1", "true", "yes"}
_HEADLESS = os.getenv("BROWSER_HEADLESS", "true").lower() in {"1", "true", "yes"}


class VisitRequest(BaseModel):
    url: str
    screenshot: bool = True

    @field_validator("url")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or len(value) > 2048:
            raise ValueError("url must be 1-2048 characters")
        return value.strip()


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "service": "sandbox"}


@app.post("/visit")
async def visit(request: VisitRequest) -> dict:
    allowed, reason = is_visitable_url(request.url, block_private=_BLOCK_PRIVATE)
    if not allowed:
        raise HTTPException(status_code=400, detail=f"blocked by sandbox policy: {reason}")

    evidence = await run_in_threadpool(
        visit_url,
        request.url,
        timeout_seconds=_TIMEOUT,
        screenshot_dir=_SCREENSHOT_DIR if request.screenshot else None,
        block_private=_BLOCK_PRIVATE,
        headless=_HEADLESS,
    )
    return evidence.to_dict()
