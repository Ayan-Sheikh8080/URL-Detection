"""Private HTTP service that runs browser visits without API credentials."""

import asyncio
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent.browser import UnsafeTarget, analyze_url

app = FastAPI(title="Phishing URL Browser Sandbox", docs_url=None, redoc_url=None)


class BrowserRequest(BaseModel):
    url: str = Field(min_length=1, max_length=8192)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(request: BrowserRequest) -> dict:
    try:
        return await asyncio.to_thread(
            analyze_url,
            request.url,
            max(1000, min(int(os.getenv("BROWSER_TIMEOUT_MS", "8000")), 15000)),
        )
    except UnsafeTarget as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
