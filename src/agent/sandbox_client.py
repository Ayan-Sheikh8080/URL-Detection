"""Client used by the agent/API to talk to the browser sandbox service.

In production the sandbox runs as its own container (`SANDBOX_URL` set), so
the API container never needs a browser or credentials. For local development
the client falls back to an in-process Playwright visit when the package is
installed.
"""

from __future__ import annotations

import httpx

from src.agent.evidence import Evidence


class SandboxClient:
    def __init__(self, base_url: str = "", *, timeout_seconds: float = 25.0):
        self.base_url = (base_url or "").rstrip("/")
        self.timeout_seconds = timeout_seconds

    def visit(self, url: str, *, screenshot: bool = True) -> dict:
        """Return an evidence dict; errors are reported inside the dict."""
        if self.base_url:
            return self._visit_remote(url, screenshot=screenshot)
        return self._visit_local(url, screenshot=screenshot)

    def _visit_remote(self, url: str, *, screenshot: bool) -> dict:
        try:
            response = httpx.post(
                f"{self.base_url}/visit",
                json={"url": url, "screenshot": screenshot},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            return Evidence(
                requested_url=url,
                error=f"sandbox service unavailable: {type(exc).__name__}: {exc}",
            ).to_dict()

    def _visit_local(self, url: str, *, screenshot: bool) -> dict:
        try:
            from src.agent.browser_sandbox import visit_url
            from src.config import Settings
        except ImportError as exc:
            return Evidence(
                requested_url=url,
                error=f"no sandbox configured and Playwright not installed: {exc}",
            ).to_dict()

        settings = Settings.from_env()
        evidence = visit_url(
            url,
            timeout_seconds=min(settings.sandbox_timeout_seconds, self.timeout_seconds),
            screenshot_dir=settings.screenshot_dir if screenshot else None,
            block_private=settings.block_private_networks,
            headless=settings.headless,
        )
        return evidence.to_dict()
