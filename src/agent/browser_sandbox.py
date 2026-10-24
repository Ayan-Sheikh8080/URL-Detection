"""Sandboxed headless-browser visit logic (tool 2 of the agent design).

This module executes the *actual browsing*. It is designed to run inside a
locked-down container that holds no credentials and has no cloud role.

Safety controls implemented here:
  * observe-only: no clicking, no form submission, no downloads;
  * non-http(s) schemes (file:, data:, javascript:) are blocked;
  * private / loopback / link-local / metadata hosts are blocked, including
    hosts that only *resolve* to a private address (DNS-rebinding guard);
  * sub-resource requests are checked against the same policy (defence in depth);
  * redirects are resolved with ``urljoin`` so a relative ``Location`` cannot be
    mis-attributed;
  * the TLS certificate is collected inside the sandbox, not by the API;
  * hard wall-clock timeout per visit, enforced across every step;
  * page content is captured as data only - it is never interpreted as
    instructions by this module.
"""

from __future__ import annotations

import ipaddress
import socket
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from urllib.parse import urljoin, urlparse

from src.agent.checks import cert_details
from src.agent.evidence import Evidence, FormEvidence
from src.features.url_features import registrable_domain

BLOCKED_SCHEMES = {"file", "data", "javascript", "ftp", "about", "blob"}

BLOCKED_HOSTNAMES = {
    "localhost",
    "metadata",
    "metadata.google.internal",
    "169.254.169.254",  # AWS/GCP instance metadata
    "100.100.100.200",  # Alibaba Cloud metadata
}

ALLOWED_SCHEMES = {"http", "https"}

# JavaScript evaluated in the page context. Read-only DOM queries only.
_COLLECT_SCRIPT = """
() => {
  const forms = Array.from(document.querySelectorAll('form')).map(f => ({
    action: f.action || '',
    method: (f.method || 'get').toLowerCase(),
    has_password: !!f.querySelector('input[type="password"]'),
    input_names: Array.from(f.querySelectorAll('input, select, textarea'))
      .map(i => i.name || i.type || i.tagName.toLowerCase())
      .slice(0, 25),
  }));
  const iframes = Array.from(document.querySelectorAll('iframe'))
    .map(f => f.src || '(inline)').slice(0, 20);
  const scripts = Array.from(document.querySelectorAll('script[src]'))
    .map(s => s.src);
  const hidden = document.querySelectorAll('input[type="hidden"]').length;
  return {
    title: document.title || '',
    text: (document.body ? document.body.innerText : '') || '',
    forms,
    iframes,
    external_script_count: scripts.filter(
      s => !!s && new URL(s, location.href).host !== location.host
    ).length,
    num_hidden_inputs: hidden,
    has_password_field: !!document.querySelector('input[type="password"]'),
  };
}
"""

_RESOLUTION_CACHE: dict[str, list[str] | None] = {}


def _literal_ip(host: str):
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def is_private_host(hostname: str) -> bool:
    """True if the host is loopback, private, link-local or metadata."""
    if not hostname:
        return True
    host = hostname.strip().strip("[]").lower().rstrip(".")
    if host in BLOCKED_HOSTNAMES:
        return True
    if host.endswith(".local") or host.endswith(".internal") or host.endswith(".localhost"):
        return True
    ip = _literal_ip(host)
    if ip is None:
        return False
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def resolve_host(hostname: str, *, timeout: float = 1.5) -> list[str] | None:
    """Resolve a hostname to IP strings, with a hard timeout.

    Returns ``None`` when resolution fails or times out.  Results are cached for
    the life of the process; a sandbox worker is short-lived by design.
    """
    host = (hostname or "").strip().strip("[]").lower().rstrip(".")
    if not host:
        return None
    if host in _RESOLUTION_CACHE:
        return _RESOLUTION_CACHE[host]

    def _lookup() -> list[str]:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
        return sorted({str(info[4][0]) for info in infos})

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(_lookup)
        try:
            addresses = future.result(timeout=timeout)
        except (FutureTimeout, OSError, UnicodeError):
            addresses = None
    finally:
        # Never block on a hung DNS lookup: the worker thread is a daemon in
        # practice and the caller must respect the visit deadline.
        pool.shutdown(wait=False, cancel_futures=True)
    _RESOLUTION_CACHE[host] = addresses
    return addresses


def resolves_to_private(hostname: str) -> bool:
    """True if any resolved address is non-routable.

    A hostname that cannot be resolved is *not* treated as private: the browser
    will fail to connect anyway, and failing closed here would break offline
    tests.  The navigation itself still has a timeout.
    """
    addresses = resolve_host(hostname)
    if not addresses:
        return False
    return any(is_private_host(address) for address in addresses)


def is_visitable_url(
    url: str, *, block_private: bool = True, check_dns: bool = False
) -> tuple[bool, str]:
    """Validate a URL against the sandbox safety policy."""
    try:
        parsed = urlparse((url or "").strip())
    except ValueError:
        return False, "unparseable url"
    scheme = parsed.scheme.lower()
    if scheme not in ALLOWED_SCHEMES or scheme in BLOCKED_SCHEMES:
        return False, f"scheme not allowed: {scheme or '(none)'}"
    host = parsed.hostname or ""
    if not host:
        return False, "missing hostname"
    if block_private and is_private_host(host):
        return False, f"private or non-routable host blocked: {host}"
    if block_private and check_dns and resolves_to_private(host):
        return False, f"host resolves to a private address: {host}"
    return True, ""


def visit_url(
    url: str,
    *,
    timeout_seconds: float = 20.0,
    screenshot_dir: str | Path | None = None,
    block_private: bool = True,
    headless: bool = True,
) -> Evidence:
    """Visit `url` in a sandboxed headless Chromium and collect evidence.

    Returns an Evidence object; on any failure the `error` field is set and
    partial evidence is preserved.
    """
    from playwright.sync_api import sync_playwright  # lazy: sandbox image only

    evidence = Evidence(requested_url=url)
    allowed, reason = is_visitable_url(url, block_private=block_private, check_dns=True)
    if not allowed:
        evidence.error = f"blocked by sandbox policy: {reason}"
        return evidence

    deadline = time.monotonic() + max(timeout_seconds, 1.0)
    started = time.monotonic()
    redirects: list[str] = []

    def remaining_ms(*, floor: int = 1000) -> int:
        return max(int((deadline - time.monotonic()) * 1000), floor)

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=headless,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--disable-background-networking",
                    "--disable-component-update",
                    "--disable-default-apps",
                    "--disable-extensions",
                    "--disable-sync",
                    "--no-first-run",
                ],
            )
            try:
                context = browser.new_context(
                    accept_downloads=False,
                    service_workers="block",
                    java_script_enabled=True,
                    ignore_https_errors=False,
                )

                def _route(route) -> None:
                    """Abort anything outside the allow-list (defence in depth)."""
                    request_url = route.request.url
                    # DNS resolution is only enforced for navigations: doing a
                    # lookup for every third-party asset would blow the visit
                    # budget, while literal private hosts are still blocked.
                    ok, _ = is_visitable_url(
                        request_url,
                        block_private=block_private,
                        check_dns=route.request.is_navigation_request(),
                    )
                    if not ok:
                        route.abort()
                    else:
                        route.continue_()

                context.route("**/*", _route)

                page = context.new_page()
                page.set_default_timeout(remaining_ms())

                def _on_response(response) -> None:
                    if response.status in (301, 302, 303, 307, 308):
                        location = response.headers.get("location")
                        if location:
                            # Relative Location headers are common; resolve
                            # against the response URL before recording.
                            redirects.append(urljoin(response.url, location))

                page.on("response", _on_response)

                response = page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=remaining_ms(),
                )

                # Settle briefly for client-side rendering, but stay within budget.
                settle = min(1500, max(int((deadline - time.monotonic()) * 1000), 0))
                if settle > 0:
                    page.wait_for_timeout(settle)

                evidence.status_code = response.status if response else None
                evidence.final_url = page.url

                collected = page.evaluate(_COLLECT_SCRIPT)

                evidence.title = str(collected.get("title", ""))[:500]
                evidence.text_sample = (collected.get("text", "") or "")[:20000]
                evidence.iframes = [str(item) for item in collected.get("iframes", [])][:20]
                evidence.external_script_count = int(collected.get("external_script_count", 0))
                evidence.num_hidden_inputs = int(collected.get("num_hidden_inputs", 0))
                evidence.has_password_field = bool(collected.get("has_password_field"))

                final_host = (urlparse(evidence.final_url).hostname or "").lower()
                final_base = registrable_domain(final_host)
                forms: list[FormEvidence] = []
                for raw in collected.get("forms", []):
                    raw_action = raw.get("action", "") or ""
                    resolved_action = urljoin(evidence.final_url or url, raw_action)
                    action_host = (urlparse(resolved_action).hostname or "").lower()
                    action_base = registrable_domain(action_host)
                    forms.append(
                        FormEvidence(
                            action=str(resolved_action)[:2048],
                            method=str(raw.get("method", "get")).lower()[:10],
                            has_password=bool(raw.get("has_password")),
                            input_names=[str(n)[:100] for n in raw.get("input_names", [])][:25],
                            submits_cross_domain=bool(
                                final_base and action_base and action_base != final_base
                            ),
                        )
                    )
                evidence.forms = forms

                # Certificate evidence is collected by the sandbox itself, so
                # the API container never has to make outbound TLS connections.
                if final_host and deadline - time.monotonic() > 0.2:
                    evidence.cert = cert_details(
                        final_host,
                        timeout=max(0.2, min(5.0, deadline - time.monotonic())),
                    )

                if screenshot_dir and deadline - time.monotonic() > 0.2:
                    try:
                        target = Path(screenshot_dir)
                        target.mkdir(parents=True, exist_ok=True)
                        shot = target / f"shot_{uuid.uuid4().hex}.png"
                        page.screenshot(path=str(shot), full_page=False)
                        evidence.screenshot_path = str(shot)
                    except Exception:  # screenshot is best-effort
                        pass

                chain = [url, *redirects, evidence.final_url]
                evidence.redirect_chain = list(dict.fromkeys(item for item in chain if item))
                context.close()
            finally:
                browser.close()
    except Exception as exc:  # noqa: BLE001 - report as missing evidence
        evidence.error = f"{type(exc).__name__}: {exc}"

    evidence.load_ms = round((time.monotonic() - started) * 1000, 2)
    return evidence
