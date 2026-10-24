"""Sandboxed headless-browser visit logic (tool 2 of the agent design).

This module executes the *actual browsing*. It is designed to run inside a
locked-down container that holds no credentials and has no cloud role.

Safety controls implemented here:
  * observe-only: no clicking, no form submission, no downloads;
  * non-http(s) schemes (file:, data:, javascript:) are blocked;
  * private / loopback / link-local / metadata hosts are blocked;
  * single-label hosts (Docker service names) and hosts whose DNS answer points
    at a private address are blocked, including decimal/hex IP tricks;
  * hard wall-clock timeout per visit;
  * page content is captured as data only - it is never interpreted as
    instructions by this module.
"""

from __future__ import annotations

import ipaddress
import socket
import time
from pathlib import Path
from urllib.parse import urlparse

from src.agent.evidence import Evidence, FormEvidence

BLOCKED_SCHEMES = {"file", "data", "javascript", "ftp", "about", "blob"}

BLOCKED_HOSTNAMES = {
    "localhost",
    "metadata",
    "metadata.google.internal",
    "169.254.169.254",  # AWS/GCP instance metadata
    "100.100.100.200",  # Alibaba Cloud metadata
}

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


def is_private_host(hostname: str) -> bool:
    """True if the host is loopback, private, link-local or metadata."""
    if not hostname:
        return True
    host = hostname.strip().strip("[]").lower().rstrip(".")
    if host in BLOCKED_HOSTNAMES:
        return True
    if host.endswith(".local") or host.endswith(".internal") or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def is_visitable_url(url: str, *, block_private: bool = True) -> tuple[bool, str]:
    """Validate a URL against the sandbox safety policy."""
    try:
        parsed = urlparse((url or "").strip())
    except ValueError:
        return False, "unparseable url"
    if parsed.scheme.lower() not in {"http", "https"}:
        return False, f"scheme not allowed: {parsed.scheme or '(none)'}"
    if parsed.scheme.lower() in BLOCKED_SCHEMES:
        return False, f"scheme not allowed: {parsed.scheme}"
    host = parsed.hostname or ""
    if not host:
        return False, "missing hostname"
    if block_private and is_private_host(host):
        return False, f"private or non-routable host blocked: {host}"
    return True, ""


def _blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    mapped = getattr(ip, "ipv4_mapped", None)  # ::ffff:10.0.0.1 style addresses
    if mapped is not None:
        ip = mapped
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def host_is_unsafe(host: str) -> str:
    """Return a reason if the host must not be visited, or '' if it is allowed.

    Stricter than is_private_host: it also blocks single-label names (Docker
    service names such as 'api'), numeric tricks such as decimal IPs, and any
    domain whose DNS answer points at a private or internal address.
    """
    host = (host or "").strip().strip("[]").lower().rstrip(".")
    if not host:
        return "missing hostname"
    if "." not in host and ":" not in host:
        return f"single-label host blocked: {host}"
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return ""  # unresolvable: the browser will fail on its own
    for info in infos:
        try:
            ip = ipaddress.ip_address(str(info[4][0]).split("%")[0])
        except ValueError:
            continue
        if _blocked_ip(ip):
            return f"{host} resolves to a private address"
    return ""


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
    allowed, reason = is_visitable_url(url, block_private=block_private)
    if not allowed:
        evidence.error = f"blocked by sandbox policy: {reason}"
        return evidence
    if block_private:
        unsafe = host_is_unsafe(urlparse(url).hostname or "")
        if unsafe:
            evidence.error = f"blocked by sandbox policy: {unsafe}"
            return evidence

    deadline = time.monotonic() + max(timeout_seconds, 1.0)
    started = time.monotonic()
    redirects: list[str] = []

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=headless,
                args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
            )
            try:
                context = browser.new_context(
                    accept_downloads=False,
                    service_workers="block",
                    java_script_enabled=True,
                    ignore_https_errors=False,
                )

                unsafe_cache: dict[str, str] = {}

                def _unsafe(host: str) -> str:
                    if host not in unsafe_cache:
                        unsafe_cache[host] = host_is_unsafe(host)
                    return unsafe_cache[host]

                def _route(route) -> None:
                    """Abort anything outside the allow-list (defence in depth)."""
                    request_url = route.request.url
                    ok, why = is_visitable_url(request_url, block_private=block_private)
                    if ok and block_private:
                        why = _unsafe(urlparse(request_url).hostname or "")
                        ok = not why
                    if not ok and not why.startswith("unparseable"):
                        route.abort()
                    else:
                        route.continue_()

                context.route("**/*", _route)

                page = context.new_page()
                page.set_default_timeout(max(int((deadline - time.monotonic()) * 1000), 1000))

                def _on_response(response) -> None:
                    if response.status in (301, 302, 303, 307, 308):
                        location = response.headers.get("location")
                        if location:
                            redirects.append(location)

                page.on("response", _on_response)

                response = page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=max(int((deadline - time.monotonic()) * 1000), 1000),
                )

                # Settle briefly for client-side rendering, but stay within budget.
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    page.wait_for_timeout(min(1500, int(remaining * 1000)))

                evidence.status_code = response.status if response else None
                evidence.final_url = page.url

                collected = page.evaluate(_COLLECT_SCRIPT)

                evidence.title = collected.get("title", "")
                evidence.text_sample = (collected.get("text", "") or "")[:20000]
                evidence.iframes = list(collected.get("iframes", []))
                evidence.external_script_count = int(collected.get("external_script_count", 0))
                evidence.num_hidden_inputs = int(collected.get("num_hidden_inputs", 0))
                evidence.has_password_field = bool(collected.get("has_password_field"))

                final_host = (urlparse(evidence.final_url).hostname or "").lower()
                forms: list[FormEvidence] = []
                for raw in collected.get("forms", []):
                    action_host = (urlparse(raw.get("action", "")).hostname or "").lower()
                    forms.append(
                        FormEvidence(
                            action=raw.get("action", ""),
                            method=raw.get("method", "get"),
                            has_password=bool(raw.get("has_password")),
                            input_names=list(raw.get("input_names", [])),
                            submits_cross_domain=bool(action_host and final_host and action_host != final_host),
                        )
                    )
                evidence.forms = forms

                if screenshot_dir:
                    try:
                        target = Path(screenshot_dir)
                        target.mkdir(parents=True, exist_ok=True)
                        shot = target / f"shot_{int(time.time() * 1000)}.png"
                        page.screenshot(path=str(shot), full_page=False)
                        evidence.screenshot_path = str(shot)
                    except Exception:  # screenshot is best-effort
                        pass

                chain = [url, *[r for r in redirects], evidence.final_url]
                evidence.redirect_chain = list(dict.fromkeys(chain))
                context.close()
            finally:
                browser.close()
    except Exception as exc:  # noqa: BLE001 - report as missing evidence
        evidence.error = f"{type(exc).__name__}: {exc}"

    evidence.load_ms = round((time.monotonic() - started) * 1000, 2)
    return evidence
