"""Time-limited, read-only browser evidence collection with SSRF guards."""

import ipaddress
import socket
import ssl
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright

from src.ml.features import parse_http_url

ROOT = Path(__file__).resolve().parents[2]
SCREENSHOT_DIR = ROOT / "data" / "screenshots"
MAX_PAGE_TEXT = 5000


class UnsafeTarget(ValueError):
    """Raised when a URL resolves to a non-public network address."""


def _public_addresses(hostname: str) -> list[str]:
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None:
        if not address.is_global:
            raise UnsafeTarget("Target resolves to a non-public address")
        return [str(address)]

    try:
        records = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except OSError as error:
        raise UnsafeTarget("Target hostname could not be resolved") from error
    addresses = sorted({record[4][0] for record in records})
    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise UnsafeTarget("Target resolves to a non-public address")
    return addresses


def validate_public_url(url: str) -> tuple[str, str, int]:
    parsed, hostname = parse_http_url(url)
    if hostname == "localhost" or hostname.endswith((".localhost", ".local")):
        raise UnsafeTarget("Local hostnames are not allowed")
    addresses = _public_addresses(hostname)
    port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
    return hostname, addresses[0], port


def _certificate(hostname: str, address: str, port: int) -> dict:
    if port != 443:
        return {"checked": False, "is_valid": False, "issuer": "", "subject": ""}
    context = ssl.create_default_context()
    try:
        with socket.create_connection((address, port), timeout=3) as raw_socket:
            with context.wrap_socket(raw_socket, server_hostname=hostname) as tls_socket:
                certificate = tls_socket.getpeercert()
        issuer = dict(pair[0] for pair in certificate.get("issuer", ()))
        subject = dict(pair[0] for pair in certificate.get("subject", ()))
        return {
            "checked": True,
            "is_valid": True,
            "issuer": issuer.get("organizationName") or issuer.get("commonName", ""),
            "subject": subject.get("commonName", ""),
            "expiration_date": certificate.get("notAfter", ""),
        }
    except (OSError, ssl.SSLError, ValueError):
        return {"checked": True, "is_valid": False, "issuer": "", "subject": ""}


def analyze_url(url: str, timeout_ms: int = 8000) -> dict:
    """Visit a public HTTP(S) URL without interacting with page controls."""
    hostname, address, port = validate_public_url(url)
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    screenshot_name = f"{uuid.uuid4().hex}.png"
    screenshot_path = SCREENSHOT_DIR / screenshot_name
    evidence = {
        "redirect_chain": [],
        "final_url": url,
        "final_domain": hostname,
        "title": "",
        "page_text": "",
        "forms": [],
        "iframes": [],
        "certificate": _certificate(hostname, address, port),
        "screenshot_path": str(screenshot_path.relative_to(ROOT)).replace("\\", "/"),
    }

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
            )
            context = browser.new_context(
                accept_downloads=False,
                ignore_https_errors=False,
                service_workers="block",
                viewport={"width": 1280, "height": 900},
            )
            validated_hosts = {hostname}

            def guard_request(route):
                requested_url = route.request.url
                if requested_url.startswith(("data:", "blob:", "about:")):
                    route.continue_()
                    return
                try:
                    _, request_hostname = parse_http_url(requested_url)
                    if request_hostname not in validated_hosts:
                        validate_public_url(requested_url)
                        validated_hosts.add(request_hostname)
                except (ValueError, UnsafeTarget):
                    route.abort("blockedbyclient")
                    return
                if route.request.resource_type in {"websocket", "eventsource"}:
                    route.abort("blockedbyclient")
                    return
                route.continue_()

            context.route("**/*", guard_request)
            page = context.new_page()
            page.set_default_navigation_timeout(timeout_ms)
            page.set_default_timeout(timeout_ms)
            page.on(
                "request",
                lambda request: evidence["redirect_chain"].append(request.url)
                if request.is_navigation_request() and request.frame == page.main_frame
                else None,
            )
            response = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            if response is not None:
                evidence["final_url"] = page.url
                _, evidence["final_domain"] = parse_http_url(page.url)

            evidence["title"] = page.title()[:300]
            evidence["page_text"] = page.evaluate(
                "(limit) => (document.body?.innerText || '').slice(0, limit)",
                MAX_PAGE_TEXT,
            )
            evidence["forms"] = page.locator("form").evaluate_all(
                """forms => forms.map(form => ({
                    action: form.getAttribute('action') || '',
                    method: (form.getAttribute('method') || 'GET').toUpperCase(),
                    inputs: Array.from(form.querySelectorAll('input')).slice(0, 50).map(input => ({
                        name: input.getAttribute('name') || '',
                        type: input.getAttribute('type') || 'text'
                    }))
                })).slice(0, 100)"""
            )
            evidence["iframes"] = page.locator("iframe").evaluate_all(
                """frames => frames.slice(0, 100).map(frame => ({
                    src: frame.getAttribute('src') || '',
                    name: frame.getAttribute('name') || ''
                }))"""
            )
            page.screenshot(path=str(screenshot_path), full_page=False, timeout=timeout_ms)
            context.close()
            browser.close()
        return {"browser_ok": True, "evidence": evidence, "error": None}
    except Exception as error:
        screenshot_path.unlink(missing_ok=True)
        return {
            "browser_ok": False,
            "evidence": {
                **evidence,
                "screenshot_path": None,
                "page_text": "",
                "forms": [],
                "iframes": [],
            },
            "error": str(error)[:300],
        }
