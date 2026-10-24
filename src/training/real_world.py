"""Real-world benchmark and prompt-injection suite.

The proposal asks for real-world testing and prompt-injection testing.
Network access is *not* required: the defaults use bundled fixtures in
``data/raw/real_world/``.  If outbound network is allowed, OpenPhish/tranco
lists can be downloaded on demand with ``--allow-network``.

Examples::

    python -m src.training.real_world
    python -m src.training.real_world --urls data/raw/real_world/real_world_urls.csv --no-browse
    python -m src.training.real_world --injection-only
    python -m src.training.real_world --allow-network --openphish-url https://...
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.reasoning import RuleBasedReasoner  # noqa: E402
from src.config import PROJECT_ROOT as ROOT  # noqa: E402
from src.ml.model_registry import ModelRegistry  # noqa: E402
from src.training.common import binary_metrics  # noqa: E402

RAW_REAL_WORLD = ROOT / "data" / "raw" / "real_world"
INJECTION_DIR = RAW_REAL_WORLD / "prompt_injection"
REPORTS_DIR = ROOT / "reports"


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Real-world and injection tests")
    parser.add_argument("--urls", type=Path, default=RAW_REAL_WORLD / "real_world_urls.csv")
    parser.add_argument("--injection-dir", type=Path, default=INJECTION_DIR)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--no-browse", action="store_true", help="URL-only benchmark")
    parser.add_argument("--injection-only", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--openphish-url", type=str, default="")
    args = parser.parse_args(argv)

    report: dict = {"url_benchmark": None, "injection_suite": None}

    if not args.injection_only:
        report["url_benchmark"] = _url_benchmark(args)

    report["injection_suite"] = _injection_suite(args.injection_dir)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "real_world_report.json"
    out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["injection_suite"], indent=2))
    return report


def _load_url_list(path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row.get("url"):
                continue
            label = row.get("label", row.get("class", 0))
            try:
                label_int = int(str(label).strip())
            except ValueError:
                label_int = 1 if "phish" in str(label).lower() else 0
            rows.append((str(row["url"]).strip(), label_int))
    return rows


def _url_benchmark(args: argparse.Namespace) -> dict:
    urls = _load_url_list(args.urls)
    if not urls:
        return {"available": False, "reason": "no real-world URLs available"}

    registry = ModelRegistry(args.models_dir)
    model = registry.load("url_only")
    if model is None:
        fallback = registry.load_best()
        if getattr(fallback, "config", None) == "url_only":
            model = fallback
    if model is None:
        return {"available": False, "reason": "no URL-only model available (run train first)"}

    try:
        from src.features.url_features import extract_url_features

        y_true = [label for _, label in urls]
        y_pred = []
        y_proba = []
        for url, _ in urls:
            proba = model.predict_proba(extract_url_features(url))
            y_proba.append(proba)
            y_pred.append(1 if proba >= 0.5 else 0)
        metrics = binary_metrics(y_true, y_pred, y_proba)
        return {"available": True, "n_samples": len(urls), "metrics": metrics}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": str(exc)[:200]}


def _injection_suite(injection_dir: Path) -> dict:
    reasoner = RuleBasedReasoner()
    if not injection_dir.exists():
        return {"available": False, "reason": f"injection fixtures missing: {injection_dir}"}

    results: list[dict] = []
    for path in sorted(injection_dir.glob("*.html")):
        html = path.read_text(encoding="utf-8", errors="ignore")
        evidence = _evidence_from_html(html, "https://paypal-secure-login.com/signin")
        verdict = reasoner.reason(
            url_score=0.9,
            combined_score=None,
            evidence=evidence,
            domain_info=evidence.get("domain_checks") or {},
        )
        results.append(
            {
                "fixture": path.name,
                "verdict": verdict.verdict,
                "confidence": verdict.confidence,
                "downgraded_to_safe": verdict.verdict == "safe",
                "reasons": verdict.reasons,
            }
        )

    downgraded = sum(1 for item in results if item["downgraded_to_safe"])
    return {
        "available": True,
        "n_fixtures": len(results),
        "downgraded_to_safe": downgraded,
        "passed": downgraded == 0,
        "results": results,
    }


def _evidence_from_html(html: str, requested_url: str) -> dict:
    lowered = html.lower()
    has_password = "type=\"password\"" in lowered or "type='password'" in lowered
    form_action = ""
    if "<form" in lowered:
        # crude extraction for fixtures only
        start = lowered.find("action=")
        if start >= 0:
            quote = html[start + 7]
            end = html.find(quote, start + 8)
            if end > start:
                form_action = html[start + 8 : end]
    title = ""
    if "<title>" in lowered and "</title>" in lowered:
        ts = lowered.find("<title>") + 7
        te = lowered.find("</title>")
        title = html[ts:te]

    evidence = {
        "requested_url": requested_url,
        "final_url": requested_url,
        "redirect_chain": [requested_url],
        "status_code": 200,
        "title": title,
        "text_sample": html[:8000],
        "forms": [
            {
                "action": form_action,
                "method": "post" if "<form" in lowered and "method" in lowered and "get" not in lowered.split("<form", 1)[1].split(">", 1)[0].lower() else "post",
                "has_password": has_password,
                "input_names": ["email", "password"] if has_password else ["q"],
                "submits_cross_domain": bool(form_action and "example" in form_action),
            }
        ],
        "iframes": [],
        "has_password_field": has_password,
        "external_script_count": 3,
        "num_hidden_inputs": 1,
        "cert": {"issuer": "R3", "subject_cn": "example.com", "days_to_expiry": 30, "self_signed": False, "host_matches": True},
        "domain_checks": {},
        "error": None,
    }
    from src.agent.checks import domain_checks

    evidence["domain_checks"] = domain_checks(requested_url, requested_url, title)
    return evidence


if __name__ == "__main__":
    main()
