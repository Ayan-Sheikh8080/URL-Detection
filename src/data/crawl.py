"""Crawl URL lists through the sandbox service and save the evidence as JSONL.

SAFETY: this script never opens a browser itself. It only talks to the sandbox
container over HTTP, and it refuses to run if the sandbox is not reachable.

Run from the project root:
  python -m src.data.crawl --phish none --limit 5     # test with legitimate sites only
  python -m src.data.crawl --workers 3                # full crawl (resumes if stopped)

Output (label: 1 = phishing, 0 = legitimate):
  data/raw/crawled_real.jsonl   {"url", "label", "evidence"} for pages that loaded
  data/raw/crawl_failed.jsonl   {"url", "label", "error"} for pages that did not load
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from pathlib import Path
from urllib.parse import urlparse

import httpx

from src.agent.sandbox_client import SandboxClient


def read_urls(path: Path) -> list[str]:
    if not path.exists():
        sys.exit(f"missing file: {path}")
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return [ln.strip() for ln in lines if ln.strip().startswith(("http://", "https://"))]


def read_done(*paths: Path) -> set[str]:
    done: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["url"])
            except (ValueError, KeyError):
                continue
    return done


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sandbox", default=os.getenv("SANDBOX_URL") or "http://localhost:8080")
    p.add_argument("--legit", default="data/raw/legit_urls.txt")
    p.add_argument("--phish", default="data/raw/phish_urls_sample.txt", help="file, or 'none'")
    p.add_argument("--out", default="data/raw/crawled_real.jsonl")
    p.add_argument("--failed", default="data/raw/crawl_failed.jsonl")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--limit", type=int, default=0, help="max URLs per class (0 = all)")
    args = p.parse_args()

    base = args.sandbox.rstrip("/")
    if not base:
        sys.exit("no sandbox URL given - refusing to run without the sandbox")
    try:
        httpx.get(f"{base}/healthz", timeout=5).raise_for_status()
    except httpx.HTTPError as exc:
        sys.exit(f"sandbox not reachable at {base} ({exc}). Start it with: docker compose up")

    legit = read_urls(Path(args.legit))
    phish = [] if args.phish.lower() == "none" else read_urls(Path(args.phish))
    if args.limit:
        legit, phish = legit[: args.limit], phish[: args.limit]

    out, failed = Path(args.out), Path(args.failed)
    done = read_done(out, failed)
    jobs: list[tuple[str, int]] = []
    for lg, ph in zip_longest(legit, phish):
        if lg and lg not in done:
            jobs.append((lg, 0))
        if ph and ph not in done:
            jobs.append((ph, 1))
    total = len(jobs)
    print(f"{total} URLs to crawl ({len(done)} already done)")
    if not jobs:
        return

    client = SandboxClient(base, timeout_seconds=40)
    lock = threading.Lock()
    counts = {"ok": 0, "failed": 0}

    def work(job: tuple[str, int]) -> None:
        url, label = job
        evidence = client.visit(url, screenshot=False)
        good = not evidence.get("error") and bool(evidence.get("final_url"))
        if good:
            record = {"url": url, "label": label, "evidence": evidence}
        else:
            record = {"url": url, "label": label, "error": str(evidence.get("error"))[:200]}
        with lock:
            with (out if good else failed).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            counts["ok" if good else "failed"] += 1
            n = counts["ok"] + counts["failed"]
            kind = "phish" if label else "legit"
            status = "ok" if good else "fail"
            print(f"[{n}/{total}] {kind:5} {status:4} {urlparse(url).netloc}")

    pool = ThreadPoolExecutor(max_workers=args.workers)
    try:
        list(pool.map(work, jobs))
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        print("stopped - run the same command again to resume")
        return
    pool.shutdown()
    print(f"done: {counts['ok']} loaded, {counts['failed']} failed")


if __name__ == "__main__":
    main()
