"""Build URL lists for the crawl.

Reads:
  data/raw/tranco.csv          (Tranco list: rank,domain  - no header)
  data/raw/phishing_urls.txt   (one phishing URL per line, e.g. OpenPhish feed)

Writes:
  data/raw/legit_urls.txt         (https://domain, randomly sampled from the top sites)
  data/raw/phish_urls_sample.txt  (deduplicated, one URL per domain, shuffled)

Run from the project root:
  python src/data/make_url_lists.py --legit 500 --phish 500
"""

import argparse
import csv
import random
from pathlib import Path
from urllib.parse import urlparse

# Infrastructure / CDN domains that rank high but have no real web page to analyse.
SKIP_WORDS = (
    "akamai", "akadns", "edgekey", "edgesuite", "cloudfront", "fastly", "googleapis",
    "gstatic", "amazonaws", "azure", "doubleclick", "googlesyndication", "root-servers",
    "gtld-servers", "cdn", "nstld", "windowsupdate", "apple-dns", "icloud-content",
)


def read_tranco(path: Path, top_n: int) -> list[str]:
    domains = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            domain = (row[1] if len(row) > 1 else row[0]).strip().lower()
            if domain in ("domain", "") or any(w in domain for w in SKIP_WORDS):
                continue
            domains.append(domain)
            if len(domains) >= top_n:
                break
    return domains


def read_phishing(path: Path) -> list[str]:
    seen_urls, seen_domains, urls = set(), set(), []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        url = line.strip()
        if not url.lower().startswith(("http://", "https://")) or url in seen_urls:
            continue
        domain = urlparse(url).netloc.lower()
        if not domain or domain in seen_domains:  # one URL per domain
            continue
        seen_urls.add(url)
        seen_domains.add(domain)
        urls.append(url)
    return urls


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tranco", default="data/raw/tranco.csv")
    p.add_argument("--phishing", default="data/raw/phishing_urls.txt")
    p.add_argument("--legit", type=int, default=500, help="how many legitimate URLs")
    p.add_argument("--phish", type=int, default=500, help="how many phishing URLs")
    p.add_argument("--top-n", type=int, default=5000, help="sample legit sites from the top N")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)

    domains = read_tranco(Path(args.tranco), args.top_n)
    legit = rng.sample(domains, min(args.legit, len(domains)))
    legit_urls = [f"https://{d}" for d in legit]
    Path("data/raw/legit_urls.txt").write_text("\n".join(legit_urls) + "\n", encoding="utf-8")

    phish = read_phishing(Path(args.phishing))
    rng.shuffle(phish)
    phish = phish[: args.phish]
    Path("data/raw/phish_urls_sample.txt").write_text("\n".join(phish) + "\n", encoding="utf-8")

    print(f"legitimate URLs written: {len(legit_urls)}  -> data/raw/legit_urls.txt")
    print(f"phishing URLs written:   {len(phish)}  -> data/raw/phish_urls_sample.txt")


if __name__ == "__main__":
    main()
