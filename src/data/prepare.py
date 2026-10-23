"""Prepare stage: validate the raw dataset, dedupe, split, persist.

Run with ``python -m src.data.prepare`` (DVC stage ``prepare``).

Supports:
  * the bundled synthetic sample (``data/raw/sample_urls.csv``);
  * the PhiUSIIL Phishing URL dataset: point ``data.raw_path`` at the downloaded
    CSV and set ``data.label_positive_is: legitimate`` (PhiUSIIL labels 1 as
    legitimate).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PROJECT_ROOT, load_params
from src.data.validate import (
    assert_valid,
    validate_evidence_records,
    validate_urls,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_raw_dataset(path: Path | str, positive_is: str = "phishing") -> pd.DataFrame:
    """Load a raw CSV and normalize to columns {url, label} with 1 = phishing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"raw dataset not found: {path}. Place a CSV with 'url'/'URL' and 'label' columns, "
            "or run with the bundled sample."
        )
    df = pd.read_csv(path)

    url_col = next((c for c in ("url", "URL", "Url") if c in df.columns), None)
    label_col = next((c for c in ("label", "Label") if c in df.columns), None)
    if url_col is None or label_col is None:
        raise ValueError(
            f"could not find url/label columns in {path.name}; found {list(df.columns)[:10]}"
        )

    out = pd.DataFrame({"url": df[url_col].astype(str).str.strip(), "label": df[label_col]})
    out["label"] = pd.to_numeric(out["label"], errors="coerce")

    # Normalize so that 1 always means phishing.
    positive_is = positive_is.strip().lower()
    if positive_is == "legitimate":
        out["label"] = 1 - out["label"]
    elif positive_is != "phishing":
        raise ValueError("data.label_positive_is must be 'phishing' or 'legitimate'")

    return out


def prepare(
    raw_path: Path | None = None,
    *,
    positive_is: str | None = None,
    test_size: float | None = None,
    random_state: int | None = None,
    crawled_path: Path | None = None,
) -> dict:
    """Run the full preparation stage and return a metadata summary."""
    params = load_params().get("data", {})
    raw_path = raw_path or PROJECT_ROOT / params.get("raw_path", "data/raw/sample_urls.csv")
    positive_is = positive_is or params.get("label_positive_is", "phishing")
    test_size = test_size if test_size is not None else float(params.get("test_size", 0.2))
    random_state = random_state if random_state is not None else int(params.get("random_state", 42))
    crawled_path = crawled_path or PROJECT_ROOT / params.get(
        "crawled_path", "data/raw/crawled_evidence.jsonl"
    )

    df = load_raw_dataset(raw_path, positive_is=positive_is)

    report = validate_urls(df)
    assert_valid(report)

    before = len(df)
    df = df.dropna(subset=["url", "label"])
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)

    # Stratify only when both classes have enough members for the split,
    # otherwise fall back to a plain random split.
    label_counts = df["label"].value_counts()
    can_stratify = df["label"].nunique() == 2 and int(label_counts.min()) >= 2 and test_size * len(df) >= 2
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"] if can_stratify else None,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    crawled_summary: dict = {"present": False}
    if crawled_path.exists():
        records = _read_jsonl(crawled_path)
        ev_report = validate_evidence_records(records)
        assert_valid(ev_report)
        crawled_summary = {
            "present": True,
            "path": str(crawled_path.relative_to(PROJECT_ROOT)),
            "rows": len(records),
            "warnings": ev_report.warnings,
        }

    meta = {
        "raw_path": str(raw_path),
        "raw_rows": before,
        "rows_after_dedupe": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "phishing_share": round(float(df["label"].mean()), 4),
        "sha256": _sha256(raw_path),
        "validation": report.to_dict(),
        "crawled": crawled_summary,
    }
    (PROCESSED_DIR / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "validation"}, indent=2))
    return meta


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_crawled_evidence(path: Path | None = None) -> list[dict]:
    """Load crawled evidence records (url, label, evidence) if present."""
    params = load_params().get("data", {})
    path = path or PROJECT_ROOT / params.get("crawled_path", "data/raw/crawled_evidence.jsonl")
    if not path.exists():
        return []
    return _read_jsonl(path)


if __name__ == "__main__":
    prepare()
