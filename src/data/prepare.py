"""Prepare stage: validate the raw dataset, dedupe, split, persist.

Run with ``python -m src.data.prepare`` (DVC stage ``prepare``).

Supports:
  * the bundled synthetic sample (``data/raw/sample_urls.csv``);
  * the PhiUSIIL Phishing URL dataset: point ``data.raw_path`` at the downloaded
    CSV and set ``data.label_positive_is: legitimate`` (PhiUSIIL labels 1 as
    legitimate);
  * a crawled page-evidence set (``data/raw/crawled_evidence.jsonl``) used by the
    content-only and combined configurations.

The stage writes reproducible train/test splits for both the URL dataset and the
crawled evidence set.  Crawled URLs are removed from the URL-only training data
so that the shared benchmark used by ``evaluate`` never leaks into the URL model.
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
CRAWLED_TRAIN = "crawled_train.jsonl"
CRAWLED_TEST = "crawled_test.jsonl"


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

    # Crawled evidence ------------------------------------------------------
    crawled_summary: dict = {"present": False}
    crawled_train: list[dict] = []
    crawled_test: list[dict] = []
    crawled_urls: set[str] = set()
    if crawled_path.exists():
        records = _read_jsonl(crawled_path)
        ev_report = validate_evidence_records(records)
        assert_valid(ev_report)

        deduped: dict[str, dict] = {}
        for record in records:
            deduped.setdefault(record["url"], record)
        records = list(deduped.values())
        crawled_urls = {record["url"] for record in records}

        crawled_train, crawled_test = _split_records(
            records, test_size=test_size, random_state=random_state
        )
        crawled_summary = {
            "present": True,
            "path": str(crawled_path.relative_to(PROJECT_ROOT))
            if crawled_path.is_relative_to(PROJECT_ROOT)
            else str(crawled_path),
            "rows": len(records),
            "train_rows": len(crawled_train),
            "test_rows": len(crawled_test),
            "warnings": ev_report.warnings,
        }

    # Keep the shared crawled benchmark out of URL training entirely.  This is
    # what makes the agent-vs-URL-only comparison in evaluate.py honest.
    if crawled_urls:
        overlap = int(df["url"].isin(crawled_urls).sum())
        if overlap:
            df = df[~df["url"].isin(crawled_urls)].reset_index(drop=True)
            crawled_summary["url_overlap_removed"] = overlap
        if df.empty or df["label"].nunique() < 2:
            raise ValueError(
                "after removing URLs that are part of crawled evidence, the URL "
                "dataset no longer has two classes; provide a larger URL dataset"
            )

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

    _write_jsonl(PROCESSED_DIR / CRAWLED_TRAIN, crawled_train)
    _write_jsonl(PROCESSED_DIR / CRAWLED_TEST, crawled_test)

    meta = {
        "raw_path": str(raw_path),
        "raw_rows": before,
        "rows_after_dedupe": len(df) + len(crawled_urls),
        "url_rows_after_overlap_removal": len(df),
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


def _split_records(records: list[dict], *, test_size: float, random_state: int) -> tuple[list[dict], list[dict]]:
    """Split crawled evidence into stratified train/test lists."""
    if len(records) < 2:
        return records, []
    labels = pd.Series([int(record["label"]) for record in records])
    counts = labels.value_counts()
    can_stratify = (
        labels.nunique() == 2
        and int(counts.min()) >= 2
        and test_size * len(records) >= 2
        and int(round(test_size * len(records))) >= 2
    )
    indices = list(range(len(records)))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels if can_stratify else None,
    )
    return [records[i] for i in train_idx], [records[i] for i in test_idx]


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_crawled_evidence(
    path: Path | None = None, *, split: str | None = None
) -> list[dict]:
    """Load crawled evidence records.

    ``split`` may be ``"train"`` or ``"test"`` to read the prepared split.  With
    no split, the raw file is returned for backward compatibility.
    """
    if split is not None:
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        processed = PROCESSED_DIR / (CRAWLED_TRAIN if split == "train" else CRAWLED_TEST)
        if not processed.exists():
            return []
        return _read_jsonl(processed)

    params = load_params().get("data", {})
    path = path or PROJECT_ROOT / params.get("crawled_path", "data/raw/crawled_evidence.jsonl")
    if not path.exists():
        return []
    return _read_jsonl(path)


if __name__ == "__main__":
    prepare()
