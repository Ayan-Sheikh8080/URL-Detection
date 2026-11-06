"""Validate, normalize, deduplicate, and version the training CSV."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from src.agent.features import (
    CONTENT_MODEL_FEATURE_NAMES,
    MAX_IFRAME_COUNT,
    MAX_PAGE_TITLE_LENGTH,
)
from src.ml.features import parse_http_url

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "PhiUSIIL_Phishing_URL_Dataset.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "urls_processed.csv"
METADATA_PATH = ROOT / "data" / "metadata" / "dataset_meta.json"
PAGE_MODEL_FEATURES = CONTENT_MODEL_FEATURE_NAMES
PHIUSIIL_PAGE_COLUMNS = {
    "has_password_field": "HasPasswordField",
    "has_external_form": "HasExternalFormSubmit",
    "iframes_count": "NoOfiFrame",
    "redirect_count": "NoOfURLRedirect",
    "title_length": "Title",
    "is_https": "IsHTTPS",
}


class DatasetValidationError(ValueError):
    """Raised when dataset structure or labels are unusable."""


def normalize_url(value: str, *, allow_credentials: bool = False) -> str:
    value = value.strip()
    if "://" not in value:
        value = f"https://{value}"
    parse_http_url(value, allow_credentials=allow_credentials)
    return value


def _numeric_feature(value: str, row_number: int, column: str, *, binary: bool = False) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise DatasetValidationError(
            f"Row {row_number} has invalid numeric value for {column}: {value!r}"
        ) from None
    if not math.isfinite(numeric) or numeric < 0 or (binary and numeric not in (0, 1)):
        raise DatasetValidationError(
            f"Row {row_number} has out-of-range value for {column}: {value!r}"
        )
    return numeric


def prepare_dataset(raw_path: Path = RAW_PATH) -> dict:
    if not raw_path.exists():
        raise FileNotFoundError(f"Dataset not found: {raw_path}")

    issues: list[dict] = []
    by_url: dict[str, dict] = {}
    duplicate_rows = 0
    input_rows = 0
    with raw_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or [])
        is_phiusiil = "URL" in fieldnames
        required = (
            {"URL", "label", *PHIUSIIL_PAGE_COLUMNS.values()}
            if is_phiusiil
            else {"url", "label", "source"}
        )
        missing_columns = required.difference(fieldnames)
        if missing_columns:
            raise DatasetValidationError(
                f"Dataset is missing required columns: {', '.join(sorted(missing_columns))}"
            )
        for row_number, row in enumerate(reader, start=2):
            input_rows += 1
            raw_url = (row.get("URL" if is_phiusiil else "url") or "").strip()
            raw_label = (row.get("label") or "").strip()
            source = "PhiUSIIL" if is_phiusiil else (row.get("source") or "").strip()
            if not raw_url or not raw_label or not source:
                issues.append({"row": row_number, "issue": "missing required value"})
                continue
            try:
                source_label = int(raw_label)
                if source_label not in (0, 1):
                    raise ValueError
            except ValueError:
                raise DatasetValidationError(
                    f"Row {row_number} has invalid binary label {raw_label!r}"
                ) from None
            label = 1 - source_label if is_phiusiil else source_label
            try:
                url = normalize_url(raw_url, allow_credentials=is_phiusiil)
            except ValueError as error:
                issues.append({"row": row_number, "issue": f"invalid URL: {error}"})
                continue
            processed_row = {"url": url, "label": label, "source": source}
            if is_phiusiil:
                for feature, column in PHIUSIIL_PAGE_COLUMNS.items():
                    value = row.get(column) or ""
                    if feature == "title_length":
                        processed_row[feature] = float(
                            min(len(value), MAX_PAGE_TITLE_LENGTH)
                        )
                    else:
                        numeric = _numeric_feature(
                            value,
                            row_number,
                            column,
                            binary=feature in {
                                "has_password_field",
                                "has_external_form",
                                "is_https",
                            },
                        )
                        processed_row[feature] = (
                            min(numeric, MAX_IFRAME_COUNT)
                            if feature == "iframes_count"
                            else numeric
                        )
            previous = by_url.get(url)
            if previous:
                if previous["label"] != label:
                    raise DatasetValidationError(
                        f"Conflicting labels for duplicate URL {url}"
                    )
                duplicate_rows += 1
                if not is_phiusiil:
                    issues.append({"row": row_number, "issue": "duplicate URL"})
                continue
            by_url[url] = processed_row

    if input_rows == 0:
        raise DatasetValidationError("Dataset contains no records")
    if not by_url:
        raise DatasetValidationError("No valid records remain after validation")
    labels = {row["label"] for row in by_url.values()}
    if labels != {0, 1}:
        raise DatasetValidationError("Training data must contain both label classes")

    processed = list(by_url.values())
    counts = Counter(str(row["label"]) for row in processed)
    hasher = hashlib.sha256()
    with raw_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    raw_hash = hasher.hexdigest()
    try:
        source_file = raw_path.relative_to(ROOT).as_posix()
    except ValueError:
        source_file = raw_path.name
    columns = ["url", "label", "source"]
    if is_phiusiil:
        columns.extend(PAGE_MODEL_FEATURES)
    metadata = {
        "version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": source_file,
        "source_sha256": raw_hash,
        "input_rows": input_rows,
        "processed_rows": len(processed),
        "duplicate_rows": duplicate_rows,
        "label_distribution": dict(counts),
        "issues": issues,
        "columns": columns,
        "source_label_mapping": (
            {"0": "phishing", "1": "legitimate"}
            if is_phiusiil
            else {"0": "legitimate", "1": "phishing"}
        ),
    }

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROCESSED_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=metadata["columns"])
        writer.writeheader()
        writer.writerows(processed)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="fail if any rows need cleanup")
    args = parser.parse_args()
    metadata = prepare_dataset()
    if args.strict and metadata["issues"]:
        print(f"Strict validation failed with {len(metadata['issues'])} issue(s)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
