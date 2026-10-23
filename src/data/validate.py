"""Dataset validation (feature 5: missing, duplicate and invalid data checks)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from urllib.parse import urlparse

import pandas as pd


class DataValidationError(ValueError):
    """Raised when the dataset fails hard validation rules."""

    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__(
            "dataset validation failed:\n- " + "\n- ".join(report.errors)
        )


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        data = asdict(self)
        data["ok"] = self.ok
        return data

    def summary(self) -> str:
        lines = [f"rows={self.stats.get('rows', 0)}", f"ok={self.ok}"]
        if self.stats.get("label_counts"):
            lines.append(f"labels={self.stats['label_counts']}")
        lines += [f"ERROR: {e}" for e in self.errors]
        lines += [f"WARN: {w}" for w in self.warnings]
        return " | ".join(lines)


def _is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(str(url).strip())
    except ValueError:
        return False
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.hostname)


def validate_urls(df: pd.DataFrame, *, required_columns=("url", "label")) -> ValidationReport:
    """Validate a URL/label dataframe.

    Errors are conditions that make training unsafe; warnings are quality
    signals that are logged but tolerated.
    """
    report = ValidationReport()

    if df is None or len(df) == 0:
        report.errors.append("dataset is empty")
        return report

    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        report.errors.append(f"missing required columns: {missing_cols}")
        return report

    report.stats["rows"] = int(len(df))

    null_urls = int(df["url"].isna().sum())
    if null_urls:
        report.errors.append(f"{null_urls} rows have a missing URL")

    null_labels = int(df["label"].isna().sum())
    if null_labels:
        report.errors.append(f"{null_labels} rows have a missing label")

    valid_mask = df["url"].map(_is_valid_url)
    invalid = int((~valid_mask).sum())
    if invalid:
        report.errors.append(
            f"{invalid} rows have an invalid URL (must be http/https with a hostname)"
        )

    duplicates = int(df["url"].duplicated().sum())
    if duplicates:
        report.warnings.append(f"{duplicates} duplicate URLs found (will be dropped)")

    labels = df["label"].dropna()
    report.stats["label_counts"] = {str(k): int(v) for k, v in labels.value_counts().items()}

    invalid_labels = sorted(set(labels.unique()) - {0, 1})
    if invalid_labels:
        report.errors.append(f"labels outside {{0, 1}} found: {invalid_labels}")
    elif len(set(labels.unique())) < 2:
        report.errors.append("dataset contains only one class; cannot train a classifier")
    else:
        phishing_share = float(labels.mean())
        report.stats["phishing_share"] = round(phishing_share, 4)
        if phishing_share < 0.2 or phishing_share > 0.8:
            report.warnings.append(
                f"strong class imbalance (phishing share {phishing_share:.2f}); "
                "consider stratified splits and class weights"
            )

    lengths = df["url"].dropna().astype(str).str.len()
    if len(lengths):
        report.stats["url_length_mean"] = round(float(lengths.mean()), 2)
        report.stats["url_length_max"] = int(lengths.max())
        if lengths.max() > 2048:
            report.warnings.append("some URLs exceed the recommended 2048 character limit")

    return report


def assert_valid(report: ValidationReport) -> None:
    if not report.ok:
        raise DataValidationError(report)


def validate_evidence_records(records: list[dict], *, required=("url", "label", "evidence")) -> ValidationReport:
    """Validate crawled evidence records used for content/combined training."""
    report = ValidationReport()
    if not records:
        report.warnings.append("no crawled evidence records provided")
        report.stats["rows"] = 0
        return report

    report.stats["rows"] = len(records)
    missing = 0
    bad_labels = 0
    bad_urls = 0
    for row in records:
        if any(key not in row for key in required):
            missing += 1
            continue
        if row["label"] not in (0, 1):
            bad_labels += 1
        if not _is_valid_url(row["url"]):
            bad_urls += 1

    if missing:
        report.errors.append(f"{missing} evidence records are missing required keys {required}")
    if bad_labels:
        report.errors.append(f"{bad_labels} evidence records have labels outside {{0, 1}}")
    if bad_urls:
        report.errors.append(f"{bad_urls} evidence records have invalid URLs")

    with_errors = sum(1 for r in records if (r.get("evidence") or {}).get("error"))
    if with_errors:
        report.warnings.append(f"{with_errors} evidence records contain browser errors")
    report.stats["browser_error_share"] = round(with_errors / max(len(records), 1), 4)
    return report
