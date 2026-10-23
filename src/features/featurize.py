"""Build pandas feature frames from URLs and evidence records."""

from __future__ import annotations

import pandas as pd

from src.features.page_features import PAGE_FEATURE_COLUMNS, extract_page_features
from src.features.url_features import URL_FEATURE_COLUMNS, extract_url_features


def url_feature_frame(urls: pd.Series) -> pd.DataFrame:
    """Vectorize a series of URL strings into the URL feature matrix."""
    rows = [extract_url_features(u) for u in urls.astype(str)]
    return pd.DataFrame(rows, columns=URL_FEATURE_COLUMNS).astype(float)


def page_feature_frame(evidence_records: list[dict]) -> pd.DataFrame:
    """Vectorize a list of evidence dicts into the page feature matrix."""
    rows = [extract_page_features(ev) for ev in evidence_records]
    return pd.DataFrame(rows, columns=PAGE_FEATURE_COLUMNS).astype(float)


def combined_feature_frame(urls: pd.Series, evidence_records: list[dict]) -> pd.DataFrame:
    """Concatenate URL and page features for the combined agent configuration."""
    if len(urls) != len(evidence_records):
        raise ValueError(
            f"urls ({len(urls)}) and evidence records ({len(evidence_records)}) "
            "must be the same length"
        )
    url_df = url_feature_frame(urls)
    page_df = page_feature_frame(evidence_records)
    return pd.concat([url_df.reset_index(drop=True), page_df], axis=1)


FEATURE_COLUMNS_BY_CONFIG = {
    "url_only": URL_FEATURE_COLUMNS,
    "content_only": PAGE_FEATURE_COLUMNS,
    "combined": URL_FEATURE_COLUMNS + PAGE_FEATURE_COLUMNS,
}
