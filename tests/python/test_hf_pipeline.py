import csv
import json

from src.training.hf_pipeline import (
    HF_ID_TO_LABEL,
    HF_LABEL_TO_ID,
    format_example,
    prepare_hf_dataset,
)
from src.training.metrics import phishing_precision_recall_f1


def test_hugging_face_label_names_match_processed_dataset_labels():
    assert HF_ID_TO_LABEL == {0: "phishing", 1: "legitimate"}
    assert HF_LABEL_TO_ID == {"phishing": 0, "legitimate": 1}


def test_phishing_metrics_use_zero_as_the_positive_label():
    result = phishing_precision_recall_f1(
        labels=[0, 0, 1, 1],
        predictions=[0, 1, 0, 1],
    )

    assert result == {"precision": 0.5, "recall": 0.5, "f1": 0.5}


def test_prepare_hf_dataset_creates_balanced_tokenization_input(tmp_path):
    source = tmp_path / "processed.csv"
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["url", "label", "source", "has_password_field", "is_https"],
        )
        writer.writeheader()
        writer.writerows([
            {"url": f"https://phish{i}.example/login", "label": 0, "source": "test", "has_password_field": 1, "is_https": 1}
            for i in range(3)
        ] + [
            {"url": f"https://legit{i}.example/home", "label": 1, "source": "test", "has_password_field": 0, "is_https": 1}
            for i in range(3)
        ])
    output = tmp_path / "hf.jsonl"
    metadata = tmp_path / "metadata.json"

    result = prepare_hf_dataset(source, output, metadata, max_per_class=2)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert result["sample_count"] == 4
    assert result["label_counts"] == {"0": 2, "1": 2}
    assert "has_password_field 1" in format_example({
        "url": "https://example.com",
        "has_password_field": "1",
    })
    assert len(rows) == 4
    assert {row["label"] for row in rows} == {0, 1}
