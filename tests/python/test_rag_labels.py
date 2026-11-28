import csv

from src.rag.pipeline import _dataset_documents


def test_dataset_documents_use_correct_processed_label_mapping(tmp_path):
    dataset = tmp_path / "processed.csv"

    with dataset.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["url", "label", "has_password_field", "is_https"],
        )
        writer.writeheader()
        writer.writerows([
            {
                "url": "https://example.com",
                "label": 0,
                "has_password_field": 0,
                "is_https": 1,
            },
            {
                "url": "https://example.org",
                "label": 1,
                "has_password_field": 1,
                "is_https": 1,
            },
        ])

    documents = _dataset_documents(dataset, max_per_label=10, seed=42)

    by_label = {document["label"]: document["text"] for document in documents}

    assert "Labeled legitimate URL example" in by_label[0]
    assert "Labeled phishing URL example" in by_label[1]
