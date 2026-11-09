import csv
import json

from src.training.hf_pipeline import format_example, prepare_hf_dataset


def test_prepare_hf_dataset_creates_balanced_tokenization_input(tmp_path):
    source = tmp_path / "processed.csv"
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["url", "label", "source", "has_password_field", "is_https"],
        )
        writer.writeheader()
        writer.writerows([
            {"url": f"https://legit{i}.example/home", "label": 0, "source": "test", "has_password_field": 0, "is_https": 1}
            for i in range(3)
        ] + [
            {"url": f"https://phish{i}.example/login", "label": 1, "source": "test", "has_password_field": 1, "is_https": 1}
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
