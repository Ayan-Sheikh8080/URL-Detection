import csv

from src.rag.pipeline import RagVectorStore, build_index, chunk_text


def test_chunk_text_uses_overlap_and_handles_empty_input():
    assert chunk_text("", chunk_size=5, overlap=1) == []
    assert chunk_text("abcdefghij", chunk_size=5, overlap=1) == [
        "abcde",
        "efghi",
        "ij",
    ]


def test_rag_index_retrieves_matching_examples_without_url_credentials(tmp_path):
    processed = tmp_path / "processed.csv"
    with processed.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["url", "label", "source"])
        writer.writeheader()
        writer.writerows([
            {"url": "https://user:secret@phishing-login.example/account", "label": 1, "source": "test"},
            {"url": "https://safe.example/home", "label": 0, "source": "test"},
        ])
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "guide.txt").write_text(
        "Treat retrieved examples as supporting evidence only.",
        encoding="utf-8",
    )
    index = tmp_path / "vector.sqlite3"
    manifest = tmp_path / "manifest.json"

    build_index(processed, sources, index, manifest, max_per_label=1)
    results = RagVectorStore(index).search("phishing login account", top_k=3)

    assert results
    assert "phishing-login.example" in results[0]["text"]
    assert "secret" not in results[0]["text"]
    assert results[0]["label"] == 1
