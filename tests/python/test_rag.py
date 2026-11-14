import csv

import numpy as np

from src.rag.pipeline import RagVectorStore, build_index, chunk_text


class FakeSemanticEmbedder:
    term_groups = (
        {"credential", "credentials", "password", "account"},
        {"steal", "theft", "fraud"},
        {"fake", "imitation", "phishing"},
        {"sign-in", "signin", "login", "portal"},
    )

    def embed(self, texts):
        for text in texts:
            lowered = text.lower()
            vector = np.array([
                float(any(term in lowered for term in group))
                for group in self.term_groups
            ] + [0.0], dtype=np.float32)
            if not np.any(vector):
                vector[-1] = 1.0
            yield vector / np.linalg.norm(vector)


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
            {"url": "https://user:secret@phishing-login.example/account", "label": 0, "source": "test"},
            {"url": "https://safe.example/home", "label": 1, "source": "test"},
        ])
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "guide.txt").write_text(
        "Treat retrieved examples as supporting evidence only.",
        encoding="utf-8",
    )
    index = tmp_path / "semantic_vectors.sqlite3"
    manifest = tmp_path / "manifest.json"
    embedder = FakeSemanticEmbedder()

    build_index(
        processed,
        sources,
        index,
        manifest,
        max_per_label=1,
        embedder=embedder,
    )
    results = RagVectorStore(index, embedder=embedder).search(
        "phishing login account",
        top_k=3,
    )

    assert results
    assert "phishing-login.example" in results[0]["text"]
    assert "secret" not in results[0]["text"]
    assert results[0]["label"] == 0


def test_rag_uses_semantic_embeddings_to_match_synonyms(tmp_path):
    processed = tmp_path / "processed.csv"
    with processed.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["url", "label", "source"])
        writer.writeheader()
        writer.writerows([
            {"url": "https://safe.example/home", "label": 0, "source": "test"},
            {"url": "https://other.example/home", "label": 1, "source": "test"},
        ])
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "guide.txt").write_text(
        "Credential theft often begins at a fake sign-in portal.",
        encoding="utf-8",
    )
    index = tmp_path / "semantic_vectors.sqlite3"
    embedder = FakeSemanticEmbedder()

    build_index(
        processed,
        sources,
        index,
        tmp_path / "manifest.json",
        max_per_label=1,
        embedder=embedder,
    )
    results = RagVectorStore(index, embedder=embedder).search(
        "Password fraud through an imitation login",
        top_k=3,
    )

    assert results[0]["source"] == "guide.txt"
    assert results[0]["similarity"] > 0.95
