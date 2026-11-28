"""Build and query a reproducible local semantic vector index."""

import argparse
import csv
import hashlib
import json
import os
import random
import sqlite3
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_PATH = ROOT / "data" / "processed" / "urls_processed.csv"
SOURCE_DIR = ROOT / "data" / "rag_source"
INDEX_PATH = ROOT / "data" / "rag" / "semantic_vectors.sqlite3"
EMBEDDING_CACHE_PATH = ROOT / "data" / "rag" / "embedding_models"
MANIFEST_PATH = ROOT / "data" / "rag" / "index_manifest.json"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_EXAMPLES_PER_LABEL = 1024
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
_embedding_model = None


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size")
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            from fastembed import TextEmbedding
        except ImportError as error:
            raise RuntimeError(
                "Semantic RAG requires fastembed; "
                "install the project requirements before indexing or searching"
            ) from error
        EMBEDDING_CACHE_PATH.mkdir(parents=True, exist_ok=True)
        _embedding_model = TextEmbedding(
            model_name=EMBEDDING_MODEL_NAME,
            cache_dir=str(EMBEDDING_CACHE_PATH),
        )
    return _embedding_model


def _encode_texts(texts: list[str], embedder=None) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    model = embedder or _get_embedding_model()
    embeddings = np.asarray(list(model.embed(texts)), dtype=np.float32)
    if (
        embeddings.ndim != 2
        or embeddings.shape[0] != len(texts)
        or embeddings.shape[1] < 1
        or not np.isfinite(embeddings).all()
    ):
        raise ValueError("The embedding model returned invalid vectors")
    return embeddings


def _dataset_documents(path: Path, max_per_label: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    samples: dict[int, list[dict]] = {0: [], 1: []}
    seen = {0: 0, 1: 0}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                label = int(row["label"])
            except (KeyError, TypeError, ValueError):
                raise ValueError("Processed dataset must have a binary label column") from None
            if label not in samples:
                raise ValueError(f"Unsupported processed label: {label}")
            seen[label] += 1
            item = dict(row)
            if len(samples[label]) < max_per_label:
                samples[label].append(item)
            else:
                index = rng.randrange(seen[label])
                if index < max_per_label:
                    samples[label][index] = item

    documents = []
    for label in sorted(samples):
        for row in samples[label]:
            try:
                parts = urlsplit(row["url"])
                safe_url = f"{parts.scheme}://{parts.hostname or ''}{parts.path[:180]}"
            except (KeyError, ValueError):
                safe_url = "URL details unavailable"
            features = [
                f"{name}={row[name]}"
                for name in (
                    "has_password_field",
                    "has_external_form",
                    "iframes_count",
                    "redirect_count",
                    "title_length",
                    "is_https",
                )
                if row.get(name) not in (None, "")
            ]
            verdict = "legitimate" if label == 0 else "phishing"
            text = (
                f"Labeled {verdict} URL example. Sanitized URL: {safe_url}. "
                f"Observed page indicators: {', '.join(features) if features else 'not available'}."
            )
            documents.append({
                "source": "dataset_example",
                "label": label,
                "text": text,
            })
    return documents


def _source_documents(source_dir: Path) -> list[dict]:
    if not source_dir.exists():
        return []
    documents = []
    for path in sorted(source_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
            documents.append({
                "source": path.relative_to(source_dir).as_posix(),
                "label": None,
                "text": path.read_text(encoding="utf-8"),
            })
    return documents


def build_index(
    processed_path: Path = PROCESSED_PATH,
    source_dir: Path = SOURCE_DIR,
    index_path: Path = INDEX_PATH,
    manifest_path: Path = MANIFEST_PATH,
    max_per_label: int = MAX_EXAMPLES_PER_LABEL,
    seed: int = 42,
    embedder=None,
) -> dict:
    if max_per_label < 1:
        raise ValueError("max_per_label must be positive")
    if not processed_path.is_file():
        raise FileNotFoundError(f"Processed dataset not found: {processed_path}")

    raw_documents = _dataset_documents(processed_path, max_per_label, seed)
    raw_documents.extend(_source_documents(source_dir))
    records_by_id = {}
    content_hash = hashlib.sha256()
    for document in raw_documents:
        for chunk_index, text in enumerate(chunk_text(document["text"])):
            record_id = hashlib.sha256(
                f'{document["source"]}\0{chunk_index}\0{text}'.encode("utf-8")
            ).hexdigest()
            if record_id in records_by_id:
                continue
            content_hash.update(record_id.encode("ascii"))
            records_by_id[record_id] = {
                **document,
                "chunk_index": chunk_index,
                "id": record_id,
                "text": text,
            }
    records = list(records_by_id.values())
    if not records:
        raise ValueError("No RAG documents were found to index")

    embeddings = _encode_texts(
        [record["text"] for record in records],
        embedder,
    )

    index_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{index_path.name}.",
        suffix=".tmp",
        dir=index_path.parent,
    )
    os.close(descriptor)
    try:
        connection = sqlite3.connect(temporary_name)
        try:
            connection.execute(
                """
                CREATE TABLE chunks (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    label INTEGER,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE TABLE index_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.executemany(
                """
                INSERT INTO chunks (id, source, chunk_index, label, text, embedding)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record["id"],
                        record["source"],
                        record["chunk_index"],
                        record["label"],
                        record["text"],
                        embeddings[index].tobytes(),
                    )
                    for index, record in enumerate(records)
                ],
            )
            connection.executemany(
                "INSERT INTO index_metadata (key, value) VALUES (?, ?)",
                [
                    ("embedding_model", EMBEDDING_MODEL_NAME),
                    ("embedding_dimensions", str(embeddings.shape[1])),
                    ("distance", "cosine_similarity"),
                ],
            )
            connection.execute("CREATE INDEX chunks_source_idx ON chunks(source)")
            connection.commit()
        finally:
            connection.close()
        os.replace(temporary_name, index_path)
    except Exception:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
        raise

    manifest = {
        "schema_version": 1,
        "vector_store": "sqlite_dense_vectors",
        "embedding_method": "fastembed",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_cache": EMBEDDING_CACHE_PATH.relative_to(ROOT).as_posix(),
        "embedding_dimensions": int(embeddings.shape[1]),
        "distance": "cosine_similarity",
        "chunk_size_characters": CHUNK_SIZE,
        "chunk_overlap_characters": CHUNK_OVERLAP,
        "max_examples_per_label": max_per_label,
        "random_seed": seed,
        "indexed_chunks": len(records),
        "corpus_sha256": content_hash.hexdigest(),
        "sources": {
            "processed_dataset": processed_path.relative_to(ROOT).as_posix()
            if processed_path.is_relative_to(ROOT)
            else str(processed_path),
            "document_directory": source_dir.relative_to(ROOT).as_posix()
            if source_dir.is_relative_to(ROOT)
            else str(source_dir),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class RagVectorStore:
    """Persistent local vector store using dense semantic embeddings."""

    def __init__(self, index_path: Path = INDEX_PATH, embedder=None):
        self.index_path = index_path
        self.embedder = embedder

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not query.strip():
            return []
        if not self.index_path.is_file():
            raise FileNotFoundError(
                f"RAG vector index not found at {self.index_path}; run `dvc repro rag_index`"
            )
        query_vector = _encode_texts([query], self.embedder)[0]
        results = []
        connection = sqlite3.connect(self.index_path)
        try:
            row = connection.execute(
                "SELECT value FROM index_metadata WHERE key = 'embedding_dimensions'"
            ).fetchone()
            if row is None:
                raise ValueError(f"Embedding dimension metadata is missing in {self.index_path}")
            dimensions = int(row[0])
            if query_vector.size != dimensions:
                raise ValueError(
                    f"Query embedding size {query_vector.size} does not match index size {dimensions}"
                )
            rows = connection.execute(
                "SELECT id, source, chunk_index, label, text, embedding FROM chunks"
            )
            for record_id, source, chunk_index, label, text, embedding in rows:
                vector = np.frombuffer(embedding, dtype=np.float32)
                if vector.size != dimensions:
                    raise ValueError(
                        f"Invalid embedding size for indexed record {record_id}"
                    )
                denominator = np.linalg.norm(query_vector) * np.linalg.norm(vector)
                similarity = (
                    float(np.dot(query_vector, vector) / denominator)
                    if denominator
                    else 0.0
                )
                results.append({
                    "id": record_id,
                    "source": source,
                    "chunk_index": chunk_index,
                    "label": label,
                    "text": text,
                    "similarity": similarity,
                })
        finally:
            connection.close()
        results.sort(key=lambda result: (-result["similarity"], result["id"]))
        return results[:top_k]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--processed", type=Path, default=PROCESSED_PATH)
    build_parser.add_argument("--sources", type=Path, default=SOURCE_DIR)
    build_parser.add_argument("--index", type=Path, default=INDEX_PATH)
    build_parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    build_parser.add_argument("--max-per-label", type=int, default=MAX_EXAMPLES_PER_LABEL)
    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("query")
    search_parser.add_argument("--index", type=Path, default=INDEX_PATH)
    search_parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    if args.command == "build":
        print(json.dumps(build_index(
            args.processed,
            args.sources,
            args.index,
            args.manifest,
            args.max_per_label,
        ), indent=2))
    else:
        print(json.dumps(RagVectorStore(args.index).search(args.query, args.top_k), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
