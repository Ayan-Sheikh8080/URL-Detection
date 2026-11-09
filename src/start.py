"""Prepare data, train missing artifacts, and run the FastAPI server."""

import os
from pathlib import Path

import uvicorn

from src.data.prepare import METADATA_PATH, prepare_dataset
from src.training.train import MODEL_PATH, train_models
from src.rag.pipeline import INDEX_PATH, build_index


def main() -> None:
    if not METADATA_PATH.exists() or not MODEL_PATH.exists():
        prepare_dataset()
        train_models()
    if not INDEX_PATH.exists():
        build_index()
    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
