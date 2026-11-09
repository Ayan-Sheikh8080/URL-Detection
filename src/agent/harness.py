"""Run the bounded analysis workflow against a local dataset sample."""

import argparse
import csv
import json
from pathlib import Path

from src.agent.orchestrator import URLAnalysisOrchestrator
from src.agent.reasoning import get_reasoner
from src.data.prepare import PROCESSED_PATH
from src.ml.features import extract_url_features
from src.ml.model_registry import ModelRegistry
from src.rag.pipeline import RagVectorStore

ROOT = Path(__file__).resolve().parents[2]


def run_harness(
    input_path: Path = PROCESSED_PATH,
    output_path: Path = ROOT / "reports" / "agent_harness.json",
    limit: int = 25,
    max_rounds: int = 3,
) -> dict:
    if limit < 1:
        raise ValueError("limit must be positive")
    registry = ModelRegistry()
    orchestrator = URLAnalysisOrchestrator(
        RagVectorStore(),
        get_reasoner(),
        max_rounds=max_rounds,
    )
    records = []
    with input_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            url = row["url"]
            prediction = registry.predict(
                "url_only",
                extract_url_features(url, allow_credentials=True),
            )
            workflow = orchestrator.run(url, prediction["score"], {}, page_text="")
            records.append({
                "url": _safe_url(url),
                "label": int(row["label"]),
                "predicted_verdict": workflow.reasoning.verdict,
                "model_score": prediction["score"],
                "confidence": workflow.reasoning.confidence,
                "rounds": workflow.rounds,
                "retrieved_count": len(workflow.retrieved_context),
                "trace": workflow.trace,
            })
            if len(records) >= limit:
                break
    report = {
        "input": input_path.name,
        "sample_count": len(records),
        "max_rounds": max_rounds,
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _safe_url(url: str) -> str:
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.hostname or ''}{parts.path[:180]}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "agent_harness.json")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--max-rounds", type=int, default=3)
    args = parser.parse_args()
    report = run_harness(args.input, args.output, args.limit, args.max_rounds)
    print(json.dumps({
        "sample_count": report["sample_count"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
