"""Bounded multi-agent harness for URL, page-evidence, and RAG analysis."""

from dataclasses import dataclass
from typing import Any

from src.agent.features import extract_content_features
from src.agent.reasoning import ReasoningInput, ReasoningOutput


@dataclass
class WorkflowResult:
    reasoning: ReasoningOutput
    retrieved_context: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    rounds: int


class URLAnalysisOrchestrator:
    """Run specialist analysis steps until confidence is sufficient or the cap is hit."""

    def __init__(
        self,
        retriever: Any,
        reasoner: Any,
        *,
        max_rounds: int = 2,
        top_k: int = 3,
        confidence_threshold: float = 0.88,
    ):
        if not 1 <= max_rounds <= 3:
            raise ValueError("max_rounds must be between 1 and 3")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not 0 < confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.retriever = retriever
        self.reasoner = reasoner
        self.max_rounds = max_rounds
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold

    def run(
        self,
        url: str,
        model_score: float,
        evidence: dict[str, Any],
        *,
        page_text: str = "",
    ) -> WorkflowResult:
        trace = [{
            "agent": "url_model_analyst",
            "status": "completed",
            "model_score": round(float(model_score), 4),
        }]
        page_features = extract_content_features(url, evidence)
        observed_signals = {
            name: value
            for name, value in page_features.items()
            if value and name not in {"page_text_length", "title_length"}
        }
        trace.append({
            "agent": "page_evidence_analyst",
            "status": "completed",
            "signals": observed_signals,
        })

        retrieval_results: list[dict[str, Any]] = []
        reasoning = None
        for round_number in range(1, self.max_rounds + 1):
            query_parts = [
                f"URL: {url}",
                f"Model score: {float(model_score):.4f}",
                f"Observed page indicators: {observed_signals}",
                f"Page text: {(page_text or '')[:500]}",
            ]
            if reasoning is not None:
                query_parts.append(f"Unresolved reasoning: {'; '.join(reasoning.reasons)}")
            retrieval_results = self.retriever.search(
                "\n".join(query_parts),
                top_k=self.top_k,
            )
            trace.append({
                "agent": "rag_retrieval_analyst",
                "round": round_number,
                "status": "completed",
                "retrieved_count": len(retrieval_results),
                "best_similarity": (
                    round(float(retrieval_results[0]["similarity"]), 4)
                    if retrieval_results
                    else None
                ),
            })
            reasoning = self.reasoner.reason(
                ReasoningInput(
                    url=url,
                    url_model_score=model_score,
                    evidence=evidence,
                    page_text=page_text,
                    retrieved_context=[
                        str(result["text"]) for result in retrieval_results
                    ],
                )
            )
            trace.append({
                "agent": "decision_analyst",
                "round": round_number,
                "status": "completed",
                "verdict": reasoning.verdict,
                "confidence": reasoning.confidence,
            })
            if (
                reasoning.confidence >= self.confidence_threshold
                or round_number == self.max_rounds
                or not retrieval_results
            ):
                break

        if reasoning is None:
            raise RuntimeError("The analysis workflow completed without a decision")
        return WorkflowResult(
            reasoning=reasoning,
            retrieved_context=retrieval_results,
            trace=trace,
            rounds=round_number,
        )
