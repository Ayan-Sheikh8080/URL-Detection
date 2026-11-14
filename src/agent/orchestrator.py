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


@dataclass
class WorkflowState:
    url: str
    model_score: float
    evidence: dict[str, Any]
    page_text: str
    page_signals: dict[str, Any]
    retrieved_context: list[dict[str, Any]]
    previous_reasons: list[str]


class URLModelAnalyst:
    def run(self, state: WorkflowState) -> dict[str, Any]:
        return {
            "agent": "url_model_analyst",
            "status": "completed",
            "model_score": round(float(state.model_score), 4),
        }


class PageEvidenceAnalyst:
    def run(self, state: WorkflowState) -> dict[str, Any]:
        features = extract_content_features(state.url, state.evidence)
        state.page_signals = {
            name: value
            for name, value in features.items()
            if value and name not in {"page_text_length", "title_length"}
        }
        return {
            "agent": "page_evidence_analyst",
            "status": "completed",
            "signals": state.page_signals,
        }


class RagRetrievalAnalyst:
    def __init__(self, retriever: Any, top_k: int):
        self.retriever = retriever
        self.top_k = top_k

    def run(self, state: WorkflowState, round_number: int) -> dict[str, Any]:
        query_parts = [
            f"URL: {state.url}",
            f"Model score: {float(state.model_score):.4f}",
            f"Observed page indicators: {state.page_signals}",
            f"Page text: {(state.page_text or '')[:500]}",
        ]
        if state.previous_reasons:
            query_parts.append(
                f"Unresolved reasoning: {'; '.join(state.previous_reasons)}"
            )
        state.retrieved_context = self.retriever.search(
            "\n".join(query_parts),
            top_k=self.top_k,
        )
        return {
            "agent": "rag_retrieval_analyst",
            "round": round_number,
            "status": "completed",
            "retrieved_count": len(state.retrieved_context),
            "best_similarity": (
                round(float(state.retrieved_context[0]["similarity"]), 4)
                if state.retrieved_context
                else None
            ),
        }


class DecisionAnalyst:
    def __init__(self, reasoner: Any):
        self.reasoner = reasoner

    def run(self, state: WorkflowState) -> ReasoningOutput:
        context = [
            {
                key: result[key]
                for key in ("id", "source", "chunk_index", "similarity", "text")
                if key in result
            }
            for result in state.retrieved_context
        ]
        reasoning = self.reasoner.reason(
            ReasoningInput(
                url=state.url,
                url_model_score=state.model_score,
                evidence=state.evidence,
                page_text=state.page_text,
                retrieved_context=context,
            )
        )
        state.previous_reasons = reasoning.reasons
        return reasoning


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
        self.url_model_analyst = URLModelAnalyst()
        self.page_evidence_analyst = PageEvidenceAnalyst()
        self.retrieval_analyst = RagRetrievalAnalyst(retriever, top_k)
        self.decision_analyst = DecisionAnalyst(reasoner)

    def run(
        self,
        url: str,
        model_score: float,
        evidence: dict[str, Any],
        *,
        page_text: str = "",
    ) -> WorkflowResult:
        state = WorkflowState(
            url=url,
            model_score=float(model_score),
            evidence=evidence,
            page_text=page_text,
            page_signals={},
            retrieved_context=[],
            previous_reasons=[],
        )
        trace = [
            self.url_model_analyst.run(state),
            self.page_evidence_analyst.run(state),
        ]
        reasoning = None
        for round_number in range(1, self.max_rounds + 1):
            trace.append(self.retrieval_analyst.run(state, round_number))
            reasoning = self.decision_analyst.run(state)
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
                or not state.retrieved_context
            ):
                break

        if reasoning is None:
            raise RuntimeError("The analysis workflow completed without a decision")
        return WorkflowResult(
            reasoning=reasoning,
            retrieved_context=state.retrieved_context,
            trace=trace,
            rounds=round_number,
        )
