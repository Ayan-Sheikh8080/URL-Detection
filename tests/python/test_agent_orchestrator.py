from src.agent.orchestrator import URLAnalysisOrchestrator
from src.agent.reasoning import ReasoningOutput


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def search(self, query, top_k=3):
        self.queries.append(query)
        return [{
            "text": "Retrieved historical case",
            "similarity": 0.8,
            "source": "test",
            "label": 1,
        }]


class FakeReasoner:
    def __init__(self):
        self.calls = 0

    def reason(self, _input_data):
        self.calls += 1
        confidence = 0.5 if self.calls == 1 else 0.95
        return ReasoningOutput(
            verdict="suspicious",
            confidence=confidence,
            reasons=["uncertain evidence"],
            top_factors=[],
            model_score=0.5,
            combined_score=0.5,
        )


def test_orchestrator_refines_uncertain_analysis_then_stops():
    retriever = FakeRetriever()
    reasoner = FakeReasoner()
    workflow = URLAnalysisOrchestrator(
        retriever,
        reasoner,
        max_rounds=3,
        confidence_threshold=0.9,
    ).run("https://example.com", 0.5, {}, page_text="Sign in")

    assert workflow.rounds == 2
    assert reasoner.calls == 2
    assert "Unresolved reasoning" in retriever.queries[1]
    assert workflow.retrieved_context[0]["source"] == "test"
    assert [step["agent"] for step in workflow.trace].count("decision_analyst") == 2
