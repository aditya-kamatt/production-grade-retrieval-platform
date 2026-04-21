from __future__ import annotations

from dataclasses import dataclass

import app.retrieval.reranker as reranker_module
from app.retrieval.reranker import CrossEncoderReranker


@dataclass
class DummyCandidate:
    chunk_id: str
    document_id: str
    text: str
    metadata: dict
    hybrid_score: float
    component_scores: dict[str, float]
    component_ranks: dict[str, int]


class FakeCrossEncoder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def predict(self, pairs):
        scores = []
        for query, text in pairs:
            if "best" in text.lower():
                scores.append(0.99)
            elif "good" in text.lower():
                scores.append(0.75)
            else:
                scores.append(0.10)
        return scores


def test_reranker_reorders_candidates(monkeypatch):
    monkeypatch.setattr(reranker_module, "CrossEncoder", FakeCrossEncoder)

    reranker = CrossEncoderReranker(model_name="fake-model")

    candidates = [
        DummyCandidate(
            chunk_id="1",
            document_id="doc-1",
            text="This is a weak match",
            metadata={},
            hybrid_score=0.9,
            component_scores={"bm25": 4.0},
            component_ranks={"bm25": 1},
        ),
        DummyCandidate(
            chunk_id="2",
            document_id="doc-2",
            text="This is the best answer",
            metadata={},
            hybrid_score=0.7,
            component_scores={"dense": 0.8},
            component_ranks={"dense": 2},
        ),
    ]

    results = reranker.rerank("query", candidates, top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == "2"
    assert results[0].rerank_score > results[1].rerank_score
    assert results[0].rank == 1


def test_reranker_empty_candidates_returns_empty_list(monkeypatch):
    monkeypatch.setattr(reranker_module, "CrossEncoder", FakeCrossEncoder)
    reranker = CrossEncoderReranker(model_name="fake-model")

    results = reranker.rerank("query", [], top_k=5)
    assert results == []