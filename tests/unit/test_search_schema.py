import pytest
from pydantic import ValidationError

from app.schemas.search import (
    SearchRequest,
    SearchResponse,
    SearchResultResponse,
)


def test_search_request_valid_defaults():
    req = SearchRequest(query="test")

    assert req.query == "test"
    assert req.candidate_k == 20
    assert req.final_k == 5
    assert req.use_reranker is True


def test_search_request_rejects_empty_query():
    with pytest.raises(ValidationError):
        SearchRequest(query="")


def test_search_request_validates_candidate_k_vs_final_k():
    with pytest.raises(ValidationError) as exc:
        SearchRequest(query="test", candidate_k=3, final_k=5)

    assert "candidate_k must be greater than or equal to final_k" in str(exc.value)


def test_search_request_enforces_bounds():
    with pytest.raises(ValidationError):
        SearchRequest(query="test", candidate_k=0)

    with pytest.raises(ValidationError):
        SearchRequest(query="test", final_k=0)

    with pytest.raises(ValidationError):
        SearchRequest(query="test", candidate_k=200)


def test_search_result_response_structure():
    result = SearchResultResponse(
        chunk_id="c1",
        document_id="d1",
        text="hello",
        metadata={"a": 1},
        fused_score=0.5,
        rerank_score=0.9,
        component_scores={"bm25": 1.0},
        component_ranks={"bm25": 1},
        rank=1,
        latency_ms=10.0,
    )

    assert result.chunk_id == "c1"
    assert result.metadata == {"a": 1}
    assert result.rank == 1


def test_search_response_contains_results():
    result = SearchResultResponse(
        chunk_id="c1",
        document_id="d1",
        text="hello",
        metadata={},
        fused_score=None,
        rerank_score=None,
        component_scores={},
        component_ranks={},
        rank=1,
        latency_ms=5.0,
    )

    response = SearchResponse(query="test", results=[result])

    assert response.query == "test"
    assert len(response.results) == 1