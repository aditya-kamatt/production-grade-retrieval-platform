from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    candidate_k: int = Field(default=20, ge=1, le=100)
    final_k: int = Field(default=5, ge=1, le=20)
    use_reranker: bool = True

    @model_validator(mode="after")
    def validate__values(self) -> "SearchRequest":
        if self.candidate_k < self.final_k:
            raise ValueError("candidate_k must be greater than or equal to final_k")
        return self

class SearchResultResponse(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any]
    fused_score: float | None = None
    rerank_score: float | None = None
    component_scores: dict[str, float] = {}
    component_ranks: dict[str, int] = {}
    rank: int
    latency_ms: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultResponse]