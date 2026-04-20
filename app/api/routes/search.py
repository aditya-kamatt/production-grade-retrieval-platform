from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_search_service
from app.core.search_service import SearchService
from app.schemas.search import SearchRequest, SearchResponse, SearchResultResponse

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search_endpoint(
    payload: SearchRequest,
    search_service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    results = search_service.search(
        query=payload.query,
        candidate_k=payload.candidate_k,
        final_k=payload.final_k,
        use_reranker=payload.use_reranker,
    )

    return SearchResponse(
        query=payload.query,
        results=[SearchResultResponse(**result) for result in results],
    )