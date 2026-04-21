from fastapi import APIRouter, Depends

from app.schemas.search import (
    SearchRequest,
    SearchResponse,
    SearchResultResponse,  
)
from app.core.search_service import SearchService  
from app.api.dependencies import get_search_service

router = APIRouter(prefix="/search", tags=["search"])

@router.post("", response_model=SearchResponse)
def search_endpoint(
    request: SearchRequest,
    search_service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    response = search_service.search(
        query=request.query,
        candidate_k=request.candidate_k,
        final_k=request.final_k,
        use_reranker=request.use_reranker,
    )

    results = response["results"]
    latency_ms = response["latency_ms"]

    return SearchResponse(
        query=request.query,
        results=[
            SearchResultResponse(
                chunk_id=result["chunk_id"],
                document_id=result["document_id"],
                text=result["text"],
                metadata=result.get("metadata", {}),
                fused_score=result["scores"]["hybrid"],      
                rerank_score=result["scores"]["reranker"],   
                component_scores={
                    "bm25": result["scores"]["bm25"],
                    "vector": result["scores"]["vector"],
                    "hybrid": result["scores"]["hybrid"],
                    "reranker": (
                        result["scores"]["reranker"]
                        if result["scores"]["reranker"] is not None
                        else 0.0
                    ),
                },           
                component_ranks={},                          
                rank=result["rank"],  
                latency_ms=latency_ms                       
            )
            for result in results
        ],
    )