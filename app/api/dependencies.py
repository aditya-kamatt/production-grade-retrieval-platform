from __future__ import annotations

from fastapi import HTTPException, Request

from app.core.search_service import SearchService


def get_search_service(request: Request) -> SearchService:
    search_service = getattr(request.app.state, "search_service", None)
    if search_service is None:
        raise HTTPException(status_code=500, detail="Search service not initialized")
    return search_service