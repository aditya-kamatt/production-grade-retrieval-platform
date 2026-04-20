from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.search import router as search_router
from app.core.search_service import SearchService


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.search_service = SearchService(
        chunks_path="data/processed/chunks.json",
        embeddings_path="data/processed/embeddings.npy",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    yield


app = FastAPI(
    title="LEC Search API",
    lifespan=lifespan,
)

app.include_router(search_router)