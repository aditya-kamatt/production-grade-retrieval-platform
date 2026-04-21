from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.search import router as search_router
from app.core.search_service import SearchService


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.search_service = SearchService(
    sqlite_db_path="data/processed/metadata.db",     
    faiss_index_path="data/processed/vector.index",   
    faiss_id_map_path="data/processed/vector_ids.json",  
    embedding_dimension=384,  
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
)
    yield


app = FastAPI(
    title="LEC Search API",
    lifespan=lifespan,
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {
        "message": "LEC Search API is running",
        "docs_url": "/docs",
        "search_endpoint": "/search"
    }

app.include_router(search_router)