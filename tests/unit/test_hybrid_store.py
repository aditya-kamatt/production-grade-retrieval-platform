def test_semantic_search_returns_joined_results():
    from app.indexing.hybrid_store import HybridRetrievalStore

    class FakeEmbeddingRepo:
        def search(self, query_vector, top_k):
            return [{"chunk_id": "c1", "score": 0.9}]

    class FakeChunkRepo:
        def get_chunk_by_id(self, cid):
            return {
                "chunk_id": cid,
                "doc_id": "d1",
                "chunk_index": 0,
                "text": "hello",
                "metadata": {},
            }

    store = HybridRetrievalStore(
        chunk_repository=FakeChunkRepo(),
        embedding_repository=FakeEmbeddingRepo(),
    )

    results = store.semantic_search([0.1, 0.2], top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["semantic_score"] == 0.9


def test_semantic_search_skips_missing_chunks():
    from app.indexing.hybrid_store import HybridRetrievalStore

    class FakeEmbeddingRepo:
        def search(self, query_vector, top_k):
            return [{"chunk_id": "missing", "score": 0.9}]

    class FakeChunkRepo:
        def get_chunk_by_id(self, cid):
            return None

    store = HybridRetrievalStore(
        chunk_repository=FakeChunkRepo(),
        embedding_repository=FakeEmbeddingRepo(),
    )

    results = store.semantic_search([0.1], top_k=1)

    assert results == []