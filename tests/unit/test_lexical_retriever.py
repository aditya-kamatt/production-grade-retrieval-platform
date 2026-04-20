from app.retrieval.lexical_retriever import BM25Retriever, simple_tokenize


def test_simple_tokenize_lowercases_and_splits():
    text = "Hybrid Search, BM25 and FAISS!"
    tokens = simple_tokenize(text)
    assert tokens == ["hybrid", "search", "bm25", "and", "faiss"]


def test_bm25_returns_ranked_results():
    chunks = [
        {
            "chunk_id": "1",
            "document_id": "doc-1",
            "text": "Hybrid search combines bm25 and dense retrieval",
            "metadata": {},
        },
        {
            "chunk_id": "2",
            "document_id": "doc-2",
            "text": "Export logistics and freight forwarding platform",
            "metadata": {},
        },
        {
            "chunk_id": "3",
            "document_id": "doc-3",
            "text": "Dense retrieval uses embeddings",
            "metadata": {},
        },
    ]

    retriever = BM25Retriever(chunks)
    results = retriever.search("bm25 hybrid", top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == "1"
    assert results[0].rank == 1
    assert results[0].score >= results[1].score


def test_bm25_empty_query_returns_empty_list():
    chunks = [
        {
            "chunk_id": "1",
            "document_id": "doc-1",
            "text": "Some text here",
            "metadata": {},
        }
    ]

    retriever = BM25Retriever(chunks)
    assert retriever.search("", top_k=5) == []
    assert retriever.search("   ", top_k=5) == []