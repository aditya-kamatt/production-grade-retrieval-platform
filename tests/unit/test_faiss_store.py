from pathlib import Path

from app.indexing.faiss_store import FAISSEmbeddingRepository


def test_faiss_store_saves_and_searches_embeddings(tmp_path: Path) -> None:
    index_path = tmp_path / "vector.index"
    id_map_path = tmp_path / "vector_ids.json"

    repo = FAISSEmbeddingRepository(
        index_path=str(index_path),
        id_map_path=str(id_map_path),
        embedding_dimension=3,
    )

    repo.replace_embeddings(
        doc_id="doc-1",
        chunk_ids=["doc-1_0_a", "doc-1_1_b"],
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )

    results = repo.search(query_vector=[1.0, 0.0, 0.0], top_k=2)

    assert len(results) >= 1
    assert results[0]["chunk_id"] == "doc-1_0_a"


def test_faiss_store_replaces_embeddings_for_same_document(tmp_path: Path) -> None:
    index_path = tmp_path / "vector.index"
    id_map_path = tmp_path / "vector_ids.json"

    repo = FAISSEmbeddingRepository(
        index_path=str(index_path),
        id_map_path=str(id_map_path),
        embedding_dimension=3,
    )

    repo.replace_embeddings(
        doc_id="doc-1",
        chunk_ids=["doc-1_0_old"],
        embeddings=[[1.0, 0.0, 0.0]],
    )

    repo.replace_embeddings(
        doc_id="doc-1",
        chunk_ids=["doc-1_0_new"],
        embeddings=[[0.0, 1.0, 0.0]],
    )

    results = repo.search(query_vector=[0.0, 1.0, 0.0], top_k=5)
    chunk_ids = [result["chunk_id"] for result in results]

    assert "doc-1_0_new" in chunk_ids
    assert "doc-1_0_old" not in chunk_ids