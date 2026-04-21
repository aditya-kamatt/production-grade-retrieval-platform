import pytest


class FakeModel:
    def encode(self, texts, **kwargs):
        import numpy as np
        return np.array([[1.0, 2.0]] * len(texts))


def test_embed_texts(monkeypatch):
    from app.services.embedding_provider import SentenceTransformerEmbeddingProvider

    monkeypatch.setattr(
        "app.services.embedding_provider.SentenceTransformer",
        lambda model_name: FakeModel(),
    )

    provider = SentenceTransformerEmbeddingProvider("fake")

    vectors = provider.embed_texts(["a", "b"])

    assert len(vectors) == 2
    assert vectors[0] == [1.0, 2.0]


def test_embed_texts_empty(monkeypatch):
    from app.services.embedding_provider import SentenceTransformerEmbeddingProvider

    monkeypatch.setattr(
        "app.services.embedding_provider.SentenceTransformer",
        lambda model_name: FakeModel(),
    )

    provider = SentenceTransformerEmbeddingProvider("fake")

    assert provider.embed_texts([]) == []


def test_embed_query(monkeypatch):
    from app.services.embedding_provider import SentenceTransformerEmbeddingProvider

    monkeypatch.setattr(
        "app.services.embedding_provider.SentenceTransformer",
        lambda model_name: FakeModel(),
    )

    provider = SentenceTransformerEmbeddingProvider("fake")

    vector = provider.embed_query("test")

    assert vector == [1.0, 2.0]