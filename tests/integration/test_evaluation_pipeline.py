from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.evaluation.runner import EvaluationRunner


class FakeEmbeddingProvider:
    def embed_texts(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeVectorItem:
    def __init__(self, chunk_id, document_id, text, metadata, score, rank):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.text = text
        self.metadata = metadata
        self.score = score
        self.rank = rank


class FakeVectorRetriever:
    def search(self, query: str, top_k: int):
        return [
            FakeVectorItem("c1", "d1", "Chunk one", {"source": "a.txt"}, 0.95, 1),
            FakeVectorItem("c2", "d2", "Chunk two", {"source": "b.txt"}, 0.85, 2),
        ][:top_k]


class FakeHybridService:
    def search(self, query: str, candidate_k: int, final_k: int, use_reranker: bool):
        results = [
            {
                "chunk_id": "c1",
                "document_id": "d1",
                "text": "Chunk one",
                "metadata": {"source": "a.txt"},
                "hybrid_score": 0.032,
                "rerank_score": 0.97 if use_reranker else None,
                "component_scores": {"bm25": 8.0, "vector": 0.95},
                "component_ranks": {"bm25": 1, "vector": 1},
                "rank": 1,
            },
            {
                "chunk_id": "c2",
                "document_id": "d2",
                "text": "Chunk two",
                "metadata": {"source": "b.txt"},
                "hybrid_score": 0.031,
                "rerank_score": 0.90 if use_reranker else None,
                "component_scores": {"bm25": 6.5, "vector": 0.85},
                "component_ranks": {"bm25": 2, "vector": 2},
                "rank": 2,
            },
        ]
        return results[:final_k]


@pytest.fixture
def runner(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    reports_dir = evaluation_dir / "reports"
    runs_dir = evaluation_dir / "runs"

    evaluation_dir.mkdir(parents=True, exist_ok=True)

    (evaluation_dir / "queries.yaml").write_text(
        """
queries:
  - query_id: q1
    text: bm25
  - query_id: q2
    text: vector search
""".strip(),
        encoding="utf-8",
    )

    (evaluation_dir / "qrels.yaml").write_text(
        """
qrels:
  q1:
    c1: 2
    c2: 1
  q2:
    c2: 2
""".strip(),
        encoding="utf-8",
    )

    runner = EvaluationRunner(
        chunks=[
            {"chunk_id": "c1", "document_id": "d1", "text": "Chunk one", "metadata": {}},
            {"chunk_id": "c2", "document_id": "d2", "text": "Chunk two", "metadata": {}},
        ],
        embeddings=np.zeros((2, 3), dtype=np.float32),
        embedding_provider=FakeEmbeddingProvider(),
        evaluation_dir=evaluation_dir,
        reports_dir=reports_dir,
        runs_dir=runs_dir,
    )

    runner.vector_retriever = FakeVectorRetriever()
    runner.hybrid_service = FakeHybridService()

    return runner


def test_evaluation_pipeline_runs_all_configs_and_writes_summary(runner):
    summary = runner.evaluate_all(candidate_k=5, final_k=2)

    assert summary["candidate_k"] == 5
    assert summary["final_k"] == 2
    assert set(summary["configs"].keys()) == {
        "semantic_only",
        "hybrid",
        "hybrid_rerank",
    }

    summary_path = runner.reports_dir / "summary.json"
    assert summary_path.exists()

    persisted = json.loads(summary_path.read_text(encoding="utf-8"))
    assert persisted["candidate_k"] == 5
    assert "hybrid" in persisted["configs"]


def test_evaluation_configuration_writes_run_and_report_files(runner):
    report = runner.evaluate_configuration("hybrid", candidate_k=4, final_k=2)

    assert report["config_name"] == "hybrid"
    assert report["candidate_k"] == 4
    assert report["final_k"] == 2
    assert "metrics" in report
    assert "latency_summary_ms" in report
    assert "query_latencies_ms" in report

    run_path = runner.runs_dir / "hybrid_run.json"
    report_path = runner.reports_dir / "hybrid_report.json"

    assert run_path.exists()
    assert report_path.exists()


def test_evaluation_pipeline_metrics_have_expected_shape(runner):
    report = runner.evaluate_configuration("hybrid_rerank", candidate_k=5, final_k=2)

    aggregate = report["metrics"]["aggregate"]

    assert "mean_precision@2" in aggregate
    assert "mean_recall@2" in aggregate
    assert "mean_ndcg@2" in aggregate
    assert aggregate["num_queries"] == 2


def test_run_configuration_raises_for_result_missing_chunk_id(runner):
    class BadHybridService:
        def search(self, query: str, candidate_k: int, final_k: int, use_reranker: bool):
            return [{"document_id": "d1"}]

    runner.hybrid_service = BadHybridService()

    with pytest.raises(KeyError, match="missing 'chunk_id'"):
        runner.run_configuration("hybrid", candidate_k=5, final_k=2)