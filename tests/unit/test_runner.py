import json
import numpy as np
import pytest
from pathlib import Path

from app.evaluation.runner import (
    EvaluationRunner,
    load_chunks,
    load_embeddings,
)


class FakeRetriever:
    def search(self, query, top_k):
        return []


class FakeHybridService:
    def search(self, query, candidate_k, final_k, use_reranker):
        return [
            {"chunk_id": "c1"},
            {"chunk_id": "c2"},
        ]


class FakeEmbeddingProvider:
    def embed_texts(self, texts):
        return [[0.1, 0.2]] * len(texts)


@pytest.fixture
def runner(tmp_path):
    queries_path = tmp_path / "queries.yaml"
    qrels_path = tmp_path / "qrels.yaml"

    queries_path.write_text("""
queries:
  - query_id: q1
    text: test query
""")

    qrels_path.write_text("""
qrels:
  q1:
    c1: 2
""")

    runner = EvaluationRunner(
        chunks=[
            {
                "chunk_id": "c1",
                "document_id": "d1",
                "text": "test text",
                "metadata": {},
            }
        ],
        embeddings=np.zeros((1, 2)),
        embedding_provider=FakeEmbeddingProvider(),
        evaluation_dir=tmp_path,
        reports_dir=tmp_path / "reports",
        runs_dir=tmp_path / "runs",
    )

    # patch retrievers
    runner.hybrid_service = FakeHybridService()
    runner.vector_retriever = FakeRetriever()

    return runner


def test_load_queries_valid(runner):
    queries = runner.load_queries()
    assert len(queries) == 1
    assert queries[0]["query_id"] == "q1"


def test_load_qrels_valid(runner):
    qrels = runner.load_qrels()
    assert "q1" in qrels
    assert qrels["q1"]["c1"] == 2


def test_run_configuration_basic(runner):
    result = runner.run_configuration("hybrid", candidate_k=5, final_k=2)

    assert "run_results" in result
    assert "q1" in result["run_results"]
    assert result["run_results"]["q1"] == ["c1", "c2"]


def test_run_configuration_invalid_config(runner):
    with pytest.raises(ValueError):
        runner.run_configuration("invalid")


def test_evaluate_configuration_writes_files(runner, tmp_path):
    report = runner.evaluate_configuration("hybrid")

    assert "metrics" in report

    run_file = runner.runs_dir / "hybrid_run.json"
    report_file = runner.reports_dir / "hybrid_report.json"

    assert run_file.exists()
    assert report_file.exists()


def test_evaluate_all_runs_all_configs(runner):
    summary = runner.evaluate_all()

    assert "configs" in summary
    assert "hybrid" in summary["configs"]
    assert "semantic_only" in summary["configs"]
    assert "hybrid_rerank" in summary["configs"]


def test_latency_summary():
    values = [10, 20, 30]

    summary = EvaluationRunner._summarise_latencies(values)

    assert summary["mean"] == pytest.approx(20)
    assert summary["p50"] == pytest.approx(20)
    assert summary["p95"] >= 20


def test_percentile_edge_cases():
    assert EvaluationRunner._percentile([], 50) == 0.0
    assert EvaluationRunner._percentile([10], 50) == 10