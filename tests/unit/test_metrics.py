import pytest

from app.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    dcg_at_k,
    idcg_at_k,
    ndcg_at_k,
    evaluate_query,
    evaluate_run,
)


def test_precision_at_k_basic():
    retrieved = ["a", "b", "c"]
    relevant = {"a": 1, "c": 2}

    assert precision_at_k(retrieved, relevant, k=2) == 0.5


def test_precision_at_k_empty():
    assert precision_at_k([], {"a": 1}, k=5) == 0.0


def test_recall_at_k_basic():
    retrieved = ["a", "b", "c"]
    relevant = {"a": 1, "c": 1}

    assert recall_at_k(retrieved, relevant, k=3) == 1.0


def test_recall_no_relevant_docs():
    assert recall_at_k(["a"], {}, k=5) == 0.0


def test_dcg_at_k_basic():
    retrieved = ["a", "b"]
    relevant = {"a": 2, "b": 1}

    assert dcg_at_k(retrieved, relevant, k=2) > 0


def test_idcg_at_k_sorted_correctly():
    relevant = {"a": 1, "b": 3, "c": 2}

    idcg = idcg_at_k(relevant, k=3)

    assert idcg > 0


def test_ndcg_at_k_perfect_ranking():
    retrieved = ["a", "b"]
    relevant = {"a": 2, "b": 1}

    assert ndcg_at_k(retrieved, relevant, k=2) == 1.0


def test_ndcg_at_k_zero_when_no_relevance():
    assert ndcg_at_k(["a"], {}, k=5) == 0.0


def test_evaluate_query_returns_all_metrics():
    retrieved = ["a", "b"]
    relevant = {"a": 1}

    result = evaluate_query(retrieved, relevant, k=2)

    assert "precision@2" in result
    assert "recall@2" in result
    assert "ndcg@2" in result


def test_evaluate_run_aggregates_results():
    run_results = {
        "q1": ["a", "b"],
        "q2": ["c"],
    }
    qrels = {
        "q1": {"a": 1},
        "q2": {"c": 1},
    }

    result = evaluate_run(run_results, qrels, k=2)

    assert "per_query" in result
    assert "aggregate" in result
    assert result["aggregate"]["num_queries"] == 2


def test_evaluate_run_handles_empty_qrels():
    result = evaluate_run({}, {}, k=5)

    assert result["aggregate"]["num_queries"] == 0
    assert result["aggregate"]["mean_precision@5"] == 0.0


def test_metrics_raise_on_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k(["a"], {"a": 1}, k=0)

    with pytest.raises(ValueError):
        recall_at_k(["a"], {"a": 1}, k=0)

    with pytest.raises(ValueError):
        dcg_at_k(["a"], {"a": 1}, k=0)

    with pytest.raises(ValueError):
        ndcg_at_k(["a"], {"a": 1}, k=0)