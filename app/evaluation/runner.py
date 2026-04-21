from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from app.evaluation.metrics import evaluate_run
from app.retrieval.hybrid_search import HybridSearchService
from app.retrieval.lexical_retriever import BM25Retriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_retriever import FaissVectorRetriever
from app.services.embedding_provider import SentenceTransformerEmbeddingProvider


class EvaluationRunner:
    """
    Runs evaluation for three configurations:
    - semantic_only
    - hybrid
    - hybrid_rerank

    Files expected under data/evaluation:
    - queries.yaml
    - qrels.yaml

    Files expected under data/processed:
    - chunks.json
    - embeddings.npy

    Why this design:
    - Keeps evaluation orchestration separate from retrieval logic.
    - Makes the three ablation configs explicit.
    - Stores both raw runs and aggregate reports for write-up evidence.
    """

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        embeddings: np.ndarray,
        embedding_provider: SentenceTransformerEmbeddingProvider,
        evaluation_dir: str | Path = "data/evaluation",
        reports_dir: str | Path = "data/evaluation/reports",
        runs_dir: str | Path = "data/evaluation/runs",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.chunks = chunks
        self.embeddings = embeddings
        self.embedding_provider = embedding_provider

        # Retriever setup
        self.lexical_retriever = BM25Retriever(chunks=self.chunks)
        self.vector_retriever = FaissVectorRetriever(
            chunks=self.chunks,
            embeddings=self.embeddings,
            embedding_function=self.embedding_provider.embed_texts,
        )
        self.reranker = CrossEncoderReranker(model_name=reranker_model_name)

        self.hybrid_service = HybridSearchService(
            lexical_retriever=self.lexical_retriever,
            vector_retriever=self.vector_retriever,
            reranker=self.reranker,
        )

        self.evaluation_dir = Path(evaluation_dir)
        self.reports_dir = Path(reports_dir)
        self.runs_dir = Path(runs_dir)

        self.queries_path = self.evaluation_dir / "queries.yaml"
        self.qrels_path = self.evaluation_dir / "qrels.yaml"

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def load_queries(self) -> list[dict[str, str]]:
        """
        Expected format:

        queries:
          - query_id: q1
            text: what is bm25
        """
        with self.queries_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        queries = data.get("queries", [])
        if not isinstance(queries, list):
            raise ValueError("queries.yaml must contain a top-level 'queries' list")

        for item in queries:
            if "query_id" not in item or "text" not in item:
                raise ValueError("Each query must contain 'query_id' and 'text'")

        return queries

    def load_qrels(self) -> dict[str, dict[str, int]]:
        """
        Expected format:

        qrels:
          q1:
            chunk_id_1: 2
            chunk_id_2: 1
        """
        with self.qrels_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        qrels = data.get("qrels", {})
        if not isinstance(qrels, dict):
            raise ValueError("qrels.yaml must contain a top-level 'qrels' mapping")

        return qrels

    def _search_semantic_only(
        self,
        query: str,
        candidate_k: int,
        final_k: int,
    ) -> list[dict[str, Any]]:
        """
        Dense-only baseline using the vector retriever directly.
        """
        dense_results = self.vector_retriever.search(query=query, top_k=candidate_k)

        return [
            {
                "chunk_id": item.chunk_id,
                "document_id": item.document_id,
                "text": item.text,
                "metadata": item.metadata,
                "hybrid_score": float(item.score),  # kept under one common field for reporting
                "rerank_score": None,
                "component_scores": {"dense": float(item.score)},
                "component_ranks": {"dense": int(item.rank)},
                "rank": idx,
            }
            for idx, item in enumerate(dense_results[:final_k], start=1)
        ]

    def _search_hybrid(
        self,
        query: str,
        candidate_k: int,
        final_k: int,
    ) -> list[dict[str, Any]]:
        """
        Hybrid retrieval without reranking.
        """
        return self.hybrid_service.search(
            query=query,
            candidate_k=candidate_k,
            final_k=final_k,
            use_reranker=False,
        )

    def _search_hybrid_rerank(
        self,
        query: str,
        candidate_k: int,
        final_k: int,
    ) -> list[dict[str, Any]]:
        """
        Hybrid retrieval with reranking.
        """
        return self.hybrid_service.search(
            query=query,
            candidate_k=candidate_k,
            final_k=final_k,
            use_reranker=True,
        )

    def run_configuration(
        self,
        config_name: str,
        candidate_k: int = 20,
        final_k: int = 5,
    ) -> dict[str, Any]:
        """
        Run all queries for one configuration and collect:
        - ordered chunk_ids per query
        - detailed results per query
        - latency per query
        - latency summary
        """
        queries = self.load_queries()

        run_results: dict[str, list[str]] = {}
        detailed_results: dict[str, list[dict[str, Any]]] = {}
        query_latencies_ms: dict[str, float] = {}

        for query_item in queries:
            query_id = query_item["query_id"]
            query_text = query_item["text"]

            start = time.perf_counter()

            if config_name == "semantic_only":
                results = self._search_semantic_only(
                    query=query_text,
                    candidate_k=candidate_k,
                    final_k=final_k,
                )
            elif config_name == "hybrid":
                results = self._search_hybrid(
                    query=query_text,
                    candidate_k=candidate_k,
                    final_k=final_k,
                )
            elif config_name == "hybrid_rerank":
                results = self._search_hybrid_rerank(
                    query=query_text,
                    candidate_k=candidate_k,
                    final_k=final_k,
                )
            else:
                raise ValueError(f"Unknown config: {config_name}")

            latency_ms = (time.perf_counter() - start) * 1000.0

            chunk_ids: list[str] = []
            for result in results:
                chunk_id = result.get("chunk_id")
                if not chunk_id:
                    raise KeyError(
                        f"Search result for query '{query_id}' is missing 'chunk_id': {result}"
                    )
                chunk_ids.append(chunk_id)

            run_results[query_id] = chunk_ids
            detailed_results[query_id] = results
            query_latencies_ms[query_id] = latency_ms

        return {
            "config_name": config_name,
            "candidate_k": candidate_k,
            "final_k": final_k,
            "run_results": run_results,
            "detailed_results": detailed_results,
            "query_latencies_ms": query_latencies_ms,
            "latency_summary_ms": self._summarise_latencies(
                list(query_latencies_ms.values())
            ),
        }

    def evaluate_configuration(
        self,
        config_name: str,
        candidate_k: int = 20,
        final_k: int = 5,
    ) -> dict[str, Any]:
        """
        Run one configuration, score it against qrels, and write output files.
        """
        qrels = self.load_qrels()

        run_output = self.run_configuration(
            config_name=config_name,
            candidate_k=candidate_k,
            final_k=final_k,
        )

        metrics_report = evaluate_run(
            run_results=run_output["run_results"],
            qrels=qrels,
            k=final_k,
        )

        full_report = {
            "config_name": config_name,
            "candidate_k": candidate_k,
            "final_k": final_k,
            "metrics": metrics_report,
            "latency_summary_ms": run_output["latency_summary_ms"],
            "query_latencies_ms": run_output["query_latencies_ms"],
        }

        run_path = self.runs_dir / f"{config_name}_run.json"
        report_path = self.reports_dir / f"{config_name}_report.json"

        with run_path.open("w", encoding="utf-8") as f:
            json.dump(run_output, f, indent=2)

        with report_path.open("w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2)

        return full_report

    def evaluate_all(
        self,
        candidate_k: int = 20,
        final_k: int = 5,
    ) -> dict[str, Any]:
        """
        Run all three required configurations and write summary.json.
        """
        config_names = ["semantic_only", "hybrid", "hybrid_rerank"]
        all_reports: dict[str, Any] = {}

        for config_name in config_names:
            all_reports[config_name] = self.evaluate_configuration(
                config_name=config_name,
                candidate_k=candidate_k,
                final_k=final_k,
            )

        summary = {
            "candidate_k": candidate_k,
            "final_k": final_k,
            "configs": {},
        }

        for config_name, report in all_reports.items():
            aggregate = report["metrics"]["aggregate"]
            latency = report["latency_summary_ms"]

            summary["configs"][config_name] = {
                f"mean_precision@{final_k}": aggregate[f"mean_precision@{final_k}"],
                f"mean_recall@{final_k}": aggregate[f"mean_recall@{final_k}"],
                f"mean_ndcg@{final_k}": aggregate[f"mean_ndcg@{final_k}"],
                "num_queries": aggregate["num_queries"],
                "mean_latency_ms": latency["mean"],
                "p50_latency_ms": latency["p50"],
                "p95_latency_ms": latency["p95"],
            }

        summary_path = self.reports_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    @staticmethod
    def _summarise_latencies(latencies_ms: list[float]) -> dict[str, float]:
        if not latencies_ms:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0}

        sorted_vals = sorted(latencies_ms)
        return {
            "mean": statistics.mean(sorted_vals),
            "p50": EvaluationRunner._percentile(sorted_vals, 50),
            "p95": EvaluationRunner._percentile(sorted_vals, 95),
        }

    @staticmethod
    def _percentile(sorted_values: list[float], percentile: int) -> float:
        if not sorted_values:
            return 0.0

        if len(sorted_values) == 1:
            return sorted_values[0]

        index = (percentile / 100.0) * (len(sorted_values) - 1)
        lower = int(index)
        upper = min(lower + 1, len(sorted_values) - 1)
        fraction = index - lower

        return sorted_values[lower] + (
            (sorted_values[upper] - sorted_values[lower]) * fraction
        )


def load_chunks(chunks_path: str | Path = "data/processed/chunks.json") -> list[dict[str, Any]]:
    """
    Expected chunks.json format: a JSON list of chunk dictionaries.
    Each chunk should contain at least:
    - chunk_id
    - document_id
    - text
    - metadata (optional)
    """
    chunks_path = Path(chunks_path)
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Missing chunks file: {chunks_path}. "
            "Your current ingestion must write chunks.json for evaluation to work."
        )

    with chunks_path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list):
        raise ValueError("chunks.json must be a JSON list")

    return chunks


def load_embeddings(embeddings_path: str | Path = "data/processed/embeddings.npy") -> np.ndarray:
    """
    Expected embeddings.npy format:
    - numpy array of shape [num_chunks, embedding_dim]
    """
    embeddings_path = Path(embeddings_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Missing embeddings file: {embeddings_path}. "
            "Your current ingestion must write embeddings.npy for evaluation to work."
        )

    embeddings = np.load(embeddings_path)
    if embeddings.ndim != 2:
        raise ValueError("embeddings.npy must be a 2D numpy array")

    return embeddings


def build_runner() -> EvaluationRunner:
    """
    Central constructor for evaluation dependencies.
    """
    chunks = load_chunks("data/processed/chunks.json")
    embeddings = load_embeddings("data/processed/embeddings.npy")

    embedding_provider = SentenceTransformerEmbeddingProvider(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return EvaluationRunner(
        chunks=chunks,
        embeddings=embeddings,
        embedding_provider=embedding_provider,
        evaluation_dir="data/evaluation",
        reports_dir="data/evaluation/reports",
        runs_dir="data/evaluation/runs",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )


def run_evaluation() -> dict[str, Any]:
    runner = build_runner()
    return runner.evaluate_all(candidate_k=20, final_k=5)


if __name__ == "__main__":
    summary = run_evaluation()
    print(json.dumps(summary, indent=2))