from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from rag_benchmark.config import BenchmarkConfig
from rag_benchmark.graph_rag import GraphRAGRunner, LazyGraphRAGRunner
from rag_benchmark.frontier_rag import FrontierRAG
from rag_benchmark.hippo_rag import HippoRAGRunner
from rag_benchmark.hybrid_rag import HybridRAG
from rag_benchmark.light_rag import LightRAGRunner
from rag_benchmark.modern_rag import AdaptiveRAGRouter, HybridDenseSparseRAG, RerankSemanticRAG
from rag_benchmark.llm_factory import TrackedLLMClient, clone_client_for_ledger, create_tracked_client
from rag_benchmark.metrics import AccuracyEvaluator, AccuracyResult, load_eval_questions
from rag_benchmark.semantic_rag import SemanticRAG
from rag_benchmark.token_tracker import TokenLedger


@dataclass
class MethodRunResult:
    method: str
    answers: list[dict[str, Any]]
    accuracy: list[AccuracyResult]
    ledger: TokenLedger
    elapsed_seconds: float
    index_seconds: float = 0.0
    query_latencies: list[float] = field(default_factory=list)


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig, tracked_client: TrackedLLMClient | None = None):
        self.config = config
        self.client = tracked_client or create_tracked_client(config)
        self.questions = load_eval_questions(config.qa_path)

    def _client_for(self, ledger: TokenLedger) -> TrackedLLMClient:
        return clone_client_for_ledger(self.client, self.config, ledger)

    def run_semantic(self) -> MethodRunResult:
        ledger = TokenLedger()
        client = self._client_for(ledger)
        rag = SemanticRAG(self.config, client, ledger)

        start = time.perf_counter()
        index_start = time.perf_counter()
        rag.build_index()
        index_seconds = time.perf_counter() - index_start
        answers, accuracy, query_latencies = self._evaluate_method(
            "semantic_rag", rag.query, client
        )
        elapsed = time.perf_counter() - start
        return MethodRunResult(
            "semantic_rag", answers, accuracy, ledger, elapsed, index_seconds, query_latencies
        )

    def _run_graph_method(
        self,
        *,
        method_name: str,
        search_method: str,
        workspace_dir: Any,
        indexing_method: str = "standard",
    ) -> MethodRunResult:
        ledger = TokenLedger()
        runner = GraphRAGRunner(
            self.config,
            ledger,
            workspace_dir=workspace_dir,
            indexing_method=indexing_method,
            search_method=search_method,
        )
        evaluator = AccuracyEvaluator(self._client_for(ledger), self.config.judge_model)

        start = time.perf_counter()
        index_start = time.perf_counter()
        runner.build_index()
        index_seconds = time.perf_counter() - index_start

        answers: list[dict[str, Any]] = []
        accuracy: list[AccuracyResult] = []
        query_latencies: list[float] = []
        for question in self.questions:
            query_start = time.perf_counter()
            result = runner.query(question.question)
            query_latencies.append(time.perf_counter() - query_start)
            answers.append(
                {
                    "question_id": question.id,
                    "question": question.question,
                    "answer": result.answer,
                    "search_method": result.search_method,
                    "query_type": question.query_type,
                }
            )
            accuracy.append(
                evaluator.evaluate(
                    method=method_name,
                    question=question,
                    prediction=result.answer,
                )
            )
        elapsed = time.perf_counter() - start
        return MethodRunResult(
            method_name, answers, accuracy, ledger, elapsed, index_seconds, query_latencies
        )

    def run_graph(self) -> MethodRunResult:
        """GraphRAG global search — best for thematic / corpus-wide questions."""
        return self._run_graph_method(
            method_name="graph_rag",
            search_method=self.config.graph_search_method,
            workspace_dir=self.config.graph_workspace,
            indexing_method=self.config.graph_indexing_method,
        )

    def run_graph_local(self) -> MethodRunResult:
        """GraphRAG local search — best for entity-centric questions."""
        return self._run_graph_method(
            method_name="graph_local_rag",
            search_method="local",
            workspace_dir=self.config.graph_workspace,
            indexing_method=self.config.graph_indexing_method,
        )

    def run_drift(self) -> MethodRunResult:
        """DRIFT — GraphRAG's native hybrid of global primer + local follow-ups."""
        return self._run_graph_method(
            method_name="drift_rag",
            search_method="drift",
            workspace_dir=self.config.graph_workspace,
            indexing_method=self.config.graph_indexing_method,
        )

    def run_lazygraph(self) -> MethodRunResult:
        ledger = TokenLedger()
        runner = LazyGraphRAGRunner(self.config, ledger)
        evaluator = AccuracyEvaluator(self._client_for(ledger), self.config.judge_model)

        start = time.perf_counter()
        index_start = time.perf_counter()
        runner.build_index()
        index_seconds = time.perf_counter() - index_start
        answers: list[dict[str, Any]] = []
        accuracy: list[AccuracyResult] = []
        query_latencies: list[float] = []
        for question in self.questions:
            query_start = time.perf_counter()
            result = runner.query(question.question)
            query_latencies.append(time.perf_counter() - query_start)
            answers.append(
                {
                    "question_id": question.id,
                    "question": question.question,
                    "answer": result.answer,
                    "search_method": result.search_method,
                    "query_type": question.query_type,
                }
            )
            accuracy.append(
                evaluator.evaluate(
                    method="lazygraph_rag",
                    question=question,
                    prediction=result.answer,
                )
            )
        elapsed = time.perf_counter() - start
        return MethodRunResult(
            "lazygraph_rag", answers, accuracy, ledger, elapsed, index_seconds, query_latencies
        )

    def run_hybrid(self) -> MethodRunResult:
        ledger = TokenLedger()
        client = self._client_for(ledger)
        rag = HybridRAG(self.config, client, ledger)

        start = time.perf_counter()
        index_start = time.perf_counter()
        rag.build_index()
        index_seconds = time.perf_counter() - index_start
        answers, accuracy, query_latencies = self._evaluate_method(
            "hybrid_rag",
            rag.query,
            client,
            extra_fields=lambda result: {
                "semantic_chunks": len(result.semantic_chunks),
                "graph_context_len": len(result.graph_answer),
            },
        )
        elapsed = time.perf_counter() - start
        return MethodRunResult(
            "hybrid_rag", answers, accuracy, ledger, elapsed, index_seconds, query_latencies
        )

    def run_lightrag(self) -> MethodRunResult:
        """HKUDS LightRAG — dual-level KG + vector (EMNLP 2025)."""
        ledger = TokenLedger()
        client = self._client_for(ledger)
        runner = LightRAGRunner(self.config, ledger)

        start = time.perf_counter()
        index_start = time.perf_counter()
        runner.build_index()
        index_seconds = time.perf_counter() - index_start
        try:
            answers, accuracy, query_latencies = self._evaluate_method(
                "light_rag",
                runner.query,
                client,
                extra_fields=lambda result: {"lightrag_mode": result.mode},
            )
        finally:
            runner.close()
        elapsed = time.perf_counter() - start
        return MethodRunResult(
            "light_rag", answers, accuracy, ledger, elapsed, index_seconds, query_latencies
        )

    def run_hybrid_dense_sparse(self) -> MethodRunResult:
        """BM25 + dense RRF — modern retrieval baseline."""
        ledger = TokenLedger()
        client = self._client_for(ledger)
        rag = HybridDenseSparseRAG(self.config, client, ledger)
        start = time.perf_counter()
        index_start = time.perf_counter()
        rag.build_index()
        index_seconds = time.perf_counter() - index_start
        answers, accuracy, query_latencies = self._evaluate_method(
            "hybrid_dense_sparse", rag.query, client
        )
        elapsed = time.perf_counter() - start
        return MethodRunResult(
            "hybrid_dense_sparse", answers, accuracy, ledger, elapsed, index_seconds, query_latencies
        )

    def run_rerank_semantic(self) -> MethodRunResult:
        """Dense retrieve + cross-encoder rerank."""
        ledger = TokenLedger()
        client = self._client_for(ledger)
        rag = RerankSemanticRAG(self.config, client, ledger)
        start = time.perf_counter()
        index_start = time.perf_counter()
        rag.build_index()
        index_seconds = time.perf_counter() - index_start
        answers, accuracy, query_latencies = self._evaluate_method(
            "rerank_semantic", rag.query, client
        )
        elapsed = time.perf_counter() - start
        return MethodRunResult(
            "rerank_semantic", answers, accuracy, ledger, elapsed, index_seconds, query_latencies
        )

    def run_adaptive(self) -> MethodRunResult:
        """Adaptive-RAG router: comparison→semantic, bridge→hybrid."""
        ledger = TokenLedger()
        client = self._client_for(ledger)
        semantic = SemanticRAG(self.config, client, ledger)
        hybrid = HybridRAG(self.config, client, ledger)

        start = time.perf_counter()
        index_start = time.perf_counter()
        semantic.build_index()
        hybrid.build_index()
        index_seconds = time.perf_counter() - index_start

        router = AdaptiveRAGRouter(
            semantic=semantic, hybrid_fn=hybrid.query, config=self.config
        )
        answers, accuracy, query_latencies = self._evaluate_method(
            "adaptive_rag",
            router.query,
            client,
            extra_fields=lambda result: {"route": result.route, "route_reason": result.reason},
        )
        elapsed = time.perf_counter() - start
        return MethodRunResult(
            "adaptive_rag", answers, accuracy, ledger, elapsed, index_seconds, query_latencies
        )

    def run_frontier(self) -> MethodRunResult:
        """FrontierRAG: Adaptive+CRAG retrieve/grade/escalate pipeline."""
        ledger = TokenLedger()
        client = self._client_for(ledger)
        rag = FrontierRAG(self.config, client, ledger)

        start = time.perf_counter()
        index_start = time.perf_counter()
        rag.build_index()
        index_seconds = time.perf_counter() - index_start
        answers, accuracy, query_latencies = self._evaluate_method(
            "frontier_rag",
            rag.query,
            client,
            extra_fields=lambda result: {
                "route": result.route,
                "route_reason": result.reason,
                "escalated": result.escalated,
                "graded_sufficient": result.graded_sufficient,
            },
        )
        elapsed = time.perf_counter() - start
        return MethodRunResult(
            "frontier_rag", answers, accuracy, ledger, elapsed, index_seconds, query_latencies
        )

    def run_hipporag(self) -> MethodRunResult:
        """HippoRAG 2 — multi-hop graph memory (OSU-NLP)."""
        ledger = TokenLedger()
        client = self._client_for(ledger)
        runner = HippoRAGRunner(self.config, ledger)

        start = time.perf_counter()
        index_start = time.perf_counter()
        runner.build_index()
        index_seconds = time.perf_counter() - index_start
        answers, accuracy, query_latencies = self._evaluate_method(
            "hippo_rag", runner.query, client
        )
        elapsed = time.perf_counter() - start
        return MethodRunResult(
            "hippo_rag", answers, accuracy, ledger, elapsed, index_seconds, query_latencies
        )

    def run_all(self, methods: list[str] | None = None) -> list[MethodRunResult]:
        runners: dict[str, Callable[[], MethodRunResult]] = {
            "semantic_rag": self.run_semantic,
            "graph_rag": self.run_graph,
            "graph_local_rag": self.run_graph_local,
            "hybrid_rag": self.run_hybrid,
            "drift_rag": self.run_drift,
            "lazygraph_rag": self.run_lazygraph,
            "light_rag": self.run_lightrag,
            "hippo_rag": self.run_hipporag,
            "hybrid_dense_sparse": self.run_hybrid_dense_sparse,
            "rerank_semantic": self.run_rerank_semantic,
            "adaptive_rag": self.run_adaptive,
            "frontier_rag": self.run_frontier,
        }
        selected = methods or [
            "semantic_rag",
            "graph_rag",
            "graph_local_rag",
            "hybrid_rag",
            "lazygraph_rag",
            "frontier_rag",
            "adaptive_rag",
            "hybrid_dense_sparse",
            "rerank_semantic",
        ]
        results: list[MethodRunResult] = []
        for name in selected:
            if name not in runners:
                raise ValueError(f"Unknown method: {name}. Choose from {list(runners)}")
            print(f"\n>>> Running {name} ...")
            results.append(runners[name]())
            print(
                f"<<< {name} done in {results[-1].elapsed_seconds:.1f}s | "
                f"tokens={results[-1].ledger.total().total_tokens}"
            )
        return results

    def _evaluate_method(
        self,
        method: str,
        query_fn: Any,
        client: TrackedLLMClient,
        extra_fields: Callable[[Any], dict[str, Any]] | None = None,
    ):
        evaluator = AccuracyEvaluator(client, self.config.judge_model)
        answers: list[dict[str, Any]] = []
        accuracy: list[AccuracyResult] = []
        query_latencies: list[float] = []

        for question in self.questions:
            query_start = time.perf_counter()
            result = query_fn(question.question)
            query_latencies.append(time.perf_counter() - query_start)
            row = {
                "question_id": question.id,
                "question": question.question,
                "answer": result.answer,
                "query_type": question.query_type,
            }
            if hasattr(result, "retrieved_chunks"):
                chunks = result.retrieved_chunks
                row["retrieved_chunks"] = len(chunks) if chunks is not None else 0
            if extra_fields:
                row.update(extra_fields(result))
            answers.append(row)
            accuracy.append(
                evaluator.evaluate(method=method, question=question, prediction=result.answer)
            )
        return answers, accuracy, query_latencies

    @staticmethod
    def to_latency_frame(results: list[MethodRunResult]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for result in results:
            for idx, latency in enumerate(result.query_latencies):
                rows.append(
                    {
                        "method": result.method,
                        "question_index": idx,
                        "query_latency_seconds": latency,
                    }
                )
            rows.append(
                {
                    "method": result.method,
                    "question_index": -1,
                    "query_latency_seconds": result.index_seconds,
                    "phase": "index",
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def to_summary_frame(results: list[MethodRunResult], config: BenchmarkConfig) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for result in results:
            accuracy_df = pd.DataFrame(
                [{"composite_score": item.composite_score()} for item in result.accuracy]
            )
            latency_series = pd.Series(result.query_latencies)
            rows.append(
                {
                    "method": result.method,
                    "mean_composite_score": float(accuracy_df["composite_score"].mean()),
                    "mean_llm_judge": float(
                        pd.DataFrame(
                            [{"s": item.llm_judge_score or 0.0} for item in result.accuracy]
                        )["s"].mean()
                    ),
                    "mean_token_f1": float(
                        pd.DataFrame([{"s": item.token_f1 or 0.0} for item in result.accuracy])[
                            "s"
                        ].mean()
                    ),
                    "exact_match_rate": float(
                        pd.DataFrame(
                            [
                                {"s": 1.0 if item.exact_match else 0.0}
                                for item in result.accuracy
                            ]
                        )["s"].mean()
                    ),
                    "contains_answer_rate": float(
                        pd.DataFrame(
                            [{"s": 1.0 if item.contains_answer else 0.0} for item in result.accuracy]
                        )["s"].mean()
                    ),
                    "total_tokens": result.ledger.total().total_tokens,
                    "prompt_tokens": result.ledger.total().prompt_tokens,
                    "completion_tokens": result.ledger.total().completion_tokens,
                    "estimated_cost_usd": result.ledger.estimate_cost_usd(config.pricing),
                    "index_seconds": result.index_seconds,
                    "mean_query_latency_seconds": float(latency_series.mean())
                    if len(latency_series)
                    else 0.0,
                    "p95_query_latency_seconds": float(latency_series.quantile(0.95))
                    if len(latency_series)
                    else 0.0,
                    "total_elapsed_seconds": result.elapsed_seconds,
                    "tokens_per_query": result.ledger.total().total_tokens
                    / max(len(result.accuracy), 1),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def to_accuracy_frame(results: list[MethodRunResult]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for result in results:
            for item in result.accuracy:
                rows.append(
                    {
                        "method": item.method,
                        "question_id": item.question_id,
                        "query_type": item.query_type,
                        "llm_judge_score": item.llm_judge_score,
                        "token_f1": item.token_f1,
                        "exact_match": item.exact_match,
                        "contains_answer": item.contains_answer,
                        "composite_score": item.composite_score(),
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def to_scenario_frame(results: list[MethodRunResult]) -> pd.DataFrame:
        """Mean composite score by method × query_type (local / global / hybrid)."""
        accuracy = BenchmarkRunner.to_accuracy_frame(results)
        if accuracy.empty:
            return accuracy
        return (
            accuracy.groupby(["method", "query_type"], as_index=False)["composite_score"]
            .mean()
            .rename(columns={"composite_score": "mean_composite_score"})
        )

    @staticmethod
    def to_token_frame(results: list[MethodRunResult], config: BenchmarkConfig) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for result in results:
            for phase, usage in result.ledger.by_phase.items():
                rows.append(
                    {
                        "method": result.method,
                        "phase": phase,
                        **usage.to_dict(),
                    }
                )
            rows.append(
                {
                    "method": result.method,
                    "phase": "__total__",
                    **result.ledger.total().to_dict(),
                    "estimated_cost_usd": result.ledger.estimate_cost_usd(config.pricing),
                    "elapsed_seconds": result.elapsed_seconds,
                }
            )
        return pd.DataFrame(rows)

    def save_results(self, results: list[MethodRunResult]) -> dict[str, Any]:
        accuracy_df = self.to_accuracy_frame(results)
        token_df = self.to_token_frame(results, self.config)
        latency_df = self.to_latency_frame(results)
        summary_df = self.to_summary_frame(results, self.config)
        scenario_df = self.to_scenario_frame(results)
        out_dir = self.config.results_dir()

        accuracy_path = out_dir / "accuracy_results.csv"
        token_path = out_dir / "token_results.csv"
        latency_path = out_dir / "latency_results.csv"
        summary_csv_path = out_dir / "summary.csv"
        scenario_path = out_dir / "scenario_results.csv"
        summary_path = out_dir / "summary.json"

        accuracy_df.to_csv(accuracy_path, index=False)
        token_df.to_csv(token_path, index=False)
        latency_df.to_csv(latency_path, index=False)
        summary_df.to_csv(summary_csv_path, index=False)
        scenario_df.to_csv(scenario_path, index=False)

        summary = {
            "methods": summary_df.to_dict(orient="records"),
            "scenarios": scenario_df.to_dict(orient="records"),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        from rag_benchmark.charts import plot_dashboard
        from rag_benchmark.decision_playbook import build_decision_artifacts
        from rag_benchmark.engineering import build_engineering_scorecard, save_engineering_scorecard

        plot_dashboard(out_dir)
        scorecard = build_engineering_scorecard(
            summary_df=summary_df,
            scenario_df=scenario_df,
            accuracy_df=accuracy_df,
        )
        eng_paths = save_engineering_scorecard(scorecard, out_dir)
        decision_paths = build_decision_artifacts(out_dir, self.config.qa_path)

        return {
            "accuracy_csv": str(accuracy_path),
            "token_csv": str(token_path),
            "latency_csv": str(latency_path),
            "summary_csv": str(summary_csv_path),
            "scenario_csv": str(scenario_path),
            "summary_json": str(summary_path),
            "engineering_json": str(eng_paths["json"]),
            "engineering_briefing": str(eng_paths["briefing"]),
            "routing_cheatsheet": str(decision_paths["cheatsheet"]),
            "choose_over_examples": str(decision_paths["examples"]),
            "accuracy_df": accuracy_df,
            "token_df": token_df,
            "latency_df": latency_df,
            "summary_df": summary_df,
            "scenario_df": scenario_df,
            "summary": summary,
            "engineering_scorecard": scorecard,
        }
