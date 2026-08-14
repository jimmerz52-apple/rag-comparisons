from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from rag_benchmark.config import BenchmarkConfig
from rag_benchmark.graph_rag import GraphRAGRunner, LazyGraphRAGRunner
from rag_benchmark.frontier_rag import FrontierRAG
from rag_benchmark.hippo_rag import HippoRAGRunner
from rag_benchmark.hybrid_rag import HybridRAG
from rag_benchmark.light_rag import LightRAGRunner
from rag_benchmark.memory_structures import ParentChildRAG, PropositionRAG, RaptorRAG
from rag_benchmark.modern_rag import AdaptiveRAGRouter, HybridDenseSparseRAG, RerankSemanticRAG
from rag_benchmark.llm_factory import TrackedLLMClient, clone_client_for_ledger, create_tracked_client
from rag_benchmark.metrics import AccuracyEvaluator, AccuracyResult, load_eval_questions
from rag_benchmark.retrieval import retrieval_scores
from rag_benchmark.semantic_rag import SemanticRAG
from rag_benchmark.token_tracker import TokenLedger


def _col_sum(frame: pd.DataFrame, name: str) -> int:
    if name not in frame.columns:
        return 0
    return int(pd.to_numeric(frame[name], errors="coerce").fillna(0).sum())


@dataclass
class MethodRunResult:
    method: str
    answers: list[dict[str, Any]]
    accuracy: list[AccuracyResult]
    ledger: TokenLedger
    elapsed_seconds: float
    index_seconds: float = 0.0
    query_latencies: list[float] = field(default_factory=list)
    per_query_tokens: list[dict[str, int]] = field(default_factory=list)


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
        answers, accuracy, query_latencies, token_rows = self._evaluate_method(
            "semantic_rag", rag.query, client
        )
        elapsed = time.perf_counter() - start
        return MethodRunResult(
            "semantic_rag", answers, accuracy, ledger, elapsed, index_seconds, query_latencies,
            per_query_tokens=token_rows,
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
        answers, accuracy, query_latencies, token_rows = self._evaluate_method(
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
            "hybrid_rag", answers, accuracy, ledger, elapsed, index_seconds, query_latencies,
            per_query_tokens=token_rows,
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
            answers, accuracy, query_latencies, token_rows = self._evaluate_method(
                "light_rag",
                runner.query,
                client,
                extra_fields=lambda result: {"lightrag_mode": result.mode},
            )
        finally:
            runner.close()
        elapsed = time.perf_counter() - start
        return MethodRunResult(
            "light_rag", answers, accuracy, ledger, elapsed, index_seconds, query_latencies,
            per_query_tokens=token_rows,
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
        answers, accuracy, query_latencies, token_rows = self._evaluate_method(
            "hybrid_dense_sparse", rag.query, client
        )
        elapsed = time.perf_counter() - start
        return MethodRunResult(
            "hybrid_dense_sparse", answers, accuracy, ledger, elapsed, index_seconds, query_latencies,
            per_query_tokens=token_rows,
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
        answers, accuracy, query_latencies, token_rows = self._evaluate_method(
            "rerank_semantic", rag.query, client
        )
        elapsed = time.perf_counter() - start
        return MethodRunResult(
            "rerank_semantic", answers, accuracy, ledger, elapsed, index_seconds, query_latencies,
            per_query_tokens=token_rows,
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
        answers, accuracy, query_latencies, token_rows = self._evaluate_method(
            "adaptive_rag",
            router.query,
            client,
            extra_fields=lambda result: {"route": result.route, "route_reason": result.reason},
        )
        elapsed = time.perf_counter() - start
        return MethodRunResult(
            "adaptive_rag", answers, accuracy, ledger, elapsed, index_seconds, query_latencies,
            per_query_tokens=token_rows,
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
        answers, accuracy, query_latencies, token_rows = self._evaluate_method(
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
            "frontier_rag", answers, accuracy, ledger, elapsed, index_seconds, query_latencies,
            per_query_tokens=token_rows,
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
        answers, accuracy, query_latencies, token_rows = self._evaluate_method(
            "hippo_rag", runner.query, client
        )
        elapsed = time.perf_counter() - start
        return MethodRunResult(
            "hippo_rag", answers, accuracy, ledger, elapsed, index_seconds, query_latencies,
            per_query_tokens=token_rows,
        )

    def run_raptor(self) -> MethodRunResult:
        """RAPTOR-lite hierarchical summary tree (Sarthi et al., ICLR 2024)."""
        ledger = TokenLedger()
        client = self._client_for(ledger)
        rag = RaptorRAG(self.config, client, ledger)
        start = time.perf_counter()
        index_start = time.perf_counter()
        rag.build_index()
        index_seconds = time.perf_counter() - index_start
        answers, accuracy, query_latencies, token_rows = self._evaluate_method(
            "raptor_rag", rag.query, client
        )
        elapsed = time.perf_counter() - start
        return MethodRunResult(
            "raptor_rag", answers, accuracy, ledger, elapsed, index_seconds, query_latencies,
            per_query_tokens=token_rows,
        )

    def run_parent_child(self) -> MethodRunResult:
        """Small-to-big: retrieve child chunks, expand to parent windows."""
        ledger = TokenLedger()
        client = self._client_for(ledger)
        rag = ParentChildRAG(self.config, client, ledger)
        start = time.perf_counter()
        index_start = time.perf_counter()
        rag.build_index()
        index_seconds = time.perf_counter() - index_start
        answers, accuracy, query_latencies, token_rows = self._evaluate_method(
            "parent_child_rag", rag.query, client
        )
        elapsed = time.perf_counter() - start
        return MethodRunResult(
            "parent_child_rag", answers, accuracy, ledger, elapsed, index_seconds, query_latencies,
            per_query_tokens=token_rows,
        )

    def run_proposition(self) -> MethodRunResult:
        """Dense-X-style proposition index (local LLM propositionizer)."""
        ledger = TokenLedger()
        client = self._client_for(ledger)
        rag = PropositionRAG(self.config, client, ledger)
        start = time.perf_counter()
        index_start = time.perf_counter()
        rag.build_index()
        index_seconds = time.perf_counter() - index_start
        answers, accuracy, query_latencies, token_rows = self._evaluate_method(
            "proposition_rag", rag.query, client
        )
        elapsed = time.perf_counter() - start
        return MethodRunResult(
            "proposition_rag", answers, accuracy, ledger, elapsed, index_seconds, query_latencies,
            per_query_tokens=token_rows,
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
            "raptor_rag": self.run_raptor,
            "parent_child_rag": self.run_parent_child,
            "proposition_rag": self.run_proposition,
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

    def _ledger_totals(client: Any) -> tuple[int, int, int]:
        ledger = getattr(client, "ledger", None)
        if ledger is None:
            return (0, 0, 0)
        t = ledger.total()
        return (int(t.prompt_tokens), int(t.completion_tokens), int(t.total_tokens))

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
        token_rows: list[dict[str, int]] = []
        total = len(self.questions)
        checkpoint_every = 25 if total >= 500 else max(5, min(25, total // 5 or 5))
        out_dir = self.config.results_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / f"_checkpoint_{method}_accuracy.csv"
        done_ids: set[str] = set()
        if ckpt_path.exists() and ckpt_path.stat().st_size > 3:
            prev = pd.read_csv(ckpt_path)
            for _, r in prev.iterrows():
                qid = str(r.get("question_id"))
                rationale = str(r.get("judge_rationale") or "")
                if rationale.startswith("error"):
                    continue
                done_ids.add(qid)
                accuracy.append(
                    AccuracyResult(
                        question_id=qid,
                        method=str(r.get("method") or method),
                        query_type=str(r.get("query_type") or "local"),
                        llm_judge_score=float(r["llm_judge_score"])
                        if pd.notna(r.get("llm_judge_score"))
                        else 0.0,
                        token_f1=float(r["token_f1"]) if pd.notna(r.get("token_f1")) else 0.0,
                        exact_match=bool(r.get("exact_match")),
                        contains_answer=bool(r.get("contains_answer")),
                        retrieval_recall=float(r["retrieval_recall"])
                        if "retrieval_recall" in r and pd.notna(r.get("retrieval_recall"))
                        else None,
                        retrieval_precision=float(r["retrieval_precision"])
                        if "retrieval_precision" in r and pd.notna(r.get("retrieval_precision"))
                        else None,
                        gold_in_context=bool(r.get("gold_in_context"))
                        if "gold_in_context" in r and pd.notna(r.get("gold_in_context"))
                        else None,
                        evidence_override=bool(r.get("evidence_override"))
                        if "evidence_override" in r and pd.notna(r.get("evidence_override"))
                        else None,
                    )
                )
                query_latencies.append(float(r.get("query_latency_seconds") or 0.0))
                token_rows.append(
                    {
                        "query_prompt_tokens": int(r.get("query_prompt_tokens") or 0),
                        "query_completion_tokens": int(r.get("query_completion_tokens") or 0),
                        "eval_prompt_tokens": int(r.get("eval_prompt_tokens") or 0),
                        "eval_completion_tokens": int(r.get("eval_completion_tokens") or 0),
                        "prompt_tokens": int(r.get("prompt_tokens") or 0),
                        "completion_tokens": int(r.get("completion_tokens") or 0),
                        "total_tokens": int(r.get("total_tokens") or 0),
                    }
                )
                answers.append({"question_id": qid, "resumed": True})
            if done_ids:
                print(
                    f"  [{method}] resume: {len(done_ids)}/{total} already scored — skipping",
                    flush=True,
                )

        for i, question in enumerate(self.questions, start=1):
            if str(question.id) in done_ids:
                continue
            snap0 = self._ledger_totals(client)
            try:
                query_start = time.perf_counter()
                result = query_fn(question.question)
                query_latencies.append(time.perf_counter() - query_start)
                snap_q = self._ledger_totals(client)
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
                acc_item = evaluator.evaluate(
                    method=method, question=question, prediction=result.answer
                )
                chunks = getattr(result, "retrieved_chunks", None) or []
                rs = retrieval_scores(
                    gold_titles=question.supporting_titles,
                    chunks=list(chunks) if chunks else [],
                    gold_answer=question.expected_answer,
                )
                acc_item.retrieval_recall = rs["retrieval_recall"]
                acc_item.retrieval_precision = rs["retrieval_precision"]
                acc_item.gold_in_context = rs["gold_in_context"]
                acc_item.evidence_override = bool(
                    rs["gold_in_context"] and not acc_item.contains_answer
                )
                accuracy.append(acc_item)
                snap1 = self._ledger_totals(client)
                token_rows.append(
                    {
                        "query_prompt_tokens": snap_q[0] - snap0[0],
                        "query_completion_tokens": snap_q[1] - snap0[1],
                        "eval_prompt_tokens": snap1[0] - snap_q[0],
                        "eval_completion_tokens": snap1[1] - snap_q[1],
                        "prompt_tokens": snap1[0] - snap0[0],
                        "completion_tokens": snap1[1] - snap0[1],
                        "total_tokens": snap1[2] - snap0[2],
                    }
                )
            except Exception as exc:  # noqa: BLE001 — keep thousand-scale runs alive
                print(f"  [{method}] ERROR on {question.id}: {exc!r}", flush=True)
                query_latencies.append(0.0)
                answers.append(
                    {
                        "question_id": question.id,
                        "question": question.question,
                        "answer": "",
                        "query_type": question.query_type,
                        "error": repr(exc),
                    }
                )
                accuracy.append(
                    AccuracyResult(
                        question_id=question.id,
                        method=method,
                        query_type=question.query_type,
                        llm_judge_score=0.0,
                        token_f1=0.0,
                        exact_match=False,
                        contains_answer=False,
                        judge_rationale=f"error: {exc!r}",
                    )
                )
                token_rows.append(
                    {
                        "query_prompt_tokens": 0,
                        "query_completion_tokens": 0,
                        "eval_prompt_tokens": 0,
                        "eval_completion_tokens": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    }
                )
            if i == 1 or i % 25 == 0 or i == total:
                mean_c = sum(a.composite_score() for a in accuracy) / len(accuracy)
                last_tok = token_rows[-1]["total_tokens"] if token_rows else 0
                print(
                    f"  [{method}] {i}/{total}  "
                    f"last_latency={query_latencies[-1]:.1f}s  "
                    f"last_tokens={last_tok}  "
                    f"running_composite={mean_c:.3f}",
                    flush=True,
                )
            if i % checkpoint_every == 0 or i == total:
                rows = []
                for item, lat, tok in zip(accuracy, query_latencies, token_rows):
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
                            "generative_score": item.generative_score(),
                            "extractive_score": item.extractive_score(),
                            "query_latency_seconds": lat,
                            "retrieval_recall": item.retrieval_recall,
                            "retrieval_precision": item.retrieval_precision,
                            "gold_in_context": item.gold_in_context,
                            "evidence_override": item.evidence_override,
                            **tok,
                        }
                    )
                pd.DataFrame(rows).to_csv(ckpt_path, index=False)
                self._publish_live_accuracy(method=method, checkpoint=ckpt_path)
        return answers, accuracy, query_latencies, token_rows

    def _publish_live_accuracy(self, *, method: str, checkpoint: Path) -> None:
        """Merge checkpoint into accuracy_results.csv so Pages/dashboard can show thousands mid-run."""
        out = self.config.results_dir() / "accuracy_results.csv"
        enriched = self.config.results_dir() / "accuracy_enriched.csv"
        fresh = pd.read_csv(checkpoint)
        if out.exists() and out.stat().st_size > 3:
            old = pd.read_csv(out)
            old = old[old["method"] != method]
            merged = pd.concat([old, fresh], ignore_index=True)
        else:
            merged = fresh
        merged.to_csv(out, index=False)
        merged.to_csv(enriched, index=False)
        # Lightweight live summary so dashboard leaderboard moves
        summary_rows = []
        for m, g in merged.groupby("method"):
            lat = (
                g["query_latency_seconds"].replace(0, pd.NA).dropna()
                if "query_latency_seconds" in g.columns
                else pd.Series(dtype=float)
            )
            tok_q = (
                float(g["total_tokens"].mean())
                if "total_tokens" in g.columns
                else 0.0
            )
            prompt_q = (
                float(g["prompt_tokens"].mean())
                if "prompt_tokens" in g.columns
                else 0.0
            )
            completion_q = (
                float(g["completion_tokens"].mean())
                if "completion_tokens" in g.columns
                else 0.0
            )
            summary_rows.append(
                {
                    "method": m,
                    "mean_composite_score": float(g["composite_score"].mean()),
                    "mean_llm_judge": float(g["llm_judge_score"].mean()),
                    "mean_token_f1": float(g["token_f1"].mean()),
                    "exact_match_rate": float(g["exact_match"].mean()),
                    "contains_answer_rate": float(g["contains_answer"].mean()),
                    "mean_retrieval_recall": float(g["retrieval_recall"].mean())
                    if "retrieval_recall" in g.columns
                    and g["retrieval_recall"].notna().any()
                    else None,
                    "gold_in_context_rate": float(g["gold_in_context"].mean())
                    if "gold_in_context" in g.columns
                    and g["gold_in_context"].notna().any()
                    else None,
                    "evidence_override_rate": float(g["evidence_override"].mean())
                    if "evidence_override" in g.columns
                    and g["evidence_override"].notna().any()
                    else None,
                    "tokens_per_query": tok_q,
                    "prompt_tokens_per_query": prompt_q,
                    "completion_tokens_per_query": completion_q,
                    "total_tokens": float(g["total_tokens"].sum()) if "total_tokens" in g.columns else 0.0,
                    "prompt_tokens": float(g["prompt_tokens"].sum()) if "prompt_tokens" in g.columns else 0.0,
                    "completion_tokens": float(g["completion_tokens"].sum()) if "completion_tokens" in g.columns else 0.0,
                    "n_scored": int(g["question_id"].nunique()),
                    "mean_query_latency_seconds": float(lat.mean()) if len(lat) else 0.0,
                    "p50_query_latency_seconds": float(lat.quantile(0.50)) if len(lat) else 0.0,
                    "p95_query_latency_seconds": float(lat.quantile(0.95)) if len(lat) else 0.0,
                }
            )
        pd.DataFrame(summary_rows).to_csv(self.config.results_dir() / "summary.csv", index=False)
        if "query_latency_seconds" in merged.columns:
            live_lat = merged[["method", "question_id", "query_latency_seconds"]].copy()
            live_lat["question_index"] = range(len(live_lat))
            live_lat.to_csv(self.config.results_dir() / "latency_results.csv", index=False)
        if "total_tokens" in merged.columns:
            tok_rows = []
            live_methods = set()
            for m, g in merged.groupby("method"):
                if g["total_tokens"].fillna(0).sum() <= 0:
                    continue
                live_methods.add(m)
                n = max(int(g["question_id"].nunique()), 1)
                q_prompt = _col_sum(g, "query_prompt_tokens")
                q_comp = _col_sum(g, "query_completion_tokens")
                e_prompt = _col_sum(g, "eval_prompt_tokens")
                e_comp = _col_sum(g, "eval_completion_tokens")
                tok_rows.append(
                    {
                        "method": m,
                        "phase": "query",
                        "prompt_tokens": q_prompt,
                        "completion_tokens": q_comp,
                        "total_tokens": q_prompt + q_comp,
                        "calls": n,
                    }
                )
                tok_rows.append(
                    {
                        "method": m,
                        "phase": "evaluation",
                        "prompt_tokens": e_prompt,
                        "completion_tokens": e_comp,
                        "total_tokens": e_prompt + e_comp,
                        "calls": n,
                    }
                )
                tok_rows.append(
                    {
                        "method": m,
                        "phase": "__total__",
                        "prompt_tokens": _col_sum(g, "prompt_tokens"),
                        "completion_tokens": _col_sum(g, "completion_tokens"),
                        "total_tokens": _col_sum(g, "total_tokens"),
                        "calls": n,
                    }
                )
            tok_path = self.config.results_dir() / "token_results.csv"
            new_tok = pd.DataFrame(tok_rows)
            if tok_path.exists() and tok_path.stat().st_size > 3 and live_methods:
                old_tok = pd.read_csv(tok_path)
                old_tok = old_tok[~old_tok["method"].isin(live_methods)]
                new_tok = pd.concat([old_tok, new_tok], ignore_index=True)
            if not new_tok.empty:
                new_tok.to_csv(tok_path, index=False)
        meta = {
            "live_partial": True,
            "updated_method": method,
            "n_rows": int(len(merged)),
            "n_questions": int(merged["question_id"].nunique()),
        }
        (self.config.results_dir() / "live_partial_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

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
            n_q = max(len(result.accuracy), 1)
            ledger_total = result.ledger.total()
            index_tokens = 0
            index_prompt = 0
            index_completion = 0
            for phase, usage in result.ledger.by_phase.items():
                if "index" in str(phase).lower():
                    index_tokens += usage.total_tokens
                    index_prompt += usage.prompt_tokens
                    index_completion += usage.completion_tokens
            serving_tokens = ledger_total.total_tokens - index_tokens
            serving_prompt = ledger_total.prompt_tokens - index_prompt
            serving_completion = ledger_total.completion_tokens - index_completion
            rows.append(
                {
                    "method": result.method,
                    "mean_composite_score": float(accuracy_df["composite_score"].mean()),
                    "mean_generative_score": float(
                        pd.DataFrame(
                            [{"s": item.generative_score()} for item in result.accuracy]
                        )["s"].mean()
                    ),
                    "mean_extractive_score": float(
                        pd.DataFrame(
                            [{"s": item.extractive_score()} for item in result.accuracy]
                        )["s"].mean()
                    ),
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
                    "total_tokens": serving_tokens,
                    "prompt_tokens": serving_prompt,
                    "completion_tokens": serving_completion,
                    "index_tokens": index_tokens,
                    "estimated_cost_usd": result.ledger.estimate_cost_usd(config.pricing),
                    "index_seconds": result.index_seconds,
                    "mean_query_latency_seconds": float(latency_series.mean())
                    if len(latency_series)
                    else 0.0,
                    "p95_query_latency_seconds": float(latency_series.quantile(0.95))
                    if len(latency_series)
                    else 0.0,
                    "total_elapsed_seconds": result.elapsed_seconds,
                    "tokens_per_query": serving_tokens / n_q,
                    "prompt_tokens_per_query": serving_prompt / n_q,
                    "completion_tokens_per_query": serving_completion / n_q,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def to_accuracy_frame(results: list[MethodRunResult]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for result in results:
            token_rows = result.per_query_tokens or [{}] * len(result.accuracy)
            for item, tok in zip(result.accuracy, token_rows):
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
                        "generative_score": item.generative_score(),
                        "extractive_score": item.extractive_score(),
                        **tok,
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

        # Refresh the multi-bench HTML dashboard when any run completes
        try:
            from scripts.build_dashboard import build as build_html_dashboard

            dash = build_html_dashboard()
        except Exception:
            try:
                import importlib.util

                spec = importlib.util.spec_from_file_location(
                    "build_dashboard",
                    self.config.project_root / "scripts" / "build_dashboard.py",
                )
                mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
                assert spec and spec.loader
                spec.loader.exec_module(mod)
                dash = mod.build()
            except Exception:
                dash = None

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
            "dashboard_html": str(dash) if dash else None,
            "accuracy_df": accuracy_df,
            "token_df": token_df,
            "latency_df": latency_df,
            "summary_df": summary_df,
            "scenario_df": scenario_df,
            "summary": summary,
            "engineering_scorecard": scorecard,
        }
