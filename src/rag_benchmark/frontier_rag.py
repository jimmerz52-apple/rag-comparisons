"""FrontierRAG — 2025–2026 production-style adaptive pipeline.

Combines practices from Adaptive-RAG (Jeong et al.), CRAG (Yan et al.), and
hybrid sparse–dense retrieval + cross-encoder rerank:

1. Classify query complexity (factoid vs multi-hop)
2. Retrieve with BM25 + dense RRF (wide pool)
3. Cross-encoder rerank
4. Grade retrieval confidence (CRAG-style)
5. Escalate to GraphRAG-local hybrid fusion when multi-hop / low confidence
6. Generate a short answer

This is the cutting-edge *system* pattern: one entrypoint that routes compute,
not a single monolithic RAG flavor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from rag_benchmark.config import BenchmarkConfig
from rag_benchmark.hybrid_rag import HybridRAG
from rag_benchmark.llm_factory import TrackedLLMClient
from rag_benchmark.modern_rag import HybridDenseSparseRAG, RerankSemanticRAG, _tokenize
from rag_benchmark.semantic_rag import ANSWER_PROMPT
from rag_benchmark.token_tracker import TokenLedger

GRADE_PROMPT = """You are a retrieval grader for RAG.

Question: {question}

Retrieved context:
{context}

Does this context contain enough information to answer the question correctly?
Reply with ONLY one word: YES or NO."""


@dataclass
class FrontierResult:
    answer: str
    route: str
    reason: str
    graded_sufficient: bool
    escalated: bool
    retrieved_chunks: list[str]


class FrontierRAG:
    """Adaptive retrieve → grade → escalate pipeline."""

    def __init__(self, config: BenchmarkConfig, tracked_client: TrackedLLMClient, ledger: TokenLedger):
        self.config = config
        self.client = tracked_client
        self.ledger = ledger
        # Reuse dense+BM25 index; rerank on top of that retrieve pool.
        self.retriever = HybridDenseSparseRAG(config, tracked_client, ledger)
        self.reranker = RerankSemanticRAG(config, tracked_client, ledger)
        self.hybrid = HybridRAG(config, tracked_client, ledger)

    def build_index(self) -> None:
        self.retriever.build_index()
        # Share chroma collection / chunks with reranker path
        self.reranker._collection = self.retriever._collection
        self.reranker._chunks = self.retriever._chunks
        # Hybrid reuses GraphRAG workspace (already built in Hotpot runs)
        self.hybrid.build_index()

    def _classify(self, question: str) -> tuple[str, str]:
        q = question.lower().strip()
        comparison = (
            q.startswith(("are ", "were ", "is ", "was ", "which ", "who is ", "who was "))
            and any(x in q for x in (" both ", " or ", "same ", "nationality", "neighborhood"))
        )
        multi_hop = any(
            x in q
            for x in (
                "who portrayed",
                "formed by",
                "based in",
                "stage name",
                "debut album",
                "companion",
                "director of",
                "arena where",
                "that was",
                "who was known",
            )
        )
        if comparison and not multi_hop:
            return "factoid", "Comparison/factoid → cheap vector path"
        if multi_hop:
            return "multi_hop", "Bridge cues → plan escalate-capable path"
        return "default", "Default → retrieve+rerank; escalate if graded insufficient"

    def _retrieve_reranked(self, question: str) -> list[str]:
        # Wide hybrid retrieve
        old_k = self.config.semantic_top_k
        wide_k = max(old_k * 4, 20)
        self.config.semantic_top_k = wide_k
        try:
            candidates = self.retriever.retrieve(question)
        finally:
            self.config.semantic_top_k = old_k

        self.reranker._ensure_reranker()
        if self.reranker._reranker is False or self.reranker._reranker is None:
            q = set(_tokenize(question))
            scored = sorted(candidates, key=lambda d: len(q & set(_tokenize(d))), reverse=True)
            return scored[:old_k]

        pairs = [(question, c) for c in candidates]
        scores = self.reranker._reranker.predict(pairs)
        order = np.argsort(scores)[::-1]
        return [candidates[i] for i in order[:old_k]]

    def _grade(self, question: str, chunks: list[str]) -> bool:
        context = "\n\n---\n\n".join(chunks[:4])
        raw = self.client.chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": GRADE_PROMPT.format(question=question, context=context),
                }
            ],
            model=self.config.chat_model,
            phase="frontier_grade",
            temperature=0.0,
        )
        return "YES" in raw.upper().split()[0:3] or raw.strip().upper().startswith("YES")

    def _generate(self, question: str, chunks: list[str]) -> str:
        context = "\n\n---\n\n".join(chunks)
        return self.client.chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": ANSWER_PROMPT.format(question=question, context=context),
                }
            ],
            model=self.config.chat_model,
            phase="frontier_generate",
            temperature=0.0,
        )

    def query(self, question: str) -> FrontierResult:
        complexity, reason = self._classify(question)
        chunks = self._retrieve_reranked(question)
        sufficient = self._grade(question, chunks)

        # CRAG: escalate whenever retrieval is insufficient, or when the query
        # looks multi-hop (even if the grader is optimistic on a 3B model).
        escalate = (complexity == "multi_hop") or (not sufficient)
        if escalate:
            hybrid_result = self.hybrid.query(question)
            return FrontierResult(
                answer=hybrid_result.answer,
                route="frontier→hybrid_escalate",
                reason=f"{reason}; grade={'YES' if sufficient else 'NO'} → escalate to hybrid",
                graded_sufficient=sufficient,
                escalated=True,
                retrieved_chunks=chunks,
            )

        answer = self._generate(question, chunks)
        return FrontierResult(
            answer=answer,
            route="frontier→rerank_generate",
            reason=f"{reason}; grade={'YES' if sufficient else 'NO'} → stay on vector path",
            graded_sufficient=sufficient,
            escalated=False,
            retrieved_chunks=chunks,
        )
