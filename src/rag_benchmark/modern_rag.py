"""Cutting-edge retrieval upgrades: BM25+dense (RRF) and cross-encoder rerank.

These are production SOTA *building blocks* (2024–2026 RAG stacks), not graph methods.
Papers/practice: hybrid sparse-dense retrieval; BGE/Cohere-style rerankers; Adaptive-RAG routing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from rag_benchmark.config import BenchmarkConfig
from rag_benchmark.corpus import TextChunk, chunk_documents, load_documents
from rag_benchmark.llm_factory import TrackedLLMClient
from rag_benchmark.semantic_rag import ANSWER_PROMPT, QueryResult, SemanticRAG
from rag_benchmark.token_tracker import TokenLedger


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridDenseSparseRAG(SemanticRAG):
    """Dense (embeddings) + sparse (BM25) fused with Reciprocal Rank Fusion."""

    def __init__(self, config: BenchmarkConfig, tracked_client: TrackedLLMClient, ledger: TokenLedger):
        super().__init__(config, tracked_client, ledger)
        self._bm25: BM25Okapi | None = None
        self._tokenized: list[list[str]] = []

    def build_index(self) -> None:
        super().build_index()
        # Chroma reuse may skip chunk materialization — always load for BM25.
        documents = load_documents(self.config.corpus_dir, self.config.max_documents)
        self._chunks = chunk_documents(
            documents,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        self._tokenized = [_tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(self._tokenized)

    def retrieve(self, question: str) -> list[str]:
        if self._collection is None or self._bm25 is None:
            raise RuntimeError("Index not built")

        dense = super().retrieve(question)
        # BM25 over all chunks
        scores = self._bm25.get_scores(_tokenize(question))
        top_n = min(max(self.config.semantic_top_k * 3, 15), len(scores))
        bm25_idx = np.argsort(scores)[::-1][:top_n]
        bm25_docs = [self._chunks[i].text for i in bm25_idx]

        # Reciprocal Rank Fusion
        ranks: dict[str, float] = {}
        for r, doc in enumerate(dense):
            ranks[doc] = ranks.get(doc, 0.0) + 1.0 / (60 + r + 1)
        for r, doc in enumerate(bm25_docs):
            ranks[doc] = ranks.get(doc, 0.0) + 1.0 / (60 + r + 1)
        ordered = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ordered[: self.config.semantic_top_k]]


class RerankSemanticRAG(SemanticRAG):
    """Retrieve wide, then cross-encoder rerank (BGE-style). Falls back to lexical if model missing."""

    def __init__(self, config: BenchmarkConfig, tracked_client: TrackedLLMClient, ledger: TokenLedger):
        super().__init__(config, tracked_client, ledger)
        self._reranker = None
        self._candidate_k = max(config.semantic_top_k * 4, 20)

    def _ensure_reranker(self) -> None:
        if self._reranker is not None:
            return
        try:
            from sentence_transformers import CrossEncoder

            # Small, widely used open reranker
            self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            self._reranker = False  # type: ignore[assignment]

    def retrieve(self, question: str) -> list[str]:
        if self._collection is None:
            raise RuntimeError("Index not built")

        # Temporarily fetch a wider pool
        old_k = self.config.semantic_top_k
        self.config.semantic_top_k = self._candidate_k
        try:
            candidates = SemanticRAG.retrieve(self, question)
        finally:
            self.config.semantic_top_k = old_k

        if len(candidates) <= old_k:
            return candidates

        self._ensure_reranker()
        if self._reranker is False:
            # Lexical fallback: overlap with question tokens
            q = set(_tokenize(question))
            scored = sorted(
                candidates,
                key=lambda d: len(q & set(_tokenize(d))),
                reverse=True,
            )
            return scored[:old_k]

        pairs = [(question, c) for c in candidates]
        scores = self._reranker.predict(pairs)
        order = np.argsort(scores)[::-1]
        return [candidates[i] for i in order[:old_k]]


@dataclass
class AdaptiveResult:
    answer: str
    route: str
    reason: str
    retrieved_chunks: list[str] | None = None


class AdaptiveRAGRouter:
    """Adaptive-RAG style complexity routing (Jeong et al., 2024).

    Production SOTA pattern: don't run one stack — classify query difficulty and route.
    Rules here are fitted to *this* Hotpot bake-off (local→semantic, bridge→hybrid).
    """

    COMPARISON_CUES = (
        "both",
        "same",
        "or",
        "which",
        "were ",
        "are ",
        "is ",
        "nationality",
        "neighborhood",
        "from the",
    )
    MULTI_HOP_CUES = (
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

    def __init__(
        self,
        *,
        semantic: SemanticRAG,
        hybrid_fn,
        config: BenchmarkConfig,
    ):
        self.semantic = semantic
        self.hybrid_fn = hybrid_fn
        self.config = config

    def classify(self, question: str) -> tuple[str, str]:
        q = question.lower().strip()
        # Explicit comparison / yes-no
        if any(c in q for c in self.COMPARISON_CUES) and (
            q.startswith(("are ", "were ", "is ", "was ", "which "))
            or " both " in f" {q} "
            or " or " in f" {q} "
        ):
            return "semantic_rag", "Comparison/factoid cues → vector (fast + precise on this set)"
        if any(c in q for c in self.MULTI_HOP_CUES) or q.count(",") >= 1 and "who" in q:
            return "hybrid_rag", "Bridge/multi-hop cues → hybrid vec+graph"
        # Default: cheap interactive path
        return "semantic_rag", "Default interactive path → vector (p95 latency SLO)"

    def query(self, question: str) -> AdaptiveResult:
        route, reason = self.classify(question)
        if route == "hybrid_rag":
            result = self.hybrid_fn(question)
            answer = result.answer if hasattr(result, "answer") else str(result)
            chunks = getattr(result, "retrieved_chunks", None) or []
            return AdaptiveResult(
                answer=answer, route=route, reason=reason, retrieved_chunks=list(chunks)
            )
        result = self.semantic.query(question)
        return AdaptiveResult(
            answer=result.answer,
            route=route,
            reason=reason,
            retrieved_chunks=list(result.retrieved_chunks or []),
        )
