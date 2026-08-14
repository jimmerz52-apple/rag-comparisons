"""Classic dense RAG method — thin wrapper over the SDK lineage store.

Uses ``rag_benchmark.sdk`` for embedder + SourceRef + incremental index.
Method bake-offs keep calling this class; apps should prefer
``RagPipelineOrchestrator`` for citations / sync APIs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rag_benchmark.config import BenchmarkConfig
from rag_benchmark.corpus import TextChunk, load_documents
from rag_benchmark.llm_factory import TrackedLLMClient
from rag_benchmark.prompts import ANSWER_PROMPT
from rag_benchmark.sdk.embedder import TrackedClientEmbedder
from rag_benchmark.sdk.vector_store import LineageVectorStore
from rag_benchmark.token_tracker import TokenLedger


@dataclass
class QueryResult:
    answer: str
    retrieved_chunks: list[str]


class SemanticRAG:
    def __init__(self, config: BenchmarkConfig, tracked_client: TrackedLLMClient, ledger: TokenLedger):
        self.config = config
        self.client = tracked_client
        self.ledger = ledger
        self._embedder = TrackedClientEmbedder(tracked_client, config.embedding_model)
        self._store = LineageVectorStore(
            root=config.project_root / ".chroma" / "sdk_lineage",
            base_collection=config.semantic_collection,
            embedder=self._embedder,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        self._chunks: list[TextChunk] = []

    @property
    def _collection(self):
        """Back-compat for HybridDenseSparseRAG / RerankSemanticRAG."""
        return self._store.collection

    @_collection.setter
    def _collection(self, value) -> None:
        # Allow subclasses/tests to poke the collection; prefer store APIs.
        if value is not None:
            self._store._collection = value

    def build_index(self) -> None:
        documents = load_documents(self.config.corpus_dir, self.config.max_documents)
        if not documents:
            raise ValueError(f"No chunks found in corpus: {self.config.corpus_dir}")

        existing = self._store.count()
        if self.config.reuse_indexes and existing >= max(1, int(0.9 * len(documents))):
            print(
                f"  Reusing semantic index ({existing} vectors ≥ 90% of {len(documents)} docs)",
                flush=True,
            )
        elif not self.config.reuse_indexes:
            print(f"  Rebuilding semantic index from scratch ({len(documents)} docs)...", flush=True)
            self._store.reset()
            self._store.upsert_documents(documents, phase="semantic_index")
        else:
            print(
                f"  Syncing semantic index (have {existing}, corpus {len(documents)})...",
                flush=True,
            )
            self._store.sync_documents(documents, drop_missing=True, phase="semantic_index")

        print(f"  Index ready: store={self._store.count()}", flush=True)

    def retrieve(self, question: str) -> list[str]:
        """Return top-k chunk texts (for hybrid fusion / method bake-off)."""
        if self._store.count() == 0:
            raise RuntimeError("Semantic index not built. Call build_index() first.")
        hits = self._store.query(
            question, top_k=self.config.semantic_top_k, phase="semantic_query"
        )
        return [text for text, _ref, _dist in hits]

    def retrieve_with_sources(self, question: str):
        """Enterprise path: texts + SourceRef lineage."""
        return self._store.query(
            question, top_k=self.config.semantic_top_k, phase="semantic_query"
        )

    def query(self, question: str) -> QueryResult:
        retrieved = self.retrieve(question)
        context = "\n\n---\n\n".join(retrieved)

        answer = self.client.chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": ANSWER_PROMPT.format(question=question, context=context),
                }
            ],
            model=self.config.chat_model,
            phase="semantic_query",
            temperature=0.0,
            num_predict=getattr(self.config, "max_answer_tokens", 96),
        )
        return QueryResult(answer=answer, retrieved_chunks=retrieved)

    @staticmethod
    def cosine_top_k(query_vector: np.ndarray, matrix: np.ndarray, k: int) -> list[int]:
        scores = matrix @ query_vector
        if k >= len(scores):
            return list(np.argsort(scores)[::-1])
        top_idx = np.argpartition(scores, -k)[-k:]
        return list(top_idx[np.argsort(scores[top_idx])[::-1]])
