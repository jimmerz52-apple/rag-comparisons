from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np

from rag_benchmark.config import BenchmarkConfig
from rag_benchmark.corpus import TextChunk, chunk_documents, load_documents
from rag_benchmark.llm_factory import TrackedLLMClient
from rag_benchmark.token_tracker import TokenLedger


ANSWER_PROMPT = """Answer the question using only the provided context.
Prefer a short, direct answer (entity name, date, number, or yes/no) when that matches the question.
Put that short answer on the FIRST line by itself. Do not hedge if the answer is clearly stated.
If the context is insufficient, say you do not have enough information.

Question: {question}

Context:
{context}

Answer:"""


@dataclass
class QueryResult:
    answer: str
    retrieved_chunks: list[str]


class SemanticRAG:
    def __init__(self, config: BenchmarkConfig, tracked_client: TrackedLLMClient, ledger: TokenLedger):
        self.config = config
        self.client = tracked_client
        self.ledger = ledger
        self._chunks: list[TextChunk] = []
        self._collection = None

    def build_index(self) -> None:
        documents = load_documents(self.config.corpus_dir, self.config.max_documents)
        self._chunks = chunk_documents(
            documents,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        if not self._chunks:
            raise ValueError(f"No chunks found in corpus: {self.config.corpus_dir}")

        chroma_path = self.config.project_root / ".chroma" / self.config.semantic_collection
        chroma = chromadb.PersistentClient(path=str(chroma_path))

        if not self.config.reuse_indexes:
            try:
                chroma.delete_collection(self.config.semantic_collection)
            except Exception:
                pass

        self._collection = chroma.get_or_create_collection(self.config.semantic_collection)

        if self.config.reuse_indexes and self._collection.count() >= len(self._chunks):
            return

        batch_size = 64
        for start in range(0, len(self._chunks), batch_size):
            batch = self._chunks[start : start + batch_size]
            embeddings = self.client.embed_texts(
                [chunk.text for chunk in batch],
                model=self.config.embedding_model,
                phase="semantic_index",
            )
            self._collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                embeddings=embeddings,
                metadatas=[
                    {"doc_id": chunk.doc_id, "source_path": str(chunk.source_path)}
                    for chunk in batch
                ],
            )

    def retrieve(self, question: str) -> list[str]:
        """Return top-k chunks without generating an answer (for hybrid fusion)."""
        if self._collection is None:
            raise RuntimeError("Semantic index not built. Call build_index() first.")

        query_embedding = self.client.embed_texts(
            [question],
            model=self.config.embedding_model,
            phase="semantic_query",
        )[0]

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=self.config.semantic_top_k,
        )
        return results.get("documents", [[]])[0]

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
        )
        return QueryResult(answer=answer, retrieved_chunks=retrieved)

    @staticmethod
    def cosine_top_k(query_vector: np.ndarray, matrix: np.ndarray, k: int) -> list[int]:
        scores = matrix @ query_vector
        if k >= len(scores):
            return list(np.argsort(scores)[::-1])
        top_idx = np.argpartition(scores, -k)[-k:]
        return list(top_idx[np.argsort(scores[top_idx])[::-1]])
