"""HKUDS LightRAG (EMNLP 2025) — dual-level graph + vector RAG.

Paper: Guo et al., "LightRAG: Simple and Fast Retrieval-Augmented Generation"
Repo: https://github.com/HKUDS/LightRAG

This is a real third-party framework (pip: lightrag-hku), not a GraphRAG CLI alias.
"""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from rag_benchmark.config import BenchmarkConfig
from rag_benchmark.corpus import load_documents
from rag_benchmark.token_tracker import TokenLedger


@dataclass
class LightRAGResult:
    answer: str
    mode: str


class LightRAGRunner:
    """Wrap LightRAG SDK for the local Ollama benchmark harness."""

    def __init__(self, config: BenchmarkConfig, ledger: TokenLedger):
        self.config = config
        self.ledger = ledger
        self.working_dir = Path(config.lightrag_workspace)
        self.mode = config.lightrag_mode
        self._rag: Any = None
        self._loop = asyncio.new_event_loop()

    def build_index(self) -> None:
        self.working_dir.mkdir(parents=True, exist_ok=True)
        marker = self.working_dir / "kv_store_full_docs.json"
        status_path = self.working_dir / "kv_store_doc_status.json"

        documents = load_documents(self.config.corpus_dir, self.config.max_documents)
        if not documents:
            raise ValueError(f"No documents in {self.config.corpus_dir}")
        texts = [doc.text for doc in documents]
        ids = [doc.doc_id for doc in documents]

        pending = 0
        if status_path.exists():
            import json

            status = json.loads(status_path.read_text(encoding="utf-8"))
            pending = sum(
                1
                for v in status.values()
                if isinstance(v, dict) and v.get("status") not in {"processed", "completed"}
            )

        # Reuse existing store, but finish any pending/failed docs after a stall/crash.
        if self.config.reuse_indexes and marker.exists() and marker.stat().st_size > 2:
            self._rag = self._run(self._init_rag())
            if pending > 0:
                print(f"LightRAG resume: {pending} docs still pending — continuing ainsert")
                self._run(self._rag.ainsert(texts, ids=ids))
            return

        if self.working_dir.exists() and any(self.working_dir.iterdir()):
            for path in self.working_dir.iterdir():
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)

        self._rag = self._run(self._init_rag())
        self._run(self._rag.ainsert(texts, ids=ids))
        for text in texts:
            self.ledger.record(
                phase="lightrag_index_estimate",
                model=self.config.chat_model,
                text=text,
                role="prompt",
            )

    def query(self, question: str) -> LightRAGResult:
        if self._rag is None:
            self._rag = self._run(self._init_rag())

        from lightrag import QueryParam

        answer = self._run(
            self._rag.aquery(
                question,
                param=QueryParam(
                    mode=self.mode,
                    response_type="Short Answer",
                    enable_rerank=False,
                ),
            )
        )
        if not isinstance(answer, str):
            answer = str(answer)

        self.ledger.record(
            phase="lightrag_query_estimate",
            model=self.config.chat_model,
            text=question,
            role="prompt",
        )
        self.ledger.record(
            phase="lightrag_query_estimate",
            model=self.config.chat_model,
            text=answer,
            role="completion",
        )
        return LightRAGResult(answer=answer.strip(), mode=self.mode)

    def close(self) -> None:
        if self._rag is not None:
            try:
                self._run(self._rag.finalize_storages())
            except Exception:
                pass
            self._rag = None
        # Leave the event loop open; LightRAG worker tasks may still be draining.

    async def _init_rag(self) -> Any:
        from lightrag import LightRAG
        from lightrag.llm.ollama import ollama_embed, ollama_model_complete
        from lightrag.utils import EmbeddingFunc

        embed_model = self.config.graphrag_embedding_model
        # nomic-embed-text → 768; override via config if needed
        embed_dim = self.config.lightrag_embedding_dim
        host = self.config.ollama_base_url

        rag = LightRAG(
            working_dir=str(self.working_dir),
            llm_model_func=ollama_model_complete,
            llm_model_name=self.config.chat_model,
            llm_model_kwargs={
                "host": host,
                "options": {"num_ctx": 8192},
                "timeout": 300,
            },
            embedding_func=EmbeddingFunc(
                embedding_dim=embed_dim,
                max_token_size=8192,
                func=partial(
                    ollama_embed.func,
                    embed_model=embed_model,
                    host=host,
                ),
            ),
        )
        await rag.initialize_storages()
        return rag

    def _run(self, coro: Any) -> Any:
        return self._loop.run_until_complete(coro)
