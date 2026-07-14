"""HippoRAG 2 (OSU-NLP) adapter for the local Ollama benchmark harness.

Paper: Gutiérrez et al., "From RAG to Memory" (ICML 2025) / HippoRAG (NeurIPS 2024)
Repo: https://github.com/OSU-NLP-Group/HippoRAG

Install (lightweight, no vLLM):
  pip install hipporag==2.0.0a3 --no-deps
  pip install igraph gritlm networkx  # gritlm may pull torch

HippoRAG's EmbeddingCache starts a multiprocessing.Manager at import time; we
stub that (and optional vllm) so macOS / Ollama setups can import cleanly.
"""

from __future__ import annotations

import os
import sys
import threading
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_benchmark.config import BenchmarkConfig
from rag_benchmark.corpus import load_documents
from rag_benchmark.token_tracker import TokenLedger


@dataclass
class HippoRAGResult:
    answer: str


def _prepare_hipporag_import() -> None:
    """Stub heavy optional deps and avoid multiprocessing.Manager at import."""
    import multiprocessing

    class _FakeManager:
        def dict(self):
            return {}

        def list(self):
            return []

        def Lock(self):
            return threading.Lock()

        def start(self):
            return None

        def shutdown(self):
            return None

    multiprocessing.Manager = lambda *a, **k: _FakeManager()  # type: ignore[assignment]

    if "vllm" not in sys.modules:
        vllm = types.ModuleType("vllm")
        vllm.LLM = type("LLM", (), {})
        vllm.SamplingParams = type("SamplingParams", (), {})
        sys.modules["vllm"] = vllm
        lora = types.ModuleType("vllm.lora.request")
        lora.LoRARequest = type("LoRARequest", (), {})
        sys.modules["vllm.lora.request"] = lora

    os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY") or "sk-local")


class HippoRAGRunner:
    """Wrap HippoRAG against Ollama's OpenAI-compatible API."""

    def __init__(self, config: BenchmarkConfig, ledger: TokenLedger):
        self.config = config
        self.ledger = ledger
        self.save_dir = Path(config.hipporag_workspace)
        self._hippo: Any = None

    def build_index(self) -> None:
        documents = load_documents(self.config.corpus_dir, self.config.max_documents)
        if not documents:
            raise ValueError(f"No documents in {self.config.corpus_dir}")
        docs = [doc.text for doc in documents]

        self.save_dir.mkdir(parents=True, exist_ok=True)
        hippo = self._get_client()

        # HippoRAG skips already-indexed content when save_dir is reused.
        if self.config.reuse_indexes and self._index_ready():
            return

        hippo.index(docs=docs)
        for text in docs:
            self.ledger.record(
                phase="hipporag_index_estimate",
                model=self.config.chat_model,
                text=text,
                role="prompt",
            )

    def query(self, question: str) -> HippoRAGResult:
        hippo = self._get_client()
        raw = hippo.rag_qa(queries=[question])
        answer = _extract_answer(raw)
        self.ledger.record(
            phase="hipporag_query_estimate",
            model=self.config.chat_model,
            text=question,
            role="prompt",
        )
        self.ledger.record(
            phase="hipporag_query_estimate",
            model=self.config.chat_model,
            text=answer,
            role="completion",
        )
        return HippoRAGResult(answer=answer)

    def _index_ready(self) -> bool:
        # HippoRAG writes under save_dir/<dataset>_<llm>_<embed>/
        if not self.save_dir.exists():
            return False
        return any(self.save_dir.rglob("*.json")) or any(self.save_dir.rglob("*.parquet"))

    def _get_client(self) -> Any:
        if self._hippo is not None:
            return self._hippo

        try:
            _prepare_hipporag_import()
            from hipporag import HippoRAG
            import hipporag.HippoRAG as hr_mod
            import hipporag.embedding_model as em
            from hipporag.embedding_model.OpenAI import OpenAIEmbeddingModel

            def _select(name: str = "nvidia/NV-Embed-v2"):
                n = name or ""
                if "GritLM" in n:
                    return em.GritLMEmbeddingModel
                if "NV-Embed-v2" in n:
                    return em.NVEmbedV2EmbeddingModel
                if "contriever" in n.lower():
                    return em.ContrieverModel
                return OpenAIEmbeddingModel

            em._get_embedding_model_class = _select
            hr_mod._get_embedding_model_class = _select
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "HippoRAG is not available. Install with:\n"
                "  pip install 'hipporag==2.0.0a3' --no-deps\n"
                "  pip install igraph gritlm networkx\n"
                f"Original error: {exc}"
            ) from exc

        base = self.config.ollama_base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"

        self._hippo = HippoRAG(
            save_dir=str(self.save_dir),
            llm_model_name=self.config.chat_model,
            llm_base_url=base,
            embedding_model_name=self.config.graphrag_embedding_model,
            embedding_base_url=base,
        )
        return self._hippo


def _extract_answer(raw: Any) -> str:
    """Normalize HippoRAG rag_qa return shapes across versions."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict):
        for key in ("answer", "response", "text"):
            if key in raw and isinstance(raw[key], str):
                return raw[key].strip()
    if isinstance(raw, (list, tuple)) and raw:
        first = raw[0]
        if isinstance(first, str):
            return first.strip()
        if isinstance(first, dict):
            return _extract_answer(first)
        if isinstance(first, (list, tuple)) and first:
            return _extract_answer(first)
        # Sometimes (answers, metadata)
        if hasattr(first, "answer"):
            return str(first.answer).strip()
    if hasattr(raw, "answer"):
        return str(raw.answer).strip()
    return str(raw).strip()
