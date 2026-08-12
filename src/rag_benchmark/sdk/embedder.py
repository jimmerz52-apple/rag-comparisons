"""Pluggable embedding models — isolated from retrieval / generation logic."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from rag_benchmark.llm_factory import TrackedLLMClient


def slug_embedder_id(model_id: str) -> str:
    """Filesystem / collection-safe id. Different models → different indexes."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_id.strip().lower()).strip("_")
    return slug or "embedder"


class Embedder(Protocol):
    """One embedding model = one vector space. Never mix models in one collection."""

    model_id: str

    def embed(self, texts: list[str], *, phase: str = "embed") -> list[list[float]]: ...


@dataclass
class TrackedClientEmbedder:
    """Adapter: use the harness LLM client's embed_texts (local ST or OpenAI)."""

    client: TrackedLLMClient
    model_id: str

    def embed(self, texts: list[str], *, phase: str = "embed") -> list[list[float]]:
        if not texts:
            return []
        return self.client.embed_texts(texts, model=self.model_id, phase=phase)
