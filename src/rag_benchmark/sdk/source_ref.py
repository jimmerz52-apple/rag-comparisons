"""Provenance attached to every stored vector (enterprise lineage)."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceRef:
    """Stable pointer from a vectorized chunk back to its origin.

    Enterprise RAG needs this for citations, ACL, deletes, and incremental sync.
    The embedding itself is never enough — keep source alongside the vector.
    """

    source_id: str  # usually Document.doc_id
    source_uri: str  # path / URL / object key
    chunk_id: str
    content_hash: str  # sha256 of *document* text (sync grain)
    chunk_hash: str  # sha256 of chunk text
    embedder_id: str
    chunk_index: int = 0
    tenant_id: str = "default"

    def to_metadata(self) -> dict[str, Any]:
        """Chroma-safe flat metadata (str/int/float/bool only)."""
        return {
            "source_id": self.source_id,
            "source_uri": self.source_uri,
            "chunk_id": self.chunk_id,
            "content_hash": self.content_hash,
            "chunk_hash": self.chunk_hash,
            "embedder_id": self.embedder_id,
            "chunk_index": int(self.chunk_index),
            "tenant_id": self.tenant_id,
            # aliases used by older SemanticRAG paths
            "doc_id": self.source_id,
            "source_path": self.source_uri,
        }

    @classmethod
    def from_metadata(cls, meta: dict[str, Any] | None, *, fallback_id: str = "") -> "SourceRef":
        m = meta or {}
        return cls(
            source_id=str(m.get("source_id") or m.get("doc_id") or fallback_id),
            source_uri=str(m.get("source_uri") or m.get("source_path") or ""),
            chunk_id=str(m.get("chunk_id") or fallback_id),
            content_hash=str(m.get("content_hash") or ""),
            chunk_hash=str(m.get("chunk_hash") or ""),
            embedder_id=str(m.get("embedder_id") or ""),
            chunk_index=int(m.get("chunk_index") or 0),
            tenant_id=str(m.get("tenant_id") or "default"),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_uri_for(path: Path | str) -> str:
    return str(path)
