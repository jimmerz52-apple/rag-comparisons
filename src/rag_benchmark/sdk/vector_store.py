"""Vector store that always keeps SourceRef metadata beside each embedding."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb

from rag_benchmark.corpus import Document, TextChunk, chunk_documents
from rag_benchmark.sdk.embedder import Embedder, slug_embedder_id
from rag_benchmark.sdk.source_ref import SourceRef, sha256_text, source_uri_for


@dataclass
class SyncReport:
    """What changed during an incremental corpus sync."""

    embedder_id: str
    collection: str
    unchanged_docs: int = 0
    reindexed_docs: int = 0
    deleted_docs: int = 0
    upserted_chunks: int = 0
    changed_source_ids: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"[{self.embedder_id}] collection={self.collection} "
            f"unchanged={self.unchanged_docs} reindexed={self.reindexed_docs} "
            f"deleted={self.deleted_docs} chunks_upserted={self.upserted_chunks}"
        )


class LineageVectorStore:
    """Chroma-backed index: one collection per embedder_id, SourceRef on every row.

    Incremental updates: hash document text → if unchanged, skip; else delete all
    chunks for that source_id and re-embed. No full-corpus revectorize required
    for small content edits (same embedder + same chunking).
    """

    def __init__(
        self,
        *,
        root: Path,
        base_collection: str,
        embedder: Embedder,
        chunk_size: int = 600,
        chunk_overlap: int = 80,
        tenant_id: str = "default",
    ):
        self.root = Path(root)
        self.embedder = embedder
        self.embedder_id = embedder.model_id
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tenant_id = tenant_id
        # Obvious separation: collection name encodes the embedding model
        self.collection_name = f"{base_collection}__{slug_embedder_id(self.embedder_id)}"
        self.root.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.root / self.collection_name))
        self._collection = self._client.get_or_create_collection(self.collection_name)
        self._chunks: list[TextChunk] = []

    @property
    def collection(self):
        return self._collection

    def count(self) -> int:
        return int(self._collection.count())

    def reset(self) -> None:
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(self.collection_name)
        self._chunks = []

    def _doc_hashes_in_store(self) -> dict[str, str]:
        """Map source_id → content_hash from existing vectors (one sample per doc)."""
        n = self._collection.count()
        if n == 0:
            return {}
        # Chroma get without ids returns all when limit set high enough
        data = self._collection.get(include=["metadatas"], limit=max(n, 1))
        out: dict[str, str] = {}
        for meta in data.get("metadatas") or []:
            if not meta:
                continue
            sid = str(meta.get("source_id") or meta.get("doc_id") or "")
            h = str(meta.get("content_hash") or "")
            if sid and h and sid not in out:
                out[sid] = h
        return out

    def delete_source(self, source_id: str) -> int:
        """Remove all vectors for one document (enterprise delete / reindex prep)."""
        try:
            existing = self._collection.get(where={"source_id": source_id}, include=[])
            ids = existing.get("ids") or []
            if ids:
                self._collection.delete(ids=ids)
                return len(ids)
        except Exception:
            # Fallback for stores that only have legacy doc_id
            existing = self._collection.get(where={"doc_id": source_id}, include=[])
            ids = existing.get("ids") or []
            if ids:
                self._collection.delete(ids=ids)
                return len(ids)
        return 0

    def upsert_documents(self, documents: list[Document], *, phase: str = "index") -> int:
        """Chunk + embed + upsert with SourceRef. Overwrites chunks for those docs."""
        if not documents:
            return 0
        for doc in documents:
            self.delete_source(doc.doc_id)

        chunks = chunk_documents(
            documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        if not chunks:
            return 0

        doc_hash = {d.doc_id: sha256_text(d.text) for d in documents}
        doc_uri = {d.doc_id: source_uri_for(d.source_path) for d in documents}
        # Track child index per doc for SourceRef.chunk_index
        counters: dict[str, int] = {}

        batch_size = 64
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = self.embedder.embed([c.text for c in batch], phase=phase)
            ids: list[str] = []
            docs: list[str] = []
            metas: list[dict[str, Any]] = []
            for chunk in batch:
                idx = counters.get(chunk.doc_id, 0)
                counters[chunk.doc_id] = idx + 1
                ref = SourceRef(
                    source_id=chunk.doc_id,
                    source_uri=doc_uri.get(chunk.doc_id, str(chunk.source_path)),
                    chunk_id=chunk.chunk_id,
                    content_hash=doc_hash[chunk.doc_id],
                    chunk_hash=sha256_text(chunk.text),
                    embedder_id=self.embedder_id,
                    chunk_index=idx,
                    tenant_id=self.tenant_id,
                )
                ids.append(chunk.chunk_id)
                docs.append(chunk.text)
                metas.append(ref.to_metadata())
            self._collection.upsert(
                ids=ids,
                documents=docs,
                embeddings=vectors,
                metadatas=metas,
            )
        self._chunks = chunks
        return len(chunks)

    def sync_documents(
        self,
        documents: list[Document],
        *,
        drop_missing: bool = False,
        phase: str = "sync",
    ) -> SyncReport:
        """Incremental sync: only re-embed docs whose content_hash changed.

        If drop_missing=True, delete vectors for source_ids no longer in ``documents``.
        """
        report = SyncReport(
            embedder_id=self.embedder_id,
            collection=self.collection_name,
        )
        stored = self._doc_hashes_in_store()
        incoming = {d.doc_id: d for d in documents}
        to_reindex: list[Document] = []

        for doc in documents:
            new_hash = sha256_text(doc.text)
            old = stored.get(doc.doc_id)
            if old == new_hash:
                report.unchanged_docs += 1
            else:
                to_reindex.append(doc)
                report.changed_source_ids.append(doc.doc_id)

        if to_reindex:
            report.upserted_chunks = self.upsert_documents(to_reindex, phase=phase)
            report.reindexed_docs = len(to_reindex)

        if drop_missing:
            stale = set(stored) - set(incoming)
            for sid in stale:
                n = self.delete_source(sid)
                if n:
                    report.deleted_docs += 1
                    report.changed_source_ids.append(sid)

        return report

    def query(
        self,
        question: str,
        *,
        top_k: int,
        phase: str = "query",
    ) -> list[tuple[str, SourceRef, float | None]]:
        """Return (text, SourceRef, distance) for top_k hits."""
        if self._collection.count() == 0:
            return []
        q = self.embedder.embed([question], phase=phase)[0]
        n = min(top_k, self._collection.count())
        res = self._collection.query(
            query_embeddings=[q],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        ids = (res.get("ids") or [[]])[0]
        out: list[tuple[str, SourceRef, float | None]] = []
        for i, text in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            if meta is not None and "chunk_id" not in meta and i < len(ids):
                meta = {**(meta or {}), "chunk_id": ids[i]}
            ref = SourceRef.from_metadata(meta, fallback_id=ids[i] if i < len(ids) else "")
            dist = float(dists[i]) if i < len(dists) and dists[i] is not None else None
            out.append((text, ref, dist))
        return out
