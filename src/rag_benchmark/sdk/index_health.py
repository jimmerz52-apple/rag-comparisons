"""Enterprise index-health auditor: flag *when* and *why* reindex is required.

This is the compliance-facing control plane sitting above incremental sync.

Signals (severity ascending):
  OK                 — index matches corpus + schema fingerprint
  CONTENT_DRIFT      — source text hash changed → partial re-embed those docs
  SOURCE_MISSING     — vectors exist for deleted sources (or corpus missing indexed docs)
  SCHEMA_DRIFT       — chunk_size / overlap / schema_version changed → FULL rebuild
  EMBEDDER_DRIFT     — embedder_id on vectors ≠ configured model → FULL rebuild
  STALE_INDEX        — indexed_at older than policy max_age → scheduled refresh
  FULL_REBUILD       — any hard drift that invalidates the vector space

Banking note: generative RAG often sits outside formal MRM scope under SR 26-2
carve-outs, but examiners still follow data lineage. These flags produce audit
evidence (what changed, who should act, partial vs full).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from rag_benchmark.corpus import Document
from rag_benchmark.sdk.source_ref import sha256_text
from rag_benchmark.sdk.vector_store import LineageVectorStore


SCHEMA_VERSION = "1.0.0"  # bump when SourceRef / chunk contract changes


class ReindexAction(str, Enum):
    NONE = "none"
    PARTIAL = "partial"  # delete+re-embed listed source_ids
    FULL = "full"  # wipe collection / rebuild all vectors


class FlagCode(str, Enum):
    OK = "OK"
    CONTENT_DRIFT = "CONTENT_DRIFT"
    SOURCE_MISSING_FROM_CORPUS = "SOURCE_MISSING_FROM_CORPUS"
    SOURCE_MISSING_FROM_INDEX = "SOURCE_MISSING_FROM_INDEX"
    SCHEMA_DRIFT = "SCHEMA_DRIFT"
    EMBEDDER_DRIFT = "EMBEDDER_DRIFT"
    STALE_INDEX = "STALE_INDEX"
    EMPTY_INDEX = "EMPTY_INDEX"


@dataclass(frozen=True)
class IndexSchemaFingerprint:
    """Immutable contract for a vector collection. Drift ⇒ full rebuild."""

    embedder_id: str
    chunk_size: int
    chunk_overlap: int
    schema_version: str = SCHEMA_VERSION

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return sha256_text(payload)[:16]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fingerprint"] = self.fingerprint()
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "IndexSchemaFingerprint":
        return cls(
            embedder_id=str(raw["embedder_id"]),
            chunk_size=int(raw["chunk_size"]),
            chunk_overlap=int(raw["chunk_overlap"]),
            schema_version=str(raw.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass
class ReindexFlag:
    code: FlagCode
    action: ReindexAction
    message: str
    source_ids: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "action": self.action.value,
            "message": self.message,
            "source_ids": self.source_ids,
            "details": self.details,
        }


@dataclass
class IndexHealthReport:
    """Audit artifact: machine-readable + human-readable reindex decision."""

    collection: str
    embedder_id: str
    generated_at: str
    required_action: ReindexAction
    flags: list[ReindexFlag]
    schema: dict[str, Any]
    stored_schema: dict[str, Any] | None
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def needs_reindex(self) -> bool:
        return self.required_action != ReindexAction.NONE

    @property
    def partial_source_ids(self) -> list[str]:
        ids: list[str] = []
        for f in self.flags:
            if f.action == ReindexAction.PARTIAL:
                ids.extend(f.source_ids)
        # preserve order, unique
        seen: set[str] = set()
        out: list[str] = []
        for i in ids:
            if i not in seen:
                seen.add(i)
                out.append(i)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection": self.collection,
            "embedder_id": self.embedder_id,
            "generated_at": self.generated_at,
            "needs_reindex": self.needs_reindex,
            "required_action": self.required_action.value,
            "partial_source_ids": self.partial_source_ids,
            "flags": [f.to_dict() for f in self.flags],
            "schema": self.schema,
            "stored_schema": self.stored_schema,
            "stats": self.stats,
        }

    def summary(self) -> str:
        if not self.needs_reindex:
            return f"[{self.collection}] OK — no reindex required"
        codes = ", ".join(f.code.value for f in self.flags if f.code != FlagCode.OK)
        return (
            f"[{self.collection}] needs_reindex={self.required_action.value} "
            f"flags=[{codes}] partial_docs={len(self.partial_source_ids)}"
        )


class IndexHealthAuditor:
    """Compare live corpus + configured schema against lineage store + registry."""

    def __init__(
        self,
        store: LineageVectorStore,
        *,
        registry_path: Path | None = None,
        max_age_hours: float | None = None,
    ):
        self.store = store
        self.registry_path = registry_path or (
            store.root / store.collection_name / "index_registry.json"
        )
        self.max_age_hours = max_age_hours
        self.expected_schema = IndexSchemaFingerprint(
            embedder_id=store.embedder_id,
            chunk_size=store.chunk_size,
            chunk_overlap=store.chunk_overlap,
            schema_version=SCHEMA_VERSION,
        )

    def load_registry(self) -> dict[str, Any] | None:
        if not self.registry_path.exists():
            return None
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def write_registry(self, *, extra: dict[str, Any] | None = None) -> Path:
        """Call after a successful build/sync to stamp the healthy state."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": self.expected_schema.to_dict(),
            "collection": self.store.collection_name,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "vector_count": self.store.count(),
            **(extra or {}),
        }
        self.registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self.registry_path

    def audit(self, documents: list[Document]) -> IndexHealthReport:
        flags: list[ReindexFlag] = []
        registry = self.load_registry()
        stored_schema_raw = (registry or {}).get("schema")
        stored_schema = (
            IndexSchemaFingerprint.from_dict(stored_schema_raw) if stored_schema_raw else None
        )

        # --- Hard drifts: full rebuild ---
        if self.store.count() == 0:
            flags.append(
                ReindexFlag(
                    code=FlagCode.EMPTY_INDEX,
                    action=ReindexAction.FULL,
                    message="Vector index is empty; full index required.",
                )
            )

        if stored_schema is None and self.store.count() > 0:
            # Legacy / unregistered index — treat schema as unknown soft warning via embedder check
            pass
        elif stored_schema is not None:
            if stored_schema.embedder_id != self.expected_schema.embedder_id:
                flags.append(
                    ReindexFlag(
                        code=FlagCode.EMBEDDER_DRIFT,
                        action=ReindexAction.FULL,
                        message=(
                            f"Embedder changed: indexed={stored_schema.embedder_id!r} "
                            f"configured={self.expected_schema.embedder_id!r}. "
                            "Vector spaces are incompatible — full rebuild required."
                        ),
                        details={
                            "indexed_embedder": stored_schema.embedder_id,
                            "configured_embedder": self.expected_schema.embedder_id,
                        },
                    )
                )
            if (
                stored_schema.chunk_size != self.expected_schema.chunk_size
                or stored_schema.chunk_overlap != self.expected_schema.chunk_overlap
                or stored_schema.schema_version != self.expected_schema.schema_version
            ):
                flags.append(
                    ReindexFlag(
                        code=FlagCode.SCHEMA_DRIFT,
                        action=ReindexAction.FULL,
                        message="Chunking/schema fingerprint changed — full rebuild required.",
                        details={
                            "indexed": stored_schema.to_dict(),
                            "configured": self.expected_schema.to_dict(),
                        },
                    )
                )

        # Spot-check embedder_id on stored vectors (defense in depth)
        sample_embedders = self._sample_embedder_ids()
        bad = [e for e in sample_embedders if e and e != self.store.embedder_id]
        if bad:
            flags.append(
                ReindexFlag(
                    code=FlagCode.EMBEDDER_DRIFT,
                    action=ReindexAction.FULL,
                    message="Stored vectors carry a different embedder_id than the active model.",
                    details={"sample_embedder_ids": sorted(set(sample_embedders))},
                )
            )

        # --- Soft / partial drifts ---
        stored_hashes = self.store._doc_hashes_in_store()
        incoming = {d.doc_id: sha256_text(d.text) for d in documents}

        content_changed = [
            sid for sid, h in incoming.items() if sid in stored_hashes and stored_hashes[sid] != h
        ]
        if content_changed:
            flags.append(
                ReindexFlag(
                    code=FlagCode.CONTENT_DRIFT,
                    action=ReindexAction.PARTIAL,
                    message=f"{len(content_changed)} document(s) changed since last index.",
                    source_ids=content_changed,
                )
            )

        missing_from_index = [sid for sid in incoming if sid not in stored_hashes]
        if missing_from_index and self.store.count() > 0:
            flags.append(
                ReindexFlag(
                    code=FlagCode.SOURCE_MISSING_FROM_INDEX,
                    action=ReindexAction.PARTIAL,
                    message=f"{len(missing_from_index)} corpus doc(s) not present in index.",
                    source_ids=missing_from_index,
                )
            )

        orphaned = [sid for sid in stored_hashes if sid not in incoming]
        if orphaned:
            flags.append(
                ReindexFlag(
                    code=FlagCode.SOURCE_MISSING_FROM_CORPUS,
                    action=ReindexAction.PARTIAL,
                    message=f"{len(orphaned)} indexed source(s) no longer in corpus (delete vectors).",
                    source_ids=orphaned,
                )
            )

        if self.max_age_hours is not None and registry and registry.get("indexed_at"):
            try:
                indexed_at = datetime.fromisoformat(registry["indexed_at"])
                age_h = (datetime.now(timezone.utc) - indexed_at).total_seconds() / 3600.0
                if age_h > self.max_age_hours:
                    flags.append(
                        ReindexFlag(
                            code=FlagCode.STALE_INDEX,
                            action=ReindexAction.FULL,
                            message=(
                                f"Index age {age_h:.1f}h exceeds policy max_age_hours="
                                f"{self.max_age_hours}."
                            ),
                            details={"age_hours": age_h, "indexed_at": registry["indexed_at"]},
                        )
                    )
            except ValueError:
                pass

        if not flags:
            flags.append(
                ReindexFlag(
                    code=FlagCode.OK,
                    action=ReindexAction.NONE,
                    message="Index matches corpus hashes and schema fingerprint.",
                )
            )

        required = ReindexAction.NONE
        if any(f.action == ReindexAction.FULL for f in flags):
            required = ReindexAction.FULL
        elif any(f.action == ReindexAction.PARTIAL for f in flags):
            required = ReindexAction.PARTIAL

        return IndexHealthReport(
            collection=self.store.collection_name,
            embedder_id=self.store.embedder_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            required_action=required,
            flags=flags,
            schema=self.expected_schema.to_dict(),
            stored_schema=stored_schema.to_dict() if stored_schema else None,
            stats={
                "vector_count": self.store.count(),
                "corpus_docs": len(documents),
                "indexed_docs": len(stored_hashes),
                "registry_path": str(self.registry_path),
            },
        )

    def _sample_embedder_ids(self, limit: int = 50) -> list[str]:
        n = self.store.count()
        if n == 0:
            return []
        data = self.store.collection.get(include=["metadatas"], limit=min(limit, n))
        out: list[str] = []
        for meta in data.get("metadatas") or []:
            if meta and meta.get("embedder_id"):
                out.append(str(meta["embedder_id"]))
        return out
