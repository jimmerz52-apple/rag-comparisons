"""RAG pipeline orchestrator — the SDK entrypoint for apps.

Not a method bake-off. Not an embedding bake-off. This wires:
  corpus → chunk → embed → lineage store → retrieve(+cite) → generate
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag_benchmark.config import BenchmarkConfig
from rag_benchmark.corpus import Document, load_documents
from rag_benchmark.llm_factory import TrackedLLMClient
from rag_benchmark.prompts import ANSWER_PROMPT
from rag_benchmark.sdk.embedder import Embedder, TrackedClientEmbedder
from rag_benchmark.sdk.index_health import IndexHealthAuditor, IndexHealthReport, ReindexAction
from rag_benchmark.sdk.source_ref import SourceRef
from rag_benchmark.sdk.vector_store import LineageVectorStore, SyncReport
from rag_benchmark.token_tracker import TokenLedger


@dataclass
class RetrievedHit:
    text: str
    source: SourceRef
    distance: float | None = None


@dataclass
class OrchestratorResult:
    answer: str
    hits: list[RetrievedHit]

    @property
    def retrieved_chunks(self) -> list[str]:
        return [h.text for h in self.hits]

    @property
    def citations(self) -> list[dict]:
        return [h.source.as_dict() for h in self.hits]


class RagPipelineOrchestrator:
    """Enterprise-facing vector RAG pipeline with source lineage + incremental sync.

    Separation:
    - Swap ``embedder`` to change embedding model (rebuild / parallel index).
    - Call ``sync_corpus`` for small content edits (no full revectorize).
    - Use BenchmarkRunner for comparing *methods*; use embedding bake-off for models.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        client: TrackedLLMClient,
        ledger: TokenLedger,
        *,
        embedder: Embedder | None = None,
        store_root: Path | None = None,
        base_collection: str | None = None,
    ):
        self.config = config
        self.client = client
        self.ledger = ledger
        self.embedder = embedder or TrackedClientEmbedder(client, config.embedding_model)
        root = store_root or (config.project_root / ".chroma" / "sdk_lineage")
        self.store = LineageVectorStore(
            root=root,
            base_collection=base_collection or config.semantic_collection,
            embedder=self.embedder,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        self.auditor = IndexHealthAuditor(self.store)
        self._last_health: IndexHealthReport | None = None

    @property
    def embedder_id(self) -> str:
        return self.embedder.model_id

    @property
    def collection_name(self) -> str:
        return self.store.collection_name

    def audit_index(self, documents: list[Document] | None = None) -> IndexHealthReport:
        """Flag whether reindex is required (partial vs full) with audit reasons."""
        docs = documents or load_documents(self.config.corpus_dir, self.config.max_documents)
        report = self.auditor.audit(docs)
        self._last_health = report
        return report

    def ensure_index(
        self,
        documents: list[Document] | None = None,
        *,
        apply: bool = True,
    ) -> tuple[IndexHealthReport, SyncReport | None]:
        """Audit then optionally apply the minimal remediation (partial or full).

        Enterprise control loop:
          1) flag (audit) → 2) act (sync/rebuild) → 3) stamp registry for examiners.
        """
        docs = documents or load_documents(self.config.corpus_dir, self.config.max_documents)
        report = self.audit_index(docs)
        if not apply or not report.needs_reindex:
            return report, None

        if report.required_action == ReindexAction.FULL:
            sync = self.build_index(docs, full=True)
        else:
            # Partial: hash sync + drop orphans (CONTENT_DRIFT / missing / orphaned)
            sync = self.sync_corpus(docs)

        # Re-audit after remediation (registry already stamped by build/sync)
        return self.audit_index(docs), sync

    def build_index(self, documents: list[Document] | None = None, *, full: bool = False) -> SyncReport:
        """Index corpus. full=True wipes collection; else incremental sync by content hash."""
        docs = documents or load_documents(self.config.corpus_dir, self.config.max_documents)
        if not docs:
            raise ValueError(f"No documents in {self.config.corpus_dir}")
        if full or not self.config.reuse_indexes:
            self.store.reset()
            n = self.store.upsert_documents(docs, phase="orchestrator_index")
            report = SyncReport(
                embedder_id=self.embedder_id,
                collection=self.collection_name,
                reindexed_docs=len(docs),
                upserted_chunks=n,
                changed_source_ids=[d.doc_id for d in docs],
            )
        else:
            report = self.store.sync_documents(docs, drop_missing=True, phase="orchestrator_sync")
        self.auditor.write_registry(extra={"last_sync": report.summary()})
        return report

    def sync_corpus(self, documents: list[Document] | None = None) -> SyncReport:
        """Incremental: re-embed only changed source_ids."""
        docs = documents or load_documents(self.config.corpus_dir, self.config.max_documents)
        report = self.store.sync_documents(docs, drop_missing=True, phase="orchestrator_sync")
        self.auditor.write_registry(extra={"last_sync": report.summary()})
        return report

    def retrieve(self, question: str, *, top_k: int | None = None) -> list[RetrievedHit]:
        k = top_k or self.config.semantic_top_k
        rows = self.store.query(question, top_k=k, phase="orchestrator_query")
        return [RetrievedHit(text=t, source=ref, distance=d) for t, ref, d in rows]

    def query(self, question: str) -> OrchestratorResult:
        hits = self.retrieve(question)
        context = "\n\n---\n\n".join(
            f"[source:{h.source.source_id} uri:{h.source.source_uri}]\n{h.text}" for h in hits
        )
        answer = self.client.chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": ANSWER_PROMPT.format(question=question, context=context),
                }
            ],
            model=self.config.chat_model,
            phase="orchestrator_generate",
            temperature=0.0,
        )
        return OrchestratorResult(answer=answer, hits=hits)
