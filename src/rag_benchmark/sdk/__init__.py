"""Enterprise RAG SDK layer (reusable pipeline primitives).

This package is intentionally separate from the benchmark harness:

  sdk/  —  Embedder · SourceRef · LineageVectorStore · Orchestrator · IndexHealth
           Stable APIs for apps: index, incremental sync, retrieve+cite, reindex flags

             used by                         used by
  Method bake-off                  Embedding bake-off
  (BenchmarkRunner)                (scripts/run_embedding_bakeoff.py)
  Compare RAG methods              Same pipeline, swap embedder
  Fixed embedder                   Parallel indexes per model

Rules of thumb:
- Different embedding models → different vector spaces → different collections.
- Content updates → incremental delete+re-embed by source_id (doc_id).
- Embedder / chunking schema change → full rebuild for that model's index.
- Call audit_index() / ensure_index() to flag and remediate drift for compliance.
"""

from rag_benchmark.sdk.embedder import Embedder, TrackedClientEmbedder, slug_embedder_id
from rag_benchmark.sdk.index_health import (
    FlagCode,
    IndexHealthAuditor,
    IndexHealthReport,
    IndexSchemaFingerprint,
    ReindexAction,
    ReindexFlag,
)
from rag_benchmark.sdk.orchestrator import OrchestratorResult, RagPipelineOrchestrator, RetrievedHit
from rag_benchmark.sdk.source_ref import SourceRef
from rag_benchmark.sdk.vector_store import LineageVectorStore, SyncReport

__all__ = [
    "Embedder",
    "TrackedClientEmbedder",
    "slug_embedder_id",
    "SourceRef",
    "LineageVectorStore",
    "SyncReport",
    "RagPipelineOrchestrator",
    "RetrievedHit",
    "OrchestratorResult",
    "IndexHealthAuditor",
    "IndexHealthReport",
    "IndexSchemaFingerprint",
    "ReindexFlag",
    "ReindexAction",
    "FlagCode",
]
