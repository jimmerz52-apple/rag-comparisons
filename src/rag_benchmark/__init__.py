"""RAG benchmark harness + enterprise SDK.

Two layers (keep them separate):

1. **sdk** (`rag_benchmark.sdk`) — Embedder, SourceRef, LineageVectorStore,
   RagPipelineOrchestrator. Apps / embedding bake-offs use this.
2. **harness** (`BenchmarkRunner`) — method bake-off across RAG architectures
   with a fixed embedder.
"""

from rag_benchmark.benchmark import BenchmarkRunner, MethodRunResult
from rag_benchmark.config import BenchmarkConfig
from rag_benchmark.graphrag_bench import build_graphrag_bench_subset
from rag_benchmark.hotpotqa import build_hotpot_subset
from rag_benchmark.llm_factory import create_tracked_client
from rag_benchmark.metrics import AccuracyEvaluator
from rag_benchmark.multihop_rag import build_multihop_rag_subset
from rag_benchmark.sdk import (
    Embedder,
    IndexHealthReport,
    LineageVectorStore,
    RagPipelineOrchestrator,
    ReindexAction,
    SourceRef,
    SyncReport,
    TrackedClientEmbedder,
)
from rag_benchmark.token_tracker import TokenLedger
from rag_benchmark.wikipedia import fetch_corpus

__all__ = [
    "BenchmarkConfig",
    "BenchmarkRunner",
    "MethodRunResult",
    "TokenLedger",
    "create_tracked_client",
    "AccuracyEvaluator",
    "fetch_corpus",
    "build_hotpot_subset",
    "build_graphrag_bench_subset",
    "build_multihop_rag_subset",
    # SDK surface
    "Embedder",
    "TrackedClientEmbedder",
    "SourceRef",
    "LineageVectorStore",
    "SyncReport",
    "RagPipelineOrchestrator",
    "IndexHealthReport",
    "ReindexAction",
]
