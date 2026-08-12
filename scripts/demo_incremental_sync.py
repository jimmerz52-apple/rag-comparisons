#!/usr/bin/env python3
"""Show that a small corpus edit does NOT require full revectorize.

Uses the SDK LineageVectorStore / RagPipelineOrchestrator.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_benchmark.config import BenchmarkConfig
from rag_benchmark.corpus import Document
from rag_benchmark.llm_factory import create_tracked_client
from rag_benchmark.sdk import RagPipelineOrchestrator, TrackedClientEmbedder
from rag_benchmark.token_tracker import TokenLedger


def main() -> None:
    config = BenchmarkConfig.from_yaml(PROJECT_ROOT)
    config.reuse_indexes = True
    ledger = TokenLedger()
    client = create_tracked_client(config, ledger)
    embedder = TrackedClientEmbedder(client, config.embedding_model)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        orch = RagPipelineOrchestrator(
            config,
            client,
            ledger,
            embedder=embedder,
            store_root=root / "chroma",
            base_collection="incremental_demo",
        )

        docs_v1 = [
            Document("a", "Doc A", "Alice founded Acme in 1999.", root / "a.txt"),
            Document("b", "Doc B", "Bob works on search systems.", root / "b.txt"),
            Document("c", "Doc C", "Carol studies retrieval quality.", root / "c.txt"),
        ]
        r1 = orch.build_index(docs_v1, full=True)
        print("Initial full index:", r1.summary())

        # Tiny edit to one doc only
        docs_v2 = [
            docs_v1[0],
            Document("b", "Doc B", "Bob works on search systems and ranking.", root / "b.txt"),
            docs_v1[2],
        ]
        r2 = orch.sync_corpus(docs_v2)
        print("After editing doc b:", r2.summary())
        assert r2.reindexed_docs == 1, r2
        assert r2.unchanged_docs == 2, r2
        assert r2.changed_source_ids == ["b"], r2.changed_source_ids

        r3 = orch.sync_corpus(docs_v2)
        print("Second sync (no edits):", r3.summary())
        assert r3.reindexed_docs == 0 and r3.unchanged_docs == 3

        hits = orch.retrieve("Who works on ranking?")
        print("Retrieve sample:")
        for h in hits:
            print(f"  source_id={h.source.source_id} uri={h.source.source_uri}")
            print(f"  text={h.text[:80]!r}")

        print("\nOK — only the changed document was re-embedded.")


if __name__ == "__main__":
    main()
