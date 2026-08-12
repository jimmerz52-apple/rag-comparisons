#!/usr/bin/env python3
"""Demonstrate enterprise reindex *flagging* (audit before act).

Shows:
  CONTENT_DRIFT  → partial
  EMBEDDER_DRIFT / SCHEMA_DRIFT → full
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_benchmark.config import BenchmarkConfig
from rag_benchmark.corpus import Document
from rag_benchmark.llm_factory import create_tracked_client
from rag_benchmark.sdk import RagPipelineOrchestrator, ReindexAction, TrackedClientEmbedder
from rag_benchmark.token_tracker import TokenLedger


def main() -> None:
    config = BenchmarkConfig.from_yaml(PROJECT_ROOT)
    config.reuse_indexes = True
    ledger = TokenLedger()
    client = create_tracked_client(config, ledger)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        orch = RagPipelineOrchestrator(
            config,
            client,
            ledger,
            embedder=TrackedClientEmbedder(client, config.embedding_model),
            store_root=root / "chroma",
            base_collection="flag_demo",
        )

        docs = [
            Document("a", "A", "Policy rate is 5 percent.", root / "a.txt"),
            Document("b", "B", "KYC requires government ID.", root / "b.txt"),
        ]
        orch.build_index(docs, full=True)
        ok = orch.audit_index(docs)
        print("1) Healthy:", ok.summary())
        assert not ok.needs_reindex

        docs_edit = [
            docs[0],
            Document("b", "B", "KYC requires government ID and proof of address.", root / "b.txt"),
        ]
        drift = orch.audit_index(docs_edit)
        print("2) After content edit:", drift.summary())
        assert drift.required_action == ReindexAction.PARTIAL
        assert "b" in drift.partial_source_ids
        print("   flags:", [f.code.value for f in drift.flags])

        health, sync = orch.ensure_index(docs_edit, apply=True)
        print("3) After ensure_index:", health.summary(), "|", sync.summary() if sync else "n/a")
        assert not health.needs_reindex

        # Schema drift: change chunk size on a new orchestrator sharing store root naming
        config.chunk_size = config.chunk_size + 50
        orch2 = RagPipelineOrchestrator(
            config,
            client,
            ledger,
            embedder=TrackedClientEmbedder(client, config.embedding_model),
            store_root=root / "chroma",
            base_collection="flag_demo",
        )
        # Point auditor at same physical collection by copying registry path expectation:
        # New chunk_size changes collection schema fingerprint via auditor expected_schema.
        # But collection name does NOT include chunk size — registry detects SCHEMA_DRIFT.
        # Force registry from orch into orch2 store path if collection names match.
        schema_report = orch2.audit_index(docs_edit)
        print("4) After chunk_size change:", schema_report.summary())
        assert schema_report.required_action == ReindexAction.FULL
        print("   flags:", [f.code.value for f in schema_report.flags])

        out = root / "health_report.json"
        out.write_text(json.dumps(schema_report.to_dict(), indent=2), encoding="utf-8")
        print(f"\nWrote sample audit artifact: {out}")
        print("OK — reindex flags distinguish partial vs full.")


if __name__ == "__main__":
    main()
