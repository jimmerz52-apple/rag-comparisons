#!/usr/bin/env python3
"""Embedding-model bake-off (separate from method bake-off).

Compares embedders with the *same* RAG pipeline (RagPipelineOrchestrator).
Each model gets its own lineage collection — vector spaces are never mixed.

Usage:
  PYTHONPATH=src python scripts/run_embedding_bakeoff.py
  PYTHONPATH=src python scripts/run_embedding_bakeoff.py all-MiniLM-L6-v2,BAAI/bge-small-en-v1.5
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_benchmark.benchmark import BenchmarkRunner
from rag_benchmark.config import BenchmarkConfig
from rag_benchmark.llm_factory import clone_client_for_ledger, create_tracked_client
from rag_benchmark.metrics import AccuracyEvaluator
from rag_benchmark.sdk import RagPipelineOrchestrator, TrackedClientEmbedder, slug_embedder_id
from rag_benchmark.token_tracker import TokenLedger


DEFAULT_MODELS = [
    "all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
]


def _run_one_embedder(
    config: BenchmarkConfig,
    base_client,
    model_id: str,
    out_dir: Path,
) -> dict:
    ledger = TokenLedger()
    client = clone_client_for_ledger(base_client, config, ledger)
    embedder = TrackedClientEmbedder(client, model_id)
    orch = RagPipelineOrchestrator(
        config,
        client,
        ledger,
        embedder=embedder,
        store_root=config.project_root / ".chroma" / "embedding_bakeoff",
        base_collection="embed_bakeoff",
    )

    print(f"\n>>> Embedding model: {model_id}")
    print(f"    collection: {orch.collection_name}")
    t0 = time.perf_counter()
    sync = orch.build_index(full=not config.reuse_indexes)
    index_s = time.perf_counter() - t0
    print(f"    index: {sync.summary()} ({index_s:.1f}s)")

    # Demo incremental path: sync again should be a no-op
    sync2 = orch.sync_corpus()
    print(f"    re-sync (expect unchanged): {sync2.summary()}")

    evaluator = AccuracyEvaluator(client, config.judge_model)
    questions = BenchmarkRunner(config, client).questions
    rows = []
    latencies = []
    for q in questions:
        tq = time.perf_counter()
        result = orch.query(q.question)
        latencies.append(time.perf_counter() - tq)
        acc = evaluator.evaluate(
            method=f"embed:{model_id}",
            question=q,
            prediction=result.answer,
        )
        rows.append(
            {
                "embedder": model_id,
                "collection": orch.collection_name,
                "question_id": q.id,
                "question": q.question,
                "gold_answer": q.expected_answer,
                "prediction": result.answer,
                "citations": json.dumps(result.citations),
                "llm_judge": acc.llm_judge_score,
                "token_f1": acc.token_f1,
                "exact_match": acc.exact_match,
                "contains_answer": acc.contains_answer,
                "composite_score": acc.composite_score(),
                "latency_s": latencies[-1],
            }
        )
        print(f"    {q.id}: composite={acc.composite_score():.2f}")

    df = pd.DataFrame(rows)
    safe = slug_embedder_id(model_id)
    df.to_csv(out_dir / f"answers__{safe}.csv", index=False)

    summary = {
        "embedder": model_id,
        "collection": orch.collection_name,
        "n_questions": len(rows),
        "mean_composite_score": float(df["composite_score"].mean()),
        "mean_llm_judge": float(df["llm_judge"].fillna(0).mean()),
        "mean_token_f1": float(df["token_f1"].fillna(0).mean()),
        "mean_contains_answer": float(df["contains_answer"].fillna(0).astype(float).mean()),
        "mean_latency_s": float(df["latency_s"].mean()),
        "index_seconds": index_s,
        "total_tokens": ledger.total().total_tokens,
        "resync_unchanged_docs": sync2.unchanged_docs,
        "resync_reindexed_docs": sync2.reindexed_docs,
    }
    return summary


def main() -> None:
    models_arg = sys.argv[1] if len(sys.argv) > 1 else None
    models = [m.strip() for m in models_arg.split(",")] if models_arg else DEFAULT_MODELS

    config = BenchmarkConfig.from_yaml(PROJECT_ROOT)
    # Keep bake-off small/fast; reuse wiki or whatever qa_path points at
    out_dir = PROJECT_ROOT / "results_embedding_bakeoff"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Embedding bake-off (SDK orchestrator) ===")
    print("Separate from method bake-off (BenchmarkRunner).")
    print(f"Corpus: {config.corpus_dir}")
    print(f"QA:     {config.qa_path}")
    print(f"Models: {models}")
    print(f"Out:    {out_dir}")

    base_client = create_tracked_client(config)
    summaries = []
    for model_id in models:
        summaries.append(_run_one_embedder(config, base_client, model_id, out_dir))

    summary_df = pd.DataFrame(summaries).sort_values("mean_composite_score", ascending=False)
    summary_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Embedding bake-off results",
                "",
                "This directory compares **embedding models** with a fixed RAG pipeline",
                "(`RagPipelineOrchestrator` in `rag_benchmark.sdk`).",
                "",
                "It is **not** the method bake-off (`results/`, `BenchmarkRunner`).",
                "",
                "Each embedder uses a separate Chroma collection:",
                "`embed_bakeoff__<slug(model)>` under `.chroma/embedding_bakeoff/`.",
                "",
                "Incremental sync: a second `sync_corpus()` after build should report",
                "all docs unchanged (no full revectorize).",
                "",
                "## Leaderboard",
                "",
                "```",
                summary_df.to_string(index=False),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("\n=== Embedding bake-off leaderboard ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {summary_path}")
    print(f"Notes: {readme}")


if __name__ == "__main__":
    main()
