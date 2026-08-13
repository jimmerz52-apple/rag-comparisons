#!/usr/bin/env python3
"""Run HotpotQA distractor benchmark — default = FULL validation (7405 Q).

GraphRAG on ~66k docs is not practical on local 3B; default methods are vector-only.
Pass graph methods explicitly if you have the budget.

Usage:
  python scripts/run_hotpot_benchmark.py              # full val, semantic+rerank
  python scripts/run_hotpot_benchmark.py all semantic_rag
  python scripts/run_hotpot_benchmark.py 100 semantic_rag,lazygraph_rag
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from rag_benchmark import BenchmarkConfig, BenchmarkRunner, build_hotpot_subset, create_tracked_client
from rag_benchmark.charts import plot_dashboard, print_leaderboard
from rag_benchmark.hotpotqa import parse_n_questions


# Full Hotpot: vector methods only by default (GraphRAG needs separate large-scale infra)
DEFAULT_METHODS_FULL = ["semantic_rag", "rerank_semantic", "hybrid_dense_sparse"]
DEFAULT_METHODS_SMALL = [
    "semantic_rag",
    "rerank_semantic",
    "lazygraph_rag",
    "hybrid_rag",
]


def main() -> None:
    n_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    n = parse_n_questions(n_arg)
    label = "all" if n is None else str(n)

    if len(sys.argv) > 2:
        methods = [m.strip() for m in sys.argv[2].split(",") if m.strip()]
    else:
        methods = list(DEFAULT_METHODS_FULL if n is None or n >= 500 else DEFAULT_METHODS_SMALL)

    print(f"Building HotpotQA distractor subset (n={label})...")
    built = build_hotpot_subset(project_root=ROOT, n_questions=n)
    print(built["meta"])

    config = BenchmarkConfig.from_yaml(ROOT)
    config.project_root = ROOT
    config.corpus_dir = built["corpus_dir"]
    config.qa_path = built["qa_path"]
    config.semantic_collection = "hotpot_semantic_full" if n is None else "hotpot_semantic"
    config.graph_workspace = ROOT / "graphrag_workspaces" / "hotpot"
    config.lazy_workspace = ROOT / "graphrag_workspaces" / "hotpot_lazy"
    config.lightrag_workspace = ROOT / "lightrag_workspaces" / "hotpot"
    config.hipporag_workspace = ROOT / "hipporag_workspaces" / "hotpot"
    config.max_documents = 100_000
    # Full corpus already indexed under hotpot_semantic_full (~66k vectors).
    # Never wipe/rebuild by default — that hung Chroma for 18+ minutes.
    config.reuse_indexes = True
    config.graph_indexing_method = "fast"
    config.semantic_top_k = 8

    n_docs = len(list(config.corpus_dir.glob("*.txt")))
    print(f"Docs={n_docs} methods={methods} reuse_indexes={config.reuse_indexes}")
    if n is None or (n is not None and n >= 500):
        print(
            "FULL/LARGE Hotpot: vector methods + REUSE index. "
            "Progress checkpoints every ~25–100 Q → results/accuracy_results.csv"
        )

    runner = BenchmarkRunner(config, create_tracked_client(config))
    results = runner.run_all(methods=methods)
    saved = runner.save_results(results)
    out = config.results_dir()
    plot_dashboard(out)
    print_leaderboard(out)
    print(f"\nCharts → {out / 'benchmark_dashboard.png'}")
    print(f"Summary → {saved['summary_csv']}")


if __name__ == "__main__":
    main()
