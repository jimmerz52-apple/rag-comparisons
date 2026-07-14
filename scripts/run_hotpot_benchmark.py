#!/usr/bin/env python3
"""Run HotpotQA distractor benchmark (Yang et al., EMNLP 2018) — closed corpus, no full-wiki index."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from rag_benchmark import BenchmarkConfig, BenchmarkRunner, build_hotpot_subset, create_tracked_client
from rag_benchmark.charts import plot_dashboard, print_leaderboard


METHODS = [
    "semantic_rag",
    "graph_rag",
    "graph_local_rag",
    "hybrid_rag",
    "lazygraph_rag",
    # light_rag: add explicitly when ready — indexing uses LLM entity extract (slow on 3B)
]


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    print(f"Building HotpotQA distractor subset (n={n})...")
    built = build_hotpot_subset(project_root=ROOT, n_questions=n)
    print(built["meta"])

    config = BenchmarkConfig.from_yaml(ROOT)
    config.project_root = ROOT
    config.corpus_dir = built["corpus_dir"]
    config.qa_path = built["qa_path"]
    config.semantic_collection = "hotpot_semantic"
    config.graph_workspace = ROOT / "graphrag_workspaces" / "hotpot"
    config.lazy_workspace = ROOT / "graphrag_workspaces" / "hotpot_lazy"
    config.lightrag_workspace = ROOT / "lightrag_workspaces" / "hotpot"
    config.hipporag_workspace = ROOT / "hipporag_workspaces" / "hotpot"
    config.max_documents = 10_000
    config.reuse_indexes = True
    config.graph_indexing_method = "fast"
    config.semantic_top_k = 8

    print(f"Docs={len(list(config.corpus_dir.glob('*.txt')))} methods={METHODS}")
    runner = BenchmarkRunner(config, create_tracked_client(config))
    results = runner.run_all(methods=METHODS)
    saved = runner.save_results(results)
    out = config.results_dir()
    plot_dashboard(out)
    print_leaderboard(out)
    print(f"\nCharts → {out / 'benchmark_dashboard.png'}")
    print(f"Summary → {saved['summary_csv']}")


if __name__ == "__main__":
    main()
