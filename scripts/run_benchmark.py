#!/usr/bin/env python3
"""Execute the RAG benchmark end-to-end (same logic as the notebook)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

load_dotenv(PROJECT_ROOT / ".env")

from rag_benchmark import BenchmarkConfig, BenchmarkRunner, create_tracked_client, fetch_corpus

METHOD_LABELS = {
    "semantic_rag": "Semantic Search",
    "graph_rag": "GraphRAG (global)",
    "hybrid_rag": "Hybrid",
    "lazygraph_rag": "LazyGraphRAG",
}

WIKI_TITLES = [
    "Albert Einstein",
    "Python (programming language)",
    "Machine learning",
    "Artificial intelligence",
    "Solar System",
    "Mars",
    "Jupiter",
    "World War II",
    "Neural network",
    "United States",
]


def main() -> None:
    config = BenchmarkConfig.from_yaml(PROJECT_ROOT)
    config.project_root = PROJECT_ROOT
    config.corpus_dir = PROJECT_ROOT / "data" / "corpus"
    config.qa_path = PROJECT_ROOT / "data" / "qa" / "eval_questions.json"

    print(f"LLM backend: {config.llm_backend}")
    print(f"Chat model: {config.chat_model}")
    print(f"Embedding model: {config.embedding_model}")

    print("Fetching Wikipedia corpus...")
    saved = fetch_corpus(WIKI_TITLES, config.corpus_dir, max_chars=10_000)
    print(f"  {len(saved)} articles ready")

    runner = BenchmarkRunner(config, create_tracked_client(config))
    print(f"Running {len(runner.questions)} eval questions across 4 RAG methods...")
    results = runner.run_all()
    saved = runner.save_results(results)

    summary_df = saved["summary_df"]
    summary_df["label"] = summary_df["method"].map(METHOD_LABELS)
    print("\n=== BENCHMARK LEADERBOARD ===")
    print(
        summary_df[
            [
                "label",
                "mean_composite_score",
                "total_tokens",
                "estimated_cost_usd",
                "index_seconds",
                "mean_query_latency_seconds",
                "total_elapsed_seconds",
            ]
        ].round(4).to_string(index=False)
    )

    out_dir = config.results_dir()
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=summary_df,
        x="label",
        y="mean_composite_score",
        hue="label",
        legend=False,
        palette="Set2",
        ax=ax,
    )
    ax.set_title("RAG Quality Comparison (Local Models)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    chart_path = out_dir / "quality_comparison.png"
    fig.savefig(chart_path, dpi=150)
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
