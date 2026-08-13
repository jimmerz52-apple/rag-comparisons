#!/usr/bin/env python3
"""Run CodeRAG-Bench basic + open-domain (Wang et al., NAACL Findings 2025).

Default: full HumanEval+MBPP+DS-1000+ODEX (~2.1k Q) over programming-solutions
+ library-documentation (~35k docs).
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from rag_benchmark import BenchmarkConfig, BenchmarkRunner, create_tracked_client
from rag_benchmark.charts import plot_dashboard, print_leaderboard
from rag_benchmark.code_rag_bench import build_code_rag_bench_subset
from rag_benchmark.decision_playbook import build_decision_artifacts
from rag_benchmark.engineering import (
    build_engineering_scorecard,
    print_engineering_briefing,
    save_engineering_scorecard,
)
from rag_benchmark.metric_autopsy import write_autopsy_artifacts

METHODS = [
    m.strip()
    for m in (
        sys.argv[1].split(",")
        if len(sys.argv) > 1
        else "semantic_rag,rerank_semantic,hybrid_dense_sparse"
    )
]


def main() -> None:
    max_q = int(sys.argv[2]) if len(sys.argv) > 2 else None
    include_so = "--stackoverflow" in sys.argv
    print(
        f"Building CodeRAG-Bench basic+open-domain; max_questions={max_q}; "
        f"stackoverflow={include_so}; methods={METHODS}"
    )
    built = build_code_rag_bench_subset(
        project_root=ROOT,
        include_humaneval=True,
        include_mbpp=True,
        include_ds1000=True,
        include_odex=True,
        include_library_docs=True,
        include_stackoverflow=include_so,
        max_questions=max_q,
    )
    print(built["meta"])

    out = ROOT / "results_code_rag"
    out.mkdir(parents=True, exist_ok=True)

    config = BenchmarkConfig.from_yaml(ROOT)
    config.project_root = ROOT
    config.corpus_dir = built["corpus_dir"]
    config.qa_path = built["qa_path"]
    config.semantic_collection = "code_rag_semantic_full"
    config.graph_workspace = ROOT / "graphrag_workspaces" / "code_rag"
    config.lazy_workspace = ROOT / "graphrag_workspaces" / "code_rag_lazy"
    config.max_documents = 100_000
    # Large library-doc corpus — reuse once built
    config.reuse_indexes = True
    config.graph_indexing_method = "fast"
    config.semantic_top_k = 8
    config.chunk_size = 800
    config.chunk_overlap = 80
    config.results_dir = lambda: out  # type: ignore[method-assign]

    n_docs = len(list(config.corpus_dir.glob("*.txt")))
    print(f"Docs={n_docs} methods={METHODS} reuse_indexes={config.reuse_indexes}")
    runner = BenchmarkRunner(config, create_tracked_client(config))
    results = runner.run_all(methods=METHODS)
    saved = runner.save_results(results)
    plot_dashboard(out)
    print_leaderboard(out)

    accuracy = out / "accuracy_results.csv"
    if accuracy.exists():
        import pandas as pd

        write_autopsy_artifacts(
            results_dir=out,
            qa_path=Path(built["qa_path"]),
            type_key="code_rag_type",
            scenario_col="code_rag_type",
        )
        summary_df = pd.read_csv(out / "summary.csv")
        scenario_path = out / "scenario_results.csv"
        scenario_df = pd.read_csv(scenario_path) if scenario_path.exists() else summary_df
        accuracy_df = pd.read_csv(accuracy)
        scorecard = build_engineering_scorecard(summary_df, scenario_df, accuracy_df)
        save_engineering_scorecard(scorecard, out)
        print_engineering_briefing(scorecard)
        build_decision_artifacts(results_dir=out, qa_path=Path(built["qa_path"]))

    print(f"\nCodeRAG results → {out}")
    print(f"Summary → {saved.get('summary_csv', out / 'summary.csv')}")


if __name__ == "__main__":
    main()
