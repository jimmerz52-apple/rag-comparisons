#!/usr/bin/env python3
"""Run methods on a GraphRAG-Bench Novel subset → results_graphrag_bench/."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from rag_benchmark import BenchmarkConfig, BenchmarkRunner, create_tracked_client
from rag_benchmark.charts import plot_dashboard, print_leaderboard
from rag_benchmark.decision_playbook import build_decision_artifacts
from rag_benchmark.engineering import (
    build_engineering_scorecard,
    print_engineering_briefing,
    save_engineering_scorecard,
)
from rag_benchmark.graphrag_bench import build_graphrag_bench_subset

METHODS = [
    m.strip()
    for m in (
        sys.argv[1].split(",")
        if len(sys.argv) > 1
        else "semantic_rag,rerank_semantic,hybrid_rag,frontier_rag"
    )
]


def main() -> None:
    n_per = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    source = sys.argv[3] if len(sys.argv) > 3 else "Novel-4128"
    print(
        f"Building GraphRAG-Bench Novel subset; n_per_type={n_per}; "
        f"source={source}; methods={METHODS}"
    )
    built = build_graphrag_bench_subset(
        project_root=ROOT, n_per_type=n_per, source=source
    )
    print(built["meta"])

    out = ROOT / "results_graphrag_bench"
    out.mkdir(parents=True, exist_ok=True)

    config = BenchmarkConfig.from_yaml(ROOT)
    config.project_root = ROOT
    config.corpus_dir = built["corpus_dir"]
    config.qa_path = built["qa_path"]
    config.semantic_collection = "graphrag_bench_semantic"
    config.graph_workspace = ROOT / "graphrag_workspaces" / "graphrag_bench"
    config.lazy_workspace = ROOT / "graphrag_workspaces" / "graphrag_bench_lazy"
    config.max_documents = 10_000
    config.reuse_indexes = True
    config.graph_indexing_method = "fast"
    config.semantic_top_k = 8

    # Redirect harness outputs away from Hotpot results/
    config.results_dir = lambda: out  # type: ignore[method-assign]

    runner = BenchmarkRunner(config, create_tracked_client(config))
    results = runner.run_all(methods=METHODS)

    accuracy_df = BenchmarkRunner.to_accuracy_frame(results)
    token_df = BenchmarkRunner.to_token_frame(results, config)
    latency_df = BenchmarkRunner.to_latency_frame(results)
    summary_df = BenchmarkRunner.to_summary_frame(results, config)
    scenario_df = BenchmarkRunner.to_scenario_frame(results)

    # Attach GraphRAG-Bench question_type for demos
    import json

    import pandas as pd

    qa = {q["id"]: q for q in json.loads(Path(built["qa_path"]).read_text(encoding="utf-8"))}
    accuracy_df["graphrag_bench_type"] = accuracy_df["question_id"].map(
        lambda i: qa.get(i, {}).get("graphrag_bench_type", "")
    )
    accuracy_df.to_csv(out / "accuracy_results.csv", index=False)
    token_df.to_csv(out / "token_results.csv", index=False)
    latency_df.to_csv(out / "latency_results.csv", index=False)
    summary_df.to_csv(out / "summary.csv", index=False)
    scenario_df.to_csv(out / "scenario_results.csv", index=False)

    # Per task-type leaderboard
    by_type = (
        accuracy_df.groupby(["graphrag_bench_type", "method"])["composite_score"]
        .mean()
        .reset_index()
        .sort_values(["graphrag_bench_type", "composite_score"], ascending=[True, False])
    )
    by_type.to_csv(out / "by_question_type.csv", index=False)

    plot_dashboard(out)
    print_leaderboard(out)
    scorecard = build_engineering_scorecard(
        summary_df=summary_df,
        scenario_df=scenario_df,
        accuracy_df=accuracy_df,
    )
    paths = save_engineering_scorecard(scorecard, out)
    print_engineering_briefing(scorecard)
    decision_paths = build_decision_artifacts(results_dir=out, qa_path=Path(built["qa_path"]))
    print(f"\nResults → {out}")
    print(f"By task type → {out / 'by_question_type.csv'}")
    print(f"Engineering → {paths['briefing']}")
    print(f"Decision → {decision_paths['cheatsheet']}")
    print(by_type.to_string(index=False))


if __name__ == "__main__":
    main()
