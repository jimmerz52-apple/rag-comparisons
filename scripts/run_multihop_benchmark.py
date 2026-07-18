#!/usr/bin/env python3
"""Run methods on MultiHop-RAG mini subset → results_multihop/."""

from __future__ import annotations

import json
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
from rag_benchmark.metric_autopsy import write_autopsy_artifacts
from rag_benchmark.multihop_rag import build_multihop_rag_subset

METHODS = [
    m.strip()
    for m in (
        sys.argv[1].split(",")
        if len(sys.argv) > 1
        else "semantic_rag,rerank_semantic,hybrid_rag,frontier_rag,lazygraph_rag"
    )
]


def main() -> None:
    n_per = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    print(f"Building MultiHop-RAG subset; n_per_type={n_per}; methods={METHODS}")
    built = build_multihop_rag_subset(project_root=ROOT, n_per_type=n_per)
    print(built["meta"])

    out = ROOT / "results_multihop"
    out.mkdir(parents=True, exist_ok=True)

    config = BenchmarkConfig.from_yaml(ROOT)
    config.project_root = ROOT
    config.corpus_dir = built["corpus_dir"]
    config.qa_path = built["qa_path"]
    config.semantic_collection = "multihop_semantic"
    config.graph_workspace = ROOT / "graphrag_workspaces" / "multihop"
    config.lazy_workspace = ROOT / "graphrag_workspaces" / "multihop_lazy"
    config.max_documents = 10_000
    config.reuse_indexes = True
    config.graph_indexing_method = "fast"
    config.semantic_top_k = 8
    config.results_dir = lambda: out  # type: ignore[method-assign]

    runner = BenchmarkRunner(config, create_tracked_client(config))
    results = runner.run_all(methods=METHODS)

    accuracy_df = BenchmarkRunner.to_accuracy_frame(results)
    qa = {q["id"]: q for q in json.loads(Path(built["qa_path"]).read_text(encoding="utf-8"))}
    accuracy_df["multihop_type"] = accuracy_df["question_id"].map(
        lambda i: qa.get(i, {}).get("multihop_type", "")
    )

    summary_df = BenchmarkRunner.to_summary_frame(results, config)
    scenario_df = BenchmarkRunner.to_scenario_frame(results)
    token_df = BenchmarkRunner.to_token_frame(results, config)
    latency_df = BenchmarkRunner.to_latency_frame(results)

    accuracy_df.to_csv(out / "accuracy_results.csv", index=False)
    summary_df.to_csv(out / "summary.csv", index=False)
    scenario_df.to_csv(out / "scenario_results.csv", index=False)
    token_df.to_csv(out / "token_results.csv", index=False)
    latency_df.to_csv(out / "latency_results.csv", index=False)

    by_type = (
        accuracy_df.groupby(["multihop_type", "method"])["generative_score"]
        .mean()
        .reset_index()
        .sort_values(["multihop_type", "generative_score"], ascending=[True, False])
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
    build_decision_artifacts(results_dir=out, qa_path=Path(built["qa_path"]))
    autopsy = write_autopsy_artifacts(
        results_dir=out,
        qa_path=Path(built["qa_path"]),
        type_key="multihop_type",
        scenario_col="multihop_type",
    )
    print(f"\nResults → {out}")
    print(f"Autopsy → {autopsy['dual']}")
    print(by_type.to_string(index=False))


if __name__ == "__main__":
    main()
