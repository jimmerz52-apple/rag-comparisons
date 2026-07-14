#!/usr/bin/env python3
"""Run opt-in methods (HippoRAG / LightRAG) on Hotpot and merge into results/."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from rag_benchmark import BenchmarkConfig, BenchmarkRunner, build_hotpot_subset, create_tracked_client
from rag_benchmark.charts import plot_dashboard, print_leaderboard
from rag_benchmark.engineering import (
    build_engineering_scorecard,
    print_engineering_briefing,
    save_engineering_scorecard,
)

METHODS = [m.strip() for m in (sys.argv[1].split(",") if len(sys.argv) > 1 else "hippo_rag,light_rag")]


def _merge_csv(path: Path, new_df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    if path.exists():
        old = pd.read_csv(path)
        # Drop overlapping methods so re-runs replace cleanly
        methods = set(new_df["method"].unique())
        if "method" in old.columns:
            old = old[~old["method"].isin(methods)]
        merged = pd.concat([old, new_df], ignore_index=True)
    else:
        merged = new_df
    merged.to_csv(path, index=False)
    return merged


def main() -> None:
    print(f"Building/reusing HotpotQA subset; methods={METHODS}")
    built = build_hotpot_subset(project_root=ROOT, n_questions=12)
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

    n_docs = len(list(config.corpus_dir.glob("*.txt")))
    print(f"Docs={n_docs} — HippoRAG/LightRAG indexing uses LLM OpenIE (slow on 3B)")

    runner = BenchmarkRunner(config, create_tracked_client(config))
    results = runner.run_all(methods=METHODS)

    out = config.results_dir()
    accuracy_df = BenchmarkRunner.to_accuracy_frame(results)
    token_df = BenchmarkRunner.to_token_frame(results, config)
    latency_df = BenchmarkRunner.to_latency_frame(results)
    summary_df = BenchmarkRunner.to_summary_frame(results, config)
    scenario_df = BenchmarkRunner.to_scenario_frame(results)

    accuracy_df = _merge_csv(out / "accuracy_results.csv", accuracy_df, ["method", "question_id"])
    token_df = _merge_csv(out / "token_results.csv", token_df, ["method", "phase"])
    latency_df = _merge_csv(out / "latency_results.csv", latency_df, ["method", "question_index"])
    summary_df = _merge_csv(out / "summary.csv", summary_df, ["method"])
    scenario_df = _merge_csv(out / "scenario_results.csv", scenario_df, ["method", "query_type"])

    plot_dashboard(out)
    print_leaderboard(out)
    scorecard = build_engineering_scorecard(
        summary_df=summary_df,
        scenario_df=scenario_df,
        accuracy_df=accuracy_df,
    )
    paths = save_engineering_scorecard(scorecard, out)
    print_engineering_briefing(scorecard)

    from rag_benchmark.decision_playbook import build_decision_artifacts

    decision_paths = build_decision_artifacts(
        results_dir=out,
        qa_path=config.qa_path,
    )
    print(f"\nMerged charts → {out / 'benchmark_dashboard.png'}")
    print(f"Engineering → {paths['briefing']}")
    print(f"Decision playbook → {decision_paths['cheatsheet']}")


if __name__ == "__main__":
    main()
