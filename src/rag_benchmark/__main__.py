"""CLI entry point: python -m rag_benchmark run | chart"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _setup_path() -> None:
    src = PROJECT_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def cmd_chart(args: argparse.Namespace) -> None:
    from rag_benchmark.charts import plot_dashboard, print_leaderboard
    from rag_benchmark.engineering import (
        build_engineering_scorecard,
        print_engineering_briefing,
        save_engineering_scorecard,
    )
    import pandas as pd

    results_dir = Path(args.results_dir) if args.results_dir else PROJECT_ROOT / "results"
    if not (results_dir / "summary.csv").exists():
        raise SystemExit(f"No results found at {results_dir}. Run: python -m rag_benchmark run")
    chart_path = plot_dashboard(results_dir)
    print_leaderboard(results_dir)

    summary_df = pd.read_csv(results_dir / "summary.csv")
    accuracy_df = pd.read_csv(results_dir / "accuracy_results.csv")
    scenario_df = (
        pd.read_csv(results_dir / "scenario_results.csv")
        if (results_dir / "scenario_results.csv").exists()
        else pd.DataFrame()
    )
    scorecard = build_engineering_scorecard(
        summary_df=summary_df,
        scenario_df=scenario_df,
        accuracy_df=accuracy_df,
    )
    paths = save_engineering_scorecard(scorecard, results_dir)
    print_engineering_briefing(scorecard)
    print(f"\nChart saved: {chart_path}")
    print(f"Engineering briefing: {paths['briefing']}")


def cmd_run(args: argparse.Namespace) -> None:
    from rag_benchmark import BenchmarkConfig, BenchmarkRunner, create_tracked_client, fetch_corpus
    from rag_benchmark.charts import plot_dashboard, print_leaderboard
    from rag_benchmark.engineering import print_engineering_briefing

    config = BenchmarkConfig.from_yaml(PROJECT_ROOT)
    config.project_root = PROJECT_ROOT
    config.corpus_dir = PROJECT_ROOT / "data" / "corpus"
    config.qa_path = PROJECT_ROOT / "data" / "qa" / "eval_questions.json"
    if args.no_reuse:
        config.reuse_indexes = False

    print(f"LLM backend: {config.llm_backend}")
    print(f"Chat model:  {config.chat_model}")
    print(f"Reuse indexes: {config.reuse_indexes}")

    if not args.skip_fetch:
        titles = [
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
        print("Fetching Wikipedia corpus...")
        saved = fetch_corpus(titles, config.corpus_dir, max_chars=10_000)
        print(f"  {len(saved)} articles ready")

    methods = args.methods
    method_list = methods.split(",") if methods else None
    n_methods = len(method_list) if method_list else 4
    runner = BenchmarkRunner(config, create_tracked_client(config))
    print(f"Running {len(runner.questions)} questions across {n_methods} methods...")
    results = runner.run_all(methods=method_list)
    saved = runner.save_results(results)

    out_dir = config.results_dir()
    plot_dashboard(out_dir)
    print_leaderboard(out_dir)
    if "engineering_scorecard" in saved:
        print_engineering_briefing(saved["engineering_scorecard"])
    print(f"\nCharts saved:")
    print(f"  {out_dir / 'benchmark_dashboard.png'}")
    print(f"  {out_dir / 'benchmark_heatmap.png'}")
    if "engineering_briefing" in saved:
        print(f"  {saved['engineering_briefing']}")


def cmd_frameworks(_args: argparse.Namespace) -> None:
    from rag_benchmark.frameworks import print_framework_catalog

    print("=== RAG framework catalog ===\n")
    print_framework_catalog()


def main() -> None:
    _setup_path()
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="RAG benchmark harness")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run full benchmark and generate charts")
    run_p.add_argument("--methods", help="Comma-separated methods (default: all)")
    run_p.add_argument("--skip-fetch", action="store_true", help="Skip Wikipedia download")
    run_p.add_argument("--no-reuse", action="store_true", help="Force rebuild all indexes")
    run_p.set_defaults(func=cmd_run)

    chart_p = sub.add_parser("chart", help="Regenerate charts + engineering scorecard")
    chart_p.add_argument("--results-dir", help="Path to results directory")
    chart_p.set_defaults(func=cmd_chart)

    fw_p = sub.add_parser("frameworks", help="List integrated / deferred RAG frameworks")
    fw_p.set_defaults(func=cmd_frameworks)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
