"""Generate benchmark comparison charts from result CSVs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

METHOD_LABELS = {
    "semantic_rag": "Semantic (vector)",
    "graph_rag": "GraphRAG global",
    "graph_local_rag": "GraphRAG local",
    "hybrid_rag": "Hybrid (vec+graph local)",
    "drift_rag": "DRIFT (GraphRAG hybrid)",
    # NOT Microsoft LazyGraphRAG — GraphRAG fast NLP index + basic search.
    "lazygraph_rag": "GraphRAG fast/basic",
    "light_rag": "LightRAG (HKUDS)",
    "hippo_rag": "HippoRAG 2",
    "hybrid_dense_sparse": "BM25+dense (RRF)",
    "rerank_semantic": "Vector + rerank",
    "adaptive_rag": "Adaptive router",
    "frontier_rag": "FrontierRAG (adaptive+CRAG)",
}

METHOD_ORDER = list(METHOD_LABELS.keys())


def _labelize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["label"] = out["method"].map(METHOD_LABELS).fillna(out["method"])
    known = [METHOD_LABELS[m] for m in METHOD_ORDER if METHOD_LABELS[m] in set(out["label"])]
    extras = [x for x in out["label"].unique() if x not in known]
    out["label"] = pd.Categorical(out["label"], categories=known + extras, ordered=True)
    return out.sort_values("label")


def plot_dashboard(results_dir: Path) -> Path:
    """Build a multi-panel benchmark dashboard PNG from saved CSVs."""
    results_dir = Path(results_dir)
    summary = _labelize(pd.read_csv(results_dir / "summary.csv"))
    accuracy = _labelize(pd.read_csv(results_dir / "accuracy_results.csv"))
    latency = pd.read_csv(results_dir / "latency_results.csv")
    latency["label"] = latency["method"].map(METHOD_LABELS).fillna(latency["method"])

    scenario_path = results_dir / "scenario_results.csv"
    scenario = _labelize(pd.read_csv(scenario_path)) if scenario_path.exists() else None

    n_methods = summary["label"].nunique()
    palette = sns.color_palette("Set2", n_colors=max(n_methods, 4))

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("RAG Benchmark Dashboard (Local GraphRAG Suite)", fontsize=18, fontweight="bold", y=0.98)

    # 1 — Quality
    ax = axes[0, 0]
    quality = summary.melt(
        id_vars="label",
        value_vars=["mean_composite_score", "mean_llm_judge", "mean_token_f1"],
        var_name="metric",
        value_name="score",
    )
    quality["metric"] = quality["metric"].map(
        {
            "mean_composite_score": "Composite",
            "mean_llm_judge": "LLM Judge",
            "mean_token_f1": "Token F1",
        }
    )
    sns.barplot(data=quality, x="label", y="score", hue="metric", ax=ax, palette="muted")
    ax.set_title("Answer Quality")
    ax.set_xlabel("")
    ax.set_ylabel("Score (0–1)")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="", fontsize=8, loc="upper right")

    # 2 — Tokens (total + per query)
    ax = axes[0, 1]
    token_cols = ["total_tokens"]
    if "tokens_per_query" in summary.columns:
        token_cols.append("tokens_per_query")
    tokens = summary[["label"] + token_cols].melt(id_vars="label", var_name="kind", value_name="tokens")
    tokens["kind"] = tokens["kind"].map(
        {"total_tokens": "Total tokens", "tokens_per_query": "Tokens / query"}
    )
    sns.barplot(data=tokens, x="label", y="tokens", hue="kind", ax=ax, palette="Pastel1")
    ax.set_title("Token Usage")
    ax.set_xlabel("")
    ax.set_ylabel("Tokens")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="", fontsize=8)

    # 3 — Query latency
    ax = axes[1, 0]
    query_lat = latency[latency["question_index"] >= 0]
    sns.boxplot(
        data=query_lat,
        x="label",
        y="query_latency_seconds",
        ax=ax,
        hue="label",
        legend=False,
        palette=palette,
    )
    ax.set_title("Per-Query Latency")
    ax.set_xlabel("")
    ax.set_ylabel("Seconds")
    ax.tick_params(axis="x", rotation=20)

    # 4 — Scenario breakdown OR timing
    ax = axes[1, 1]
    if scenario is not None and not scenario.empty:
        sns.barplot(
            data=scenario,
            x="query_type",
            y="mean_composite_score",
            hue="label",
            ax=ax,
            palette=palette,
        )
        ax.set_title("Quality by Question Scenario")
        ax.set_xlabel("Scenario (local / global / hybrid)")
        ax.set_ylabel("Mean composite")
        ax.set_ylim(0, 1.05)
        ax.legend(title="", fontsize=7, loc="upper right")
    else:
        timing = summary[
            ["label", "index_seconds", "mean_query_latency_seconds", "total_elapsed_seconds"]
        ].melt(id_vars="label", var_name="phase", value_name="seconds")
        timing["phase"] = timing["phase"].map(
            {
                "index_seconds": "Index",
                "mean_query_latency_seconds": "Mean query",
                "total_elapsed_seconds": "Total",
            }
        )
        sns.barplot(data=timing, x="label", y="seconds", hue="phase", ax=ax)
        ax.set_title("Timing Breakdown")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(title="", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = results_dir / "benchmark_dashboard.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Heatmap
    pivot = accuracy.pivot_table(
        index="question_id",
        columns="label",
        values="composite_score",
        aggfunc="mean",
        observed=False,
    )
    fig2, ax2 = plt.subplots(figsize=(max(10, n_methods * 1.6), max(5, len(pivot) * 0.45)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGn", vmin=0, vmax=1, ax=ax2, linewidths=0.5)
    ax2.set_title("Composite Score by Question × Method")
    ax2.set_xlabel("")
    ax2.set_ylabel("Question")
    fig2.tight_layout()
    heatmap_path = results_dir / "benchmark_heatmap.png"
    fig2.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # Token efficiency: quality vs tokens
    fig3, ax3 = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=summary,
        x="total_tokens",
        y="mean_composite_score",
        hue="label",
        s=180,
        ax=ax3,
        palette=palette,
    )
    for _, row in summary.iterrows():
        ax3.annotate(str(row["label"]), (row["total_tokens"], row["mean_composite_score"]), fontsize=8)
    ax3.set_title("Quality vs Token Cost (upper-left is better)")
    ax3.set_xlabel("Total tokens")
    ax3.set_ylabel("Mean composite score")
    ax3.set_ylim(0, 1.05)
    ax3.legend(title="", fontsize=8)
    fig3.tight_layout()
    scatter_path = results_dir / "quality_vs_tokens.png"
    fig3.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig3)

    return out_path


def render_notebook_dashboard(
    saved: dict | None = None,
    results_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Render inline notebook charts; returns dataframes for tables."""
    results_dir = Path(results_dir) if results_dir else None
    if saved:
        summary = _labelize(saved["summary_df"])
        accuracy = _labelize(saved["accuracy_df"])
        latency = saved["latency_df"].copy()
        scenario = _labelize(saved["scenario_df"]) if "scenario_df" in saved else None
    elif results_dir and (results_dir / "summary.csv").exists():
        summary = _labelize(pd.read_csv(results_dir / "summary.csv"))
        accuracy = _labelize(pd.read_csv(results_dir / "accuracy_results.csv"))
        latency = pd.read_csv(results_dir / "latency_results.csv")
        scenario = (
            _labelize(pd.read_csv(results_dir / "scenario_results.csv"))
            if (results_dir / "scenario_results.csv").exists()
            else None
        )
    else:
        raise FileNotFoundError("No benchmark results found. Run the benchmark cell first.")

    latency["label"] = latency["method"].map(METHOD_LABELS).fillna(latency["method"])
    n_methods = summary["label"].nunique()
    palette = sns.color_palette("Set2", n_colors=max(n_methods, 4))
    sns.set_theme(style="whitegrid", context="notebook")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("RAG Benchmark Dashboard", fontsize=16, fontweight="bold")

    quality = summary.melt(
        id_vars="label",
        value_vars=["mean_composite_score", "mean_llm_judge", "mean_token_f1"],
        var_name="metric",
        value_name="score",
    )
    quality["metric"] = quality["metric"].map(
        {
            "mean_composite_score": "Composite",
            "mean_llm_judge": "LLM Judge",
            "mean_token_f1": "Token F1",
        }
    )
    sns.barplot(data=quality, x="label", y="score", hue="metric", ax=axes[0, 0], palette="muted")
    axes[0, 0].set_title("Answer Quality")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].tick_params(axis="x", rotation=20)
    axes[0, 0].legend(fontsize=8, loc="upper right")

    token_y = "tokens_per_query" if "tokens_per_query" in summary.columns else "total_tokens"
    sns.barplot(
        data=summary,
        x="label",
        y=token_y,
        ax=axes[0, 1],
        hue="label",
        legend=False,
        palette=palette,
    )
    axes[0, 1].set_title("Tokens per Query" if token_y == "tokens_per_query" else "Total Tokens")
    axes[0, 1].tick_params(axis="x", rotation=20)

    query_lat = latency[latency["question_index"] >= 0]
    sns.boxplot(
        data=query_lat,
        x="label",
        y="query_latency_seconds",
        ax=axes[1, 0],
        hue="label",
        legend=False,
        palette=palette,
    )
    axes[1, 0].set_title("Per-Query Latency (seconds)")
    axes[1, 0].tick_params(axis="x", rotation=20)

    if scenario is not None and not scenario.empty:
        sns.barplot(
            data=scenario,
            x="query_type",
            y="mean_composite_score",
            hue="label",
            ax=axes[1, 1],
            palette=palette,
        )
        axes[1, 1].set_title("Quality by Scenario")
        axes[1, 1].set_ylim(0, 1.05)
        axes[1, 1].legend(fontsize=7)
    else:
        sns.barplot(
            data=summary,
            x="label",
            y="mean_query_latency_seconds",
            ax=axes[1, 1],
            hue="label",
            legend=False,
            palette=palette,
        )
        axes[1, 1].set_title("Mean Query Latency")
        axes[1, 1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    plt.show()

    pivot = accuracy.pivot_table(
        index="question_id", columns="label", values="composite_score", observed=False
    )
    fig2, ax2 = plt.subplots(figsize=(max(10, n_methods * 1.5), max(4, len(pivot) * 0.4)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGn", vmin=0, vmax=1, ax=ax2)
    ax2.set_title("Composite Score: Question × Method")
    fig2.tight_layout()
    plt.show()

    if results_dir:
        plot_dashboard(results_dir)

    return {
        "summary": summary,
        "accuracy": accuracy,
        "latency": latency,
        "scenario": scenario if scenario is not None else pd.DataFrame(),
    }


def print_leaderboard(results_dir: Path) -> None:
    summary = _labelize(pd.read_csv(results_dir / "summary.csv"))
    cols = [
        c
        for c in [
            "label",
            "mean_composite_score",
            "mean_llm_judge",
            "mean_token_f1",
            "total_tokens",
            "tokens_per_query",
            "index_seconds",
            "mean_query_latency_seconds",
            "total_elapsed_seconds",
        ]
        if c in summary.columns
    ]
    print("\n=== BENCHMARK LEADERBOARD ===")
    print(summary[cols].round(3).to_string(index=False))

    scenario_path = Path(results_dir) / "scenario_results.csv"
    if scenario_path.exists():
        scenario = _labelize(pd.read_csv(scenario_path))
        print("\n=== QUALITY BY SCENARIO ===")
        pivot = scenario.pivot_table(
            index="label", columns="query_type", values="mean_composite_score", observed=False
        )
        print(pivot.round(3).to_string())
