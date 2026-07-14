"""Engineering-oriented scorecard: routing, cost, latency SLOs — not just academic averages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from rag_benchmark.charts import METHOD_LABELS

# Default engineering thresholds (teams can override)
DEFAULT_LATENCY_SLO_S = 5.0
DEFAULT_MIN_QUALITY = 0.35
# Within this quality margin, prefer the cheaper method (tok/query).
TIE_MARGIN = 0.03

# Hotpot / harness query_type → eng-facing scenario names
SCENARIO_LABELS = {
    "local": "local_factoid",
    "hybrid": "multi_hop",
    "global": "thematic",
    "bridge": "multi_hop",
    "comparison": "local_factoid",
}


def _label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _scenario_label(query_type: str) -> str:
    return SCENARIO_LABELS.get(query_type, query_type)


def build_engineering_scorecard(
    *,
    summary_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    accuracy_df: pd.DataFrame,
    latency_slo_seconds: float = DEFAULT_LATENCY_SLO_S,
    min_quality: float = DEFAULT_MIN_QUALITY,
) -> dict[str, Any]:
    """Produce a decision-ready scorecard for engineering teams."""
    summary = summary_df.copy()
    summary["label"] = summary["method"].map(_label)
    summary["quality_per_1k_tokens"] = summary["mean_composite_score"] / (
        summary["tokens_per_query"].clip(lower=1) / 1000.0
    )
    summary["meets_latency_slo"] = summary["p95_query_latency_seconds"] <= latency_slo_seconds
    summary["meets_quality_bar"] = summary["mean_composite_score"] >= min_quality

    if len(accuracy_df) and "method" in accuracy_df.columns:
        n_queries = int(accuracy_df.groupby("method").size().iloc[0])
    else:
        n_queries = 0

    # Usable answer ≈ judge OK or contains gold span (more useful than EM for generative RAG)
    usable = (
        accuracy_df.assign(
            usable=lambda d: (d["llm_judge_score"].fillna(0) >= 0.5)
            | (d["contains_answer"].fillna(False).astype(bool))
        )
        .groupby("method", as_index=False)["usable"]
        .mean()
        .rename(columns={"usable": "usable_answer_rate"})
    )
    summary = summary.merge(usable, on="method", how="left")

    tok_by_method = summary.set_index("method")["tokens_per_query"].to_dict()

    # Per-scenario winner: prefer head-to-head win rate over tiny mean gaps (n is small).
    routing: list[dict[str, Any]] = []
    if not scenario_df.empty:
        for qtype, group in scenario_df.groupby("query_type"):
            ranked = group.sort_values("mean_composite_score", ascending=False)
            top = ranked.iloc[0]
            second = ranked.iloc[1] if len(ranked) > 1 else None
            margin = (
                float(top["mean_composite_score"] - second["mean_composite_score"])
                if second is not None
                else float(top["mean_composite_score"])
            )
            pick = top
            tie_broke = False
            h2h_note = ""
            if second is not None:
                h2h = _head_to_head(
                    accuracy_df,
                    query_type=str(qtype),
                    method_a=str(top["method"]),
                    method_b=str(second["method"]),
                )
                # Mean lead can flip vs win rate on tiny n — trust H2H when mean Δ is thin.
                if margin < TIE_MARGIN and h2h["wins_a"] < h2h["wins_b"]:
                    pick = second
                    tie_broke = True
                    h2h_note = (
                        f" meanΔ={margin:.3f} but H2H {h2h['wins_a']}W-{h2h['wins_b']}L "
                        f"→ prefer head-to-head winner"
                    )
                elif margin < TIE_MARGIN:
                    top_tok = float(tok_by_method.get(str(top["method"]), 1e12))
                    sec_tok = float(tok_by_method.get(str(second["method"]), 1e12))
                    if sec_tok < top_tok and h2h["wins_a"] <= h2h["wins_b"]:
                        pick = second
                        tie_broke = True
                        h2h_note = " quality tie → cheaper tokens"
                    else:
                        h2h_note = (
                            f" H2H {h2h['wins_a']}W-{h2h['wins_b']}L vs runner-up "
                            f"(meanΔ={margin:.3f}; treat as soft)"
                        )

            scenario = _scenario_label(str(qtype))
            guidance = _routing_guidance(scenario, str(pick["method"]), tie_broke) + h2h_note
            routing.append(
                {
                    "query_type": str(qtype),
                    "scenario": scenario,
                    "recommended_method": str(pick["method"]),
                    "recommended_label": _label(str(pick["method"])),
                    "quality": float(pick["mean_composite_score"]),
                    "margin_over_next": margin,
                    "tie_broke_on_cost": tie_broke,
                    "tokens_per_query": float(tok_by_method.get(str(pick["method"]), 0)),
                    "guidance": guidance.strip(),
                }
            )

    # Default production pick: best quality among methods that meet latency SLO
    slo_ok = summary[summary["meets_latency_slo"]]
    if not slo_ok.empty:
        default_row = slo_ok.sort_values(
            ["mean_composite_score", "quality_per_1k_tokens"], ascending=False
        ).iloc[0]
        default_reason = f"Best quality among methods with p95 ≤ {latency_slo_seconds}s"
    else:
        default_row = summary.sort_values("mean_query_latency_seconds").iloc[0]
        default_reason = "No method met latency SLO; picking fastest mean latency"

    efficiency = summary.sort_values("quality_per_1k_tokens", ascending=False)[
        [
            "method",
            "label",
            "mean_composite_score",
            "usable_answer_rate",
            "tokens_per_query",
            "quality_per_1k_tokens",
            "p95_query_latency_seconds",
        ]
    ]

    scorecard = {
        "audience": "engineering",
        "latency_slo_seconds": latency_slo_seconds,
        "min_quality_bar": min_quality,
        "n_eval_queries": n_queries,
        "caveats": _caveats(n_queries, summary),
        "default_production_method": {
            "method": str(default_row["method"]),
            "label": str(default_row["label"]),
            "reason": default_reason,
            "quality": float(default_row["mean_composite_score"]),
            "usable_answer_rate": float(default_row.get("usable_answer_rate", 0) or 0),
            "p95_latency_s": float(default_row["p95_query_latency_seconds"]),
            "tokens_per_query": float(default_row["tokens_per_query"]),
        },
        "routing_by_query_type": routing,
        "methods": summary.to_dict(orient="records"),
        "efficiency_ranking": efficiency.to_dict(orient="records"),
        "recommendations": _team_recommendations(summary, routing, latency_slo_seconds),
    }
    return scorecard


def _head_to_head(
    accuracy_df: pd.DataFrame,
    *,
    query_type: str,
    method_a: str,
    method_b: str,
) -> dict[str, int]:
    subset = accuracy_df[accuracy_df["query_type"] == query_type]
    if subset.empty or "composite_score" not in subset.columns:
        return {"wins_a": 0, "wins_b": 0, "ties": 0}
    pivot = subset.pivot_table(
        index="question_id", columns="method", values="composite_score", aggfunc="mean"
    )
    if method_a not in pivot.columns or method_b not in pivot.columns:
        return {"wins_a": 0, "wins_b": 0, "ties": 0}
    delta = pivot[method_a] - pivot[method_b]
    return {
        "wins_a": int((delta > 0).sum()),
        "wins_b": int((delta < 0).sum()),
        "ties": int((delta == 0).sum()),
    }


def _caveats(n_queries: int, summary: pd.DataFrame) -> list[str]:
    notes = [
        f"Eval set size n={n_queries} — treat rankings as directional, not production SLAs.",
        "Overlapping quality CIs are common at this n; trust large gaps (latency/tokens/local EM) over 0.01 composite deltas.",
        "Composite mixes EM/F1/judge; generative graph answers often lose on EM even when useful.",
        "Absolute latency depends on hardware + model size; use relative ordering for stack choice.",
    ]
    if summary["index_seconds"].max() > 60:
        notes.append(
            "Graph index build dominates cold-start cost; measure amortized $/query at your QPS."
        )
    return notes


def _routing_guidance(scenario: str, method: str, tie_broke: bool) -> str:
    label = _label(method)
    suffix = " (quality tie → cheaper tokens)" if tie_broke else ""
    if scenario == "local_factoid":
        return f"Single-entity / comparison factoids → {label}{suffix}."
    if scenario == "thematic":
        return f"Corpus-wide themes / overviews → {label}{suffix}."
    if scenario == "multi_hop":
        return f"Bridge / cross-doc multi-hop → {label}{suffix}."
    return f"Route {scenario} queries to {label}{suffix}."


def _team_recommendations(
    summary: pd.DataFrame,
    routing: list[dict[str, Any]],
    latency_slo: float,
) -> list[str]:
    tips: list[str] = []
    tips.append(
        "Ship a query router, not one RAG stack: local factoids vs multi-hop need different paths."
    )
    for route in routing:
        tie = " [cost tie-break]" if route.get("tie_broke_on_cost") else ""
        tips.append(
            f"{route['scenario'].upper()}: {route['recommended_label']} "
            f"(q={route['quality']:.2f}, Δ={route['margin_over_next']:.2f}, "
            f"{route['tokens_per_query']:.0f} tok/q){tie}."
        )

    # Highlight usable vs composite disagreement (common with verbose graph answers)
    if "usable_answer_rate" in summary.columns:
        by_usable = summary.sort_values("usable_answer_rate", ascending=False).iloc[0]
        by_comp = summary.sort_values("mean_composite_score", ascending=False).iloc[0]
        if by_usable["method"] != by_comp["method"]:
            tips.append(
                f"Usable-answer leader ({by_usable['label']}, "
                f"{by_usable['usable_answer_rate']:.0%}) differs from composite leader "
                f"({by_comp['label']}) — pick the metric that matches your UX."
            )

    expensive = summary.sort_values("tokens_per_query", ascending=False).iloc[0]
    tips.append(
        f"Highest token cost: {expensive['label']} "
        f"({expensive['tokens_per_query']:.0f} tok/q) — gate behind hard-query classifier."
    )

    slow = summary[summary["p95_query_latency_seconds"] > latency_slo]
    if not slow.empty:
        names = ", ".join(slow["label"].astype(str))
        tips.append(
            f"Misses p95≤{latency_slo}s SLO: {names}. Keep interactive path on vector; "
            "run graph/hybrid async or on stronger GPUs."
        )

    tips.append(
        "Rebuild GraphRAG indexes only on corpus change; amortize index_seconds over expected volume."
    )
    return tips


def save_engineering_scorecard(
    scorecard: dict[str, Any],
    out_dir: Path,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "engineering_scorecard.json"
    json_path.write_text(json.dumps(scorecard, indent=2, default=str), encoding="utf-8")

    methods = pd.DataFrame(scorecard["methods"])
    csv_path = out_dir / "engineering_scorecard.csv"
    keep = [
        c
        for c in [
            "label",
            "method",
            "mean_composite_score",
            "usable_answer_rate",
            "tokens_per_query",
            "quality_per_1k_tokens",
            "mean_query_latency_seconds",
            "p95_query_latency_seconds",
            "meets_latency_slo",
            "meets_quality_bar",
            "index_seconds",
        ]
        if c in methods.columns
    ]
    methods[keep].to_csv(csv_path, index=False)

    routing = pd.DataFrame(scorecard["routing_by_query_type"])
    routing_path = out_dir / "routing_recommendations.csv"
    if not routing.empty:
        routing.to_csv(routing_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.set_theme(style="whitegrid", context="notebook")

    if not methods.empty:
        sns.scatterplot(
            data=methods,
            x="tokens_per_query",
            y="mean_composite_score",
            hue="label",
            size="p95_query_latency_seconds",
            sizes=(80, 280),
            ax=axes[0],
        )
        axes[0].axhline(scorecard["min_quality_bar"], ls="--", color="gray", lw=1)
        axes[0].set_title("Quality vs tokens (size = p95 latency)")
        axes[0].set_xlabel("Tokens / query")
        axes[0].set_ylabel("Composite quality")
        axes[0].legend(fontsize=7, loc="best")

    if not routing.empty:
        plot_df = routing.copy()
        plot_df["scenario"] = plot_df["scenario"].astype(str)
        sns.barplot(
            data=plot_df,
            x="scenario",
            y="quality",
            hue="recommended_label",
            ax=axes[1],
            dodge=False,
        )
        axes[1].set_title("Recommended method by scenario")
        axes[1].set_ylim(0, 1.05)
        axes[1].legend(fontsize=7)

    fig.tight_layout()
    chart_path = out_dir / "engineering_decision.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    default = scorecard["default_production_method"]
    lines = [
        "# Engineering scorecard",
        "",
        f"Default interactive path: **{default['label']}**",
        f"- Reason: {default['reason']}",
        f"- Quality: {default['quality']:.3f} | usable: {default.get('usable_answer_rate', 0):.0%}",
        f"- p95 latency: {default['p95_latency_s']:.2f}s | tokens/query: {default['tokens_per_query']:.0f}",
        "",
        "## Routing (by scenario)",
    ]
    for route in scorecard["routing_by_query_type"]:
        lines.append(
            f"- **{route['scenario']}** → {route['recommended_label']} "
            f"(q={route['quality']:.3f}, {route['tokens_per_query']:.0f} tok/q) — {route['guidance']}"
        )
    lines.append("")
    lines.append("## Recommendations")
    for tip in scorecard["recommendations"]:
        lines.append(f"- {tip}")
    lines.append("")
    lines.append("## Caveats")
    for note in scorecard.get("caveats", []):
        lines.append(f"- {note}")
    md_path = out_dir / "engineering_briefing.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "json": json_path,
        "csv": csv_path,
        "routing_csv": routing_path,
        "chart": chart_path,
        "briefing": md_path,
    }


def print_engineering_briefing(scorecard: dict[str, Any]) -> None:
    default = scorecard["default_production_method"]
    print("\n=== ENGINEERING SCORECARD ===")
    print(f"Default interactive path: {default['label']}")
    print(f"  {default['reason']}")
    print(
        f"  quality={default['quality']:.3f}  usable={default.get('usable_answer_rate', 0):.0%}  "
        f"p95={default['p95_latency_s']:.2f}s  tok/q={default['tokens_per_query']:.0f}"
    )
    print("\nRouting:")
    for route in scorecard["routing_by_query_type"]:
        print(
            f"  [{route['scenario']}] → {route['recommended_label']} "
            f"(q={route['quality']:.3f}, {route['tokens_per_query']:.0f} tok/q)"
        )
    print("\nRecommendations:")
    for tip in scorecard["recommendations"]:
        print(f"  • {tip}")
    if scorecard.get("caveats"):
        print("\nCaveats:")
        for note in scorecard["caveats"]:
            print(f"  • {note}")
