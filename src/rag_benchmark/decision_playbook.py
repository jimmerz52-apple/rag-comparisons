"""Data-driven choose-A-over-B playbook + BenchmarkQED-lite pairwise win rates.

Uses your Hotpot accuracy_results.csv — not hand-wavy marketing claims.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from rag_benchmark.charts import METHOD_LABELS


def load_qa(qa_path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(qa_path).read_text(encoding="utf-8"))
    return {q["id"]: q for q in payload}


def pairwise_win_rates(accuracy_df: pd.DataFrame) -> pd.DataFrame:
    """For each method pair, fraction of questions where A beats B on composite."""
    methods = sorted(accuracy_df["method"].unique())
    pivot = accuracy_df.pivot_table(
        index="question_id", columns="method", values="composite_score", aggfunc="mean"
    )
    rows = []
    for i, a in enumerate(methods):
        for b in methods[i + 1 :]:
            if a not in pivot.columns or b not in pivot.columns:
                continue
            delta = pivot[a] - pivot[b]
            rows.append(
                {
                    "method_a": a,
                    "label_a": METHOD_LABELS.get(a, a),
                    "method_b": b,
                    "label_b": METHOD_LABELS.get(b, b),
                    "wins_a": int((delta > 0).sum()),
                    "wins_b": int((delta < 0).sum()),
                    "ties": int((delta == 0).sum()),
                    "n": int(delta.notna().sum()),
                    "win_rate_a": float((delta > 0).mean()),
                    "mean_delta_a_minus_b": float(delta.mean()),
                }
            )
    return pd.DataFrame(rows).sort_values("win_rate_a", ascending=False)


def choose_over_examples(
    accuracy_df: pd.DataFrame,
    qa_by_id: dict[str, dict[str, Any]],
    *,
    min_margin: float = 0.15,
) -> pd.DataFrame:
    """Concrete questions where one method clearly beats another."""
    pivot = accuracy_df.pivot_table(
        index="question_id", columns="method", values="composite_score", aggfunc="mean"
    )
    rows: list[dict[str, Any]] = []
    methods = list(pivot.columns)
    for qid, row in pivot.iterrows():
        ranked = row.dropna().sort_values(ascending=False)
        if len(ranked) < 2:
            continue
        best_m, best_s = ranked.index[0], float(ranked.iloc[0])
        second_m, second_s = ranked.index[1], float(ranked.iloc[1])
        margin = best_s - second_s
        if margin < min_margin:
            continue
        q = qa_by_id.get(str(qid), {})
        rows.append(
            {
                "question": q.get("question", qid),
                "gold": q.get("expected_answer", ""),
                "hotpot_type": q.get("hotpot_type", q.get("query_type", "")),
                "choose": METHOD_LABELS.get(best_m, best_m),
                "choose_method": best_m,
                "choose_score": best_s,
                "over": METHOD_LABELS.get(second_m, second_m),
                "over_method": second_m,
                "over_score": second_s,
                "margin": margin,
                "rule": _rule(best_m, second_m, q),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("margin", ascending=False)


def _rule(best: str, worse: str, q: dict[str, Any]) -> str:
    qtype = q.get("hotpot_type") or q.get("query_type")
    if best == "semantic_rag" and qtype in {"comparison", "local"}:
        return "Simple comparison/factoid → prefer vector RAG (cheap + EM-friendly)."
    if best in {"hybrid_rag", "lazygraph_rag", "hippo_rag"} and qtype in {"bridge", "hybrid"}:
        return "Multi-hop bridge → prefer graph-augmented / hybrid path."
    if best == "hybrid_rag" and worse.startswith("graph"):
        return "Pure GraphRAG prose failed short-span QA; hybrid kept vector facts."
    if worse == "graph_rag":
        return "GraphRAG global is wrong tool for Hotpot short answers."
    return f"On this question, {METHOD_LABELS.get(best, best)} beat {METHOD_LABELS.get(worse, worse)}."


def routing_cheat_sheet(accuracy_df: pd.DataFrame, scenario_df: pd.DataFrame) -> str:
    lines = [
        "# When to choose which RAG (from *your* Hotpot data)",
        "",
        "Cutting-edge production pattern (Adaptive-RAG): **route by query type**, "
        "don't crown a single stack.",
        "",
    ]
    # Prefer head-to-head within each query_type (means lie on tiny n)
    if not accuracy_df.empty and "query_type" in accuracy_df.columns:
        for qtype, group in accuracy_df.groupby("query_type"):
            pivot = group.pivot_table(
                index="question_id", columns="method", values="composite_score", aggfunc="mean"
            )
            methods = list(pivot.columns)
            # score = mean, but break ties with win-rate vs runner-up
            means = pivot.mean().sort_values(ascending=False)
            best = means.index[0]
            if len(means) > 1:
                second = means.index[1]
                delta = pivot[best] - pivot[second]
                if float(means.iloc[0] - means.iloc[1]) < 0.03 and int((delta > 0).sum()) < int(
                    (delta < 0).sum()
                ):
                    best = second
            label = METHOD_LABELS.get(str(best), str(best))
            scenario = "multi-hop / bridge" if qtype == "hybrid" else "comparison / local factoid"
            lines.append(
                f"- **{scenario}** (`{qtype}`) → **{label}** "
                f"(mean {float(means[best]):.2f})"
            )
    elif not scenario_df.empty:
        for qtype, group in scenario_df.groupby("query_type"):
            best = group.loc[group["mean_composite_score"].idxmax()]
            label = METHOD_LABELS.get(str(best["method"]), str(best["method"]))
            scenario = "multi-hop / bridge" if qtype == "hybrid" else "comparison / local factoid"
            lines.append(
                f"- **{scenario}** (`{qtype}`) → **{label}** "
                f"(mean composite {float(best['mean_composite_score']):.2f})"
            )
    lines.append("")
    lines.append("## Pairwise takeaways")
    wins = pairwise_win_rates(accuracy_df)
    if not wins.empty:
        for _, r in wins.iterrows():
            pair = {r["method_a"], r["method_b"]}
            if pair == {"semantic_rag", "hybrid_rag"}:
                lines.append(
                    f"- Semantic vs Hybrid: {r['label_a']} {r['wins_a']}W–{r['wins_b']}W {r['label_b']} "
                    f"(mean Δ={r['mean_delta_a_minus_b']:+.2f})"
                )
            if pair == {"hybrid_rag", "graph_rag"}:
                lines.append(
                    f"- Hybrid vs GraphRAG global: {r['wins_a']}W–{r['wins_b']}W "
                    f"(mean Δ={r['mean_delta_a_minus_b']:+.2f})"
                )
            if pair == {"semantic_rag", "graph_rag"}:
                lines.append(
                    f"- Semantic vs GraphRAG global: {r['wins_a']}W–{r['wins_b']}W "
                    f"(mean Δ={r['mean_delta_a_minus_b']:+.2f})"
                )
    lines.append("")
    lines.append(
        "## Research honesty: is this cutting edge?\n"
        "- **Yes as an engineering bake-off**: FrontierRAG (Adaptive+CRAG escalate) + "
        "BM25/dense RRF + cross-encoder rerank + GraphRAG modes + HippoRAG 2/LightRAG "
        "is a 2025–2026-relevant *system* stack.\n"
        "- **No as a SOTA paper claim**: n=12 Hotpot subset, local 3B judge=generator, "
        "no full BenchmarkQED AutoQ/AutoE LLM pairwise, LazyGraphRAG itself still not OSS.\n"
        "- **Cutting-edge move**: ship the *router* that grades retrieval and escalates "
        "compute; keep the cost/latency scorecard; use stronger models for OpenIE and judging."
    )
    return "\n".join(lines)


def build_decision_artifacts(
    results_dir: Path,
    qa_path: Path,
) -> dict[str, Path]:
    results_dir = Path(results_dir)
    accuracy = pd.read_csv(results_dir / "accuracy_results.csv")
    scenario = (
        pd.read_csv(results_dir / "scenario_results.csv")
        if (results_dir / "scenario_results.csv").exists()
        else pd.DataFrame()
    )
    qa = load_qa(qa_path)

    examples = choose_over_examples(accuracy, qa)
    wins = pairwise_win_rates(accuracy)
    md = routing_cheat_sheet(accuracy, scenario)

    if not examples.empty:
        md += "\n\n## Concrete choose-A-over-B examples\n\n"
        for _, r in examples.head(12).iterrows():
            md += (
                f"### Choose **{r['choose']}** over {r['over']}\n"
                f"- Q: {r['question']}\n"
                f"- Gold: `{r['gold']}` ({r['hotpot_type']})\n"
                f"- Scores: {r['choose_score']:.2f} vs {r['over_score']:.2f} (margin {r['margin']:.2f})\n"
                f"- Why: {r['rule']}\n\n"
            )

    examples_path = results_dir / "choose_over_examples.csv"
    wins_path = results_dir / "pairwise_win_rates.csv"
    md_path = results_dir / "routing_cheatsheet.md"
    examples.to_csv(examples_path, index=False)
    wins.to_csv(wins_path, index=False)
    md_path.write_text(md, encoding="utf-8")
    return {"examples": examples_path, "pairwise": wins_path, "cheatsheet": md_path}
