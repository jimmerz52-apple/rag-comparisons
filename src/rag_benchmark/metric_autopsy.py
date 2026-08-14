"""Shared metric autopsy helpers — canvas-style analysis for notebooks / CSVs.

Cutting-edge reading rule:
  - generative_score = mean(llm_judge, contains)  → fair for GraphRAG prose
  - extractive_score = mean(token_f1, exact_match) → Hotpot short-span EM
  - composite_score averages all four (can hide graph wins on multi-hop)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from rag_benchmark.charts import METHOD_LABELS


def enrich_accuracy(accuracy_df: pd.DataFrame) -> pd.DataFrame:
    df = accuracy_df.copy()
    df["contains_answer"] = df["contains_answer"].astype(bool)
    df["exact_match"] = df["exact_match"].astype(bool)
    df["generative_score"] = (
        df["llm_judge_score"].fillna(0.0) + df["contains_answer"].astype(float)
    ) / 2.0
    df["extractive_score"] = (
        df["token_f1"].fillna(0.0) + df["exact_match"].astype(float)
    ) / 2.0
    if "composite_score" not in df.columns:
        df["composite_score"] = (
            df["llm_judge_score"].fillna(0.0)
            + df["token_f1"].fillna(0.0)
            + df["exact_match"].astype(float)
            + df["contains_answer"].astype(float)
        ) / 4.0
    return df


def method_metric_profile(accuracy_df: pd.DataFrame) -> pd.DataFrame:
    df = enrich_accuracy(accuracy_df)
    out = (
        df.groupby("method", as_index=False)
        .agg(
            llm_judge=("llm_judge_score", "mean"),
            contains=("contains_answer", "mean"),
            token_f1=("token_f1", "mean"),
            exact_match=("exact_match", "mean"),
            generative=("generative_score", "mean"),
            extractive=("extractive_score", "mean"),
            composite=("composite_score", "mean"),
        )
        .sort_values("generative", ascending=False)
    )
    out["label"] = out["method"].map(lambda m: METHOD_LABELS.get(m, m))
    return out


def question_catalog(
    accuracy_df: pd.DataFrame,
    qa_path: Path,
    *,
    type_key: str = "hotpot_type",
) -> pd.DataFrame:
    """Q# → id / question / gold + avg metrics across methods."""
    qa = json.loads(Path(qa_path).read_text(encoding="utf-8"))
    df = enrich_accuracy(accuracy_df)
    rows: list[dict[str, Any]] = []
    for i, q in enumerate(qa):
        sub = df[df["question_id"] == q["id"]]
        qtype = q.get(type_key) or q.get("graphrag_bench_type") or q.get("multihop_type") or q.get(
            "query_type", ""
        )
        if sub.empty:
            rows.append(
                {
                    "label": f"Q{i+1}",
                    "question_id": q["id"],
                    "type": qtype,
                    "question": q["question"],
                    "gold": q.get("expected_answer", ""),
                    "em_rate": None,
                    "avg_judge": None,
                    "avg_f1": None,
                    "contains_rate": None,
                    "avg_generative": None,
                    "avg_extractive": None,
                    "best_method_generative": None,
                }
            )
            continue
        best = sub.loc[sub["generative_score"].idxmax()]
        rows.append(
            {
                "label": f"Q{i+1}",
                "question_id": q["id"],
                "type": qtype,
                "question": q["question"],
                "gold": q.get("expected_answer", ""),
                "em_rate": float(sub["exact_match"].mean()),
                "avg_judge": float(sub["llm_judge_score"].mean()),
                "avg_f1": float(sub["token_f1"].mean()),
                "contains_rate": float(sub["contains_answer"].mean()),
                "avg_generative": float(sub["generative_score"].mean()),
                "avg_extractive": float(sub["extractive_score"].mean()),
                "best_method_generative": METHOD_LABELS.get(best["method"], best["method"]),
            }
        )
    return pd.DataFrame(rows)


def disagreement_stats(accuracy_df: pd.DataFrame) -> dict[str, Any]:
    df = enrich_accuracy(accuracy_df)
    n = len(df)
    return {
        "n_rows": n,
        "judge_ge_05_but_em_0": int(
            ((df["llm_judge_score"] >= 0.5) & (~df["exact_match"])).sum()
        ),
        "contains_but_not_em": int((df["contains_answer"] & (~df["exact_match"])).sum()),
        "graph_em_rate": float(
            df.loc[df["method"].str.contains("graph", case=False), "exact_match"].mean()
        )
        if df["method"].str.contains("graph", case=False).any()
        else None,
        "top_judge_method": df.groupby("method")["llm_judge_score"].mean().idxmax(),
        "top_judge_value": float(df.groupby("method")["llm_judge_score"].mean().max()),
        "top_generative_multihop": None,
    }


def scenario_dual_leaderboard(
    accuracy_df: pd.DataFrame,
    *,
    scenario_col: str = "query_type",
) -> pd.DataFrame:
    """Side-by-side generative vs extractive winners by scenario/type."""
    df = enrich_accuracy(accuracy_df)
    if scenario_col not in df.columns:
        raise KeyError(scenario_col)
    rows = []
    for scenario, sub in df.groupby(scenario_col):
        by_m = sub.groupby("method").agg(
            generative=("generative_score", "mean"),
            extractive=("extractive_score", "mean"),
            composite=("composite_score", "mean"),
            judge=("llm_judge_score", "mean"),
        )
        g_best = by_m["generative"].idxmax()
        e_best = by_m["extractive"].idxmax()
        c_best = by_m["composite"].idxmax()
        rows.append(
            {
                "scenario": scenario,
                "generative_winner": METHOD_LABELS.get(g_best, g_best),
                "generative_score": round(float(by_m.loc[g_best, "generative"]), 3),
                "extractive_winner": METHOD_LABELS.get(e_best, e_best),
                "extractive_score": round(float(by_m.loc[e_best, "extractive"]), 3),
                "composite_winner": METHOD_LABELS.get(c_best, c_best),
                "composite_score": round(float(by_m.loc[c_best, "composite"]), 3),
                "ranking_flips": g_best != c_best,
            }
        )
    return pd.DataFrame(rows)


CORE_TRIO = ("semantic_rag", "graph_rag", "lazygraph_rag")


def method_clear_wins(
    accuracy_df: pd.DataFrame,
    qa_path: Path,
    *,
    methods: tuple[str, ...] = CORE_TRIO,
    score_col: str = "composite_score",
    min_margin: float = 0.12,
    min_vs_each: float = 0.05,
) -> pd.DataFrame:
    """Questions where method M clearly beats all other methods in `methods`."""
    qa_by_id = {q["id"]: q for q in json.loads(Path(qa_path).read_text(encoding="utf-8"))}
    df = enrich_accuracy(accuracy_df)
    pivot = df.pivot_table(
        index="question_id", columns="method", values=score_col, aggfunc="mean"
    )
    rows: list[dict[str, Any]] = []
    for qid in pivot.index:
        if not all(m in pivot.columns for m in methods):
            continue
        scores = {m: float(pivot.loc[qid, m]) for m in methods if pd.notna(pivot.loc[qid, m])}
        if len(scores) < len(methods):
            continue
        for winner, w_score in scores.items():
            rivals = {m: s for m, s in scores.items() if m != winner}
            if not rivals:
                continue
            best_rival_m = max(rivals, key=rivals.get)  # type: ignore[arg-type]
            best_rival_s = rivals[best_rival_m]
            margin = w_score - best_rival_s
            if margin < min_margin:
                continue
            if any(w_score - s < min_vs_each for s in rivals.values()):
                continue
            q = qa_by_id.get(str(qid), {})
            sub = df[(df.question_id == qid) & (df.method == winner)].iloc[0]
            rows.append(
                {
                    "winner": winner,
                    "winner_label": METHOD_LABELS.get(winner, winner),
                    "question_id": qid,
                    "question": q.get("question", qid),
                    "gold": q.get("expected_answer", ""),
                    "hotpot_type": q.get("hotpot_type", q.get("query_type", "")),
                    "score_col": score_col,
                    "winner_score": round(w_score, 3),
                    "runner_up": best_rival_m,
                    "runner_up_label": METHOD_LABELS.get(best_rival_m, best_rival_m),
                    "runner_up_score": round(best_rival_s, 3),
                    "margin": round(margin, 3),
                    "llm_judge": round(float(sub["llm_judge_score"]), 3),
                    "token_f1": round(float(sub["token_f1"]), 3),
                    "exact_match": bool(sub["exact_match"]),
                    "contains_answer": bool(sub["contains_answer"]),
                    "generative_score": round(float(sub["generative_score"]), 3),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["winner", "margin"], ascending=[True, False])


def format_method_win_examples_markdown(wins_df: pd.DataFrame) -> str:
    if wins_df.empty:
        return "_No clear wins at the chosen margin threshold._"
    lines = [
        "# Clear wins: Semantic vs GraphRAG vs LazyGraph",
        "",
        "A **clear win** means the method beats *both* rivals in the trio by at least "
        "the margin threshold on the chosen score column.",
        "",
    ]
    for method in CORE_TRIO:
        sub = wins_df[wins_df.winner == method]
        label = METHOD_LABELS.get(method, method)
        lines.append(f"## {label}")
        if sub.empty:
            lines.append("_No clear wins on this slice at this threshold._\n")
            continue
        for _, r in sub.iterrows():
            lines.append(f"### {r['question'][:80]}{'…' if len(str(r['question'])) > 80 else ''}")
            lines.append(f"- **Gold:** `{r['gold']}` ({r['hotpot_type']})")
            lines.append(
                f"- **{r['score_col']}:** {r['winner_score']:.2f} vs "
                f"{r['runner_up_label']} {r['runner_up_score']:.2f} "
                f"(margin **{r['margin']:.2f}**)"
            )
            lines.append(
                f"- **Breakdown:** judge={r['llm_judge']:.2f}, contains={int(r['contains_answer'])}, "
                f"F1={r['token_f1']:.2f}, EM={int(r['exact_match'])}, "
                f"generative={r['generative_score']:.2f}"
            )
            lines.append("")
    return "\n".join(lines)


def write_autopsy_artifacts(
    *,
    results_dir: Path,
    qa_path: Path,
    type_key: str = "hotpot_type",
    scenario_col: str = "query_type",
) -> dict[str, Path]:
    results_dir = Path(results_dir)
    acc = pd.read_csv(results_dir / "accuracy_results.csv")
    acc = enrich_accuracy(acc)

    # Attach dataset-specific type columns from QA when missing on accuracy rows
    # (e.g. code_rag_type, graphrag_bench_type — BenchmarkRunner only stores query_type).
    qa = json.loads(Path(qa_path).read_text(encoding="utf-8"))
    by_id = {str(q["id"]): q for q in qa}
    for col in (type_key, scenario_col, "code_rag_type", "graphrag_bench_type", "multihop_type"):
        if col == "query_type":
            continue
        if col not in acc.columns or acc[col].isna().all():
            acc[col] = acc["question_id"].map(
                lambda qid, c=col: by_id.get(str(qid), {}).get(c)
            )

    # Persist enriched accuracy for notebooks / GitHub
    acc.to_csv(results_dir / "accuracy_enriched.csv", index=False)

    profile = method_metric_profile(acc)
    profile.to_csv(results_dir / "metric_breakdown_by_method.csv", index=False)

    catalog = question_catalog(acc, qa_path, type_key=type_key)
    catalog_path = results_dir / "question_catalog.csv"
    catalog.to_csv(catalog_path, index=False)

    dual_col = scenario_col if scenario_col in acc.columns and acc[scenario_col].notna().any() else "query_type"
    dual = scenario_dual_leaderboard(acc, scenario_col=dual_col)
    dual.to_csv(results_dir / "dual_scoreboard.csv", index=False)

    stats = disagreement_stats(acc)
    (results_dir / "metric_disagreement.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    composite_wins = method_clear_wins(acc, qa_path, score_col="composite_score")
    generative_wins = method_clear_wins(acc, qa_path, score_col="generative_score")
    composite_wins.to_csv(results_dir / "method_clear_wins_composite.csv", index=False)
    generative_wins.to_csv(results_dir / "method_clear_wins_generative.csv", index=False)
    wins_md = format_method_win_examples_markdown(composite_wins)
    wins_md += "\n\n---\n\n## Same filter on generative score (judge + contains)\n\n"
    wins_md += format_method_win_examples_markdown(generative_wins)
    wins_md_path = results_dir / "method_win_examples.md"
    wins_md_path.write_text(wins_md, encoding="utf-8")

    return {
        "enriched": results_dir / "accuracy_enriched.csv",
        "profile": results_dir / "metric_breakdown_by_method.csv",
        "catalog": catalog_path,
        "dual": results_dir / "dual_scoreboard.csv",
        "disagreement": results_dir / "metric_disagreement.json",
        "wins_composite": results_dir / "method_clear_wins_composite.csv",
        "wins_generative": results_dir / "method_clear_wins_generative.csv",
        "wins_md": wins_md_path,
    }
