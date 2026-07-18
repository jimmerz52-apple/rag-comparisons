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
    # Persist enriched accuracy for notebooks / GitHub
    acc.to_csv(results_dir / "accuracy_enriched.csv", index=False)

    profile = method_metric_profile(acc)
    profile.to_csv(results_dir / "metric_breakdown_by_method.csv", index=False)

    catalog = question_catalog(acc, qa_path, type_key=type_key)
    catalog_path = results_dir / "question_catalog.csv"
    catalog.to_csv(catalog_path, index=False)

    dual = scenario_dual_leaderboard(acc, scenario_col=scenario_col)
    dual.to_csv(results_dir / "dual_scoreboard.csv", index=False)

    stats = disagreement_stats(acc)
    (results_dir / "metric_disagreement.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    return {
        "enriched": results_dir / "accuracy_enriched.csv",
        "profile": results_dir / "metric_breakdown_by_method.csv",
        "catalog": catalog_path,
        "dual": results_dir / "dual_scoreboard.csv",
        "disagreement": results_dir / "metric_disagreement.json",
    }
