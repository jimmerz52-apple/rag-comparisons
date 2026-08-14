"""Token / latency / quality breakdown by question type.

Used by per-bench notebooks (`notebooks/*_tokens.ipynb`) and the dashboard.
LLM tok/q is per-question when `accuracy_results.csv` has token columns; otherwise
we still break down question/gold size (tiktoken) and latency by type.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from rag_benchmark.charts import METHOD_LABELS
from rag_benchmark.token_tracker import count_tokens

BENCHES: dict[str, dict[str, Any]] = {
    "hotpot": {
        "id": "hotpot",
        "title": "HotpotQA distractor",
        "results": "results",
        "qa": "data/qa/hotpot_eval.json",
        "meta": "data/qa/hotpot_meta.json",
        "type_key": "hotpot_type",
        "types_help": (
            "**bridge** = multi-hop (needs two Wikipedia paragraphs; harness "
            "`query_type=hybrid`). **comparison** = same-nationality / which-of-two "
            "(`query_type=local`)."
        ),
    },
    "code_rag": {
        "id": "code_rag",
        "title": "CodeRAG-Bench",
        "results": "results_code_rag",
        "qa": "data/qa/code_rag_eval.json",
        "meta": "data/qa/code_rag_meta.json",
        "type_key": "code_rag_type",
        "types_help": (
            "**humaneval** / **mbpp** = basic programming (leave-gold-out). "
            "**ds1000** / **odex** = open-domain against library docs."
        ),
    },
    "graphrag_bench": {
        "id": "graphrag_bench",
        "title": "GraphRAG-Bench Novel",
        "results": "results_graphrag_bench",
        "qa": "data/qa/graphrag_bench_eval.json",
        "meta": "data/qa/graphrag_bench_meta.json",
        "type_key": "graphrag_bench_type",
        "types_help": (
            "Paper axis: **Fact Retrieval** (graphs often tie) → **Complex Reasoning** "
            "→ **Contextual Summarize** → **Creative Generation** (graphs more likely to help)."
        ),
    },
    "multihop": {
        "id": "multihop",
        "title": "MultiHop-RAG",
        "results": "results_multihop",
        "qa": "data/qa/multihop_eval.json",
        "meta": "data/qa/multihop_meta.json",
        "type_key": "multihop_type",
        "types_help": (
            "**inference_query** / **comparison_query** / **temporal_query** — "
            "evidence is spread across 2–4 news docs."
        ),
    },
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size < 4:
        return pd.DataFrame()
    return pd.read_csv(path)


def _qa_frame(qa_path: Path, type_key: str) -> pd.DataFrame:
    qa = json.loads(Path(qa_path).read_text(encoding="utf-8"))
    rows = []
    for q in qa:
        question = str(q.get("question") or "")
        gold = str(q.get("expected_answer") or "")
        qtype = (
            q.get(type_key)
            or q.get("hotpot_type")
            or q.get("code_rag_type")
            or q.get("graphrag_bench_type")
            or q.get("multihop_type")
            or q.get("query_type")
            or "unknown"
        )
        rows.append(
            {
                "question_id": str(q["id"]),
                "question_type": qtype,
                "query_type": q.get("query_type", ""),
                "question": question,
                "gold": gold,
                "question_tokens": count_tokens(question),
                "gold_tokens": count_tokens(gold),
            }
        )
    return pd.DataFrame(rows)


def _attach_latency(acc: pd.DataFrame, lat_path: Path) -> pd.DataFrame:
    out = acc.copy()
    lat = _read_csv(lat_path)
    if lat.empty or "query_latency_seconds" not in lat.columns:
        return out
    if "phase" in lat.columns:
        lat = lat[lat["phase"].fillna("").astype(str).str.lower().ne("index")]
    lat = lat[pd.to_numeric(lat["query_latency_seconds"], errors="coerce").fillna(0) > 0]
    if lat.empty:
        return out
    if "question_id" in lat.columns:
        merged = out.merge(
            lat[["method", "question_id", "query_latency_seconds"]].drop_duplicates(
                ["method", "question_id"]
            ),
            on=["method", "question_id"],
            how="left",
            suffixes=("", "_lat"),
        )
        if "query_latency_seconds_lat" in merged.columns:
            merged["query_latency_seconds"] = merged["query_latency_seconds"].where(
                merged["query_latency_seconds"].notna(), merged["query_latency_seconds_lat"]
            )
            merged = merged.drop(columns=["query_latency_seconds_lat"])
        return merged
    if "question_index" not in lat.columns:
        return out
    attached = []
    for method, g in out.groupby("method", sort=False):
        lg = lat.loc[lat["method"] == method].sort_values("question_index")
        lg = lg[pd.to_numeric(lg["question_index"], errors="coerce") >= 0]
        if len(lg) != len(g):
            continue
        chunk = g.copy()
        chunk["query_latency_seconds"] = lg["query_latency_seconds"].to_numpy()
        attached.append(chunk)
    if not attached:
        return out
    used = {chunk["method"].iloc[0] for chunk in attached}
    rest = out[~out["method"].isin(used)]
    return pd.concat(attached + ([rest] if len(rest) else []), ignore_index=True)


def load_bench_frame(root: Path, bench_id: str) -> dict[str, Any]:
    spec = BENCHES[bench_id]
    root = Path(root)
    results = root / spec["results"]
    qa_path = root / spec["qa"]
    meta_path = root / spec["meta"]
    type_key = spec["type_key"]
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    catalog = _qa_frame(qa_path, type_key) if qa_path.exists() else pd.DataFrame()
    acc = _read_csv(results / "accuracy_enriched.csv")
    if acc.empty:
        acc = _read_csv(results / "accuracy_results.csv")
    tok_method = _read_csv(results / "token_results.csv")
    notes: list[str] = []
    if acc.empty:
        notes.append("No accuracy_results.csv yet — catalog-only (question/gold size by type).")
        frame = catalog.copy()
        frame["method"] = None
        has_llm_tokens = False
    else:
        acc["question_id"] = acc["question_id"].astype(str)
        if "query_latency_seconds" not in acc.columns:
            acc = _attach_latency(acc, results / "latency_results.csv")
        if not catalog.empty:
            acc = acc.merge(
                catalog[
                    [
                        "question_id",
                        "question_type",
                        "question_tokens",
                        "gold_tokens",
                    ]
                ],
                on="question_id",
                how="left",
            )
        else:
            acc["question_type"] = acc.get("query_type", "unknown")
        if type_key in acc.columns and acc["question_type"].isna().any():
            acc["question_type"] = acc["question_type"].fillna(acc[type_key])
        acc["question_type"] = acc["question_type"].fillna(acc.get("query_type", "unknown"))
        frame = acc
        has_llm_tokens = "total_tokens" in frame.columns and float(
            pd.to_numeric(frame["total_tokens"], errors="coerce").fillna(0).sum()
        ) > 0
        if not has_llm_tokens:
            notes.append(
                "Per-question LLM tokens are not in this accuracy file yet "
                "(live scorer started before the token-stats patch). "
                "Question/gold size and latency still split by type; "
                "method-level `token_results.csv` is the LLM total."
            )
    type_catalog = (
        catalog.groupby("question_type", as_index=False)
        .agg(
            n_indexed=("question_id", "nunique"),
            mean_question_tokens=("question_tokens", "mean"),
            p50_question_tokens=("question_tokens", "median"),
            p95_question_tokens=("question_tokens", lambda s: float(s.quantile(0.95))),
            mean_gold_tokens=("gold_tokens", "mean"),
        )
        .sort_values("n_indexed", ascending=False)
        if not catalog.empty
        else pd.DataFrame()
    )
    by_type = summarize_by_type(frame, has_llm_tokens=has_llm_tokens)
    if not by_type.empty:
        by_type.to_csv(results / "token_by_type.csv", index=False)
    return {
        "spec": spec,
        "meta": meta,
        "catalog": catalog,
        "frame": frame,
        "type_catalog": type_catalog,
        "by_type": by_type,
        "tok_method": tok_method,
        "has_llm_tokens": has_llm_tokens,
        "notes": notes,
        "results_dir": results,
    }


def summarize_by_type(frame: pd.DataFrame, *, has_llm_tokens: bool) -> pd.DataFrame:
    if frame.empty or "question_type" not in frame.columns:
        return pd.DataFrame()
    scored = frame
    if "method" in frame.columns:
        scored = frame[frame["method"].notna()].copy()
    if scored.empty or "method" not in scored.columns:
        return pd.DataFrame()
    if "question_id" not in scored.columns:
        scored["question_id"] = range(len(scored))

    named: dict[str, tuple[str, str]] = {
        "n": ("question_id", "nunique"),
    }
    if "question_tokens" in scored.columns:
        named["mean_question_tokens"] = ("question_tokens", "mean")
    if "gold_tokens" in scored.columns:
        named["mean_gold_tokens"] = ("gold_tokens", "mean")
    if "composite_score" in scored.columns:
        named["mean_composite"] = ("composite_score", "mean")
    if "llm_judge_score" in scored.columns:
        named["mean_judge"] = ("llm_judge_score", "mean")
    if "retrieval_recall" in scored.columns:
        named["mean_retrieval_recall"] = ("retrieval_recall", "mean")
    if "gold_in_context" in scored.columns:
        named["gold_in_context_rate"] = ("gold_in_context", "mean")
    if "evidence_override" in scored.columns:
        named["evidence_override_rate"] = ("evidence_override", "mean")
    if "query_latency_seconds" in scored.columns:
        named["mean_latency_s"] = ("query_latency_seconds", "mean")
    if has_llm_tokens:
        named["tokens_per_query"] = ("total_tokens", "mean")
        if "prompt_tokens" in scored.columns:
            named["prompt_tokens_per_query"] = ("prompt_tokens", "mean")
        if "completion_tokens" in scored.columns:
            named["completion_tokens_per_query"] = ("completion_tokens", "mean")
        if "query_prompt_tokens" in scored.columns:
            q_comp = (
                pd.to_numeric(scored["query_completion_tokens"], errors="coerce").fillna(0)
                if "query_completion_tokens" in scored.columns
                else 0
            )
            e_prompt = (
                pd.to_numeric(scored["eval_prompt_tokens"], errors="coerce").fillna(0)
                if "eval_prompt_tokens" in scored.columns
                else 0
            )
            e_comp = (
                pd.to_numeric(scored["eval_completion_tokens"], errors="coerce").fillna(0)
                if "eval_completion_tokens" in scored.columns
                else 0
            )
            scored["query_tokens"] = (
                pd.to_numeric(scored["query_prompt_tokens"], errors="coerce").fillna(0) + q_comp
            )
            scored["eval_tokens"] = e_prompt + e_comp
            named["query_tokens_per_query"] = ("query_tokens", "mean")
            named["eval_tokens_per_query"] = ("eval_tokens", "mean")

    out = scored.groupby(["method", "question_type"], as_index=False).agg(**named)
    if "query_latency_seconds" in scored.columns:
        p95 = (
            scored.groupby(["method", "question_type"])["query_latency_seconds"]
            .quantile(0.95)
            .reset_index(name="p95_latency_s")
        )
        out = out.merge(p95, on=["method", "question_type"], how="left")
    out["method_label"] = out["method"].map(lambda m: METHOD_LABELS.get(m, m))
    return out.sort_values(["question_type", "method_label"])


def plot_type_overview(payload: dict[str, Any], *, figsize: tuple[float, float] = (11, 8)):
    """Matplotlib figures for notebooks. Returns the pyplot module after drawing."""
    import matplotlib.pyplot as plt

    type_catalog: pd.DataFrame = payload["type_catalog"]
    by_type: pd.DataFrame = payload["by_type"]
    has_llm = payload["has_llm_tokens"]
    title = payload["spec"]["title"]

    n_plots = 2 if type_catalog.empty else 3
    if by_type.empty:
        n_plots = 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(figsize[0], 3.6 * n_plots))
    if n_plots == 1:
        axes = [axes]
    ax_i = 0

    if not type_catalog.empty:
        ax = axes[ax_i]
        ax_i += 1
        cats = type_catalog.sort_values("mean_question_tokens")
        ax.barh(cats["question_type"], cats["mean_question_tokens"], color="#0b5fff", label="Question")
        ax.barh(cats["question_type"], cats["mean_gold_tokens"], color="#7c3aed", alpha=0.7, label="Gold")
        ax.set_xlabel("Mean tiktoken tokens (cl100k)")
        ax.set_title(f"{title} — input size by question type (full catalog)")
        ax.legend()

    if not by_type.empty:
        ax = axes[ax_i]
        ax_i += 1
        metric = "tokens_per_query" if has_llm and "tokens_per_query" in by_type.columns else "mean_question_tokens"
        pivot = by_type.pivot(index="method_label", columns="question_type", values=metric)
        im = ax.imshow(pivot.fillna(0).to_numpy(), aspect="auto", cmap="Blues")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(list(pivot.columns), rotation=20, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(list(pivot.index))
        label = "LLM tokens / query" if metric == "tokens_per_query" else "Question tokens (mean)"
        ax.set_title(f"{title} — {label} by method × type")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for y, method in enumerate(pivot.index):
            for x, qtype in enumerate(pivot.columns):
                val = pivot.loc[method, qtype]
                if pd.notna(val):
                    ax.text(x, y, f"{val:.0f}", ha="center", va="center", fontsize=8, color="#15202b")

        if has_llm and {"prompt_tokens_per_query", "completion_tokens_per_query"} <= set(by_type.columns):
            ax = axes[ax_i]
            stacked = by_type.copy()
            stacked["xtick"] = stacked["method_label"].astype(str) + " · " + stacked["question_type"].astype(str)
            stacked = stacked.sort_values("tokens_per_query")
            ax.barh(stacked["xtick"], stacked["prompt_tokens_per_query"], color="#0b5fff", label="Prompt / q")
            ax.barh(
                stacked["xtick"],
                stacked["completion_tokens_per_query"],
                left=stacked["prompt_tokens_per_query"],
                color="#7c3aed",
                label="Completion / q",
            )
            ax.set_xlabel("LLM tokens / query")
            ax.set_title(f"{title} — prompt vs completion by method × type")
            ax.legend()
        elif ax_i < len(axes):
            ax = axes[ax_i]
            lat_col = "mean_latency_s" if "mean_latency_s" in by_type.columns else None
            if lat_col:
                pivot_l = by_type.pivot(index="method_label", columns="question_type", values=lat_col)
                im = ax.imshow(pivot_l.fillna(0).to_numpy(), aspect="auto", cmap="Oranges")
                ax.set_xticks(range(len(pivot_l.columns)))
                ax.set_xticklabels(list(pivot_l.columns), rotation=20, ha="right")
                ax.set_yticks(range(len(pivot_l.index)))
                ax.set_yticklabels(list(pivot_l.index))
                ax.set_title(f"{title} — mean latency (s) by method × type")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.axis("off")

    fig.tight_layout()
    return fig
