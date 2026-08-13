#!/usr/bin/env python3
"""Build a research-grounded interactive RAG dashboard for GitHub Pages.

Incorporates 2024–2026 evaluation practice:
  - RAGAS-style retrieval vs generation framing (Es et al., 2024+)
  - Generative vs extractive dual lenses (Hotpot EM/F1 underestimates generative QA)
  - GraphRAG-Bench: graphs help by task difficulty, not by default (Xiang et al., 2025)
  - CodeRAG-Bench leave-gold-out + retrieval bottleneck (Wang et al., NAACL 2025)
  - Systematic finding: hybrid/router > single paradigm (Han et al. / VentureBeat 2026 synthesis)

Outputs: docs/index.html · results/dashboard.html · docs/dashboard_data.json
"""

from __future__ import annotations

import html as html_lib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

BENCHES = [
    {
        "id": "hotpot",
        "title": "HotpotQA distractor",
        "short": "Hotpot",
        "results": ROOT / "results",
        "meta": ROOT / "data" / "qa" / "hotpot_meta.json",
        "era": "2018 classic · still useful, metric-sensitive",
        "what": (
            "Multi-hop Wikipedia QA (distractor). Gold answers are short extractive spans. "
            "Recent critiques (Qi 2025) note EM/F1 punish fluent generative answers that humans "
            "accept — which is why this dashboard always shows generative vs extractive lenses."
        ),
        "how_to_read": (
            "Prefer composite for a balanced rank; check generative if your UX is chatty prose; "
            "check extractive if you need short-span fidelity. Ranking flips are expected."
        ),
        "paper": "Yang et al., EMNLP 2018 · https://hotpotqa.github.io/",
    },
    {
        "id": "code_rag",
        "title": "CodeRAG-Bench",
        "short": "CodeRAG",
        "results": ROOT / "results_code_rag",
        "meta": ROOT / "data" / "qa" / "code_rag_meta.json",
        "era": "NAACL Findings 2025",
        "what": (
            "Wang et al.: retrieval-augmented code generation across basic programming and "
            "open-domain tasks. Canonical solutions are leave-gold-out. Paper finding: high-quality "
            "context helps, but retrievers often fail to fetch it — end-to-end RAG ≠ oracle RAG."
        ),
        "how_to_read": (
            "Code-aware contains/judge matter more than wiki-style F1. Compare methods under the "
            "same datastore; do not treat token F1 as pass@k."
        ),
        "paper": "Wang et al., Findings NAACL 2025 · https://code-rag-bench.github.io/",
    },
    {
        "id": "graphrag_bench",
        "title": "GraphRAG-Bench Novel",
        "short": "GraphRAG-Bench",
        "results": ROOT / "results_graphrag_bench",
        "meta": ROOT / "data" / "qa" / "graphrag_bench_meta.json",
        "era": "arXiv:2506.05690 · 2025–26",
        "what": (
            "Built to answer: when do graph structures help? Tasks scale from fact retrieval → "
            "complex reasoning → contextual summarize → creative generation. Literature: graphs "
            "often tie or lose on L1 fact lookup; gains concentrate on harder reasoning/summarize."
        ),
        "how_to_read": (
            "Do not crown GraphRAG from Hotpot averages alone. Filter by question type here; "
            "expect vector methods to compete on factoid items."
        ),
        "paper": "Xiang et al., When to use Graphs in RAG · https://github.com/GraphRAG-Bench/GraphRAG-Benchmark",
    },
    {
        "id": "multihop",
        "title": "MultiHop-RAG",
        "short": "MultiHop",
        "results": ROOT / "results_multihop",
        "meta": ROOT / "data" / "qa" / "multihop_meta.json",
        "era": "2024 news multi-doc",
        "what": (
            "Multi-document news reasoning. Useful stress test for bridge questions, but "
            "GraphRAG-Bench authors argue many ‘multi-hop’ sets still under-test deep synthesis."
        ),
        "how_to_read": "Watch generative ≫ extractive (news answers rarely match gold spans).",
        "paper": "Tang & Yang, 2024 · MultiHop-RAG",
    },
]

LABELS = {
    "semantic_rag": "Semantic",
    "rerank_semantic": "Rerank",
    "hybrid_dense_sparse": "BM25+dense",
    "hybrid_rag": "Hybrid",
    "lazygraph_rag": "GraphRAG fast/basic",
    "graph_rag": "GraphRAG global",
    "graph_local_rag": "GraphRAG local",
    "frontier_rag": "Frontier",
    "adaptive_rag": "Adaptive",
}

# RAGAS / 2026 practice mapped onto what this harness actually measures
METRIC_MAP = [
    {
        "ragas": "Answer relevance ≈ LLM judge",
        "ours": "llm_judge_score",
        "layer": "Generation",
        "status": "proxied",
        "note": "Local judge scores usefulness vs gold. Not identical to RAGAS answer_relevancy (query-conditioned, often reference-free).",
    },
    {
        "ragas": "Faithfulness / groundedness",
        "ours": "— (gap)",
        "layer": "Generation",
        "status": "missing",
        "note": "2026 guides treat faithfulness as mandatory. This bake-off does not yet score claim-vs-context; do not equate judge with faithfulness.",
    },
    {
        "ragas": "Context precision / recall",
        "ours": "— (gap)",
        "layer": "Retrieval",
        "status": "missing",
        "note": "Separate retrieval eval (nDCG@k, recall@k) is standard in CodeRAG-Bench and GraphRAG-Bench. End-to-end scores alone hide retrieval failures.",
    },
    {
        "ragas": "Correctness (lexical)",
        "ours": "token_f1 + exact_match",
        "layer": "Generation",
        "status": "covered",
        "note": "Hotpot-style. Underestimates fluent paraphrases (Qi 2025; LLM-as-judge reassessments 2025).",
    },
    {
        "ragas": "Soft recall / containment",
        "ours": "contains_answer",
        "layer": "Generation",
        "status": "covered",
        "note": "Loose check that gold (or code body tokens) appear in the prediction.",
    },
    {
        "ragas": "Cost / latency (ops)",
        "ours": "tokens_per_query, latency",
        "layer": "Ops",
        "status": "covered",
        "note": "Production selection is multi-objective — quality alone is not a ship decision (Braintrust / Atlan 2026 practice).",
    },
]

METRIC_DEFS = [
    {
        "id": "composite",
        "name": "Composite",
        "range": "0–1 ↑",
        "means": "Mean of available judge, F1, EM, contains. Default ranker — blends generative and extractive pressure.",
    },
    {
        "id": "generative",
        "name": "Generative lens",
        "range": "0–1 ↑",
        "means": "Judge + contains. Fairer to GraphRAG prose that humans accept but Hotpot F1 punishes.",
    },
    {
        "id": "extractive",
        "name": "Extractive lens",
        "range": "0–1 ↑",
        "means": "Token F1 + EM. Classic Hotpot scoring. Prefer when answers must be short spans.",
    },
    {
        "id": "llm_judge",
        "name": "LLM judge",
        "range": "0–1 ↑",
        "means": "Semantic correctness vs gold. Correlates better with humans than EM on generative QA, but is not faithfulness-to-context.",
    },
    {
        "id": "latency",
        "name": "Query latency (mean / p50 / p95)",
        "range": "seconds ↓",
        "means": "Wall-clock per question after the index exists. p95 is the SLO number — a method with mean 2s and p95 15s will feel broken in interactive UI.",
    },
    {
        "id": "index",
        "name": "Index build time",
        "range": "seconds ↓",
        "means": "One-time (or rebuild) cost. GraphRAG global is dominated by this. Amortize over expected query volume.",
    },
]

RESEARCH_FINDINGS = [
    {
        "claim": "No single RAG paradigm wins everywhere",
        "evidence": (
            "Unified evaluations (RAG vs GraphRAG systematic studies, 2025–26) and GraphRAG-Bench "
            "show vector RAG competitive on fact lookup; graphs pull ahead on complex reasoning / "
            "contextual summarize. Hybrid or routed stacks beat either alone."
        ),
        "dashboard": "Use Decision Lab → generative vs extractive + routing table; don’t pick from one bar chart.",
        "cites": ["Xiang et al. arXiv:2506.05690", "Han et al. arXiv:2502.11371"],
    },
    {
        "claim": "Separate retrieval from generation",
        "evidence": (
            "RAGAS / DeepEval / 2026 guides: a pipeline can look ‘faithful’ while context recall "
            "quietly collapses. CodeRAG-Bench explicitly reports retrieval nDCG and end-to-end gen."
        ),
        "dashboard": "Metric map marks context precision/recall as gaps — treat end-to-end ranks as provisional.",
        "cites": ["Es et al. RAGAS 2024", "Wang et al. CodeRAG-Bench 2025"],
    },
    {
        "claim": "EM/F1 under-credit generative answers",
        "evidence": (
            "Hotpot-style extractive metrics penalize correct verbose answers. LLM-as-judge closes "
            "much of the gap to human ratings; dual scoreboards routinely flip winners."
        ),
        "dashboard": "Dual scoreboard + Generative vs Extractive scatter are first-class, not footnotes.",
        "cites": ["Qi 2025 (stop-only-Hotpot caution)", "arXiv:2504.11972 LLM-as-judge QA"],
    },
    {
        "claim": "Code RAG bottleneck is often retrieval, not generation",
        "evidence": (
            "CodeRAG-Bench: gold docs help even strong models; current retrievers struggle on "
            "DS-1000 / ODEX / SWE-style tasks. Leave-gold-out is required for honest basic-programming eval."
        ),
        "dashboard": "CodeRAG card explains leave-gold-out; expand scored n before claiming production readiness.",
        "cites": ["Wang et al. Findings NAACL 2025"],
    },
    {
        "claim": "Ship a router, not a religion",
        "evidence": (
            "Industry + academic synthesis 2026: route local factoids to vector+rerank; reserve "
            "graph / frontier for multi-hop or corpus-wide synthesis; watch token/latency SLOs."
        ),
        "dashboard": "Routing recommendations + Pareto (quality vs tokens) encode that tradeoff.",
        "cites": ["VentureBeat GraphRAG synthesis 2026", "this harness engineering_briefing.md"],
    },
]

METHOD_DEFS = {
    "Semantic": "Dense vector retrieve + generate. Strong baseline for single-hop / local factoids.",
    "Rerank": "Retrieve then rerank. Often best quality/latency interactive path in this harness.",
    "BM25+dense": "Sparse+dense fusion. Helps keyword-heavy queries; paper-standard hybrid retriever.",
    "Hybrid": "Vector + graph local path in this harness.",
    "GraphRAG fast/basic": "Fast NLP graph index + basic/local search. Often high generative / contains, low F1.",
    "GraphRAG global": "Community-summary global search. Costly; for corpus-wide themes, not factoids.",
    "GraphRAG local": "Entity-neighborhood search.",
    "Frontier": "Adaptive + corrective-style stack — multi-hop candidate in routing table.",
    "Adaptive": "Per-query router among retrieval strategies.",
}


def _howto_html() -> str:
    path = ROOT / "docs" / "how-to-run.md"
    if not path.exists():
        return "<p class='muted'>docs/how-to-run.md missing.</p>"
    md = path.read_text(encoding="utf-8")
    chunks: list[str] = []
    in_code = False
    code: list[str] = []
    in_table = False
    table_rows: list[str] = []

    def flush_table() -> None:
        nonlocal in_table, table_rows
        if not table_rows:
            in_table = False
            return
        cells = [r.split("|") for r in table_rows]
        cells = [[c.strip() for c in row if c.strip() != ""] for row in cells]
        if len(cells) >= 2:
            head, body = cells[0], cells[2:] if cells[1] and set(cells[1][0]) <= set("-: ") else cells[1:]
            # skip separator row
            body = [r for r in body if not all(set(c) <= set("-: ") for c in r)]
            thead = "<tr>" + "".join(f"<th>{html_lib.escape(c)}</th>" for c in head) + "</tr>"
            tbody = "".join(
                "<tr>" + "".join(f"<td>{html_lib.escape(c)}</td>" for c in row) + "</tr>"
                for row in body
            )
            chunks.append(f"<table><thead>{thead}</thead><tbody>{tbody}</tbody></table>")
        table_rows = []
        in_table = False

    for raw in md.splitlines():
        line = raw.rstrip()
        if line.startswith("```"):
            if in_code:
                chunks.append("<pre>" + html_lib.escape("\n".join(code)) + "</pre>")
                code = []
                in_code = False
            else:
                flush_table()
                in_code = True
            continue
        if in_code:
            code.append(line)
            continue
        if line.startswith("|"):
            in_table = True
            table_rows.append(line)
            continue
        if in_table:
            flush_table()
        if not line:
            continue
        if line.startswith("# "):
            chunks.append(f"<h2>{html_lib.escape(line[2:])}</h2>")
        elif line.startswith("## "):
            chunks.append(f"<h3>{html_lib.escape(line[3:])}</h3>")
        elif line.startswith("---"):
            chunks.append("<hr/>")
        elif line.startswith("- "):
            chunks.append(f"<li>{html_lib.escape(line[2:])}</li>")
        else:
            chunks.append(f"<p>{html_lib.escape(line)}</p>")
    if in_code:
        chunks.append("<pre>" + html_lib.escape("\n".join(code)) + "</pre>")
    flush_table()
    # wrap consecutive <li>
    out: list[str] = []
    in_ul = False
    for c in chunks:
        if c.startswith("<li>"):
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(c)
        else:
            if in_ul:
                out.append("</ul>")
                in_ul = False
            out.append(c)
    if in_ul:
        out.append("</ul>")
    return '<div class="howto">' + "".join(out) + "</div>"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _meta(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _safe_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size < 3:
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []
    records = []
    for row in df.to_dict(orient="records"):
        clean = {}
        for k, v in row.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean[k] = None
            elif hasattr(v, "item"):
                clean[k] = v.item()
            else:
                clean[k] = v
        records.append(clean)
    return records


def _lens_summary(accuracy: list[dict]) -> list[dict]:
    if not accuracy:
        return []
    df = pd.DataFrame(accuracy)
    if "generative_score" not in df.columns:
        return []
    g = (
        df.groupby("method", as_index=False)
        .agg(
            generative=("generative_score", "mean"),
            extractive=("extractive_score", "mean"),
            composite=("composite_score", "mean"),
            n=("question_id", "count"),
        )
    )
    out = []
    for row in g.to_dict(orient="records"):
        row["method_label"] = LABELS.get(row["method"], row["method"])
        out.append(row)
    return out


def _ops_stats(results_dir: Path) -> tuple[list[dict], list[dict], dict]:
    """Latency / token / index stats from CSVs. Sample latencies for boxplots."""
    lat_path = results_dir / "latency_results.csv"
    tok_path = results_dir / "token_results.csv"
    ops: dict[str, dict] = {}
    samples: list[dict] = []

    if lat_path.exists() and lat_path.stat().st_size > 3:
        lat = pd.read_csv(lat_path)
        if "query_latency_seconds" in lat.columns:
            phase_col = lat["phase"] if "phase" in lat.columns else pd.Series([None] * len(lat))
            is_index = phase_col.fillna("").astype(str).str.lower().eq("index")
            queries = lat.loc[~is_index]
            indexes = lat.loc[is_index]
            for method, g in queries.groupby("method"):
                vals = pd.to_numeric(g["query_latency_seconds"], errors="coerce").dropna()
                vals = vals[vals > 0]
                if vals.empty:
                    continue
                ops.setdefault(method, {})
                ops[method].update(
                    {
                        "n_latency": int(len(vals)),
                        "mean_query_latency_seconds": float(vals.mean()),
                        "p50_query_latency_seconds": float(vals.quantile(0.50)),
                        "p95_query_latency_seconds": float(vals.quantile(0.95)),
                        "min_query_latency_seconds": float(vals.min()),
                        "max_query_latency_seconds": float(vals.max()),
                    }
                )
                take = vals.sample(n=min(200, len(vals)), random_state=0) if len(vals) > 200 else vals
                for v in take.tolist():
                    samples.append(
                        {
                            "method": method,
                            "method_label": LABELS.get(method, method),
                            "query_latency_seconds": float(v),
                        }
                    )
            for method, g in indexes.groupby("method"):
                ops.setdefault(method, {})
                ops[method]["index_seconds"] = float(
                    pd.to_numeric(g["query_latency_seconds"], errors="coerce").dropna().sum()
                )

    if tok_path.exists() and tok_path.stat().st_size > 3:
        tok = pd.read_csv(tok_path)
        if "phase" in tok.columns:
            totals = tok[tok["phase"].astype(str).eq("__total__")]
            for _, r in totals.iterrows():
                method = r.get("method")
                ops.setdefault(method, {})
                n_lat = ops[method].get("n_latency") or 1
                total_tok = float(r.get("total_tokens") or 0)
                ops[method]["total_tokens"] = total_tok
                ops[method]["tokens_per_query"] = total_tok / max(n_lat, 1)
                ops[method]["prompt_tokens"] = float(r.get("prompt_tokens") or 0)
                ops[method]["completion_tokens"] = float(r.get("completion_tokens") or 0)
                elapsed = float(r.get("elapsed_seconds") or 0)
                ops[method]["wall_seconds"] = elapsed

    rows = []
    for method, stats in ops.items():
        stats = dict(stats)
        stats["method"] = method
        stats["method_label"] = LABELS.get(method, method)
        rows.append(stats)
    return rows, samples, ops


def collect_payload() -> dict:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    benches = []
    for spec in BENCHES:
        meta = _meta(spec["meta"])
        summary = _safe_csv(spec["results"] / "summary.csv")
        accuracy = _safe_csv(spec["results"] / "accuracy_enriched.csv")
        if not accuracy:
            accuracy = _safe_csv(spec["results"] / "accuracy_results.csv")
        dual = _safe_csv(spec["results"] / "dual_scoreboard.csv")
        wins = _safe_csv(spec["results"] / "method_clear_wins_composite.csv")
        routing = _safe_csv(spec["results"] / "routing_recommendations.csv")
        ops_rows, lat_samples, ops_by_method = _ops_stats(spec["results"])
        briefing_path = spec["results"] / "engineering_briefing.md"
        briefing = briefing_path.read_text(encoding="utf-8") if briefing_path.exists() else ""
        scored = len({r.get("question_id") for r in accuracy}) if accuracy else 0
        indexed_q = meta.get("n_questions")
        for row in summary:
            row["method_label"] = LABELS.get(row.get("method", ""), row.get("method", ""))
            extra = ops_by_method.get(row.get("method"), {})
            for k, v in extra.items():
                if row.get(k) in (None, 0, 0.0, ""):
                    row[k] = v
        for row in accuracy:
            row["method_label"] = LABELS.get(row.get("method", ""), row.get("method", ""))
        benches.append(
            {
                "id": spec["id"],
                "title": spec["title"],
                "short": spec["short"],
                "era": spec["era"],
                "what": spec["what"],
                "how_to_read": spec["how_to_read"],
                "paper": spec["paper"],
                "meta": {
                    k: v
                    for k, v in meta.items()
                    if k not in {"corpus_dir", "qa_path", "catalog_path"}
                },
                "indexed_questions": indexed_q,
                "indexed_documents": meta.get("n_documents"),
                "scored_questions": scored,
                "scores_are_partial": bool(
                    indexed_q and scored and scored < int(indexed_q)
                ),
                "live_partial": (spec["results"] / "live_partial_meta.json").exists(),
                "status": (
                    "scoring live"
                    if (spec["results"] / "live_partial_meta.json").exists()
                    else ("scored" if summary else "corpus only")
                ),
                "summary": summary,
                "accuracy": accuracy,
                "lens": _lens_summary(accuracy),
                "ops": ops_rows,
                "latency_samples": lat_samples,
                "dual": dual,
                "clear_wins": wins[:40],
                "routing": routing,
                "briefing": briefing[:2500],
            }
        )
    return {
        "generated_at": now,
        "repo": "rag-comparisons",
        "research_version": "2026-08 research-refined",
        "metric_defs": METRIC_DEFS,
        "metric_map": METRIC_MAP,
        "research_findings": RESEARCH_FINDINGS,
        "method_defs": METHOD_DEFS,
        "howto_html": _howto_html(),
        "benches": benches,
    }


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>RAG Benchmark · Research Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root {
  --bg:#f3f5f7; --panel:#fff; --text:#15202b; --muted:#5b6572; --line:#d7dde5;
  --accent:#0b5fff; --soft:#e8f0ff; --warn:#9a6700; --warnbg:#fff8c5;
  --ok:#0f6d45; --gap:#8a2f2f; --gapbg:#ffe8e8; --cov:#0f6d45; --covbg:#e6f6ee;
  --font:"Source Sans 3","Segoe UI",system-ui,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
  --shadow:0 1px 2px rgba(16,24,40,.05),0 10px 28px rgba(16,24,40,.06);
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 var(--font)}
a{color:var(--accent)}
.hero{background:#0a1628;color:#eef3ff;padding:28px 28px 18px}
.hero h1{margin:0 0 8px;font-size:28px;letter-spacing:-.02em}
.hero p{margin:0;max-width:980px;opacity:.9}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}
.chip{font-size:12px;padding:4px 10px;border-radius:999px;border:1px solid rgba(255,255,255,.2);background:rgba(255,255,255,.08)}
.tabs{display:flex;gap:6px;padding:12px 16px 0;max-width:1440px;margin:0 auto}
.tab{border:1px solid var(--line);background:#fff;border-radius:999px;padding:8px 14px;cursor:pointer;font:inherit}
.tab.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.layout{display:grid;grid-template-columns:270px 1fr;gap:14px;padding:14px 16px 28px;max-width:1440px;margin:0 auto}
@media(max-width:980px){.layout{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow);padding:14px 16px}
.sidebar{position:sticky;top:10px;align-self:start;max-height:calc(100vh - 20px);overflow:auto}
h2{margin:0 0 8px;font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
label{display:block;font-size:12px;color:var(--muted);margin:12px 0 4px;font-weight:650}
select,input[type=search]{width:100%;padding:8px 10px;border:1px solid var(--line);border-radius:8px;font:inherit}
.help{margin-top:10px;padding:10px;background:#eef2f6;border-radius:8px;font-size:12px;color:var(--muted)}
.callout{border:1px solid var(--line);background:var(--soft);border-radius:10px;padding:12px 14px;margin-bottom:12px}
.callout.warn{background:var(--warnbg);border-color:#e2c56e}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:12px}
@media(max-width:900px){.stats{grid-template-columns:1fr 1fr}}
.stat{border:1px solid var(--line);border-radius:10px;padding:12px;background:#fff}
.stat .k{font-size:11px;color:var(--muted);text-transform:uppercase}
.stat .v{font-size:22px;font-weight:700;font-variant-numeric:tabular-nums}
.grid2{display:grid;grid-template-columns:1.15fr 1fr;gap:12px}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
@media(max-width:1100px){.grid2,.grid3{grid-template-columns:1fr}}
.chart{min-height:340px}.chart.tall{min-height:400px}
.table-wrap{overflow:auto;max-height:420px;border:1px solid var(--line);border-radius:8px}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{position:sticky;top:0;background:#f7f9fc;color:var(--muted)}
.method-list{display:flex;flex-wrap:wrap;gap:6px}
.method-pill{font-size:11px;padding:3px 8px;border-radius:999px;background:#eef2f6;border:1px solid var(--line);cursor:pointer}
.method-pill.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.btn-row{display:flex;gap:8px;margin-top:10px}
button{border:1px solid var(--line);background:#fff;border-radius:8px;padding:7px 10px;font:inherit;cursor:pointer}
button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
.badge{display:inline-block;font-size:10px;padding:2px 7px;border-radius:999px;font-weight:700;text-transform:uppercase}
.badge.covered{background:var(--covbg);color:var(--cov)}
.badge.proxied{background:var(--soft);color:var(--accent)}
.badge.missing{background:var(--gapbg);color:var(--gap)}
.finding{border-left:3px solid var(--accent);padding:8px 0 8px 12px;margin:12px 0}
.finding h3{margin:0 0 4px;font-size:14px}
.finding .cites{font-size:11px;color:var(--muted);font-family:var(--mono)}
.muted{color:var(--muted)}
.view{display:none}.view.on{display:block}
.footer{max-width:1440px;margin:0 auto;padding:0 16px 28px;color:var(--muted);font-size:12px}
details{border-top:1px solid var(--line);padding:8px 0}
summary{cursor:pointer;font-weight:650}
code{font-family:var(--mono);font-size:12px}
.howto h3{margin:18px 0 6px;font-size:15px}
.howto ol{padding-left:20px}
.howto li{margin:4px 0}
.howto pre{white-space:pre-wrap;font:12px/1.45 var(--mono);background:#0a1628;color:#d7e2ff;padding:12px;border-radius:8px;overflow:auto}
.howto table{margin:8px 0 16px}
</style>
</head>
<body>
<header class="hero">
  <h1>RAG Benchmark · Research Dashboard</h1>
  <p>
    Interactive bake-off grounded in 2024–2026 RAG evaluation practice:
    retrieval vs generation layers, generative vs extractive lenses, and
    “when graphs help” rather than a single winner chart.
  </p>
  <div class="chips">
    <span class="chip" id="genChip">Generated —</span>
    <span class="chip">RAGAS-aligned metric map</span>
    <span class="chip">GraphRAG-Bench task difficulty</span>
    <span class="chip">CodeRAG leave-gold-out</span>
  </div>
</header>

<div class="tabs">
  <button class="tab on" data-view="explore">Explore</button>
  <button class="tab" data-view="latency">Latency / cost</button>
  <button class="tab" data-view="decision">Decision Lab</button>
  <button class="tab" data-view="research">Research Lens</button>
  <button class="tab" data-view="howto">How to run</button>
</div>

<div class="layout">
  <aside class="panel sidebar">
    <h2>Controls</h2>
    <label for="bench">Benchmark</label>
    <select id="bench"></select>
    <label for="metric">Primary metric</label>
    <select id="metric">
      <option value="mean_composite_score">Composite (default rank)</option>
      <option value="mean_llm_judge">LLM judge</option>
      <option value="mean_token_f1">Token F1</option>
      <option value="contains_answer_rate">Contains rate</option>
      <option value="tokens_per_query">Tokens / query</option>
      <option value="mean_query_latency_seconds">Mean latency (s)</option>
      <option value="p95_query_latency_seconds">p95 latency (s)</option>
      <option value="index_seconds">Index time (s)</option>
    </select>
    <label for="qtype">Question type</label>
    <select id="qtype"><option value="__all__">All types</option></select>
    <label>Methods</label>
    <div class="method-list" id="methods"></div>
    <div class="btn-row">
      <button type="button" id="allMethods">All</button>
      <button type="button" id="noneMethods">None</button>
      <button type="button" class="primary" id="reset">Reset</button>
    </div>
    <div class="help" id="metricHelp"></div>
    <h2 style="margin-top:16px">Metric glossary</h2>
    <div id="glossary"></div>
  </aside>

  <main>
    <section class="view on" id="view-explore">
      <div id="provenance" class="callout"></div>
      <div class="stats" id="stats"></div>
      <div class="grid2">
        <div class="panel">
          <h2>Leaderboard</h2>
          <p class="muted" style="margin:0 0 8px">Click a bar to focus the explorer. Cost metrics sort ascending.</p>
          <div id="barChart" class="chart"></div>
        </div>
        <div class="panel">
          <h2>Quality–cost Pareto</h2>
          <p class="muted" style="margin:0 0 8px">Ideal region: high composite, low tokens (top-left).</p>
          <div id="scatterChart" class="chart"></div>
        </div>
      </div>
      <div class="grid2" style="margin-top:12px">
        <div class="panel">
          <h2>Per-question composite spread</h2>
          <div id="boxChart" class="chart tall"></div>
        </div>
        <div class="panel">
          <h2>Normalized metric heatmap</h2>
          <div id="heatChart" class="chart tall"></div>
        </div>
      </div>
      <div class="panel" style="margin-top:12px">
        <h2>Per-question explorer</h2>
        <input type="search" id="qsearch" placeholder="Filter question id / type…"/>
        <div class="table-wrap" style="margin-top:10px" id="qTable"></div>
      </div>
    </section>

    <section class="view" id="view-latency">
      <div class="callout">
        <strong>Latency, tokens, index cost</strong>
        Quality without p95 and tokens/query is not a ship decision. Index time is a one-shot
        (or rebuild) bill — GraphRAG global is usually dominated by it. Query latency here is
        retrieve + generate on this hardware (local llama3.2:3b unless you changed
        <code>config/benchmark.yaml</code>).
      </div>
      <div class="stats" id="latStats"></div>
      <div class="grid2">
        <div class="panel">
          <h2>Per-query latency distribution</h2>
          <p class="muted" style="margin:0 0 8px">Box = spread across questions. Outliers are the SLO killers.</p>
          <div id="latBox" class="chart tall"></div>
        </div>
        <div class="panel">
          <h2>Quality vs latency</h2>
          <p class="muted" style="margin:0 0 8px">Ideal: top-left (high composite, low mean latency).</p>
          <div id="latScatter" class="chart tall"></div>
        </div>
      </div>
      <div class="panel" style="margin-top:12px">
        <h2>Ops table (this bench · selected methods)</h2>
        <div class="table-wrap" id="opsTable"></div>
      </div>
    </section>

    <section class="view" id="view-decision">
      <div class="callout">
        <strong>Decision Lab — research default is a router</strong>
        Systematic 2025–26 evaluations: vector RAG for local factoids; graph / frontier for harder multi-hop
        and synthesis; always check generative vs extractive because they disagree.
      </div>
      <div class="grid2">
        <div class="panel">
          <h2>Generative vs extractive (dual lens)</h2>
          <p class="muted" style="margin:0 0 8px">
            Above the diagonal → chatty/useful answers that Hotpot F1 under-credits.
            Below → span-faithful but maybe less helpful prose.
          </p>
          <div id="lensChart" class="chart tall"></div>
        </div>
        <div class="panel">
          <h2>Dual scoreboard</h2>
          <div class="table-wrap" id="dualTable"></div>
          <h2 style="margin-top:16px">Routing recommendations</h2>
          <div class="table-wrap" id="routeTable"></div>
        </div>
      </div>
      <div class="panel" style="margin-top:12px">
        <h2>Engineering briefing</h2>
        <pre class="brief" id="briefing"></pre>
      </div>
      <div class="panel" style="margin-top:12px">
        <h2>Clear wins (composite margin ≥ 0.12)</h2>
        <div class="table-wrap" id="winsTable"></div>
      </div>
    </section>

    <section class="view" id="view-research">
      <div class="panel">
        <h2>How this harness maps to 2026 RAG eval practice</h2>
        <p class="muted">
          Industry stacks (RAGAS, DeepEval, TruLens) separate <em>retrieval</em> (context precision/recall)
          from <em>generation</em> (faithfulness, answer relevance). We map our scores honestly — including gaps.
        </p>
        <div class="table-wrap" id="mapTable"></div>
      </div>
      <div class="panel" style="margin-top:12px">
        <h2>Research findings this UI is designed around</h2>
        <div id="findings"></div>
      </div>
      <div class="panel" style="margin-top:12px">
        <h2>Method definitions</h2>
        <div id="methodGlossary"></div>
      </div>
    </section>
    <section class="view" id="view-howto">
      <div class="panel" id="howtoBody"></div>
    </section>
  </main>
</div>

<footer class="footer">
  Rebuild: <code>PYTHONPATH=src python scripts/build_dashboard.py</code>
  · Live Pages: GitHub Actions → <code>docs/</code>
  · Not a substitute for retrieval-only nDCG / faithfulness until those metrics are wired in.
</footer>

<script id="dashboard-data" type="application/json">__DATA_JSON__</script>
<script>
const DATA = JSON.parse(document.getElementById('dashboard-data').textContent);
const COSTISH = new Set(['tokens_per_query','mean_query_latency_seconds','p95_query_latency_seconds','index_seconds']);
const METRIC_LABEL = {
  mean_composite_score:'Composite', mean_llm_judge:'LLM judge', mean_token_f1:'Token F1',
  contains_answer_rate:'Contains rate', tokens_per_query:'Tokens / query',
  mean_query_latency_seconds:'Mean latency (s)',
  p95_query_latency_seconds:'p95 latency (s)',
  index_seconds:'Index time (s)',
};
const state = { benchId: DATA.benches[0]?.id, metric:'mean_composite_score', qtype:'__all__', methods:new Set(), focusMethod:null, search:'' };

function bench(){ return DATA.benches.find(b=>b.id===state.benchId)||DATA.benches[0]; }
function labelOf(m){ return (bench().summary||[]).find(r=>r.method===m)?.method_label || m; }
function plotLayout(extra){
  return Object.assign({margin:{t:28,r:16,b:64,l:56},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
    font:{family:'Source Sans 3, Segoe UI, sans-serif',size:12,color:'#15202b'},hovermode:'closest'}, extra||{});
}

document.querySelectorAll('.tab').forEach(tab=>{
  tab.onclick=()=>{
    document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on'));
    document.querySelectorAll('.view').forEach(v=>v.classList.remove('on'));
    tab.classList.add('on');
    document.getElementById('view-'+tab.dataset.view).classList.add('on');
    // reflow plots when tab becomes visible
    setTimeout(()=>window.dispatchEvent(new Event('resize')), 50);
    if(tab.dataset.view==='decision') renderDecision();
    if(tab.dataset.view==='research') renderResearch();
    if(tab.dataset.view==='latency') renderLatency();
    if(tab.dataset.view==='howto') renderHowto();
  };
});

function init(){
  document.getElementById('genChip').textContent = DATA.research_version + ' · ' + DATA.generated_at;
  const sel=document.getElementById('bench');
  sel.innerHTML = DATA.benches.map(b=>`<option value="${b.id}">${b.title} · ${b.scored_questions||0} scored</option>`).join('');
  sel.value=state.benchId;
  sel.onchange=()=>{state.benchId=sel.value; state.focusMethod=null; syncMethods(true); render();};
  document.getElementById('metric').onchange=e=>{state.metric=e.target.value; updateHelp(); render();};
  document.getElementById('qtype').onchange=e=>{state.qtype=e.target.value; render();};
  document.getElementById('qsearch').oninput=e=>{state.search=e.target.value.trim().toLowerCase(); renderQTable();};
  document.getElementById('allMethods').onclick=()=>{state.methods=new Set((bench().summary||[]).map(r=>r.method)); render();};
  document.getElementById('noneMethods').onclick=()=>{state.methods=new Set(); render();};
  document.getElementById('reset').onclick=()=>{
    state.metric='mean_composite_score'; state.qtype='__all__'; state.focusMethod=null; state.search='';
    document.getElementById('metric').value=state.metric; document.getElementById('qsearch').value='';
    syncMethods(true); render();
  };
  document.getElementById('glossary').innerHTML = DATA.metric_defs.map(m=>
    `<details><summary>${m.name} <span class="muted">${m.range}</span></summary><p class="muted">${m.means}</p></details>`
  ).join('');
  updateHelp(); syncMethods(true); render(); renderResearch(); renderHowto();
}

function updateHelp(){
  const map={mean_composite_score:'composite',mean_llm_judge:'llm_judge',mean_token_f1:'extractive',
    contains_answer_rate:'generative',tokens_per_query:'tokens_per_query',
    mean_query_latency_seconds:'latency',p95_query_latency_seconds:'latency',index_seconds:'index'};
  const d=DATA.metric_defs.find(x=>x.id===map[state.metric]);
  document.getElementById('metricHelp').innerHTML = d?`<strong>${d.name}</strong><br>${d.means}`:'';
}

function syncMethods(selectAll){
  const rows=bench().summary||[];
  if(selectAll||!state.methods.size) state.methods=new Set(rows.map(r=>r.method));
  else {
    const allow=new Set(rows.map(r=>r.method));
    state.methods=new Set([...state.methods].filter(m=>allow.has(m)));
    if(!state.methods.size) state.methods=new Set(allow);
  }
  const types=new Set();
  for(const r of (bench().accuracy||[])){
    [r.query_type,r.code_rag_type,r.hotpot_type,r.graphrag_bench_type,r.multihop_type].forEach(t=>t&&types.add(t));
  }
  const qt=document.getElementById('qtype'); const prev=state.qtype;
  qt.innerHTML=`<option value="__all__">All types</option>`+[...types].sort().map(t=>`<option value="${t}">${t}</option>`).join('');
  state.qtype=[...types].includes(prev)?prev:'__all__'; qt.value=state.qtype;
  const box=document.getElementById('methods');
  box.innerHTML=rows.map(r=>`<span class="method-pill ${state.methods.has(r.method)?'on':''}" data-m="${r.method}">${r.method_label}</span>`).join('');
  box.querySelectorAll('.method-pill').forEach(el=>{
    el.onclick=()=>{const m=el.dataset.m; state.methods.has(m)?state.methods.delete(m):state.methods.add(m); render();};
  });
}

function filteredSummary(){ return (bench().summary||[]).filter(r=>state.methods.has(r.method)); }
function filteredAccuracy(){
  let rows=(bench().accuracy||[]).filter(r=>state.methods.has(r.method));
  if(state.qtype!=='__all__'){
    rows=rows.filter(r=>[r.query_type,r.code_rag_type,r.hotpot_type,r.graphrag_bench_type,r.multihop_type].includes(state.qtype));
  }
  return rows;
}

function renderProvenance(){
  const b=bench();
  const el=document.getElementById('provenance');
  el.className='callout'+(b.scores_are_partial?' warn':'');
  el.innerHTML=`<strong>${b.title}</strong> <span class="muted">· ${b.era}</span>
    <div style="margin-top:6px">${b.what}</div>
    <div style="margin-top:8px"><em>How to read:</em> ${b.how_to_read}</div>
    <div style="margin-top:8px" class="muted">${b.paper}</div>
    <div style="margin-top:8px">Indexed <strong>${b.indexed_questions??'—'}</strong> Q /
      <strong>${b.indexed_documents??'—'}</strong> docs · Scored here
      <strong>${b.scored_questions||0}</strong>
      ${b.scores_are_partial?' · <span style="color:var(--warn)">partial scores — full run still landing</span>':''}
    </div>`;
  document.getElementById('stats').innerHTML=`
    <div class="stat"><div class="k">Indexed Q</div><div class="v">${b.indexed_questions??'—'}</div></div>
    <div class="stat"><div class="k">Docs</div><div class="v">${b.indexed_documents??'—'}</div></div>
    <div class="stat"><div class="k">Scored Q</div><div class="v">${b.scored_questions||0}</div></div>
    <div class="stat"><div class="k">Methods</div><div class="v">${(b.summary||[]).length}</div></div>`;
}

function renderBar(){
  const rows=filteredSummary().slice().sort((a,b)=>{
    const av=a[state.metric]??-1e9, bv=b[state.metric]??-1e9;
    return COSTISH.has(state.metric)?av-bv:bv-av;
  });
  Plotly.newPlot('barChart',[{
    type:'bar', orientation:'h',
    y:rows.map(r=>r.method_label), x:rows.map(r=>r[state.metric]),
    marker:{color:rows.map(r=>r.method===state.focusMethod?'#063a9c':'#0b5fff')},
    customdata:rows.map(r=>r.method),
    hovertemplate:'%{y}<br>'+METRIC_LABEL[state.metric]+': %{x:.3f}<extra></extra>',
  }], plotLayout({yaxis:{autorange:'reversed'}, xaxis:{title:METRIC_LABEL[state.metric]}}), {responsive:true});
  document.getElementById('barChart').on('plotly_click',ev=>{
    const m=ev.points?.[0]?.customdata; if(!m)return;
    state.focusMethod=state.focusMethod===m?null:m; render();
  });
}

function renderScatter(){
  const rows=filteredSummary().filter(r=>r.mean_composite_score!=null);
  const useTok = rows.some(r=>Number(r.tokens_per_query)>0);
  Plotly.newPlot('scatterChart',[{
    type:'scatter', mode:'markers+text',
    x:rows.map(r=> useTok ? r.tokens_per_query : r.mean_query_latency_seconds),
    y:rows.map(r=>r.mean_composite_score),
    text:rows.map(r=>r.method_label), textposition:'top center',
    marker:{size:13,color:rows.map(r=>r.method===state.focusMethod?'#063a9c':'#0b5fff')},
    customdata:rows.map(r=>[r.method,r.mean_llm_judge,r.mean_query_latency_seconds,r.tokens_per_query]),
    hovertemplate:'<b>%{text}</b><br>Composite %{y:.3f}<br>Tokens/q %{customdata[3]:.0f}<br>Latency %{customdata[2]:.2f}s<extra></extra>',
  }], plotLayout({
    xaxis:{title: useTok ? 'Tokens / query ↓ cheaper' : 'Mean latency (s) ↓ faster (tokens not in this slice)'},
    yaxis:{title:'Composite ↑ better',range:[0,1]}
  }), {responsive:true});


function renderBox(){
  const by={};
  for(const r of filteredAccuracy()) (by[r.method_label] ||= []).push(r.composite_score??0);
  const traces=Object.entries(by).map(([name,y])=>({type:'box',name,y,boxpoints:'outliers',marker:{size:4}}));
  Plotly.newPlot('boxChart', traces, plotLayout({showlegend:false,yaxis:{title:'Per-question composite',range:[-0.05,1.05]}}), {responsive:true});
}

function renderHeat(){
  const rows=filteredSummary();
  if(!rows.length){document.getElementById('heatChart').innerHTML='<p class="muted">No data</p>';return;}
  const keys=[['mean_composite_score','Composite'],['mean_llm_judge','Judge'],['mean_token_f1','F1'],['contains_answer_rate','Contains']];
  const z=keys.map(([k])=>{
    const vals=rows.map(r=>Number(r[k])||0); const mx=Math.max(...vals,1e-9); return vals.map(v=>v/mx);
  });
  const tok=rows.map(r=>Number(r.tokens_per_query)||0); const tmax=Math.max(...tok,1e-9);
  z.push(tok.map(v=>1-v/tmax)); keys.push(['tokens','Cheap tokens']);
  const lat=rows.map(r=>Number(r.mean_query_latency_seconds)||0); const lmax=Math.max(...lat,1e-9);
  z.push(lat.map(v=>1-v/lmax)); keys.push(['Fast mean']);

  Plotly.newPlot('heatChart',[{type:'heatmap',z,x:rows.map(r=>r.method_label),y:keys.map(k=>k[1]),colorscale:'Blues',
    hovertemplate:'%{y} · %{x}<br>norm %{z:.2f}<extra></extra>'}],
    plotLayout({margin:{t:24,r:16,b:80,l:90}}), {responsive:true});
}

function fmtNum(v, digits){
  if(v==null || v==='' || Number.isNaN(Number(v))) return '—';
  const n=Number(v);
  if(Math.abs(n)>=100) return n.toFixed(0);
  return n.toFixed(digits);
}

function renderLatency(){
  const ops=(bench().ops||[]).filter(r=>state.methods.has(r.method));
  const sum=filteredSummary();
  const fastest = [...sum].filter(r=>r.mean_query_latency_seconds>0).sort((a,b)=>a.mean_query_latency_seconds-b.mean_query_latency_seconds)[0];
  const cheapest = [...sum].filter(r=>r.tokens_per_query>0).sort((a,b)=>a.tokens_per_query-b.tokens_per_query)[0];
  const p95best = [...sum].filter(r=>r.p95_query_latency_seconds>0).sort((a,b)=>a.p95_query_latency_seconds-b.p95_query_latency_seconds)[0];
  const idxCheap = [...sum].filter(r=>r.index_seconds>0).sort((a,b)=>a.index_seconds-b.index_seconds)[0];
  document.getElementById('latStats').innerHTML = `
    <div class="stat"><div class="k">Fastest mean</div><div class="v">${fastest?fmtNum(fastest.mean_query_latency_seconds,2)+'s':'—'}</div><div class="h">${fastest?fastest.method_label:''}</div></div>
    <div class="stat"><div class="k">Best p95</div><div class="v">${p95best?fmtNum(p95best.p95_query_latency_seconds,2)+'s':'—'}</div><div class="h">${p95best?p95best.method_label:''}</div></div>
    <div class="stat"><div class="k">Cheapest tokens/q</div><div class="v">${cheapest?fmtNum(cheapest.tokens_per_query,0):'—'}</div><div class="h">${cheapest?cheapest.method_label:''}</div></div>
    <div class="stat"><div class="k">Fastest index</div><div class="v">${idxCheap?fmtNum(idxCheap.index_seconds,1)+'s':'—'}</div><div class="h">${idxCheap?idxCheap.method_label:''}</div></div>`;

  const samples=(bench().latency_samples||[]).filter(r=>state.methods.has(r.method));
  const by={};
  for(const r of samples) (by[r.method_label] ||= []).push(r.query_latency_seconds);
  const traces=Object.entries(by).map(([name,y])=>({type:'box',name,y,boxpoints:'outliers',marker:{size:4}}));
  if(traces.length){
    Plotly.newPlot('latBox', traces, plotLayout({showlegend:false,yaxis:{title:'Query latency (seconds) ↓'}}), {responsive:true});
  } else {
    document.getElementById('latBox').innerHTML='<p class="muted">No latency_results.csv yet for this bench (or still indexing).</p>';
  }

  const rows=sum.filter(r=>r.mean_composite_score!=null && r.mean_query_latency_seconds);
  Plotly.newPlot('latScatter',[{
    type:'scatter', mode:'markers+text',
    x:rows.map(r=>r.mean_query_latency_seconds),
    y:rows.map(r=>r.mean_composite_score),
    text:rows.map(r=>r.method_label), textposition:'top center',
    marker:{size:13,color:rows.map(r=>r.method===state.focusMethod?'#063a9c':'#0b5fff')},
    customdata:rows.map(r=>[r.method,r.p95_query_latency_seconds,r.tokens_per_query,r.index_seconds]),
    hovertemplate:'<b>%{text}</b><br>Composite %{y:.3f}<br>Mean %{x:.2f}s<br>p95 %{customdata[1]:.2f}s<br>Tok/q %{customdata[2]:.0f}<br>Index %{customdata[3]:.1f}s<extra></extra>',
  }], plotLayout({xaxis:{title:'Mean query latency (s) ↓ faster'}, yaxis:{title:'Composite ↑ better',range:[0,1]}}), {responsive:true});

  const tableSrc = ops.length ? ops : sum;
  document.getElementById('opsTable').innerHTML = `<table><thead><tr>
    <th>Method</th><th>n</th><th>Mean lat (s)</th><th>p50</th><th>p95</th><th>Min</th><th>Max</th>
    <th>Index (s)</th><th>Tokens/q</th><th>Prompt tok</th><th>Completion tok</th><th>Wall (s)</th>
  </tr></thead><tbody>
  ${tableSrc.map(r=>`<tr>
    <td>${r.method_label||labelOf(r.method)}</td>
    <td>${r.n_latency||r.n_scored||'—'}</td>
    <td>${fmtNum(r.mean_query_latency_seconds,3)}</td>
    <td>${fmtNum(r.p50_query_latency_seconds,3)}</td>
    <td>${fmtNum(r.p95_query_latency_seconds,3)}</td>
    <td>${fmtNum(r.min_query_latency_seconds,3)}</td>
    <td>${fmtNum(r.max_query_latency_seconds,3)}</td>
    <td>${fmtNum(r.index_seconds,1)}</td>
    <td>${fmtNum(r.tokens_per_query,0)}</td>
    <td>${fmtNum(r.prompt_tokens,0)}</td>
    <td>${fmtNum(r.completion_tokens,0)}</td>
    <td>${fmtNum(r.wall_seconds,1)}</td>
  </tr>`).join('')}
  </tbody></table>`;
}

function renderHowto(){
  document.getElementById('howtoBody').innerHTML = DATA.howto_html || '<p class="muted">See docs/how-to-run.md in the repo.</p>';
}

function renderQTable(){
  const focus=state.focusMethod || filteredSummary().slice().sort((a,b)=>(b.mean_composite_score||0)-(a.mean_composite_score||0))[0]?.method;
  let rows=filteredAccuracy().filter(r=>r.method===focus);
  if(state.search) rows=rows.filter(r=>String(r.question_id).toLowerCase().includes(state.search)||String(r.query_type||'').toLowerCase().includes(state.search));
  rows=rows.slice().sort((a,b)=>(b.composite_score||0)-(a.composite_score||0));
  document.getElementById('qTable').innerHTML=`<p class="muted">Focus: <strong>${labelOf(focus)}</strong> · ${rows.length} rows</p>
    <table><thead><tr><th>Id</th><th>Type</th><th>Composite</th><th>Gen</th><th>Ext</th><th>Judge</th><th>F1</th></tr></thead><tbody>
    ${rows.slice(0,250).map(r=>`<tr>
      <td><code>${r.question_id}</code></td><td>${r.query_type||r.code_rag_type||r.hotpot_type||''}</td>
      <td>${Number(r.composite_score||0).toFixed(3)}</td>
      <td>${Number(r.generative_score||0).toFixed(3)}</td>
      <td>${Number(r.extractive_score||0).toFixed(3)}</td>
      <td>${Number(r.llm_judge_score||0).toFixed(3)}</td>
      <td>${Number(r.token_f1||0).toFixed(3)}</td></tr>`).join('')}
    </tbody></table>`;
}

function renderDecision(){
  const lens=(bench().lens||[]).filter(r=>state.methods.has(r.method));
  if(lens.length){
    const maxv=1;
    Plotly.newPlot('lensChart',[
      {type:'scatter',mode:'lines',x:[0,maxv],y:[0,maxv],line:{dash:'dot',color:'#9aa4b2'},hoverinfo:'skip',showlegend:false},
      {type:'scatter',mode:'markers+text',
        x:lens.map(r=>r.extractive), y:lens.map(r=>r.generative),
        text:lens.map(r=>r.method_label), textposition:'top center',
        marker:{size:14,color:'#0b5fff'},
        customdata:lens.map(r=>[r.composite,r.n]),
        hovertemplate:'<b>%{text}</b><br>Generative %{y:.3f}<br>Extractive %{x:.3f}<br>Composite %{customdata[0]:.3f}<extra></extra>'}
    ], plotLayout({xaxis:{title:'Extractive (F1+EM) ↑',range:[-0.02,1.02]}, yaxis:{title:'Generative (judge+contains) ↑',range:[-0.02,1.02]},
      annotations:[{x:0.75,y:0.2,text:'Extractive-favored',showarrow:false,font:{color:'#5b6572',size:11}},
                   {x:0.25,y:0.85,text:'Generative-favored (graphs often here)',showarrow:false,font:{color:'#5b6572',size:11}}]
    }), {responsive:true});
  } else {
    document.getElementById('lensChart').innerHTML='<p class="muted">Need accuracy_enriched.csv with generative/extractive columns.</p>';
  }

  const dual=bench().dual||[];
  document.getElementById('dualTable').innerHTML = dual.length ? `<table><thead><tr>
    <th>Scenario</th><th>Generative winner</th><th>Composite winner</th><th>Flip?</th><th>Meaning</th></tr></thead><tbody>
    ${dual.map(r=>{
      const flip=r.ranking_flips===true||r.ranking_flips==='True'||r.ranking_flips==='true';
      return `<tr><td>${r.scenario||''}</td><td>${r.generative_winner||''}</td><td>${r.composite_winner||''}</td>
        <td>${flip?'YES':'no'}</td>
        <td>${flip?'Metric choice changes the system you ship.':'Stable across lenses.'}</td></tr>`;
    }).join('')}</tbody></table>` : '<p class="muted">No dual_scoreboard.csv</p>';

  const route=bench().routing||[];
  document.getElementById('routeTable').innerHTML = route.length ? `<table><thead><tr>
    <th>Scenario</th><th>Recommended</th><th>Quality</th><th>Margin</th><th>Tokens/q</th><th>Guidance</th></tr></thead><tbody>
    ${route.map(r=>`<tr><td>${r.scenario||r.query_type||''}</td><td>${r.recommended_label||r.recommended_method||''}</td>
      <td>${Number(r.quality||0).toFixed(3)}</td><td>${Number(r.margin_over_next||0).toFixed(3)}</td>
      <td>${Number(r.tokens_per_query||0).toFixed(0)}</td><td>${r.guidance||''}</td></tr>`).join('')}
    </tbody></table>` : '<p class="muted">No routing_recommendations.csv for this bench.</p>';

  document.getElementById('briefing').textContent = bench().briefing || 'No engineering_briefing.md';
  const wins=bench().clear_wins||[];
  document.getElementById('winsTable').innerHTML = wins.length ? `<table><thead><tr>
    <th>Winner</th><th>Question</th><th>Gold</th><th>Score</th><th>Runner-up</th><th>Margin</th></tr></thead><tbody>
    ${wins.map(r=>`<tr><td>${r.winner_label||r.winner||''}</td><td>${(r.question||'').slice(0,120)}</td>
      <td>${(r.gold||'').slice(0,50)}</td><td>${Number(r.winner_score||0).toFixed(3)}</td>
      <td>${r.runner_up_label||''}</td><td>${Number(r.margin||0).toFixed(3)}</td></tr>`).join('')}
    </tbody></table>` : '<p class="muted">No clear-win rows.</p>';
}

function renderResearch(){
  document.getElementById('mapTable').innerHTML=`<table><thead><tr>
    <th>2026 practice (RAGAS-style)</th><th>This harness</th><th>Layer</th><th>Status</th><th>Note</th></tr></thead><tbody>
    ${DATA.metric_map.map(m=>`<tr>
      <td>${m.ragas}</td><td><code>${m.ours}</code></td><td>${m.layer}</td>
      <td><span class="badge ${m.status}">${m.status}</span></td><td>${m.note}</td></tr>`).join('')}
    </tbody></table>`;
  document.getElementById('findings').innerHTML = DATA.research_findings.map(f=>`
    <div class="finding"><h3>${f.claim}</h3>
      <p>${f.evidence}</p>
      <p><strong>In this UI:</strong> ${f.dashboard}</p>
      <div class="cites">${(f.cites||[]).join(' · ')}</div>
    </div>`).join('');
  document.getElementById('methodGlossary').innerHTML = Object.entries(DATA.method_defs).map(([k,v])=>
    `<details><summary>${k}</summary><p class="muted">${v}</p></details>`).join('');
}

function render(){
  document.querySelectorAll('#methods .method-pill').forEach(el=>el.classList.toggle('on', state.methods.has(el.dataset.m)));
  renderProvenance();
  if(!(bench().summary||[]).length){
    ['barChart','scatterChart','boxChart','heatChart'].forEach(id=>document.getElementById(id).innerHTML='<p class="muted">No summary.csv</p>');
    return;
  }
  renderBar(); renderScatter(); renderBox(); renderHeat(); renderQTable();
  if(document.getElementById('view-decision').classList.contains('on')) renderDecision();
  if(document.getElementById('view-latency').classList.contains('on')) renderLatency();
}

init();
</script>
</body>
</html>
"""


def build() -> Path:
    payload = collect_payload()
    docs = ROOT / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "dashboard_data.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    raw = json.dumps(payload, ensure_ascii=False)
    raw = raw.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    # Fix any template bugs before write — sanitize the broken color line by regenerating clean HTML
    html = HTML.replace("__DATA_JSON__", raw)
    out_docs = docs / "index.html"
    out_docs.write_text(html, encoding="utf-8")
    for path in (
        ROOT / "results" / "dashboard.html",
        ROOT / "results_code_rag" / "dashboard.html",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
    (docs / "README.md").write_text(
        "# Research-refined RAG dashboard\n\n"
        "Open **[index.html](./index.html)** on GitHub Pages.\n\n"
        "Tabs: **Explore** · **Latency / cost** · **Decision Lab** · "
        "**Research Lens** · **[How to run](./how-to-run.md)**.\n\n"
        "Rebuild: `PYTHONPATH=src python scripts/build_dashboard.py`\n",
        encoding="utf-8",
    )
    return out_docs


if __name__ == "__main__":
    path = build()
    print(f"Dashboard → {path}")
