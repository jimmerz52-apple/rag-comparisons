#!/usr/bin/env python3
"""Build an interactive (Tableau-style) multi-benchmark dashboard for GitHub Pages.

Outputs:
  docs/index.html          — GitHub Pages entry (interactive Plotly)
  results/dashboard.html   — same file for local browsing
  docs/dashboard_data.json — embedded twin for debugging / external tools

Rebuild:
  PYTHONPATH=src python scripts/build_dashboard.py
"""

from __future__ import annotations

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
        "what": (
            "Multi-hop Wikipedia QA (distractor setting). "
            "Each question needs evidence from ≥2 paragraphs; the corpus also "
            "includes distractor paragraphs that look related but are not gold."
        ),
        "how_to_read": (
            "Higher composite is better. Tokens/query is cost. "
            "Graph methods often win on LLM-judge but lose on extractive F1 — "
            "check Dual scoreboard and the Quality vs Tokens chart."
        ),
    },
    {
        "id": "code_rag",
        "title": "CodeRAG-Bench",
        "short": "CodeRAG",
        "results": ROOT / "results_code_rag",
        "meta": ROOT / "data" / "qa" / "code_rag_meta.json",
        "what": (
            "Code generation with retrieval (Wang et al., NAACL Findings 2025). "
            "HumanEval / MBPP / DS-1000 / ODEX. Canonical solutions are "
            "leave-gold-out of the datastore."
        ),
        "how_to_read": (
            "Code-aware metrics: contains/judge look for function bodies and "
            "key tokens, not wiki-style short answers. Prefer composite + "
            "judge together."
        ),
    },
    {
        "id": "graphrag_bench",
        "title": "GraphRAG-Bench Novel",
        "short": "GraphRAG-Bench",
        "results": ROOT / "results_graphrag_bench",
        "meta": ROOT / "data" / "qa" / "graphrag_bench_meta.json",
        "what": (
            "Questions designed to test when graph structure helps "
            "(Novel-4128 Pepys diary corpus)."
        ),
        "how_to_read": (
            "If GraphRAG beats semantic here but not on Hotpot, the win is "
            "task-specific — do not generalize without checking Hotpot."
        ),
    },
    {
        "id": "multihop",
        "title": "MultiHop-RAG",
        "short": "MultiHop",
        "results": ROOT / "results_multihop",
        "meta": ROOT / "data" / "qa" / "multihop_meta.json",
        "what": "News multi-document reasoning; answers often need several articles.",
        "how_to_read": "Compare bridge-style multi-doc questions vs single-hop filters.",
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

METRIC_DEFS = [
    {
        "id": "composite",
        "name": "Composite score",
        "range": "0–1, higher better",
        "means": (
            "Primary ranking metric. Blend of LLM-judge (generative quality) and "
            "token F1 / exact-match style extractive overlap. Use this to pick a "
            "default method unless your product is judge-only or F1-only."
        ),
    },
    {
        "id": "llm_judge",
        "name": "LLM judge",
        "range": "0–1, higher better",
        "means": (
            "A local judge model scores whether the answer is correct / useful "
            "relative to gold. Rewards fluent, correct answers even when wording "
            "differs from gold. Can be generous to chatty graph answers."
        ),
    },
    {
        "id": "token_f1",
        "name": "Token F1",
        "range": "0–1, higher better",
        "means": (
            "Token overlap between prediction and gold (classic Hotpot-style). "
            "Strict: paraphrases score lower. Good for short factual answers."
        ),
    },
    {
        "id": "contains",
        "name": "Contains answer",
        "range": "rate 0–1",
        "means": (
            "Fraction of questions where gold (or key gold tokens / code body) "
            "appears in the prediction. Loose recall-style check."
        ),
    },
    {
        "id": "tokens_per_query",
        "name": "Tokens / query",
        "range": "count, lower cheaper",
        "means": (
            "Average prompt+completion tokens per question. Proxy for $ and "
            "latency. A method with slightly lower quality but far fewer tokens "
            "may still be the right production choice."
        ),
    },
    {
        "id": "latency",
        "name": "Query latency",
        "range": "seconds, lower faster",
        "means": "Mean wall-clock time per question after the index exists.",
    },
    {
        "id": "dual",
        "name": "Dual scoreboard / flips",
        "range": "winner labels",
        "means": (
            "Generative winner (judge-heavy) vs composite winner. A flip means "
            "the metric you optimize changes who wins — do not trust a single "
            "leaderboard number without checking this."
        ),
    },
]

METHOD_DEFS = {
    "Semantic": "Dense vector retrieval (embeddings) + LLM generate over top-k chunks.",
    "Rerank": "Semantic retrieve, then cross-encoder / LLM rerank before generate.",
    "BM25+dense": "Hybrid sparse (BM25) + dense retrieval fused before generate.",
    "Hybrid": "Graph + vector hybrid path used in this harness.",
    "GraphRAG fast/basic": "Microsoft GraphRAG-style fast NLP index + basic/local search (not global).",
    "GraphRAG global": "Community-summary / global GraphRAG search over the knowledge graph.",
    "GraphRAG local": "Entity-neighborhood local GraphRAG search.",
    "Frontier": "Adaptive routing + corrective RAG style frontier stack.",
    "Adaptive": "Query router that picks retrieval strategy per question.",
}


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
    # JSON-safe
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


def _scored_n(accuracy: list[dict]) -> int:
    if not accuracy:
        return 0
    return len({r.get("question_id") for r in accuracy})


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
        scored = _scored_n(accuracy)
        indexed_q = meta.get("n_questions")
        for row in summary:
            row["method_label"] = LABELS.get(row.get("method", ""), row.get("method", ""))
        for row in accuracy:
            row["method_label"] = LABELS.get(row.get("method", ""), row.get("method", ""))
        benches.append(
            {
                "id": spec["id"],
                "title": spec["title"],
                "short": spec["short"],
                "what": spec["what"],
                "how_to_read": spec["how_to_read"],
                "meta": {
                    k: v
                    for k, v in meta.items()
                    if k
                    not in {
                        "corpus_dir",
                        "qa_path",
                        "catalog_path",
                    }
                },
                "indexed_questions": indexed_q,
                "indexed_documents": meta.get("n_documents"),
                "scored_questions": scored,
                "scores_are_partial": bool(
                    indexed_q and scored and scored < int(indexed_q)
                ),
                "status": "scored" if summary else "corpus only",
                "summary": summary,
                "accuracy": accuracy,
                "dual": dual,
                "clear_wins": wins[:40],
            }
        )
    return {
        "generated_at": now,
        "repo": "rag-comparisons",
        "metric_defs": METRIC_DEFS,
        "method_defs": METHOD_DEFS,
        "benches": benches,
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>RAG Benchmark Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root {
  --bg: #f4f6f8;
  --panel: #ffffff;
  --text: #1a1d23;
  --muted: #5c6570;
  --line: #d8dee6;
  --accent: #1f6feb;
  --accent-soft: #e8f0fe;
  --ok: #0d7a4f;
  --warn: #9a6700;
  --warn-bg: #fff8c5;
  --danger: #cf222e;
  --chip: #eef1f5;
  --shadow: 0 1px 2px rgba(16,24,40,.06), 0 8px 24px rgba(16,24,40,.06);
  --font: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}
* { box-sizing: border-box; }
body {
  margin: 0; color: var(--text); background: var(--bg);
  font: 14px/1.5 var(--font);
}
a { color: var(--accent); }
.top {
  background: linear-gradient(180deg, #0b1f3a 0%, #123056 100%);
  color: #eef3ff; padding: 28px 28px 22px;
}
.top h1 { margin: 0 0 6px; font-size: 26px; font-weight: 650; letter-spacing: -0.02em; }
.top .sub { margin: 0; opacity: .85; max-width: 920px; }
.top .meta-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
.chip {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 10px; border-radius: 999px; font-size: 12px;
  background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.18);
}
.layout {
  display: grid; grid-template-columns: 280px 1fr;
  gap: 16px; padding: 16px; max-width: 1400px; margin: 0 auto;
}
@media (max-width: 960px) {
  .layout { grid-template-columns: 1fr; }
}
.panel {
  background: var(--panel); border: 1px solid var(--line);
  border-radius: 12px; box-shadow: var(--shadow); padding: 14px 16px;
}
.sidebar { position: sticky; top: 12px; align-self: start; max-height: calc(100vh - 24px); overflow: auto; }
.sidebar h2, .main h2 {
  margin: 0 0 8px; font-size: 13px; text-transform: uppercase;
  letter-spacing: .06em; color: var(--muted); font-weight: 700;
}
label { display: block; font-size: 12px; color: var(--muted); margin: 12px 0 4px; font-weight: 600; }
select, input[type="search"] {
  width: 100%; padding: 8px 10px; border-radius: 8px;
  border: 1px solid var(--line); background: #fff; font: inherit;
}
.help {
  font-size: 12px; color: var(--muted); margin-top: 10px;
  padding: 10px; background: var(--chip); border-radius: 8px;
}
.callout {
  border-radius: 10px; padding: 12px 14px; margin-bottom: 14px;
  border: 1px solid var(--line); background: var(--accent-soft);
}
.callout.warn { background: var(--warn-bg); border-color: #e2c56e; }
.callout strong { display: block; margin-bottom: 4px; }
.stats {
  display: grid; grid-template-columns: repeat(4, minmax(0,1fr));
  gap: 10px; margin-bottom: 14px;
}
@media (max-width: 900px) { .stats { grid-template-columns: repeat(2, 1fr); } }
.stat {
  background: var(--panel); border: 1px solid var(--line);
  border-radius: 10px; padding: 12px 14px;
}
.stat .k { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
.stat .v { font-size: 22px; font-weight: 700; margin-top: 2px; font-variant-numeric: tabular-nums; }
.stat .h { font-size: 11px; color: var(--muted); margin-top: 2px; }
.grid2 { display: grid; grid-template-columns: 1.2fr 1fr; gap: 14px; }
.grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }
@media (max-width: 1100px) {
  .grid2, .grid3 { grid-template-columns: 1fr; }
}
.chart { min-height: 340px; }
.chart.tall { min-height: 420px; }
.table-wrap { overflow: auto; max-height: 420px; border: 1px solid var(--line); border-radius: 8px; }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { position: sticky; top: 0; background: #f8fafc; color: var(--muted); font-weight: 700; }
tr:hover td { background: #f5f8ff; cursor: pointer; }
tr.active td { background: var(--accent-soft); }
.glossary details { border-top: 1px solid var(--line); padding: 8px 0; }
.glossary summary { cursor: pointer; font-weight: 600; }
.glossary p { margin: 6px 0 0; color: var(--muted); font-size: 12px; }
.footer {
  max-width: 1400px; margin: 0 auto; padding: 8px 16px 28px;
  color: var(--muted); font-size: 12px;
}
.method-list { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.method-pill {
  font-size: 11px; padding: 3px 8px; border-radius: 999px;
  background: var(--chip); border: 1px solid var(--line); cursor: pointer;
}
.method-pill.on { background: var(--accent); color: #fff; border-color: var(--accent); }
.btn-row { display: flex; gap: 8px; margin-top: 10px; }
button {
  border: 1px solid var(--line); background: #fff; border-radius: 8px;
  padding: 7px 10px; font: inherit; cursor: pointer;
}
button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
.muted { color: var(--muted); }
code { font-family: var(--mono); font-size: 12px; }
</style>
</head>
<body>
<header class="top">
  <h1>RAG Benchmark Dashboard</h1>
  <p class="sub">
    Interactive comparison of retrieval methods (semantic, hybrid, GraphRAG, …)
    across HotpotQA, CodeRAG-Bench, GraphRAG-Bench, and MultiHop-RAG.
    Filter like Tableau: pick a benchmark, metric, and methods — charts and the
    question table update together.
  </p>
  <div class="meta-row">
    <span class="chip" id="genChip">Generated —</span>
    <span class="chip">Higher quality ↑ · Lower tokens/latency ↓</span>
    <span class="chip">GitHub Pages live view</span>
  </div>
</header>

<div class="layout">
  <aside class="panel sidebar">
    <h2>Controls</h2>
    <label for="bench">Benchmark</label>
    <select id="bench"></select>

    <label for="metric">Primary metric (Y / bars)</label>
    <select id="metric">
      <option value="mean_composite_score">Composite (recommended)</option>
      <option value="mean_llm_judge">LLM judge</option>
      <option value="mean_token_f1">Token F1</option>
      <option value="contains_answer_rate">Contains answer rate</option>
      <option value="tokens_per_query">Tokens / query (cost)</option>
      <option value="mean_query_latency_seconds">Mean latency (s)</option>
    </select>

    <label for="qtype">Question type filter</label>
    <select id="qtype"><option value="__all__">All types</option></select>

    <label>Methods (click to toggle)</label>
    <div class="method-list" id="methods"></div>
    <div class="btn-row">
      <button type="button" id="allMethods">All</button>
      <button type="button" id="noneMethods">None</button>
      <button type="button" class="primary" id="reset">Reset</button>
    </div>

    <div class="help" id="metricHelp"></div>

    <h2 style="margin-top:18px">What the metrics mean</h2>
    <div class="glossary" id="glossary"></div>

    <h2 style="margin-top:18px">Methods</h2>
    <div class="glossary" id="methodGlossary"></div>
  </aside>

  <main class="main">
    <div id="provenance" class="callout"></div>
    <div class="stats" id="stats"></div>

    <div class="grid2">
      <div class="panel">
        <h2>Leaderboard</h2>
        <p class="muted" style="margin-top:0">Click a bar to focus that method in the question table.</p>
        <div id="barChart" class="chart"></div>
      </div>
      <div class="panel">
        <h2>Quality vs cost</h2>
        <p class="muted" style="margin-top:0">X = tokens/query (cheaper left). Y = composite. Ideal = top-left.</p>
        <div id="scatterChart" class="chart"></div>
      </div>
    </div>

    <div class="grid2" style="margin-top:14px">
      <div class="panel">
        <h2>Per-question score distribution</h2>
        <p class="muted" style="margin-top:0">Box = spread of composite across questions. Hover for quartiles.</p>
        <div id="boxChart" class="chart tall"></div>
      </div>
      <div class="panel">
        <h2>Metric heatmap</h2>
        <p class="muted" style="margin-top:0">Normalized 0–1 within each column so cost and quality share a scale.</p>
        <div id="heatChart" class="chart tall"></div>
      </div>
    </div>

    <div class="panel" style="margin-top:14px">
      <h2>Dual scoreboard (generative vs composite)</h2>
      <p class="muted" style="margin-top:0">
        If winners differ, optimizing “sounds right” vs “matches gold tokens” picks different systems.
      </p>
      <div class="table-wrap" id="dualTable"></div>
    </div>

    <div class="panel" style="margin-top:14px">
      <h2>Per-question explorer</h2>
      <p class="muted" style="margin-top:0">
        Search or click a row. Sorted by selected method’s composite (desc).
      </p>
      <input type="search" id="qsearch" placeholder="Filter question id / type…"/>
      <div class="table-wrap" style="margin-top:10px" id="qTable"></div>
    </div>

    <div class="panel" style="margin-top:14px">
      <h2>Clear wins (composite margin ≥ 0.12 vs both rivals)</h2>
      <p class="muted" style="margin-top:0">
        Stricter than “highest average.” Empty = no method dominated the core trio on that slice.
      </p>
      <div class="table-wrap" id="winsTable"></div>
    </div>
  </main>
</div>

<footer class="footer">
  Rebuild locally: <code>PYTHONPATH=src python scripts/build_dashboard.py</code>
  · Source data: <code>results*/summary.csv</code> + <code>accuracy_enriched.csv</code>
  · This page is static + Plotly.js (works on GitHub Pages).
</footer>

<script id="dashboard-data" type="application/json">__DATA_JSON__</script>
<script>
const DATA = JSON.parse(document.getElementById('dashboard-data').textContent);
const METRIC_LABEL = {
  mean_composite_score: 'Composite',
  mean_llm_judge: 'LLM judge',
  mean_token_f1: 'Token F1',
  contains_answer_rate: 'Contains rate',
  tokens_per_query: 'Tokens / query',
  mean_query_latency_seconds: 'Latency (s)',
};
const COSTISH = new Set(['tokens_per_query', 'mean_query_latency_seconds']);

const state = {
  benchId: DATA.benches[0]?.id,
  metric: 'mean_composite_score',
  qtype: '__all__',
  methods: new Set(),
  focusMethod: null,
  search: '',
};

function bench() {
  return DATA.benches.find(b => b.id === state.benchId) || DATA.benches[0];
}

function labelOf(method) {
  const row = (bench().summary || []).find(r => r.method === method);
  return row?.method_label || method;
}

function initControls() {
  document.getElementById('genChip').textContent = 'Generated ' + DATA.generated_at;
  const sel = document.getElementById('bench');
  sel.innerHTML = DATA.benches.map(b =>
    `<option value="${b.id}">${b.title} (${b.scored_questions || 0} scored)</option>`
  ).join('');
  sel.value = state.benchId;
  sel.onchange = () => { state.benchId = sel.value; state.focusMethod = null; syncMethods(); render(); };

  document.getElementById('metric').onchange = (e) => {
    state.metric = e.target.value; updateMetricHelp(); render();
  };
  document.getElementById('qtype').onchange = (e) => { state.qtype = e.target.value; render(); };
  document.getElementById('qsearch').oninput = (e) => { state.search = e.target.value.trim().toLowerCase(); renderQuestionTable(); };
  document.getElementById('allMethods').onclick = () => {
    state.methods = new Set((bench().summary || []).map(r => r.method));
    render();
  };
  document.getElementById('noneMethods').onclick = () => { state.methods = new Set(); render(); };
  document.getElementById('reset').onclick = () => {
    state.metric = 'mean_composite_score';
    state.qtype = '__all__';
    state.focusMethod = null;
    state.search = '';
    document.getElementById('metric').value = state.metric;
    document.getElementById('qsearch').value = '';
    syncMethods(true);
    render();
  };

  const g = document.getElementById('glossary');
  g.innerHTML = DATA.metric_defs.map(m =>
    `<details><summary>${m.name} <span class="muted">(${m.range})</span></summary><p>${m.means}</p></details>`
  ).join('');
  const mg = document.getElementById('methodGlossary');
  mg.innerHTML = Object.entries(DATA.method_defs).map(([k,v]) =>
    `<details><summary>${k}</summary><p>${v}</p></details>`
  ).join('');
  updateMetricHelp();
  syncMethods(true);
}

function updateMetricHelp() {
  const map = {
    mean_composite_score: DATA.metric_defs.find(m => m.id === 'composite'),
    mean_llm_judge: DATA.metric_defs.find(m => m.id === 'llm_judge'),
    mean_token_f1: DATA.metric_defs.find(m => m.id === 'token_f1'),
    contains_answer_rate: DATA.metric_defs.find(m => m.id === 'contains'),
    tokens_per_query: DATA.metric_defs.find(m => m.id === 'tokens_per_query'),
    mean_query_latency_seconds: DATA.metric_defs.find(m => m.id === 'latency'),
  };
  const d = map[state.metric];
  document.getElementById('metricHelp').innerHTML = d
    ? `<strong>${d.name}</strong><br/>${d.means}`
    : '';
}

function syncMethods(selectAll) {
  const rows = bench().summary || [];
  if (selectAll || state.methods.size === 0) {
    state.methods = new Set(rows.map(r => r.method));
  } else {
    const allowed = new Set(rows.map(r => r.method));
    state.methods = new Set([...state.methods].filter(m => allowed.has(m)));
    if (state.methods.size === 0) state.methods = new Set(allowed);
  }
  const types = new Set();
  for (const r of (bench().accuracy || [])) {
    if (r.query_type) types.add(r.query_type);
    if (r.code_rag_type) types.add(r.code_rag_type);
    if (r.hotpot_type) types.add(r.hotpot_type);
  }
  const qt = document.getElementById('qtype');
  const prev = state.qtype;
  qt.innerHTML = `<option value="__all__">All types</option>` +
    [...types].sort().map(t => `<option value="${t}">${t}</option>`).join('');
  state.qtype = [...types].includes(prev) ? prev : '__all__';
  qt.value = state.qtype;

  const box = document.getElementById('methods');
  box.innerHTML = rows.map(r => {
    const on = state.methods.has(r.method) ? 'on' : '';
    return `<span class="method-pill ${on}" data-m="${r.method}">${r.method_label}</span>`;
  }).join('');
  box.querySelectorAll('.method-pill').forEach(el => {
    el.onclick = () => {
      const m = el.dataset.m;
      if (state.methods.has(m)) state.methods.delete(m); else state.methods.add(m);
      el.classList.toggle('on');
      render();
    };
  });
}

function filteredSummary() {
  return (bench().summary || []).filter(r => state.methods.has(r.method));
}

function filteredAccuracy() {
  let rows = (bench().accuracy || []).filter(r => state.methods.has(r.method));
  if (state.qtype !== '__all__') {
    rows = rows.filter(r =>
      r.query_type === state.qtype ||
      r.code_rag_type === state.qtype ||
      r.hotpot_type === state.qtype
    );
  }
  return rows;
}

function renderProvenance() {
  const b = bench();
  const el = document.getElementById('provenance');
  const partial = b.scores_are_partial;
  el.className = 'callout' + (partial ? ' warn' : '');
  el.innerHTML = `
    <strong>${b.title}</strong>
    <div>${b.what}</div>
    <div style="margin-top:8px"><em>How to read:</em> ${b.how_to_read}</div>
    <div style="margin-top:8px">
      Indexed: <strong>${b.indexed_questions ?? '—'}</strong> questions /
      <strong>${b.indexed_documents ?? '—'}</strong> docs
      · Currently scored in this dashboard:
      <strong>${b.scored_questions || 0}</strong> questions
      ${partial ? ' · <span style="color:var(--warn)">Scores are a subset — full-scale run still in progress or not yet written to results/</span>' : ''}
    </div>
  `;
  document.getElementById('stats').innerHTML = `
    <div class="stat"><div class="k">Indexed Q</div><div class="v">${b.indexed_questions ?? '—'}</div><div class="h">on disk corpus</div></div>
    <div class="stat"><div class="k">Indexed docs</div><div class="v">${b.indexed_documents ?? '—'}</div><div class="h">retrieval corpus</div></div>
    <div class="stat"><div class="k">Scored Q</div><div class="v">${b.scored_questions || 0}</div><div class="h">in summary/accuracy CSVs</div></div>
    <div class="stat"><div class="k">Methods</div><div class="v">${(b.summary||[]).length}</div><div class="h">${b.status}</div></div>
  `;
}

function plotLayout(extra) {
  return Object.assign({
    margin: { t: 24, r: 16, b: 64, l: 56 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'IBM Plex Sans, Segoe UI, sans-serif', size: 12, color: '#1a1d23' },
    hovermode: 'closest',
  }, extra || {});
}

function renderBar() {
  const rows = filteredSummary().slice().sort((a,b) => {
    const av = a[state.metric] ?? -Infinity;
    const bv = b[state.metric] ?? -Infinity;
    return COSTISH.has(state.metric) ? av - bv : bv - av;
  });
  const y = rows.map(r => r.method_label);
  const x = rows.map(r => r[state.metric]);
  const colors = rows.map(r => r.method === state.focusMethod ? '#0b3d91' : '#1f6feb');
  Plotly.newPlot('barChart', [{
    type: 'bar', orientation: 'h',
    y, x, marker: { color: colors },
    customdata: rows.map(r => r.method),
    hovertemplate: '%{y}<br>' + METRIC_LABEL[state.metric] + ': %{x:.3f}<extra></extra>',
  }], plotLayout({
    yaxis: { autorange: 'reversed' },
    xaxis: { title: METRIC_LABEL[state.metric] },
  }), {responsive: true, displayModeBar: true});

  document.getElementById('barChart').on('plotly_click', (ev) => {
    const m = ev.points?.[0]?.customdata;
    if (!m) return;
    state.focusMethod = state.focusMethod === m ? null : m;
    render();
  });
}

function renderScatter() {
  const rows = filteredSummary().filter(r => r.mean_composite_score != null && r.tokens_per_query != null);
  Plotly.newPlot('scatterChart', [{
    type: 'scatter', mode: 'markers+text',
    x: rows.map(r => r.tokens_per_query),
    y: rows.map(r => r.mean_composite_score),
    text: rows.map(r => r.method_label),
    textposition: 'top center',
    marker: { size: 14, color: rows.map(r => r.method === state.focusMethod ? '#0b3d91' : '#1f6feb') },
    customdata: rows.map(r => [r.method, r.mean_llm_judge, r.mean_query_latency_seconds]),
    hovertemplate:
      '<b>%{text}</b><br>Composite: %{y:.3f}<br>Tokens/q: %{x:.0f}' +
      '<br>Judge: %{customdata[1]:.3f}<br>Latency: %{customdata[2]:.2f}s<extra></extra>',
  }], plotLayout({
    xaxis: { title: 'Tokens / query (lower = cheaper)' },
    yaxis: { title: 'Composite (higher = better)', range: [0, 1] },
  }), {responsive: true});

  document.getElementById('scatterChart').on('plotly_click', (ev) => {
    const m = ev.points?.[0]?.customdata?.[0];
    if (!m) return;
    state.focusMethod = state.focusMethod === m ? null : m;
    render();
  });
}

function renderBox() {
  const acc = filteredAccuracy();
  const by = {};
  for (const r of acc) {
    (by[r.method_label] ||= []).push(r.composite_score ?? 0);
  }
  const traces = Object.entries(by).map(([name, vals]) => ({
    type: 'box', name, y: vals, boxpoints: 'outliers',
    marker: { size: 4 },
  }));
  Plotly.newPlot('boxChart', traces, plotLayout({
    showlegend: false,
    yaxis: { title: 'Per-question composite', range: [-0.05, 1.05] },
  }), {responsive: true});
}

function renderHeat() {
  const rows = filteredSummary();
  const metrics = [
    ['mean_composite_score', 'Composite'],
    ['mean_llm_judge', 'Judge'],
    ['mean_token_f1', 'F1'],
    ['contains_answer_rate', 'Contains'],
  ];
  // include tokens inverted so high = good
  const z = metrics.map(([key]) => {
    const vals = rows.map(r => Number(r[key]) || 0);
    const max = Math.max(...vals, 1e-9);
    return vals.map(v => v / max);
  });
  // tokens: invert
  const tok = rows.map(r => Number(r.tokens_per_query) || 0);
  const tmax = Math.max(...tok, 1e-9);
  z.push(tok.map(v => 1 - (v / tmax)));
  metrics.push(['tokens_per_query', 'Cheap tokens']);

  Plotly.newPlot('heatChart', [{
    type: 'heatmap',
    z,
    x: rows.map(r => r.method_label),
    y: metrics.map(m => m[1]),
    colorscale: 'Blues',
    hovertemplate: '%{y} · %{x}<br>normalized %{z:.2f}<extra></extra>',
  }], plotLayout({
    margin: { t: 24, r: 16, b: 80, l: 90 },
  }), {responsive: true});
}

function renderDual() {
  const rows = bench().dual || [];
  if (!rows.length) {
    document.getElementById('dualTable').innerHTML = '<p class="muted">No dual_scoreboard.csv for this bench.</p>';
    return;
  }
  document.getElementById('dualTable').innerHTML = `
    <table><thead><tr>
      <th>Scenario</th><th>Generative winner</th><th>Composite winner</th><th>Flip?</th><th>What it means</th>
    </tr></thead><tbody>
    ${rows.map(r => {
      const flip = r.ranking_flips === true || r.ranking_flips === 'True' || r.ranking_flips === 'true';
      return `<tr>
        <td>${r.scenario ?? ''}</td>
        <td>${r.generative_winner ?? ''}</td>
        <td>${r.composite_winner ?? ''}</td>
        <td>${flip ? 'yes — metric choice changes winner' : 'no'}</td>
        <td>${flip
          ? 'Judge-heavy ranking ≠ composite. Decide which product metric matters.'
          : 'Same winner under both lenses — more trustworthy pick.'}</td>
      </tr>`;
    }).join('')}
    </tbody></table>`;
}

function renderQuestionTable() {
  const acc = filteredAccuracy();
  // pivot-ish: list unique questions with focus method score
  const focus = state.focusMethod || (filteredSummary()[0] && filteredSummary().sort((a,b)=>(b.mean_composite_score||0)-(a.mean_composite_score||0))[0]?.method);
  let rows = acc.filter(r => r.method === focus);
  if (state.search) {
    rows = rows.filter(r =>
      String(r.question_id).toLowerCase().includes(state.search) ||
      String(r.query_type||'').toLowerCase().includes(state.search) ||
      String(r.code_rag_type||'').toLowerCase().includes(state.search)
    );
  }
  rows = rows.slice().sort((a,b) => (b.composite_score||0) - (a.composite_score||0));
  const label = labelOf(focus);
  document.getElementById('qTable').innerHTML = `
    <p class="muted">Showing <strong>${label}</strong> · ${rows.length} questions</p>
    <table><thead><tr>
      <th>Question id</th><th>Type</th><th>Composite</th><th>Judge</th><th>F1</th><th>Contains</th>
    </tr></thead><tbody>
    ${rows.slice(0, 200).map(r => `
      <tr>
        <td><code>${r.question_id}</code></td>
        <td>${r.query_type || r.code_rag_type || r.hotpot_type || ''}</td>
        <td>${Number(r.composite_score||0).toFixed(3)}</td>
        <td>${Number(r.llm_judge_score||0).toFixed(3)}</td>
        <td>${Number(r.token_f1||0).toFixed(3)}</td>
        <td>${r.contains_answer}</td>
      </tr>`).join('')}
    </tbody></table>`;
}

function renderWins() {
  const rows = bench().clear_wins || [];
  if (!rows.length) {
    document.getElementById('winsTable').innerHTML = '<p class="muted">No clear-win rows for this bench/slice.</p>';
    return;
  }
  document.getElementById('winsTable').innerHTML = `
    <table><thead><tr>
      <th>Winner</th><th>Question</th><th>Gold</th><th>Score</th><th>Runner-up</th><th>Margin</th>
    </tr></thead><tbody>
    ${rows.map(r => `
      <tr>
        <td>${r.winner_label || r.winner || ''}</td>
        <td>${(r.question || '').slice(0, 140)}</td>
        <td>${(r.gold || '').slice(0, 60)}</td>
        <td>${Number(r.winner_score||0).toFixed(3)}</td>
        <td>${r.runner_up_label || r.runner_up || ''}</td>
        <td>${Number(r.margin||0).toFixed(3)}</td>
      </tr>`).join('')}
    </tbody></table>`;
}

function render() {
  // refresh method pills visual
  document.querySelectorAll('#methods .method-pill').forEach(el => {
    el.classList.toggle('on', state.methods.has(el.dataset.m));
  });
  renderProvenance();
  if (!(bench().summary || []).length) {
    ['barChart','scatterChart','boxChart','heatChart'].forEach(id => {
      document.getElementById(id).innerHTML = '<p class="muted">No summary.csv yet.</p>';
    });
    renderDual(); renderWins();
    document.getElementById('qTable').innerHTML = '<p class="muted">No accuracy rows yet.</p>';
    return;
  }
  renderBar();
  renderScatter();
  renderBox();
  renderHeat();
  renderDual();
  renderQuestionTable();
  renderWins();
}

initControls();
render();
</script>
</body>
</html>
"""


def build() -> Path:
    payload = collect_payload()
    docs = ROOT / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    data_path = docs / "dashboard_data.json"
    data_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Embed JSON safely inside <script type="application/json">
    raw = json.dumps(payload, ensure_ascii=False)
    raw = raw.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    html = HTML_TEMPLATE.replace("__DATA_JSON__", raw)

    out_docs = docs / "index.html"
    out_docs.write_text(html, encoding="utf-8")
    out_results = ROOT / "results" / "dashboard.html"
    out_results.parent.mkdir(parents=True, exist_ok=True)
    out_results.write_text(html, encoding="utf-8")
    code_rag_dash = ROOT / "results_code_rag" / "dashboard.html"
    code_rag_dash.parent.mkdir(parents=True, exist_ok=True)
    code_rag_dash.write_text(html, encoding="utf-8")

    # Lightweight Pages README
    (docs / "README.md").write_text(
        "# RAG Benchmark Dashboard (GitHub Pages)\n\n"
        "Open **[index.html](./index.html)** on GitHub Pages for the interactive view.\n\n"
        "Rebuild: `PYTHONPATH=src python scripts/build_dashboard.py`\n",
        encoding="utf-8",
    )
    return out_docs


if __name__ == "__main__":
    path = build()
    print(f"Dashboard → {path}")
    print(f"Local twin → {ROOT / 'results' / 'dashboard.html'}")
