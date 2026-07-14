"""Concrete win/fail examples for notebook / CLI — grounded in Hotpot results."""

from __future__ import annotations

from typing import Any


# Explains the confusing "Fast/basic (Lazy-style) 0.357 / 0.298" row.
FAST_BASIC_EXPLAINER = """
## What “GraphRAG fast/basic” (formerly Fast/basic Lazy-style) means

It is **not** Microsoft LazyGraphRAG. It is Microsoft GraphRAG with:
- **fast** indexing = spaCy NLP noun-phrase graph (cheap, no LLM entity extract)
- **basic** search = lightweight retrieval over that graph

The two numbers are **mean composite quality by scenario** on our HotpotQA subset (n=12):

| Scenario | Score | What it is |
|---|---|---|
| **0.357** | `query_type=hybrid` | Hotpot **bridge** multi-hop questions (need 2 documents) |
| **0.298** | `query_type=local` | Hotpot **comparison** questions (often yes/no / which-of-two) |

So: decent on multi-hop relative to pure GraphRAG global/local, weaker than vector RAG on simple comparison factoids. Exact match is usually 0 because answers are verbose — `contains_answer` is the fairer lens.
""".strip()


# Hand-picked from results/accuracy_results.csv + hotpot_eval.json
METHOD_EXAMPLES: list[dict[str, Any]] = [
    {
        "method": "semantic_rag",
        "label": "Semantic (vector)",
        "excels_when": "Answer lives in one passage / comparison yes-no with clear entity facts.",
        "breaks_when": "Bridge questions where the linking entity is not in the top-k chunks.",
        "example_excel": {
            "question": "Are Giuseppe Verdi and Ambroise Thomas both Opera composers?",
            "gold": "yes",
            "why": "Both facts appear as simple entity attributes; vector retrieve + short answer works (composite ≈ 0.88, EM).",
        },
        "example_break": {
            "question": "What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?",
            "gold": "Animorphs",
            "why": "Multi-hop description; vector alone scored 0.0 here while hybrid/lazygraph found the series name.",
        },
    },
    {
        "method": "lazygraph_rag",
        "label": "GraphRAG fast/basic",
        "excels_when": "Cheap graph index helps surface related entities without full GraphRAG LLM indexing.",
        "breaks_when": "Short yes/no comparisons (verbose graph prose fails EM; sometimes misses neighborhood facts).",
        "example_excel": {
            "question": "2014 S/S is the debut album of a South Korean boy group that was formed by who?",
            "gold": "YG Entertainment",
            "why": "Bridge multi-hop: album → group → agency. fast/basic composite ≈ 0.46 with contains=True (mean hybrid scenario 0.357).",
        },
        "example_break": {
            "question": "Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?",
            "gold": "no",
            "why": "Comparison factoid; fast/basic scored 0.0 while semantic got EM=yes/no correctly (local scenario mean 0.298).",
        },
    },
    {
        "method": "hybrid_rag",
        "label": "Hybrid (vec + graph local)",
        "excels_when": "Need both a precise chunk and a graph path (classic Hotpot bridge).",
        "breaks_when": "Simple comparisons where graph context dilutes the prompt; also highest token cost / misses p95 SLO.",
        "example_excel": {
            "question": "Who was known by his stage name Aladin and helped organizations improve their performance as a consultant?",
            "gold": "Eenasul Fateh",
            "why": "Hybrid hit EM=1.0; pure GraphRAG global scored 0.0.",
        },
        "example_break": {
            "question": "Are Random House Tower and 888 7th Avenue both used for real estate?",
            "gold": "no",
            "why": "Semantic EM=True; hybrid scored 0.0 — fusion over-complicated a comparison.",
        },
    },
    {
        "method": "graph_rag",
        "label": "GraphRAG global",
        "excels_when": "Thematic / corpus-wide summaries (not this Hotpot short-answer set).",
        "breaks_when": "Factoid QA requiring a short span — global community reports never hit gold on our 6 bridge questions.",
        "example_excel": {
            "question": "(Thematic) What are the main topics across this corpus?",
            "gold": "n/a — not in Hotpot eval",
            "why": "Global search is designed for overview questions; Hotpot EM/F1 punish long reports.",
        },
        "example_break": {
            "question": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
            "gold": "Chief of Protocol",
            "why": "Bridge multi-hop; GraphRAG global composite 0.0 on all 6 bridge items in this run.",
        },
    },
    {
        "method": "graph_local_rag",
        "label": "GraphRAG local",
        "excels_when": "Entity-centric questions when the graph has clean LLM-extracted entities (standard index + strong model).",
        "breaks_when": "Fast NLP graphs are noisy; local search still near-zero on Hotpot bridge here.",
        "example_excel": {
            "question": "Are Random House Tower and 888 7th Avenue both used for real estate?",
            "gold": "no",
            "why": "Best of the pure-graph methods on some local items (contains sometimes), still far behind semantic.",
        },
        "example_break": {
            "question": "What science fantasy young adult series…?",
            "gold": "Animorphs",
            "why": "Bridge; graph_local composite ≈ 0.0–0.02 on hybrid scenario overall.",
        },
    },
    {
        "method": "light_rag",
        "label": "LightRAG (HKUDS)",
        "excels_when": "Dual-level KG + vector on mid-size corpora with a capable LLM (paper setting).",
        "breaks_when": "Tiny local 3B models produce malformed OpenIE triples; answers can collapse to reference lists.",
        "example_excel": {
            "question": "What are the main relationships between entities in this domain?",
            "gold": "n/a — needs stronger LLM than llama3.2:3b for fair comparison",
            "why": "LightRAG's dual-level retrieval shines when entity extraction is reliable.",
        },
        "example_break": {
            "question": "Which team won the Memorial Cup? (smoke test)",
            "gold": "Vancouver Giants",
            "why": "On 3B smoke test, response degraded to '### References' without the short answer.",
        },
    },
    {
        "method": "hippo_rag",
        "label": "HippoRAG 2",
        "excels_when": "Multi-hop associativity (Hotpot/MuSiQue-style) — paper reports strong gains vs GraphRAG.",
        "breaks_when": "OpenIE + seed entities need a capable LLM; heavy deps; not for thematic global summaries.",
        "example_excel": {
            "question": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
            "gold": "Chief of Protocol",
            "why": "Canonical bridge: actress → person → office. HippoRAG is built for this hop pattern.",
        },
        "example_break": {
            "question": "Summarize the themes across the entire corpus.",
            "gold": "n/a",
            "why": "HippoRAG optimizes passage/fact linking, not GraphRAG-style community reports.",
        },
    },
]


def format_examples_markdown() -> str:
    lines = [FAST_BASIC_EXPLAINER, "", "## When each method excels vs breaks", ""]
    for item in METHOD_EXAMPLES:
        lines.append(f"### {item['label']} (`{item['method']}`)")
        lines.append(f"- **Excels when:** {item['excels_when']}")
        lines.append(f"- **Breaks when:** {item['breaks_when']}")
        ex = item["example_excel"]
        br = item["example_break"]
        lines.append(f"- **Example query (excels):** {ex['question']}")
        lines.append(f"  - Gold: `{ex['gold']}` — {ex['why']}")
        lines.append(f"- **Example query (breaks):** {br['question']}")
        lines.append(f"  - Gold: `{br['gold']}` — {br['why']}")
        lines.append("")
    return "\n".join(lines)


def examples_frame():
    import pandas as pd

    rows = []
    for item in METHOD_EXAMPLES:
        rows.append(
            {
                "method": item["label"],
                "excels_when": item["excels_when"],
                "example_excel_query": item["example_excel"]["question"],
                "breaks_when": item["breaks_when"],
                "example_break_query": item["example_break"]["question"],
            }
        )
    return pd.DataFrame(rows)
