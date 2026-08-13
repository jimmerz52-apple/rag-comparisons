# RAG comparisons

Local bake-off of retrieval methods (semantic, rerank, BM25+dense, GraphRAG, hybrid, …)
on HotpotQA, CodeRAG-Bench, GraphRAG-Bench, and MultiHop-RAG.

## Interactive dashboard (GitHub Pages)

**[Open the live dashboard →](https://jimmerz52-apple.github.io/rag-comparisons/)**

Tableau-style filters: pick a benchmark, metric, and methods — leaderboard, quality-vs-cost scatter, per-question boxes, heatmap, and question explorer update together.

Metric glossary and “indexed vs scored” provenance are built into the page so partial runs are not mistaken for full validation.

Rebuild after new scores:

```bash
PYTHONPATH=src python scripts/build_dashboard.py
# writes docs/index.html (Pages) and results/dashboard.html (local)
```

## Full-scale data

| Bench | Indexed | Default scoring |
|-------|--------:|-----------------|
| HotpotQA distractor | **7,405 Q / ~66k docs** | vector methods (`run_hotpot_benchmark.py all`) |
| CodeRAG-Bench | **~2.1k Q / ~35k docs** | HE+MBPP+DS-1000+ODEX (`run_code_rag_benchmark.py`) |
| MultiHop-RAG | 150 Q | expandable |
| GraphRAG-Bench Novel | 72 Q | Novel-4128 |

## Quick start

```bash
python scripts/run_hotpot_benchmark.py all
python scripts/run_code_rag_benchmark.py
python scripts/build_dashboard.py
```
