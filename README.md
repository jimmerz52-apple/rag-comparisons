# RAG comparisons

Local bake-off of retrieval methods (semantic, rerank, BM25+dense, GraphRAG, hybrid, …)
on HotpotQA, CodeRAG-Bench, GraphRAG-Bench, and MultiHop-RAG.

## Interactive dashboard (GitHub Pages)

**[Open the live dashboard →](https://jimmerz52-apple.github.io/rag-comparisons/)**  
**[How to run on the source data →](docs/how-to-run.md)** (also a tab on the dashboard)

Research-refined (2024–2026 eval practice), Tableau-style:

| Tab | What you get |
|-----|----------------|
| **Explore** | Filters, leaderboard, quality–cost Pareto, per-question explorer |
| **Latency / cost** | p50/p95 latency, index time, tokens/query, quality vs latency |
| **Decision Lab** | Generative vs extractive scatter, dual scoreboard, routing, engineering briefing |
| **Research Lens** | RAGAS-style metric map (incl. honest gaps), GraphRAG-Bench / CodeRAG findings |
| **How to run** | Concrete commands from HuggingFace / your `.txt` files → scores |

Grounded in: RAGAS retrieval-vs-generation split · GraphRAG-Bench “when graphs help” · CodeRAG-Bench leave-gold-out · Hotpot EM/F1 under-crediting generative answers.

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
