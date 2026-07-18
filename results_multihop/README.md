# MultiHop-RAG mini results (COLM 2024)

Closed-world subset: gold evidence news articles + distractors (9 questions × 3 types).

**Read with the dual scoreboard** (`dual_scoreboard.csv`):

| Score | Meaning |
|-------|---------|
| **generative** | mean(judge, contains) — primary for multi-doc GraphRAG |
| **extractive** | mean(F1, EM) — harsh on prose |
| **composite** | mean(all four) — can under-credit graphs |

On this run, **Hybrid wins generative** on `inference_query` and `comparison_query`.

Notebook: `notebooks/multihop_rag_bench.ipynb`  
CLI: `PYTHONPATH=src python scripts/run_multihop_benchmark.py`
