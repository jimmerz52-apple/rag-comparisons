# GraphRAG-Bench mini results (ICLR 2026)

Local bake-off on the **Novel** split (`Novel-4128`, Samuel Pepys diary), ~2 questions × 4 task levels.

| Artifact | Purpose |
|----------|---------|
| `by_question_type.csv` | Winner per GraphRAG-Bench level |
| `accuracy_results.csv` | Per-question metrics (+ `graphrag_bench_type`) |
| `summary.csv` / `scenario_results.csv` | Overall + local/hybrid routing views |
| `engineering_briefing.md` | Tokens / latency / default path |
| `routing_cheatsheet.md` | Choose-A-over-B from this run |

**Notebook demo:** `notebooks/rag_benchmark.ipynb`  
**Rebuild / re-run:**

```bash
PYTHONPATH=src python scripts/run_graphrag_bench.py semantic_rag,rerank_semantic,hybrid_rag,frontier_rag 2 Novel-4128
```

Paper: [arXiv:2506.05690](https://arxiv.org/abs/2506.05690) · [Leaderboard](https://graphrag-bench.github.io/)
