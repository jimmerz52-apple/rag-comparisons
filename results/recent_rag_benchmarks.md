# Recent RAG benchmarks (≈2024–2026)

| Benchmark | Year | Scale in this repo | Status |
|-----------|------|--------------------|--------|
| **HotpotQA** distractor | 2018 | **Full validation: 7,405 Q / ~66k docs** | `python scripts/run_hotpot_benchmark.py all` (vector default) |
| **CodeRAG-Bench** | 2024–25 | **Basic + open-domain: ~2,103 Q / ~35k docs** | HE+MBPP+DS-1000+ODEX · `notebooks/code_rag_bench.ipynb` |
| **MultiHop-RAG** | 2024 | 150 Q (expandable to 2,556) | `notebooks/multihop_rag_bench.ipynb` |
| **GraphRAG-Bench** | 2025–26 | Novel-4128 all 72 Q | `notebooks/rag_benchmark.ipynb` |

## CodeRAG-Bench (large)

| Slice | Questions | Corpus |
|-------|----------:|--------|
| HumanEval | 164 | programming-solutions (leave-gold-out) |
| MBPP | 500 | programming-solutions |
| DS-1000 | 1,000 | library-documentation (~34k) |
| ODEX | 439 | library-documentation |
| **Total** | **~2,103** | **~35k docs** |

Optional: `--stackoverflow` adds up to 20k SO posts.

```bash
python scripts/run_code_rag_benchmark.py
python scripts/run_code_rag_benchmark.py semantic_rag,rerank_semantic --stackoverflow
```

## Hotpot full scale

```bash
# Full 7405 Q — vector methods (default)
python scripts/run_hotpot_benchmark.py all

# GraphRAG on full corpus is days/weeks on local 3B — opt-in only:
python scripts/run_hotpot_benchmark.py all semantic_rag,lazygraph_rag
```

## Dashboard

```bash
python scripts/build_dashboard.py   # → results/dashboard.html
```
