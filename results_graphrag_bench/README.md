# GraphRAG-Bench (ICLR 2026) — local demo notes

Paper: [When to use Graphs in RAG](https://arxiv.org/abs/2506.05690) (Xiang et al.)  
Official: https://github.com/GraphRAG-Bench/GraphRAG-Benchmark · https://graphrag-bench.github.io

## Task ladder
1. Fact Retrieval — basic RAG competitive (Obs.1)
2. Complex Reasoning — GraphRAG helps (Obs.2)
3. Contextual Summarize — GraphRAG helps
4. Creative Generation — GraphRAG helps

## This repo
- Builder: `src/rag_benchmark/graphrag_bench.py`
- Runner: `scripts/run_graphrag_bench.py`
- Notebook: `notebooks/rag_benchmark.ipynb`
- Outputs: `results_graphrag_bench/`
- Catalog: `results/graphrag_bench_question_catalog.csv`

```bash
PYTHONPATH=src python scripts/run_graphrag_bench.py semantic_rag,rerank_semantic,hybrid_rag,frontier_rag 2
```
