# Recent RAG research this dashboard encodes (2024–2026)

| Finding | Source | Dashboard surface |
|---------|--------|-------------------|
| Separate retrieval vs generation metrics | RAGAS / DeepEval / 2026 guides | **Research Lens → metric map** (faithfulness & context P/R marked as gaps) |
| Graphs help by *task difficulty*, not by default | Xiang et al. GraphRAG-Bench (arXiv:2506.05690) | Bench cards + Decision Lab router framing |
| Hybrid / router beats single paradigm | Systematic RAG vs GraphRAG evals (2025–26) | Routing table + engineering briefing |
| EM/F1 underestimates generative QA | Qi 2025; LLM-as-judge QA reassessments | Generative vs extractive scatter + dual scoreboard |
| Code RAG often bottlenecked by retrieval; leave-gold-out required | Wang et al., CodeRAG-Bench, Findings NAACL 2025 | CodeRAG provenance + code-aware metrics note |
| Quality alone ≠ ship decision | Ops practice (tokens, latency SLOs) | Quality–cost Pareto |

## Scale in this repo

| Bench | Indexed | Notes |
|-------|--------:|-------|
| HotpotQA distractor | 7,405 Q / ~66k docs | Full val materialised; scored slice may still be partial |
| CodeRAG-Bench | ~2.1k Q / ~35k docs target | HE+MBPP+DS-1000+ODEX |
| GraphRAG-Bench Novel | 72 Q | Task-difficulty lens |
| MultiHop-RAG | 150 Q | News multi-doc |

## Commands

```bash
python scripts/build_dashboard.py
python scripts/run_hotpot_benchmark.py all
python scripts/run_code_rag_benchmark.py
```
