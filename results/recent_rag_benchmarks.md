# Recent RAG benchmarks (≈2024–2026)

How this HotpotQA harness relates to newer evaluation suites. Use these when graduating beyond n=12 distractor Hotpot.

| Benchmark | Year | What it stresses | Why it matters vs Hotpot |
|-----------|------|------------------|---------------------------|
| **[MultiHop-RAG](https://arxiv.org/abs/2401.15391)** (COLM 2024) | 2024 | 2,556 multi-hop queries over news; evidence across 2–4 docs + metadata | Closer to “real RAG” multi-doc synthesis than Wikipedia bridge spans |
| **[CRAG](https://arxiv.org/abs/2406.04744)** (Meta / KDD Cup 2024) | 2024 | 4,409 QA pairs; web + KG mock APIs; dynamism, popularity, complexity | Trustworthiness / hallucination under changing facts — not just EM |
| **[BenchmarkQED](https://github.com/microsoft/benchmark-qed)** (Microsoft) | 2025 | AutoQ / AutoE / AutoD — synthetic queries + LLM-as-judge pairwise | Scalable eval without hand-labeled Hotpot; matches GraphRAG research tooling |
| **[GraphRAG-Bench](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark)** (ICLR’26) | 2025–26 | Domain CS reasoning; graph construction → retrieval → generation | Asks *when* graphs help; Hotpot alone under-tests GraphRAG |
| **RGB / RECALL** (cited baseline) | 2023–24 | Noise, counterfactuals, generation robustness | Good for hallucination stress; weak on multi-hop retrieval |
| **FlashRAG toolkit** | 2024+ | Many datasets in one harness (Hotpot, MultiHop-RAG, …) | Reproducible pipeline zoo — complementary to this engineering scorecard |

## Takeaways for this repo

1. **HotpotQA** — smoke / routing bake-off. See `notebooks/hotpot_metric_autopsy.ipynb` (GitHub-rendered).
2. **MultiHop-RAG** — news multi-doc reasoning. See `notebooks/multihop_rag_bench.ipynb` + `scripts/run_multihop_benchmark.py`.
3. **GraphRAG-Bench** — task ladder (fact → creative). See `notebooks/rag_benchmark.ipynb`.
4. **Dual scoreboard (required):** `generative = mean(judge, contains)` vs `extractive = mean(F1, EM)`.
   On Hotpot multi-hop, GraphRAG fast/basic **wins generative** while composite can crown Frontier — do not conclude “graphs lose multi-hop” from composite alone.
5. Next: BenchmarkQED AutoE with a **stronger separate judge**; official GraphRAG-Bench `Evaluation/` ROUGE/coverage.
