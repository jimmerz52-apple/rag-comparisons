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

1. **HotpotQA remains the lingua franca** for multi-hop GraphRAG papers — keep it as the smoke / routing bake-off.
2. **Next upgrades:** MultiHop-RAG (retrieval quality) + CRAG-style trust scores + BenchmarkQED AutoE pairwise (judge ≠ generator).
3. **GraphRAG-Bench** is the right “is graph worth it?” follow-on after FrontierRAG routing looks good on Hotpot.
4. Pair **EM/F1** (extractive) with **contains + stronger judge** — newer suites already treat EM-only as insufficient for generative RAG.
