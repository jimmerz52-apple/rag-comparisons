# When to choose which RAG (from *your* Hotpot data)

Cutting-edge production pattern (Adaptive-RAG): **route by query type**, don't crown a single stack.

- **multi-hop / bridge** (`hybrid`) → **Semantic (vector)** (mean 0.29)
- **comparison / local factoid** (`local`) → **Hybrid (vec+graph local)** (mean 0.32)

## Pairwise takeaways
- Semantic vs Hybrid: Hybrid (vec+graph local) 3W–3W Semantic (vector) (mean Δ=+0.02)

## Research honesty: is this cutting edge?
- **Yes as an engineering bake-off**: FrontierRAG (Adaptive+CRAG escalate) + BM25/dense RRF + cross-encoder rerank + GraphRAG modes + HippoRAG 2/LightRAG is a 2025–2026-relevant *system* stack.
- **No as a SOTA paper claim**: n=12 Hotpot subset, local 3B judge=generator, no full BenchmarkQED AutoQ/AutoE LLM pairwise, LazyGraphRAG itself still not OSS.
- **Cutting-edge move**: ship the *router* that grades retrieval and escalates compute; keep the cost/latency scorecard; use stronger models for OpenIE and judging.

## Concrete choose-A-over-B examples

### Choose **Hybrid (vec+graph local)** over FrontierRAG (adaptive+CRAG)
- Q: What company, according to TechCrunch, experienced a 38% decrease in reported sexual assault rates between its first and second safety reports, faces criticism for inadequate background checks aimed at quick driver sign-up, and reported $394 million in operating income and $219 million in net income in the third quarter?
- Gold: `Uber` (inference_query)
- Scores: 0.26 vs 0.00 (margin 0.26)
- Why: On this question, Hybrid (vec+graph local) beat FrontierRAG (adaptive+CRAG).

### Choose **Hybrid (vec+graph local)** over Vector + rerank
- Q: Does 'The Independent - Life and Style' article suggest that Taylor Swift is secretive about her relationship with Travis Kelce, while the 'FOX News - Lifestyle' article indicates that she engaged with a viral TikTok video, and does the other 'The Independent - Life and Style' article claim that she has a firm commitment to her Eras Tour schedule?
- Gold: `no` (comparison_query)
- Scores: 0.45 vs 0.25 (margin 0.20)
- Why: On this question, Hybrid (vec+graph local) beat Vector + rerank.

