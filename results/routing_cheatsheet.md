# When to choose which RAG (from *your* Hotpot data)

Cutting-edge production pattern (Adaptive-RAG): **route by query type**, don't crown a single stack.

- **multi-hop / bridge** (`hybrid`) → **FrontierRAG (adaptive+CRAG)** (mean 0.41)
- **comparison / local factoid** (`local`) → **Vector + rerank** (mean 0.56)

## Pairwise takeaways
- Semantic vs Hybrid: Hybrid (vec+graph local) 4W–5W Semantic (vector) (mean Δ=-0.06)
- Hybrid vs GraphRAG global: 2W–10W (mean Δ=-0.24)
- Semantic vs GraphRAG global: 2W–8W (mean Δ=-0.30)

## Research honesty: is this cutting edge?
- **Yes as an engineering bake-off**: FrontierRAG (Adaptive+CRAG escalate) + BM25/dense RRF + cross-encoder rerank + GraphRAG modes + HippoRAG 2/LightRAG is a 2025–2026-relevant *system* stack.
- **No as a SOTA paper claim**: n=12 Hotpot subset, local 3B judge=generator, no full BenchmarkQED AutoQ/AutoE LLM pairwise, LazyGraphRAG itself still not OSS.
- **Cutting-edge move**: ship the *router* that grades retrieval and escalates compute; keep the cost/latency scorecard; use stronger models for OpenIE and judging.

## Concrete choose-A-over-B examples

### Choose **Semantic (vector)** over GraphRAG local
- Q: Are Random House Tower and 888 7th Avenue both used for real estate?
- Gold: `no` (comparison)
- Scores: 0.62 vs 0.45 (margin 0.17)
- Why: Simple comparison/factoid → prefer vector RAG (cheap + EM-friendly).

