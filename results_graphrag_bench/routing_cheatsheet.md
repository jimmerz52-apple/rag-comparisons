# When to choose which RAG (from *your* Hotpot data)

Cutting-edge production pattern (Adaptive-RAG): **route by query type**, don't crown a single stack.

- **multi-hop / bridge** (`hybrid`) → **Semantic (vector)** (mean 0.11)
- **comparison / local factoid** (`local`) → **FrontierRAG (adaptive+CRAG)** (mean 0.25)

## Pairwise takeaways
- Semantic vs Hybrid: Hybrid (vec+graph local) 1W–6W Semantic (vector) (mean Δ=-0.06)

## Research honesty: is this cutting edge?
- **Yes as an engineering bake-off**: FrontierRAG (Adaptive+CRAG escalate) + BM25/dense RRF + cross-encoder rerank + GraphRAG modes + HippoRAG 2/LightRAG is a 2025–2026-relevant *system* stack.
- **No as a SOTA paper claim**: n=12 Hotpot subset, local 3B judge=generator, no full BenchmarkQED AutoQ/AutoE LLM pairwise, LazyGraphRAG itself still not OSS.
- **Cutting-edge move**: ship the *router* that grades retrieval and escalates compute; keep the cost/latency scorecard; use stronger models for OpenIE and judging.