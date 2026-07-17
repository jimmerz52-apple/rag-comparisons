# When to choose which RAG (from *your* Hotpot data)

Cutting-edge production pattern (Adaptive-RAG): **route by query type**, don't crown a single stack.

- **multi-hop / bridge** (`hybrid`) → **FrontierRAG (adaptive+CRAG)** (mean 0.08)
- **comparison / local factoid** (`local`) → **Hybrid (vec+graph local)** (mean 0.21)

## Pairwise takeaways
- Semantic vs Hybrid: Hybrid (vec+graph local) 4W–2W Semantic (vector) (mean Δ=+0.07)

## Research honesty: is this cutting edge?
- **Yes as an engineering bake-off**: FrontierRAG (Adaptive+CRAG escalate) + BM25/dense RRF + cross-encoder rerank + GraphRAG modes + HippoRAG 2/LightRAG is a 2025–2026-relevant *system* stack.
- **No as a SOTA paper claim**: n=12 Hotpot subset, local 3B judge=generator, no full BenchmarkQED AutoQ/AutoE LLM pairwise, LazyGraphRAG itself still not OSS.
- **Cutting-edge move**: ship the *router* that grades retrieval and escalates compute; keep the cost/latency scorecard; use stronger models for OpenIE and judging.

## Concrete choose-A-over-B examples

### Choose **Hybrid (vec+graph local)** over Semantic (vector)
- Q: How did the Lady treat Samuel Pepys when he visited her at the Wardrobe, as detailed in his diary account?
- Gold: `Kindly` (Fact Retrieval)
- Scores: 0.42 vs 0.12 (margin 0.30)
- Why: On this question, Hybrid (vec+graph local) beat Semantic (vector).

