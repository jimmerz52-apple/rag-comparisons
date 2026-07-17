# Engineering scorecard

Default interactive path: **Vector + rerank**
- Reason: No method met latency SLO; picking fastest mean latency
- Quality: 0.024 | usable: 0%
- p95 latency: 7.65s | tokens/query: 4518

## Routing (by scenario)
- **multi_hop** → FrontierRAG (adaptive+CRAG) (q=0.085, 6940 tok/q) — Bridge / cross-doc multi-hop → FrontierRAG (adaptive+CRAG) (quality tie → cheaper tokens). meanΔ=0.024 but H2H 2W-3L → prefer head-to-head winner
- **local_factoid** → Hybrid (vec+graph local) (q=0.210, 5049 tok/q) — Single-entity / comparison factoids → Hybrid (vec+graph local).

## Recommendations
- Ship a query router, not one RAG stack: local factoids vs multi-hop need different paths.
- MULTI_HOP: FrontierRAG (adaptive+CRAG) (q=0.08, Δ=0.02, 6940 tok/q) [cost tie-break].
- LOCAL_FACTOID: Hybrid (vec+graph local) (q=0.21, Δ=0.15, 5049 tok/q).
- Highest token cost: Semantic (vector) (7969 tok/q) — gate behind hard-query classifier.
- Misses p95≤5.0s SLO: Semantic (vector), Vector + rerank, Hybrid (vec+graph local), FrontierRAG (adaptive+CRAG). Keep interactive path on vector; run graph/hybrid async or on stronger GPUs.
- Rebuild GraphRAG indexes only on corpus change; amortize index_seconds over expected volume.

## Caveats
- Eval set size n=7 — treat rankings as directional, not production SLAs.
- Overlapping quality CIs are common at this n; trust large gaps (latency/tokens/local EM) over 0.01 composite deltas.
- Composite mixes EM/F1/judge; generative graph answers often lose on EM even when useful.
- Absolute latency depends on hardware + model size; use relative ordering for stack choice.
- Graph index build dominates cold-start cost; measure amortized $/query at your QPS.
