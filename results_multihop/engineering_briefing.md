# Engineering scorecard

Default interactive path: **Vector + rerank**
- Reason: No method met latency SLO; picking fastest mean latency
- Quality: 0.252 | usable: 89%
- p95 latency: 13.10s | tokens/query: 5310

## Routing (by scenario)
- **multi_hop** → Semantic (vector) (q=0.287, 5317 tok/q) — Bridge / cross-doc multi-hop → Semantic (vector).
- **local_factoid** → Hybrid (vec+graph local) (q=0.317, 5893 tok/q) — Single-entity / comparison factoids → Hybrid (vec+graph local).

## Recommendations
- Ship a query router, not one RAG stack: local factoids vs multi-hop need different paths.
- MULTI_HOP: Semantic (vector) (q=0.29, Δ=0.03, 5317 tok/q).
- LOCAL_FACTOID: Hybrid (vec+graph local) (q=0.32, Δ=0.07, 5893 tok/q).
- Usable-answer leader (Vector + rerank, 89%) differs from composite leader (Hybrid (vec+graph local)) — pick the metric that matches your UX.
- Highest token cost: FrontierRAG (adaptive+CRAG) (7675 tok/q) — gate behind hard-query classifier.
- Misses p95≤5.0s SLO: Semantic (vector), Vector + rerank, Hybrid (vec+graph local), FrontierRAG (adaptive+CRAG). Keep interactive path on vector; run graph/hybrid async or on stronger GPUs.
- Rebuild GraphRAG indexes only on corpus change; amortize index_seconds over expected volume.

## Caveats
- Eval set size n=9 — treat rankings as directional, not production SLAs.
- Overlapping quality CIs are common at this n; trust large gaps (latency/tokens/local EM) over 0.01 composite deltas.
- Composite mixes EM/F1/judge; generative graph answers often lose on EM even when useful.
- Absolute latency depends on hardware + model size; use relative ordering for stack choice.
