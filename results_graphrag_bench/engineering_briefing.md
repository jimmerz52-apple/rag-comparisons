# Engineering scorecard

Default interactive path: **Vector + rerank**
- Reason: No method met latency SLO; picking fastest mean latency
- Quality: 0.119 | usable: 29%
- p95 latency: 6.78s | tokens/query: 4531

## Routing (by scenario)
- **multi_hop** → Semantic (vector) (q=0.107, 4370 tok/q) — Bridge / cross-doc multi-hop → Semantic (vector) (quality tie → cheaper tokens). meanΔ=0.007 but H2H 2W-3L → prefer head-to-head winner
- **local_factoid** → Vector + rerank (q=0.248, 4531 tok/q) — Single-entity / comparison factoids → Vector + rerank (quality tie → cheaper tokens). quality tie → cheaper tokens

## Recommendations
- Ship a query router, not one RAG stack: local factoids vs multi-hop need different paths.
- MULTI_HOP: Semantic (vector) (q=0.11, Δ=0.01, 4370 tok/q) [cost tie-break].
- LOCAL_FACTOID: Vector + rerank (q=0.25, Δ=0.00, 4531 tok/q) [cost tie-break].
- Usable-answer leader (Hybrid (vec+graph local), 43%) differs from composite leader (FrontierRAG (adaptive+CRAG)) — pick the metric that matches your UX.
- Highest token cost: FrontierRAG (adaptive+CRAG) (6902 tok/q) — gate behind hard-query classifier.
- Misses p95≤5.0s SLO: Semantic (vector), Vector + rerank, Hybrid (vec+graph local), FrontierRAG (adaptive+CRAG). Keep interactive path on vector; run graph/hybrid async or on stronger GPUs.
- Rebuild GraphRAG indexes only on corpus change; amortize index_seconds over expected volume.

## Caveats
- Eval set size n=7 — treat rankings as directional, not production SLAs.
- Overlapping quality CIs are common at this n; trust large gaps (latency/tokens/local EM) over 0.01 composite deltas.
- Composite mixes EM/F1/judge; generative graph answers often lose on EM even when useful.
- Absolute latency depends on hardware + model size; use relative ordering for stack choice.
