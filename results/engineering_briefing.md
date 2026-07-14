# Engineering scorecard

Default interactive path: **Vector + rerank**
- Reason: Best quality among methods with p95 ≤ 5.0s
- Quality: 0.426 | usable: 75%
- p95 latency: 2.70s | tokens/query: 1265

## Routing (by scenario)
- **multi_hop** → FrontierRAG (adaptive+CRAG) (q=0.406, 2059 tok/q) — Bridge / cross-doc multi-hop → FrontierRAG (adaptive+CRAG).
- **local_factoid** → Vector + rerank (q=0.557, 1265 tok/q) — Single-entity / comparison factoids → Vector + rerank. H2H 1W-1L vs runner-up (meanΔ=0.021; treat as soft)

## Recommendations
- Ship a query router, not one RAG stack: local factoids vs multi-hop need different paths.
- MULTI_HOP: FrontierRAG (adaptive+CRAG) (q=0.41, Δ=0.05, 2059 tok/q).
- LOCAL_FACTOID: Vector + rerank (q=0.56, Δ=0.02, 1265 tok/q).
- Usable-answer leader (GraphRAG fast/basic, 83%) differs from composite leader (Vector + rerank) — pick the metric that matches your UX.
- Highest token cost: FrontierRAG (adaptive+CRAG) (2059 tok/q) — gate behind hard-query classifier.
- Misses p95≤5.0s SLO: GraphRAG global, GraphRAG local, Hybrid (vec+graph local), GraphRAG fast/basic, Adaptive router, FrontierRAG (adaptive+CRAG). Keep interactive path on vector; run graph/hybrid async or on stronger GPUs.
- Rebuild GraphRAG indexes only on corpus change; amortize index_seconds over expected volume.

## Caveats
- Eval set size n=12 — treat rankings as directional, not production SLAs.
- Overlapping quality CIs are common at this n; trust large gaps (latency/tokens/local EM) over 0.01 composite deltas.
- Composite mixes EM/F1/judge; generative graph answers often lose on EM even when useful.
- Absolute latency depends on hardware + model size; use relative ordering for stack choice.
- Graph index build dominates cold-start cost; measure amortized $/query at your QPS.
