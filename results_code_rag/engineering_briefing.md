# Engineering scorecard

Default interactive path: **Semantic (vector)**
- Reason: No method met latency SLO; picking fastest mean latency
- Quality: 0.270 | usable: 85%
- p95 latency: 5.03s | tokens/query: 6905

## Routing (by scenario)
- **local_factoid** → Vector + rerank (q=0.302, 6818 tok/q) — Single-entity / comparison factoids → Vector + rerank.

## Recommendations
- Ship a query router, not one RAG stack: local factoids vs multi-hop need different paths.
- LOCAL_FACTOID: Vector + rerank (q=0.30, Δ=0.03, 6818 tok/q).
- Usable-answer leader (Semantic (vector), 85%) differs from composite leader (Vector + rerank) — pick the metric that matches your UX.
- Highest token cost: Semantic (vector) (6905 tok/q) — gate behind hard-query classifier.
- Misses p95≤5.0s SLO: Semantic (vector), Vector + rerank. Keep interactive path on vector; run graph/hybrid async or on stronger GPUs.
- Rebuild GraphRAG indexes only on corpus change; amortize index_seconds over expected volume.

## Caveats
- Eval set size n=20 — treat rankings as directional, not production SLAs.
- Overlapping quality CIs are common at this n; trust large gaps (latency/tokens/local EM) over 0.01 composite deltas.
- Composite mixes EM/F1/judge; generative graph answers often lose on EM even when useful.
- Absolute latency depends on hardware + model size; use relative ordering for stack choice.
