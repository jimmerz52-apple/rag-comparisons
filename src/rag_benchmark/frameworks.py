"""Catalog of RAG frameworks considered for this harness.

Status key:
  integrated — runnable via BenchmarkRunner method id
  partial — related pieces only (not the full paper system)
  deferred — noteworthy, not wired (deps / not OSS / too heavy for local 3B)
"""

from __future__ import annotations

FRAMEWORKS: list[dict[str, str]] = [
    {
        "id": "semantic_rag",
        "name": "Semantic / vector RAG",
        "status": "integrated",
        "notes": "Chroma + local embeddings baseline.",
    },
    {
        "id": "graph_rag",
        "name": "Microsoft GraphRAG (global / local / DRIFT)",
        "status": "integrated",
        "notes": "Official OSS CLI. DRIFT available as drift_rag.",
        "url": "https://github.com/microsoft/graphrag",
    },
    {
        "id": "lazygraph_rag",
        "name": "Microsoft LazyGraphRAG",
        "status": "partial",
        "notes": (
            "Full LazyGraphRAG (iterative-deepening relevance tests + budget) is NOT in "
            "open-source graphrag yet (2026-07). Available via Microsoft Discovery / Azure Local. "
            "Our method id runs GraphRAG fast NLP index + basic search (FastGraphRAG-style)."
        ),
        "url": "https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/",
    },
    {
        "id": "light_rag",
        "name": "LightRAG (HKUDS, EMNLP 2025)",
        "status": "integrated",
        "notes": "Dual-level KG + vector; pip package lightrag-hku. Strong GraphRAG alternative.",
        "url": "https://github.com/HKUDS/LightRAG",
    },
    {
        "id": "hybrid_rag",
        "name": "Hybrid (vector retrieve + GraphRAG local fuse)",
        "status": "integrated",
        "notes": "In-house fusion; not a published framework.",
    },
    {
        "id": "hippo_rag",
        "name": "HippoRAG 2 (OSU-NLP)",
        "status": "integrated",
        "notes": (
            "Multi-hop graph memory. Wired via Ollama OpenAI-compatible API. "
            "Install: pip install hipporag==2.0.0a3 --no-deps && pip install igraph gritlm networkx. "
            "Opt-in: --methods hippo_rag (OpenIE indexing is slow on 3B)."
        ),
        "url": "https://github.com/OSU-NLP-Group/HippoRAG",
    },
    {
        "id": "frontier_rag",
        "name": "FrontierRAG (Adaptive + CRAG escalate)",
        "status": "integrated",
        "notes": "Classify → BM25+dense RRF → cross-encoder rerank → grade → escalate to hybrid. 2025–2026 production pattern.",
    },
    {
        "id": "adaptive_rag",
        "name": "Adaptive-RAG router (Jeong et al., 2024)",
        "status": "integrated",
        "notes": "Complexity-based routing: comparison→vector, bridge→hybrid. Production SOTA pattern.",
        "url": "https://arxiv.org/abs/2403.14403",
    },
    {
        "id": "hybrid_dense_sparse",
        "name": "BM25 + dense (RRF)",
        "status": "integrated",
        "notes": "Modern retrieval baseline; sparse+dense fusion before generation.",
    },
    {
        "id": "rerank_semantic",
        "name": "Vector + cross-encoder rerank",
        "status": "integrated",
        "notes": "Retrieve wide then MiniLM cross-encoder rerank (BGE/Cohere-style stage).",
    },
    {
        "id": "ket_rag",
        "name": "KET-RAG",
        "status": "deferred",
        "notes": "Cost-efficient GraphRAG-family indexing; community alternative while LazyGraphRAG OSS lands.",
        "url": "https://github.com/waetr/KET-RAG",
    },
    {
        "id": "minirag",
        "name": "MiniRAG (HKUDS)",
        "status": "deferred",
        "notes": "SLM-oriented; shares LightRAG ecosystem. Add if benchmarking tiny models on-device.",
        "url": "https://github.com/HKUDS/MiniRAG",
    },
    {
        "id": "raptor",
        "name": "RAPTOR",
        "status": "deferred",
        "notes": "Tree-of-summaries; cited in LazyGraphRAG comparisons. Heavy to reimplement locally.",
    },
]


def print_framework_catalog() -> None:
    for item in FRAMEWORKS:
        print(f"[{item['status']:10}] {item['id']:16} {item['name']}")
        print(f"             {item['notes']}")
        if item.get("url"):
            print(f"             {item['url']}")
