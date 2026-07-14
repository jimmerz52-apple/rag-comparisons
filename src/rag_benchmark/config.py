from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BenchmarkConfig:
    project_root: Path
    corpus_dir: Path
    qa_path: Path
    max_documents: int = 25
    chunk_size: int = 800
    chunk_overlap: int = 100
    semantic_top_k: int = 5
    semantic_collection: str = "semantic_rag_corpus"
    graph_workspace: Path = field(default_factory=lambda: Path("graphrag_workspaces/full"))
    lazy_workspace: Path = field(default_factory=lambda: Path("graphrag_workspaces/lazy"))
    hybrid_graph_workspace: Path = field(
        default_factory=lambda: Path("graphrag_workspaces/hybrid")
    )
    lightrag_workspace: Path = field(default_factory=lambda: Path("lightrag_workspaces/default"))
    lightrag_mode: str = "hybrid"
    lightrag_embedding_dim: int = 768
    hipporag_workspace: Path = field(default_factory=lambda: Path("hipporag_workspaces/default"))
    graph_search_method: str = "global"
    graph_indexing_method: str = "fast"
    hybrid_graph_search_method: str = "local"
    lazy_search_method: str = "basic"
    lazy_relevance_budget: int = 500
    community_level: int = 2
    llm_backend: str = "local"
    ollama_base_url: str = "http://127.0.0.1:11434"
    openai_api_key: str | None = None
    chat_model: str = "llama3.2:3b"
    embedding_model: str = "all-MiniLM-L6-v2"
    judge_model: str = "llama3.2:3b"
    graphrag_chat_model: str = "llama3.2:3b"
    graphrag_embedding_model: str = "nomic-embed-text"
    pricing: dict[str, dict[str, float]] = field(default_factory=dict)
    random_seed: int = 42
    reuse_indexes: bool = False

    @classmethod
    def from_yaml(cls, project_root: Path, config_path: Path | None = None) -> "BenchmarkConfig":
        project_root = project_root.resolve()
        config_path = config_path or project_root / "config" / "benchmark.yaml"
        with config_path.open("r", encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle)

        benchmark = raw.get("benchmark", {})
        corpus = raw.get("corpus", {})
        semantic = raw.get("semantic_rag", {})
        graph = raw.get("graph_rag", {})
        hybrid = raw.get("hybrid_rag", {})
        lazy = raw.get("lazygraph_rag", {})
        light = raw.get("light_rag", {})
        hippo = raw.get("hippo_rag", {})
        evaluation = raw.get("evaluation", {})
        pricing = raw.get("pricing", {})
        llm = raw.get("llm", {})

        backend = os.getenv("LLM_BACKEND", llm.get("backend", "local"))
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", llm.get("ollama_base_url", "http://127.0.0.1:11434"))
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if backend == "local":
            chat_model = os.getenv("LOCAL_CHAT_MODEL", llm.get("chat_model", "llama3.2:3b"))
            embedding_model = os.getenv(
                "LOCAL_EMBEDDING_MODEL", llm.get("embedding_model", "all-MiniLM-L6-v2")
            )
            judge_model = os.getenv("LOCAL_JUDGE_MODEL", llm.get("judge_model", chat_model))
            graphrag_chat = llm.get("graphrag_chat_model", chat_model)
            graphrag_embed = llm.get("graphrag_embedding_model", "nomic-embed-text")
        else:
            chat_model = os.getenv("OPENAI_CHAT_MODEL", llm.get("chat_model", "gpt-4o-mini"))
            embedding_model = os.getenv(
                "OPENAI_EMBEDDING_MODEL", llm.get("embedding_model", "text-embedding-3-small")
            )
            judge_model = os.getenv("OPENAI_JUDGE_MODEL", llm.get("judge_model", chat_model))
            graphrag_chat = chat_model
            graphrag_embed = embedding_model

        return cls(
            project_root=project_root,
            corpus_dir=project_root / corpus.get("source_dir", "data/corpus"),
            qa_path=project_root / evaluation.get("qa_path", "data/qa/eval_questions.json"),
            max_documents=corpus.get("max_documents", 25),
            chunk_size=corpus.get("chunk_size", 800),
            chunk_overlap=corpus.get("chunk_overlap", 100),
            semantic_top_k=semantic.get("top_k", 5),
            semantic_collection=semantic.get("collection_name", "semantic_rag_corpus"),
            graph_workspace=project_root / graph.get("workspace_dir", "graphrag_workspaces/full"),
            lazy_workspace=project_root / lazy.get("workspace_dir", "graphrag_workspaces/lazy"),
            hybrid_graph_workspace=project_root
            / hybrid.get("workspace_dir", "graphrag_workspaces/hybrid"),
            lightrag_workspace=project_root
            / light.get("workspace_dir", "lightrag_workspaces/default"),
            lightrag_mode=light.get("mode", "hybrid"),
            lightrag_embedding_dim=int(light.get("embedding_dim", 768)),
            hipporag_workspace=project_root
            / hippo.get("workspace_dir", "hipporag_workspaces/default"),
            graph_search_method=graph.get("search_method", "global"),
            graph_indexing_method=graph.get(
                "indexing_method",
                "fast" if backend == "local" else "standard",
            ),
            hybrid_graph_search_method=hybrid.get("graph_search_method", "local"),
            lazy_search_method=lazy.get("search_method", "basic"),
            lazy_relevance_budget=lazy.get("relevance_budget", 500),
            community_level=graph.get("community_level", 2),
            llm_backend=backend,
            ollama_base_url=ollama_base_url,
            openai_api_key=openai_api_key,
            chat_model=chat_model,
            embedding_model=embedding_model,
            judge_model=judge_model,
            graphrag_chat_model=graphrag_chat,
            graphrag_embedding_model=graphrag_embed,
            pricing=pricing,
            random_seed=benchmark.get("random_seed", 42),
            reuse_indexes=benchmark.get("reuse_indexes", False),
        )

    def results_dir(self) -> Path:
        path = self.project_root / "results"
        path.mkdir(parents=True, exist_ok=True)
        return path
