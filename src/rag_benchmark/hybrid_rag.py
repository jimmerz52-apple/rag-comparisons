"""Hybrid RAG: vector retrieval + GraphRAG local, fused in one generation call.

State-of-the-art pattern (MS GraphRAG docs):
- Semantic/basic RAG for phrase-overlap fact lookup
- Graph local for entity/relationship context
- One fusion generation that *preserves specific facts* from both
  (avoid the common failure of regenerating from a graph prose answer
   after already paying for a semantic answer — that wastes tokens and
   often dilutes precise dates/names with a weak local LLM).
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_benchmark.config import BenchmarkConfig
from rag_benchmark.graph_rag import GraphRAGRunner
from rag_benchmark.llm_factory import TrackedLLMClient
from rag_benchmark.semantic_rag import SemanticRAG
from rag_benchmark.token_tracker import TokenLedger

HYBRID_PROMPT = """You are answering with hybrid retrieval: vector chunks + knowledge-graph context.

Rules:
1. Prefer concrete facts (dates, names, numbers, places, yes/no) present in either source.
2. If both sources agree, state the fact confidently in a SHORT answer when possible.
3. If one source has a specific detail the other lacks, INCLUDE that detail.
4. Do not invent facts. Do not hedge away concrete evidence already in the context.
5. For HotpotQA-style questions, answer with the minimal correct span (name, yes/no, number).
6. Put the short answer on the FIRST line by itself; optional explanation only after a blank line.

Question: {question}

=== Vector-retrieved document chunks ===
{semantic_context}

=== GraphRAG local (entities / relationships / community hints) ===
{graph_context}

Final answer:"""


@dataclass
class HybridQueryResult:
    answer: str
    semantic_chunks: list[str]
    graph_answer: str


class HybridRAG:
    """True hybrid: retrieve-only semantic + GraphRAG local → single fusion call."""

    def __init__(
        self,
        config: BenchmarkConfig,
        tracked_client: TrackedLLMClient,
        ledger: TokenLedger,
    ):
        self.config = config
        self.client = tracked_client
        self.ledger = ledger
        self.semantic = SemanticRAG(config, tracked_client, ledger)
        # Reuse the main GraphRAG workspace (same index as graph_rag / graph_local)
        # so hybrid does not rebuild a third expensive standard index.
        self.graph = GraphRAGRunner(
            config,
            ledger,
            workspace_dir=config.graph_workspace,
            indexing_method=config.graph_indexing_method,
            search_method=config.hybrid_graph_search_method,
        )

    def build_index(self) -> None:
        self.semantic.build_index()
        self.graph.build_index()

    def query(self, question: str) -> HybridQueryResult:
        # Retrieve chunks only — do NOT generate a semantic answer first.
        # That old pattern burned ~2x tokens and often diluted quality.
        chunks = self.semantic.retrieve(question)
        graph_result = self.graph.query(question)

        semantic_context = "\n\n---\n\n".join(chunks) if chunks else "(no chunks)"
        answer = self.client.chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": HYBRID_PROMPT.format(
                        question=question,
                        semantic_context=semantic_context,
                        graph_context=graph_result.answer,
                    ),
                }
            ],
            model=self.config.chat_model,
            phase="hybrid_query",
            temperature=0.0,
        )
        return HybridQueryResult(
            answer=answer,
            semantic_chunks=chunks,
            graph_answer=graph_result.answer,
        )
