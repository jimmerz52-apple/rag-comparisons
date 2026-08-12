"""Research-backed RAG index / memory data structures (local lite).

Fidelity notes (read before treating scores as paper reproductions):

1. **RAPTOR** (Sarthi et al., ICLR 2024) — recursive embed→cluster→summarize tree.
   - We use sklearn GaussianMixture + BIC for cluster count (paper: GMM).
   - Soft membership: nodes with P(cluster) ≥ threshold join multiple summaries.
   - Querying: **collapsed tree** (paper's preferred method; beats tree traversal).
   - Deferred vs paper: UMAP pre-reduction + two-step global/local clustering;
     SBERT multi-qa-mpnet; GPT-3.5 summarizer. Our embedder/chat models differ.

2. **Parent–child / small-to-big** — retrieve fine units, expand to parent context
   (LlamaIndex ParentDocumentRetriever / common RAG engineering). Distinct from
   RAPTOR: adjacency windows, not semantic clustering trees.

3. **Proposition index** (Chen et al., Dense X Retrieval, EMNLP 2024) — atomic,
   self-contained factoids with coreference resolved. We LLM-propositionize with
   the local chat model (paper: Flan-T5 Propositionizer). Sentence splitting alone
   is NOT proposition indexing.

Removed earlier ``tiered_memory_rag`` (working/episodic/semantic fusion labeled
"MemGPT"): MemGPT (Packer et al. 2023) is an OS-style agent paging architecture
(working context + FIFO + archival/recall via tool calls), not a static multi-index
RAG fusion. That label was research-incorrect.

Prefer generative_score for RAPTOR (summary nodes). Proposition index is closer
to extractive F1/EM when answers are short spans.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
import numpy as np
from sklearn.mixture import GaussianMixture

from rag_benchmark.config import BenchmarkConfig
from rag_benchmark.corpus import Document, TextChunk, chunk_documents, load_documents
from rag_benchmark.llm_factory import TrackedLLMClient
from rag_benchmark.prompts import ANSWER_PROMPT
from rag_benchmark.semantic_rag import QueryResult, SemanticRAG
from rag_benchmark.token_tracker import TokenLedger

SUMMARIZE_PROMPT = """Summarize the following passages into 2–4 dense sentences.
Preserve named entities, numbers, and causal links. No preamble.

Passages:
{text}

Summary:"""

PROPOSITIONIZE_PROMPT = """Decompose the passage into atomic propositions.

Rules (Dense X Retrieval / proposition style):
- Each proposition is one self-contained factoid in plain English.
- Resolve pronouns/coreferences to full entity names from the passage.
- Do not invent facts absent from the passage.
- Output one proposition per line. No numbering, bullets, or preamble.

Passage:
{text}

Propositions:"""


@dataclass
class TreeNode:
    node_id: str
    text: str
    level: int  # 0 = leaf chunk
    children: list[str] = field(default_factory=list)


class RaptorRAG:
    """RAPTOR-lite: GMM cluster → summarize → collapsed-tree retrieve."""

    def __init__(self, config: BenchmarkConfig, tracked_client: TrackedLLMClient, ledger: TokenLedger):
        self.config = config
        self.client = tracked_client
        self.ledger = ledger
        self.workspace = Path(config.raptor_workspace)
        self.max_levels = config.raptor_max_levels
        self.cluster_size = max(2, config.raptor_cluster_size)
        self.soft_threshold = 0.15  # GMM soft membership (paper: soft clustering)
        self._nodes: list[TreeNode] = []
        self._collection = None

    def build_index(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        tree_path = self.workspace / "tree.json"
        chroma_path = self.workspace / "chroma"
        chroma = chromadb.PersistentClient(path=str(chroma_path))
        collection_name = "raptor_nodes"

        if self.config.reuse_indexes and tree_path.exists():
            payload = json.loads(tree_path.read_text(encoding="utf-8"))
            self._nodes = [TreeNode(**n) for n in payload["nodes"]]
            self._collection = chroma.get_or_create_collection(collection_name)
            if self._collection.count() >= len(self._nodes):
                return

        documents = load_documents(self.config.corpus_dir, self.config.max_documents)
        chunks = chunk_documents(
            documents,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        if not chunks:
            raise ValueError(f"No chunks for RAPTOR in {self.config.corpus_dir}")

        nodes: list[TreeNode] = [
            TreeNode(node_id=c.chunk_id, text=c.text, level=0) for c in chunks
        ]
        level_ids = [n.node_id for n in nodes]
        id_to_node = {n.node_id: n for n in nodes}

        embeddings: dict[str, list[float]] = {}
        batch = 32
        for start in range(0, len(nodes), batch):
            part = nodes[start : start + batch]
            vecs = self.client.embed_texts(
                [n.text for n in part],
                model=self.config.embedding_model,
                phase="raptor_embed_leaf",
            )
            for n, v in zip(part, vecs):
                embeddings[n.node_id] = v

        current_level = 0
        while current_level < self.max_levels and len(level_ids) >= self.cluster_size * 2:
            clusters = self._gmm_cluster(level_ids, embeddings)
            if len(clusters) <= 1:
                break
            next_ids: list[str] = []
            summarized_members: set[str] = set()
            for ci, member_ids in enumerate(clusters):
                if len(member_ids) < 2:
                    continue
                joined = "\n\n".join(id_to_node[mid].text[:1200] for mid in member_ids[:8])
                summary = self.client.chat_completion(
                    messages=[
                        {
                            "role": "user",
                            "content": SUMMARIZE_PROMPT.format(text=joined[:6000]),
                        }
                    ],
                    model=self.config.chat_model,
                    phase="raptor_summarize",
                    temperature=0.0,
                ).strip()
                nid = f"L{current_level + 1}_C{ci}"
                parent = TreeNode(
                    node_id=nid,
                    text=summary,
                    level=current_level + 1,
                    children=list(member_ids),
                )
                nodes.append(parent)
                id_to_node[nid] = parent
                emb = self.client.embed_texts(
                    [summary],
                    model=self.config.embedding_model,
                    phase="raptor_embed_summary",
                )[0]
                embeddings[nid] = emb
                next_ids.append(nid)
                summarized_members.update(member_ids)
            # Carry unclustered / singleton nodes up so they remain searchable
            for mid in level_ids:
                if mid not in summarized_members:
                    next_ids.append(mid)
            seen_ids: set[str] = set()
            deduped: list[str] = []
            for nid in next_ids:
                if nid not in seen_ids:
                    seen_ids.add(nid)
                    deduped.append(nid)
            level_ids = deduped
            if not any(id_to_node[i].level == current_level + 1 for i in level_ids):
                break
            current_level += 1

        self._nodes = nodes
        tree_path.write_text(
            json.dumps(
                {
                    "nodes": [n.__dict__ for n in nodes],
                    "max_level": current_level,
                    "n_leaves": sum(1 for n in nodes if n.level == 0),
                    "retrieval": "collapsed_tree",
                    "clustering": "gmm_bic_soft",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        try:
            chroma.delete_collection(collection_name)
        except Exception:
            pass
        self._collection = chroma.get_or_create_collection(collection_name)
        for start in range(0, len(nodes), batch):
            part = nodes[start : start + batch]
            self._collection.upsert(
                ids=[n.node_id for n in part],
                documents=[n.text for n in part],
                embeddings=[embeddings[n.node_id] for n in part],
                metadatas=[{"level": n.level} for n in part],
            )

    def _gmm_cluster(self, ids: list[str], embeddings: dict[str, list[float]]) -> list[list[str]]:
        """GMM + BIC (paper); soft membership above probability threshold."""
        if len(ids) <= self.cluster_size:
            return [ids]
        mat = np.array([embeddings[i] for i in ids], dtype=np.float64)
        # L2-normalize — embeddings are often cosine-oriented
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat = mat / np.clip(norms, 1e-9, None)

        max_k = min(len(ids) // self.cluster_size, len(ids) // 2, 12)
        max_k = max(2, max_k)
        best_k = 2
        best_bic = np.inf
        best_model: GaussianMixture | None = None
        for k in range(2, max_k + 1):
            try:
                model = GaussianMixture(
                    n_components=k,
                    covariance_type="full",
                    random_state=self.config.random_seed,
                    max_iter=100,
                    n_init=1,
                )
                model.fit(mat)
                bic = model.bic(mat)
                if bic < best_bic:
                    best_bic = bic
                    best_k = k
                    best_model = model
            except Exception:
                continue
        if best_model is None:
            # Fallback: contiguous groups of cluster_size
            return [
                ids[i : i + self.cluster_size]
                for i in range(0, len(ids), self.cluster_size)
                if len(ids[i : i + self.cluster_size]) >= 2
            ]

        probs = best_model.predict_proba(mat)
        hard = probs.argmax(axis=1)
        clusters: list[list[str]] = [[] for _ in range(best_k)]
        for idx, nid in enumerate(ids):
            assigned = {int(hard[idx])}
            for j, p in enumerate(probs[idx]):
                if p >= self.soft_threshold:
                    assigned.add(j)
            for j in assigned:
                clusters[j].append(nid)
        return [c for c in clusters if len(c) >= 2] or [ids]

    def retrieve(self, question: str) -> list[str]:
        """Collapsed-tree retrieval (Sarthi et al.): top-k over all levels jointly."""
        if self._collection is None:
            raise RuntimeError("RAPTOR index not built")
        q = self.client.embed_texts(
            [question], model=self.config.embedding_model, phase="raptor_query"
        )[0]
        n = min(self.config.semantic_top_k, max(1, self._collection.count()))
        res = self._collection.query(query_embeddings=[q], n_results=n)
        return list(res.get("documents", [[]])[0])

    def query(self, question: str) -> QueryResult:
        chunks = self.retrieve(question)
        context = "\n\n---\n\n".join(chunks)
        answer = self.client.chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": ANSWER_PROMPT.format(question=question, context=context),
                }
            ],
            model=self.config.chat_model,
            phase="raptor_generate",
            temperature=0.0,
        )
        return QueryResult(answer=answer, retrieved_chunks=chunks)


class ParentChildRAG(SemanticRAG):
    """Small-to-big retrieval: score child chunks, return parent windows.

    Research / practice family: ParentDocumentRetriever (LlamaIndex), dense
    hierarchical retrieval (doc→passage), small-to-big context expansion.
    """

    def __init__(self, config: BenchmarkConfig, tracked_client: TrackedLLMClient, ledger: TokenLedger):
        super().__init__(config, tracked_client, ledger)
        self.workspace = Path(config.parent_child_workspace)
        self.child_size = config.parent_child_child_size
        self.parent_size = config.parent_child_parent_size
        self._parent_text: dict[str, str] = {}
        self._child_to_parent: dict[str, str] = {}
        self._collection_name = f"{config.semantic_collection}_parent_child"

    def build_index(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        map_path = self.workspace / "parent_child_map.json"
        documents = load_documents(self.config.corpus_dir, self.config.max_documents)

        if self.config.reuse_indexes and map_path.exists():
            payload = json.loads(map_path.read_text(encoding="utf-8"))
            self._parent_text = payload["parents"]
            self._child_to_parent = payload["child_to_parent"]
            self._chunks = [
                TextChunk(
                    chunk_id=c["chunk_id"],
                    doc_id=c["doc_id"],
                    text=c["text"],
                    source_path=Path(c["source_path"]),
                )
                for c in payload["children"]
            ]
        else:
            parents = chunk_documents(
                documents,
                chunk_size=self.parent_size,
                chunk_overlap=max(50, self.parent_size // 10),
            )
            children: list[TextChunk] = []
            child_to_parent: dict[str, str] = {}
            parent_text = {p.chunk_id: p.text for p in parents}

            for parent in parents:
                # Re-chunk parent text into smaller children that share parent id
                fake_doc = Document(
                    doc_id=parent.chunk_id,
                    title=parent.chunk_id,
                    text=parent.text,
                    source_path=parent.source_path,
                )
                kids = chunk_documents(
                    [fake_doc],
                    chunk_size=self.child_size,
                    chunk_overlap=max(20, self.child_size // 5),
                )
                for kid in kids:
                    children.append(kid)
                    child_to_parent[kid.chunk_id] = parent.chunk_id

            self._chunks = children
            self._parent_text = parent_text
            self._child_to_parent = child_to_parent
            map_path.write_text(
                json.dumps(
                    {
                        "parents": parent_text,
                        "child_to_parent": child_to_parent,
                        "children": [
                            {
                                "chunk_id": c.chunk_id,
                                "doc_id": c.doc_id,
                                "text": c.text,
                                "source_path": str(c.source_path),
                            }
                            for c in children
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        chroma = chromadb.PersistentClient(path=str(self.workspace / "chroma"))
        if not self.config.reuse_indexes:
            try:
                chroma.delete_collection(self._collection_name)
            except Exception:
                pass
        self._collection = chroma.get_or_create_collection(self._collection_name)
        if self.config.reuse_indexes and self._collection.count() >= len(self._chunks):
            return

        batch_size = 64
        for start in range(0, len(self._chunks), batch_size):
            batch = self._chunks[start : start + batch_size]
            embeddings = self.client.embed_texts(
                [c.text for c in batch],
                model=self.config.embedding_model,
                phase="parent_child_index",
            )
            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                embeddings=embeddings,
                metadatas=[
                    {"parent_id": self._child_to_parent.get(c.chunk_id, c.chunk_id)}
                    for c in batch
                ],
            )

    def retrieve(self, question: str) -> list[str]:
        """Retrieve by child similarity; expand to unique parent windows."""
        if self._collection is None:
            raise RuntimeError("Parent–child index not built")
        q = self.client.embed_texts(
            [question], model=self.config.embedding_model, phase="parent_child_query"
        )[0]
        # Over-fetch children so parent expansion still yields diversity
        n = min(max(self.config.semantic_top_k * 3, 6), max(1, self._collection.count()))
        res = self._collection.query(query_embeddings=[q], n_results=n)
        ids = res.get("ids", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        ordered_parents: list[str] = []
        seen: set[str] = set()
        for cid, meta in zip(ids, metas):
            pid = (meta or {}).get("parent_id") or self._child_to_parent.get(cid, cid)
            if pid in seen:
                continue
            seen.add(pid)
            text = self._parent_text.get(pid)
            if text:
                ordered_parents.append(text)
            if len(ordered_parents) >= self.config.semantic_top_k:
                break
        return ordered_parents


class PropositionRAG(SemanticRAG):
    """Dense-X-style proposition index via local LLM propositionizer."""

    def __init__(self, config: BenchmarkConfig, tracked_client: TrackedLLMClient, ledger: TokenLedger):
        super().__init__(config, tracked_client, ledger)
        self.workspace = Path(config.proposition_workspace)
        self._prop_collection_name = f"{config.semantic_collection}_propositions"

    def build_index(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        prop_path = self.workspace / "propositions.json"
        documents = load_documents(self.config.corpus_dir, self.config.max_documents)
        base_chunks = chunk_documents(
            documents,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

        if self.config.reuse_indexes and prop_path.exists():
            rows = json.loads(prop_path.read_text(encoding="utf-8"))
            self._chunks = [
                TextChunk(
                    chunk_id=r["chunk_id"],
                    doc_id=r["doc_id"],
                    text=r["text"],
                    source_path=Path(r["source_path"]),
                )
                for r in rows
            ]
        else:
            props: list[TextChunk] = []
            for chunk in base_chunks:
                for i, prop in enumerate(self._propositionize(chunk.text)):
                    props.append(
                        TextChunk(
                            chunk_id=f"{chunk.chunk_id}_p{i}",
                            doc_id=chunk.doc_id,
                            text=prop,
                            source_path=chunk.source_path,
                        )
                    )
            self._chunks = props or base_chunks
            prop_path.write_text(
                json.dumps(
                    [
                        {
                            "chunk_id": c.chunk_id,
                            "doc_id": c.doc_id,
                            "text": c.text,
                            "source_path": str(c.source_path),
                        }
                        for c in self._chunks
                    ],
                    indent=2,
                ),
                encoding="utf-8",
            )

        chroma = chromadb.PersistentClient(path=str(self.workspace / "chroma"))
        if not self.config.reuse_indexes:
            try:
                chroma.delete_collection(self._prop_collection_name)
            except Exception:
                pass
        self._collection = chroma.get_or_create_collection(self._prop_collection_name)
        if self.config.reuse_indexes and self._collection.count() >= len(self._chunks):
            return

        batch_size = 64
        for start in range(0, len(self._chunks), batch_size):
            batch = self._chunks[start : start + batch_size]
            embeddings = self.client.embed_texts(
                [c.text for c in batch],
                model=self.config.embedding_model,
                phase="proposition_index",
            )
            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                embeddings=embeddings,
                metadatas=[{"doc_id": c.doc_id} for c in batch],
            )

    def _propositionize(self, text: str) -> list[str]:
        """LLM propositionizer with sentence-split fallback."""
        snippet = text[:3500]
        raw = self.client.chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": PROPOSITIONIZE_PROMPT.format(text=snippet),
                }
            ],
            model=self.config.chat_model,
            phase="propositionize",
            temperature=0.0,
        )
        props = _parse_proposition_lines(raw)
        if len(props) >= 2:
            return props[:30]
        # Fallback: sentence units (weaker than true propositions)
        return _split_sentences(snippet)[:20]


def _parse_proposition_lines(raw: str) -> list[str]:
    out: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        line = re.sub(r"^[\-\*\d\.\)\]]+\s*", "", line)
        if len(line) < 15:
            continue
        if line.lower().startswith(("passage", "proposition", "here are", "output")):
            continue
        out.append(line)
    return out


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) >= 20]
