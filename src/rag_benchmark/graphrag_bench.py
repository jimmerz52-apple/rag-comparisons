"""GraphRAG-Bench subset loader (Xiang et al., ICLR 2026).

Paper: "When to use Graphs in RAG" — arXiv:2506.05690
Dataset: https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench
Repo: https://github.com/GraphRAG-Bench/GraphRAG-Benchmark

Four task levels of increasing difficulty:
  Fact Retrieval → Complex Reasoning → Contextual Summarize → Creative Generation

Paper finding we demo locally:
  Obs.1 Basic RAG ≈ GraphRAG on fact retrieval
  Obs.2 GraphRAG wins on complex / summarize / creative
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

QUESTION_TYPES = (
    "Fact Retrieval",
    "Complex Reasoning",
    "Contextual Summarize",
    "Creative Generation",
)

# Map GraphRAG-Bench levels → our scenario column (routing / charts)
TYPE_TO_QUERY = {
    "Fact Retrieval": "local",  # vector should compete / win
    "Complex Reasoning": "hybrid",
    "Contextual Summarize": "hybrid",
    "Creative Generation": "hybrid",
}

GITHUB_RAW = "https://raw.githubusercontent.com/GraphRAG-Bench/GraphRAG-Benchmark/main/Datasets"
CORPUS_URL = f"{GITHUB_RAW}/Corpus/novel.json"
QUESTIONS_URL = f"{GITHUB_RAW}/Questions/novel_questions.json"


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    return slug.strip("_")[:80] or "doc"


def _download_json(url: str, dest: Path) -> list | dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        return json.loads(dest.read_text(encoding="utf-8"))
    print(f"Downloading {url} …")
    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = resp.read()
    dest.write_bytes(payload)
    return json.loads(payload.decode("utf-8"))


def _load_novel_assets(cache_dir: Path) -> tuple[list[dict], list[dict]]:
    """Return (corpus_rows, questions) for the Novel split."""
    # Prefer a sibling clone if present (offline / faster)
    local_root = Path("/tmp/GraphRAG-Benchmark/Datasets")
    if (local_root / "Corpus/novel.json").exists():
        corpus = json.loads((local_root / "Corpus/novel.json").read_text(encoding="utf-8"))
        questions = json.loads(
            (local_root / "Questions/novel_questions.json").read_text(encoding="utf-8")
        )
        return corpus, questions

    corpus = _download_json(CORPUS_URL, cache_dir / "novel_corpus.json")
    questions = _download_json(QUESTIONS_URL, cache_dir / "novel_questions.json")
    return corpus, questions


def _pick_source(corpus: list[dict], questions: list[dict]) -> str:
    """Smallest novel that covers all four question types."""
    by_src: dict[str, Counter] = defaultdict(Counter)
    for q in questions:
        by_src[q["source"]][q["question_type"]] += 1
    sizes = {row["corpus_name"]: len(row["context"]) for row in corpus}
    ranked = []
    for src, counts in by_src.items():
        if src not in sizes:
            continue
        if not all(qt in counts for qt in QUESTION_TYPES):
            continue
        ranked.append((sizes[src], src, dict(counts)))
    if not ranked:
        # Fallback: densest source
        src, _ = max(((s, sum(c.values())) for s, c in by_src.items()), key=lambda x: x[1])
        return src
    ranked.sort()
    return ranked[0][1]


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def _qa_looks_coherent(question: dict) -> bool:
    """Drop obviously misaligned GraphRAG-Bench rows (bad upstream pairs).

    Example failure mode: Port William question paired with a Pepys gold answer.
    Heuristic: require some content-word overlap between Q and A, or between A and
    evidence snippets when evidence is available.
    """
    q = str(question.get("question", ""))
    a = str(question.get("answer", ""))
    q_tok, a_tok = _tokenize(q), _tokenize(a)
    if not a_tok:
        return False
    # Proper-name-ish tokens in the question should not be totally absent from answer
    # when the answer is short (fact / complex). Skip if Q∩A is empty AND answer
    # also fails to overlap evidence.
    q_cap = set(re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", q)) - {
        "What",
        "How",
        "Which",
        "When",
        "Where",
        "Who",
        "Why",
        "During",
        "Rewrite",
        "Describe",
        "Explain",
        "The",
        "This",
    }
    if q_cap and not any(name.lower() in a.lower() for name in q_cap):
        evidence = question.get("evidence") or []
        ev_text = " ".join(str(e) for e in evidence[:8])
        if a_tok.isdisjoint(_tokenize(ev_text)) and q_tok.isdisjoint(a_tok):
            return False
    # Soft positive: any token overlap Q↔A or A↔evidence
    evidence = question.get("evidence") or []
    ev_tok = _tokenize(" ".join(str(e) for e in evidence[:12]))
    if q_tok & a_tok:
        return True
    if a_tok & ev_tok:
        return True
    # Creative / summarize answers can be long paraphrases — allow if answer is long
    return len(a) >= 120


def _chunk_novel(text: str, *, max_chars: int = 3500) -> list[str]:
    """Split a novel into paragraph packs (hard-split oversized paragraphs)."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        paras = [text]

    # Hard-split any paragraph longer than max_chars
    flat: list[str] = []
    for p in paras:
        if len(p) <= max_chars:
            flat.append(p)
            continue
        for i in range(0, len(p), max_chars):
            flat.append(p[i : i + max_chars])

    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for p in flat:
        if size + len(p) > max_chars and buf:
            chunks.append("\n\n".join(buf))
            buf, size = [], 0
        buf.append(p)
        size += len(p) + 2
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks or [text[:max_chars]]


def build_graphrag_bench_subset(
    *,
    project_root: Path,
    n_per_type: int = 3,
    source: str | None = None,
) -> dict:
    """Materialize a Novel-split GraphRAG-Bench demo corpus + QA JSON."""
    cache_dir = project_root / "data" / "graphrag_bench_cache"
    corpus_rows, questions = _load_novel_assets(cache_dir)

    source_name = source or _pick_source(corpus_rows, questions)
    novel = next(r for r in corpus_rows if r["corpus_name"] == source_name)
    pool = [q for q in questions if q["source"] == source_name]

    by_type: dict[str, list[dict]] = defaultdict(list)
    for q in pool:
        by_type[q["question_type"]].append(q)

    selected: list[dict] = []
    for qt in QUESTION_TYPES:
        pool_qt = [q for q in by_type.get(qt, []) if _qa_looks_coherent(q)]
        if len(pool_qt) < n_per_type:
            # Fall back to remaining rows if filter is too aggressive
            seen = {q["id"] for q in pool_qt}
            pool_qt.extend(q for q in by_type.get(qt, []) if q["id"] not in seen)
        # Prefer mid-length answers (very short often noisy; very long hard for EM)
        ranked = sorted(
            pool_qt,
            key=lambda r: abs(len(str(r.get("answer", ""))) - 80),
        )
        selected.extend(ranked[:n_per_type])

    corpus_dir = project_root / "data" / "corpus_graphrag_bench"
    qa_path = project_root / "data" / "qa" / "graphrag_bench_eval.json"
    catalog_path = project_root / "results" / "graphrag_bench_question_catalog.csv"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)

    for stale in corpus_dir.glob("*.txt"):
        stale.unlink()

    chunks = _chunk_novel(novel["context"])
    for i, chunk in enumerate(chunks):
        (corpus_dir / f"{_slug(source_name)}_{i:03d}.txt").write_text(
            f"# {source_name} part {i}\n\n{chunk}\n", encoding="utf-8"
        )

    eval_items: list[dict] = []
    catalog_rows: list[str] = [
        "label,question_id,question_type,query_type,question,gold_answer,source"
    ]
    for i, q in enumerate(selected):
        qt = q["question_type"]
        item = {
            "id": q["id"],
            "question": q["question"],
            "expected_answer": q["answer"],
            "query_type": TYPE_TO_QUERY.get(qt, "hybrid"),
            "best_method": "hybrid_rag" if qt != "Fact Retrieval" else "semantic_rag",
            "source_doc": source_name,
            "rationale": (
                f"GraphRAG-Bench Novel ({qt}) — Xiang et al., ICLR 2026 / arXiv:2506.05690"
            ),
            "graphrag_bench_type": qt,
            "hotpot_type": qt,  # reuse catalog helpers that look for hotpot_type
            "evidence": q.get("evidence", []),
        }
        eval_items.append(item)
        q_esc = q["question"].replace('"', '""')
        a_esc = str(q["answer"]).replace('"', '""')
        catalog_rows.append(
            f'Q{i+1},{q["id"]},{qt},{item["query_type"]},"{q_esc}","{a_esc}",{source_name}'
        )

    qa_path.write_text(json.dumps(eval_items, indent=2), encoding="utf-8")
    catalog_path.write_text("\n".join(catalog_rows) + "\n", encoding="utf-8")

    type_counts = Counter(q["graphrag_bench_type"] for q in eval_items)
    meta = {
        "dataset": "GraphRAG-Bench",
        "paper": "Xiang et al., ICLR 2026",
        "citation": "https://arxiv.org/abs/2506.05690",
        "subset": "novel",
        "source": source_name,
        "n_questions": len(eval_items),
        "n_documents": len(list(corpus_dir.glob("*.txt"))),
        "type_counts": dict(type_counts),
        "corpus_dir": str(corpus_dir),
        "qa_path": str(qa_path),
        "catalog_path": str(catalog_path),
        "findings": {
            "obs1": "Basic RAG matches GraphRAG on Fact Retrieval",
            "obs2": "GraphRAG excels on Complex Reasoning / Summarize / Creative",
        },
    }
    (project_root / "data" / "qa" / "graphrag_bench_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return {"corpus_dir": corpus_dir, "qa_path": qa_path, "meta": meta}
