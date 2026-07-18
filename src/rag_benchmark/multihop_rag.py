"""MultiHop-RAG subset loader (Tang & Yang, COLM 2024).

Paper: https://arxiv.org/abs/2401.15391
Dataset: https://huggingface.co/datasets/yixuantt/MultiHopRAG

Query types: inference_query, comparison_query, temporal_query, null_query.
Evidence is spread across 2–4 news documents — the multi-hop retrieval stress test.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

QUESTION_TYPES = (
    "inference_query",
    "comparison_query",
    "temporal_query",
)

# Map to harness scenario column
TYPE_TO_QUERY = {
    "inference_query": "hybrid",
    "comparison_query": "local",
    "temporal_query": "hybrid",
    "null_query": "local",
}


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    return slug.strip("_")[:80] or "doc"


def build_multihop_rag_subset(
    *,
    project_root: Path,
    n_per_type: int = 3,
    n_distractors: int = 8,
) -> dict:
    """Closed-world MultiHop-RAG mini corpus: gold evidence docs + distractors."""
    from datasets import load_dataset

    questions = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split="train")
    corpus = load_dataset("yixuantt/MultiHopRAG", "corpus", split="train")
    by_title = {row["title"]: row for row in corpus}

    by_type: dict[str, list] = defaultdict(list)
    for q in questions:
        if q["question_type"] in QUESTION_TYPES:
            # Prefer questions whose evidence titles all resolve in the corpus
            titles = [e["title"] for e in q["evidence_list"]]
            if titles and all(t in by_title for t in titles):
                by_type[q["question_type"]].append(q)

    selected: list = []
    for qt in QUESTION_TYPES:
        # Prefer shorter answers for readable EM/F1, mid evidence count
        pool = sorted(
            by_type.get(qt, []),
            key=lambda r: (abs(len(r["evidence_list"]) - 3), len(str(r["answer"]))),
        )
        selected.extend(pool[:n_per_type])

    gold_titles: set[str] = set()
    for q in selected:
        for e in q["evidence_list"]:
            gold_titles.add(e["title"])

    # Distractors: other corpus docs not in gold set
    distractors = [row for row in corpus if row["title"] not in gold_titles][:n_distractors]
    docs = [by_title[t] for t in sorted(gold_titles)] + distractors

    corpus_dir = project_root / "data" / "corpus_multihop"
    qa_path = project_root / "data" / "qa" / "multihop_eval.json"
    catalog_path = project_root / "results" / "multihop_question_catalog.csv"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)

    for stale in corpus_dir.glob("*.txt"):
        stale.unlink()

    for i, row in enumerate(docs):
        body = row.get("body") or ""
        header = f"# {row['title']}\nSource: {row.get('source')}\n\n"
        (corpus_dir / f"{i:03d}_{_slug(row['title'])}.txt").write_text(
            header + body + "\n", encoding="utf-8"
        )

    eval_items: list[dict] = []
    catalog_rows = [
        "label,question_id,question_type,query_type,question,gold_answer,n_evidence"
    ]
    for i, q in enumerate(selected):
        qid = f"mh-{q['question_type'][:3]}-{i:03d}-{_slug(q['answer'])[:24]}"
        qt = q["question_type"]
        item = {
            "id": qid,
            "question": q["query"],
            "expected_answer": q["answer"],
            "query_type": TYPE_TO_QUERY.get(qt, "hybrid"),
            "best_method": "hybrid_rag",
            "source_doc": None,
            "rationale": f"MultiHop-RAG ({qt}) — Tang & Yang, COLM 2024",
            "multihop_type": qt,
            "hotpot_type": qt,  # reuse helpers that look for hotpot_type
            "evidence_titles": [e["title"] for e in q["evidence_list"]],
        }
        eval_items.append(item)
        q_esc = q["query"].replace('"', '""')
        a_esc = str(q["answer"]).replace('"', '""')
        catalog_rows.append(
            f'Q{i+1},{qid},{qt},{item["query_type"]},"{q_esc}","{a_esc}",{len(q["evidence_list"])}'
        )

    qa_path.write_text(json.dumps(eval_items, indent=2), encoding="utf-8")
    catalog_path.write_text("\n".join(catalog_rows) + "\n", encoding="utf-8")

    meta = {
        "dataset": "MultiHop-RAG",
        "paper": "Tang & Yang, COLM 2024",
        "citation": "https://arxiv.org/abs/2401.15391",
        "n_questions": len(eval_items),
        "n_documents": len(docs),
        "n_gold_docs": len(gold_titles),
        "n_distractors": len(distractors),
        "type_counts": dict(Counter(q["multihop_type"] for q in eval_items)),
        "corpus_dir": str(corpus_dir),
        "qa_path": str(qa_path),
        "catalog_path": str(catalog_path),
        "note": (
            "Closed-world: gold evidence articles + distractors. "
            "Use generative_score (judge+contains) for multi-hop fairness."
        ),
    }
    (project_root / "data" / "qa" / "multihop_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return {"corpus_dir": corpus_dir, "qa_path": qa_path, "meta": meta}
