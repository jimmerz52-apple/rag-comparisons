"""HotpotQA distractor subset loader (Yang et al., EMNLP 2018).

Standard multi-hop QA benchmark used across GraphRAG literature.
Distractor setting: 2 gold + 8 distractor Wikipedia paragraphs per question —
a closed corpus, so we do not index all of Wikipedia.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


def _slug(title: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", title.strip().lower())
    return slug.strip("_")[:80] or "doc"


def _load_hotpot_validation(n: int) -> list[dict]:
    from datasets import load_dataset

    # Prefer hard multi-hop examples for GraphRAG stress tests.
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    hard = [row for row in ds if row.get("level") == "hard"]
    bridge = [row for row in hard if row.get("type") == "bridge"]
    compare = [row for row in hard if row.get("type") == "comparison"]

    # Balanced sample: half bridge (multi-hop), half comparison
    n_bridge = n // 2
    n_compare = n - n_bridge
    selected = bridge[:n_bridge] + compare[:n_compare]
    if len(selected) < n:
        selected = hard[:n] if hard else list(ds)[:n]
    return selected[:n]


def build_hotpot_subset(
    *,
    project_root: Path,
    n_questions: int = 20,
    seed: int = 42,
    prefer_hard: bool = True,
) -> dict:
    """Materialize a tiny HotpotQA distractor corpus + eval JSON."""
    del seed, prefer_hard  # deterministic slice from HF validation hard split
    selected = _load_hotpot_validation(n_questions)

    corpus_dir = project_root / "data" / "corpus_hotpot"
    qa_path = project_root / "data" / "qa" / "hotpot_eval.json"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    qa_path.parent.mkdir(parents=True, exist_ok=True)

    for stale in corpus_dir.glob("*.txt"):
        stale.unlink()

    written: set[str] = set()
    eval_items: list[dict] = []

    for row in selected:
        qid = row["id"]
        answer = row["answer"]
        qtype = row.get("type", "bridge")
        level = row.get("level", "hard")
        ctx = row["context"]
        titles = ctx["title"]
        sentences = ctx["sentences"]

        for title, sents in zip(titles, sentences):
            slug = _slug(title)
            if slug in written:
                continue
            text = " ".join(sents).strip()
            if not text:
                continue
            (corpus_dir / f"{slug}.txt").write_text(f"# {title}\n\n{text}\n", encoding="utf-8")
            written.add(slug)

        # Hotpot bridge ≈ multi-hop / hybrid; comparison often local+entity
        query_type = "hybrid" if qtype == "bridge" else "local"
        best_method = "hybrid_rag" if qtype == "bridge" else "semantic_rag"

        eval_items.append(
            {
                "id": qid,
                "question": row["question"],
                "expected_answer": answer,
                "query_type": query_type,
                "best_method": best_method,
                "source_doc": None,
                "rationale": f"HotpotQA distractor ({level}/{qtype}) — Yang et al. EMNLP 2018",
                "hotpot_type": qtype,
                "hotpot_level": level,
            }
        )

    qa_path.write_text(json.dumps(eval_items, indent=2), encoding="utf-8")
    meta = {
        "dataset": "HotpotQA",
        "paper": "Yang et al., EMNLP 2018",
        "citation": "https://hotpotqa.github.io/",
        "setting": "distractor",
        "source": "hotpotqa/hotpot_qa (HuggingFace)",
        "n_questions": len(eval_items),
        "n_documents": len(written),
        "type_counts": dict(Counter(i["hotpot_type"] for i in eval_items)),
        "corpus_dir": str(corpus_dir),
        "qa_path": str(qa_path),
    }
    (project_root / "data" / "qa" / "hotpot_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return {"corpus_dir": corpus_dir, "qa_path": qa_path, "meta": meta}
