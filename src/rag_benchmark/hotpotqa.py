"""HotpotQA distractor subset loader (Yang et al., EMNLP 2018).

Standard multi-hop QA benchmark used across GraphRAG literature.
Distractor setting: 2 gold + 8 distractor Wikipedia paragraphs per question —
a closed corpus, so we do not index all of Wikipedia.

Full distractor validation is 7,405 hard questions (~66k unique paragraphs).
Use n_questions=None / "all" for the full split; prefer 100–500 for local runs.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path


def _slug(title: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", title.strip().lower())
    return slug.strip("_")[:80] or "doc"


def _load_hotpot_validation(n: int | None) -> list[dict]:
    from datasets import load_dataset

    # Prefer hard multi-hop examples for GraphRAG stress tests.
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    hard = [row for row in ds if row.get("level") == "hard"]
    bridge = [row for row in hard if row.get("type") == "bridge"]
    compare = [row for row in hard if row.get("type") == "comparison"]

    if n is None:
        # Full hard validation (all distractor val is hard).
        return hard

    # Balanced sample: half bridge (multi-hop), half comparison
    n_bridge = n // 2
    n_compare = n - n_bridge
    selected = bridge[:n_bridge] + compare[:n_compare]
    if len(selected) < n:
        selected = hard[:n] if hard else list(ds)[:n]
    return selected[:n]


def parse_n_questions(value: str | int | None) -> int | None:
    """CLI helper: 'all' / 'full' → None (full val); else int."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if text in {"all", "full", "*"}:
        return None
    return int(text)


def build_hotpot_subset(
    *,
    project_root: Path,
    n_questions: int | None = 100,
    seed: int = 42,
    prefer_hard: bool = True,
    force: bool = False,
) -> dict:
    """Materialize HotpotQA distractor corpus + eval JSON.

    n_questions=None materializes the **full** hard validation split (7,405 Q).
    Skips rewrite when disk already matches the requested size (unless force=True).
    """
    del seed, prefer_hard  # deterministic slice from HF validation hard split

    corpus_dir = project_root / "data" / "corpus_hotpot"
    qa_path = project_root / "data" / "qa" / "hotpot_eval.json"
    meta_path = project_root / "data" / "qa" / "hotpot_meta.json"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    qa_path.parent.mkdir(parents=True, exist_ok=True)

    target_q = 7405 if n_questions is None else int(n_questions)
    if not force and qa_path.exists() and meta_path.exists():
        try:
            existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            n_docs_disk = len(list(corpus_dir.glob("*.txt")))
            n_q_meta = int(existing_meta.get("n_questions") or 0)
            n_docs_meta = int(existing_meta.get("n_documents") or 0)
            if (
                n_q_meta == target_q
                and n_docs_disk >= int(0.98 * n_docs_meta)
                and n_docs_meta >= 100
            ):
                print(
                    f"Hotpot already on disk: {n_q_meta} Q / {n_docs_disk} docs — skip rebuild",
                    flush=True,
                )
                existing_meta["corpus_dir"] = str(corpus_dir)
                existing_meta["qa_path"] = str(qa_path)
                enrich_hotpot_supporting_titles(qa_path)
                return {
                    "corpus_dir": corpus_dir,
                    "qa_path": qa_path,
                    "meta": existing_meta,
                }
        except Exception:
            pass

    selected = _load_hotpot_validation(n_questions)

    # Write into a staging dir then swap — never leave a half-wiped corpus.
    staging = corpus_dir.parent / f".corpus_hotpot_staging_{os.getpid()}"
    if staging.exists():
        import shutil

        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    written: set[str] = set()
    eval_items: list[dict] = []

    for i, row in enumerate(selected, start=1):
        if i == 1 or i % 500 == 0 or i == len(selected):
            print(f"  Hotpot materialize {i}/{len(selected)} questions...", flush=True)
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
            (staging / f"{slug}.txt").write_text(f"# {title}\n\n{text}\n", encoding="utf-8")
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
                "supporting_titles": list(dict.fromkeys(row.get("supporting_facts", {}).get("title") or [])),
            }
        )

    import shutil

    if corpus_dir.exists():
        shutil.rmtree(corpus_dir)
    staging.rename(corpus_dir)

    qa_path.write_text(json.dumps(eval_items, indent=2), encoding="utf-8")
    meta = {
        "dataset": "HotpotQA",
        "paper": "Yang et al., EMNLP 2018",
        "citation": "https://hotpotqa.github.io/",
        "setting": "distractor",
        "source": "hotpotqa/hotpot_qa (HuggingFace)",
        "n_questions": len(eval_items),
        "n_documents": len(written),
        "full_validation": n_questions is None,
        "type_counts": dict(Counter(i["hotpot_type"] for i in eval_items)),
        "corpus_dir": str(corpus_dir),
        "qa_path": str(qa_path),
        "note": (
            "Full distractor validation = 7405 hard Q / ~66k paragraphs. "
            "Local GraphRAG runs are practical at n≈100–500; use n=all only with "
            "vector-only methods or a large compute budget."
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"corpus_dir": corpus_dir, "qa_path": qa_path, "meta": meta}


def enrich_hotpot_supporting_titles(qa_path: Path) -> int:
    """Patch existing eval JSON with gold Wikipedia titles (no corpus rewrite)."""
    qa_path = Path(qa_path)
    items = json.loads(qa_path.read_text(encoding="utf-8"))
    if items and items[0].get("supporting_titles"):
        return 0
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    by_id = {}
    for row in ds:
        titles = list(dict.fromkeys(row.get("supporting_facts", {}).get("title") or []))
        by_id[row["id"]] = titles
    n = 0
    for item in items:
        titles = by_id.get(item["id"])
        if titles:
            item["supporting_titles"] = titles
            n += 1
    qa_path.write_text(json.dumps(items, indent=2), encoding="utf-8")
    print(f"Enriched {n}/{len(items)} Hotpot questions with supporting_titles", flush=True)
    return n
