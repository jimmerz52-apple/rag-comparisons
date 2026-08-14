"""CodeRAG-Bench loader (Wang et al., NAACL Findings 2025).

Paper: https://arxiv.org/abs/2406.14497 · https://code-rag-bench.github.io/
HF: https://huggingface.co/code-rag-bench

Default slice = **basic + open-domain** (full categories, not a toy subset):

| Questions | Count | Retrieval corpus |
|-----------|------:|------------------|
| HumanEval | 164 | programming-solutions (leave-gold-out) |
| MBPP | 500 | programming-solutions (leave-gold-out) |
| DS-1000 | 1,000 | library-documentation (~34k) |
| ODEX | 439 | library-documentation (~34k) |
| **Total** | **~2,103** | **~35k docs** |

Optional: StackOverflow posts (~76k) via `include_stackoverflow=True`.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip().lower())
    return slug.strip("_")[:80] or "doc"


def _write_doc(path: Path, title: str, body: str, *, header_extra: str = "") -> None:
    text = body.strip()
    if not text:
        return
    extra = f"{header_extra}\n" if header_extra else ""
    path.write_text(f"# {title}\n{extra}\n{text}\n", encoding="utf-8")


def build_code_rag_bench_subset(
    *,
    project_root: Path,
    include_humaneval: bool = True,
    include_mbpp: bool = True,
    include_ds1000: bool = True,
    include_odex: bool = True,
    include_library_docs: bool = True,
    include_stackoverflow: bool = False,
    max_stackoverflow: int | None = 20_000,
    max_questions: int | None = None,
) -> dict:
    """Materialize CodeRAG-Bench basic + open-domain questions and retrieval corpora."""
    from datasets import load_dataset

    corpus_dir = project_root / "data" / "corpus_code_rag"
    qa_path = project_root / "data" / "qa" / "code_rag_eval.json"
    catalog_path = project_root / "results" / "code_rag_question_catalog.csv"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)

    for stale in corpus_dir.glob("*.txt"):
        stale.unlink()

    eval_items: list[dict] = []

    # ── Questions ──────────────────────────────────────────────
    if include_humaneval:
        he = load_dataset("code-rag-bench/humaneval", split="train")
        for row in he:
            tid = str(row["task_id"])
            prompt = str(row["prompt"]).strip()
            eval_items.append(
                {
                    "id": tid.replace("/", "-"),
                    "question": (
                        "Complete the following Python function. "
                        "Return only the full function implementation.\n\n"
                        f"{prompt}"
                    ),
                    "expected_answer": (row["prompt"] + row["canonical_solution"]).strip(),
                    "query_type": "local",
                    "best_method": "semantic_rag",
                    "source_doc": tid,
                    "rationale": "CodeRAG-Bench HumanEval — Wang et al., NAACL Findings 2025",
                    "code_rag_type": "humaneval",
                    "hotpot_type": "humaneval",
                    "entry_point": row.get("entry_point"),
                    "test": row.get("test"),
                    "canonical_task_id": tid,
                }
            )

    if include_mbpp:
        mb = load_dataset("code-rag-bench/mbpp", split="train")
        for row in mb:
            tid = str(row["task_id"])
            eval_items.append(
                {
                    "id": f"MBPP-{tid}",
                    "question": (
                        "Write a Python function for the following problem. "
                        "Return only code.\n\n"
                        f"{row['text']}"
                    ),
                    "expected_answer": str(row["code"]).strip(),
                    "query_type": "local",
                    "best_method": "semantic_rag",
                    "source_doc": f"MBPP/{tid}",
                    "rationale": "CodeRAG-Bench MBPP — Wang et al., NAACL Findings 2025",
                    "code_rag_type": "mbpp",
                    "hotpot_type": "mbpp",
                    "test_list": list(row.get("test_list") or []),
                    "canonical_task_id": tid,
                }
            )

    if include_ds1000:
        ds = load_dataset("code-rag-bench/ds1000", split="train")
        for i, row in enumerate(ds):
            meta = row.get("metadata") or {}
            lib = meta.get("library", "unknown") if isinstance(meta, dict) else "unknown"
            pid = meta.get("problem_id", i) if isinstance(meta, dict) else i
            qid = f"DS1000-{lib}-{pid}"
            eval_items.append(
                {
                    "id": qid,
                    "question": (
                        "Solve the following data-science coding problem. "
                        "Return only Python code.\n\n"
                        f"{row['prompt']}"
                    ),
                    "expected_answer": str(row.get("reference_code") or "").strip(),
                    "query_type": "hybrid",
                    "best_method": "semantic_rag",
                    "source_doc": qid,
                    "rationale": "CodeRAG-Bench DS-1000 — Wang et al., NAACL Findings 2025",
                    "code_rag_type": "ds1000",
                    "hotpot_type": "ds1000",
                    "library": lib,
                    "canonical_task_id": qid,
                }
            )

    if include_odex:
        ox = load_dataset("code-rag-bench/odex", split="train")
        for row in ox:
            tid = str(row["task_id"])
            intent = str(row.get("intent") or row.get("prompt") or "").strip()
            prompt = str(row.get("prompt") or "")
            solution = str(row.get("canonical_solution") or "").strip()
            eval_items.append(
                {
                    "id": f"ODEX-{tid}",
                    "question": (
                        "Implement the following open-domain Python intent. "
                        "Return only code.\n\n"
                        f"Intent: {intent}\n\nStub:\n{prompt}"
                    ),
                    "expected_answer": solution,
                    "query_type": "hybrid",
                    "best_method": "semantic_rag",
                    "source_doc": tid,
                    "rationale": "CodeRAG-Bench ODEX — Wang et al., NAACL Findings 2025",
                    "code_rag_type": "odex",
                    "hotpot_type": "odex",
                    "canonical_task_id": tid,
                }
            )

    if max_questions is not None and len(eval_items) > max_questions:
        # Keep proportional heads per type
        by_type: dict[str, list] = {}
        for q in eval_items:
            by_type.setdefault(q["code_rag_type"], []).append(q)
        types = list(by_type.keys())
        per = max(1, max_questions // max(1, len(types)))
        trimmed: list[dict] = []
        for t in types:
            trimmed.extend(by_type[t][:per])
        eval_items = trimmed[:max_questions]

    # ── Retrieval corpora ──────────────────────────────────────
    written = 0
    exclude_task_ids = {
        str(q["canonical_task_id"])
        for q in eval_items
        if q["code_rag_type"] in {"humaneval", "mbpp"}
    }

    # Programming solutions (basic-programming datastore)
    solutions = load_dataset("code-rag-bench/programming-solutions", split="train")
    for row in solutions:
        meta = row.get("meta") or {}
        tid = str(meta.get("task_id") or "")
        if tid in exclude_task_ids:
            continue
        title = row.get("title") or tid or f"sol_{written}"
        text = (row.get("text") or "").strip()
        if not text:
            continue
        _write_doc(
            corpus_dir / f"ps_{written:05d}_{_slug(title)}.txt",
            str(title),
            text,
            header_extra=f"Task: {tid}",
        )
        written += 1

    # Library documentation (~34k) — open-domain datastore
    if include_library_docs:
        docs = load_dataset("code-rag-bench/library-documentation", split="train")
        for i, row in enumerate(docs):
            doc_id = str(row.get("doc_id") or f"libdoc_{i}")
            content = str(row.get("doc_content") or "").strip()
            if not content:
                continue
            _write_doc(
                corpus_dir / f"lib_{i:05d}_{_slug(doc_id)}.txt",
                doc_id,
                content,
            )
            written += 1

    # Optional StackOverflow (~76k) — can be capped
    if include_stackoverflow:
        so = load_dataset("code-rag-bench/stackoverflow-posts", split="train")
        n_so = 0
        for i, row in enumerate(so):
            if max_stackoverflow is not None and n_so >= max_stackoverflow:
                break
            # Schema may vary; take common text fields
            title = str(row.get("title") or row.get("doc_id") or f"so_{i}")
            body = str(
                row.get("text")
                or row.get("body")
                or row.get("doc_content")
                or row.get("content")
                or ""
            ).strip()
            if not body:
                continue
            _write_doc(
                corpus_dir / f"so_{n_so:05d}_{_slug(title)}.txt",
                title,
                body,
            )
            written += 1
            n_so += 1

    catalog_rows = [
        "label,question_id,code_rag_type,query_type,question,gold_preview"
    ]
    for i, q in enumerate(eval_items):
        q_esc = q["question"].replace('"', '""')[:200]
        a_esc = q["expected_answer"].replace('"', '""')[:120]
        catalog_rows.append(
            f'Q{i+1},{q["id"]},{q["code_rag_type"]},{q["query_type"]},"{q_esc}","{a_esc}"'
        )

    qa_path.write_text(json.dumps(eval_items, indent=2), encoding="utf-8")
    catalog_path.write_text("\n".join(catalog_rows) + "\n", encoding="utf-8")

    meta = {
        "dataset": "CodeRAG-Bench",
        "paper": "Wang et al., NAACL Findings 2025",
        "citation": "https://arxiv.org/abs/2406.14497",
        "homepage": "https://code-rag-bench.github.io/",
        "slice": "basic_plus_open_domain",
        "source": (
            "humaneval + mbpp + ds1000 + odex · "
            "programming-solutions + library-documentation"
            + (" + stackoverflow-posts" if include_stackoverflow else "")
        ),
        "n_questions": len(eval_items),
        "n_documents": written,
        "type_counts": dict(Counter(q["code_rag_type"] for q in eval_items)),
        "corpus_dir": str(corpus_dir),
        "qa_path": str(qa_path),
        "catalog_path": str(catalog_path),
        "protocol": (
            "Leave-gold-out for HumanEval/MBPP solutions; open-domain tasks "
            "retrieve against library docs (± StackOverflow)."
        ),
        "note": (
            "Full CodeRAG-Bench also includes RepoEval / SWE-bench-Lite / "
            "CodeSearchNet-Py (repo-level + retrieval). This harness ships the "
            "complete basic + open-domain categories (~2.1k Q / ~35k docs)."
        ),
    }
    (project_root / "data" / "qa" / "code_rag_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return {"corpus_dir": corpus_dir, "qa_path": qa_path, "meta": meta}
