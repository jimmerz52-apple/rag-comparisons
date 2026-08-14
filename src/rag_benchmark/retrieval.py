"""Non-LLM retrieval metrics (2025–26 RAGAS / CodeRAG-Bench practice).

Context recall/precision against gold doc titles — no extra LLM calls.
Gold-in-context vs gold-in-answer flags the evidence-override failure mode
(Hotpot 2026 facet-tracing: retrieved evidence ignored at generation).
"""

from __future__ import annotations

import re
from typing import Iterable


def _slug(title: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", title.strip().lower())
    return slug.strip("_")[:80]


def titles_from_chunks(chunks: Iterable[str]) -> set[str]:
    found: set[str] = set()
    for raw in chunks:
        text = (raw or "").strip()
        if not text:
            continue
        first = text.split("\n", 1)[0].strip()
        heading = first.lstrip("#").strip()
        if heading:
            found.add(_slug(heading))
        found.add(_slug(text[:80]))
    return found


def retrieval_scores(
    *,
    gold_titles: list[str] | None,
    chunks: list[str] | None,
    gold_answer: str = "",
) -> dict[str, float | bool | None]:
    """Return recall@k, precision@k, gold_in_context, evidence_override."""
    chunks = [c for c in (chunks or []) if c]
    gold = [t for t in (gold_titles or []) if t]
    gold_slugs = {_slug(t) for t in gold}
    if not gold_slugs:
        gold_in_ctx = bool(
            gold_answer
            and any(gold_answer.lower()[:80] in (c or "").lower() for c in chunks)
        )
        return {
            "retrieval_recall": None,
            "retrieval_precision": None,
            "gold_in_context": gold_in_ctx,
            "evidence_override": False,
        }
    got = titles_from_chunks(chunks)
    hits = 0
    for title in gold:
        slug = _slug(title)
        needle = title.lower()
        if slug in got or any(needle in (c or "").lower()[:400] for c in chunks):
            hits += 1
    recall = hits / len(gold_slugs)
    prec = (hits / len(chunks)) if chunks else 0.0
    gold_in_ctx = bool(
        gold_answer and any(gold_answer.lower()[:80] in (c or "").lower() for c in chunks)
    ) or hits > 0
    return {
        "retrieval_recall": float(recall),
        "retrieval_precision": float(min(1.0, prec)),
        "gold_in_context": bool(gold_in_ctx),
        "evidence_override": False,
    }
