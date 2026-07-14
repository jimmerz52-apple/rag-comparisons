"""Fetch Wikipedia articles and save them as plain-text corpus files."""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "RAGBenchmark/1.0 (local research; contact: local)"


def fetch_article(title: str) -> str | None:
    """Return plain-text extract for a Wikipedia article title."""
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": "1",
            "titles": title,
        }
    )
    request = urllib.request.Request(
        f"{WIKIPEDIA_API}?{params}",
        headers={"User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    pages = payload.get("query", {}).get("pages", {})
    for page in pages.values():
        if page.get("missing"):
            return None
        text = page.get("extract", "").strip()
        return text or None
    return None


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", title.strip().lower())
    return slug.strip("_")


def fetch_corpus(
    titles: list[str],
    output_dir: Path,
    *,
    max_chars: int = 12_000,
) -> list[Path]:
    """Download Wikipedia articles and write one .txt file per article."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for title in titles:
        text = fetch_article(title)
        if not text:
            continue
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + "…"

        path = output_dir / f"{_slugify(title)}.txt"
        path.write_text(f"# {title}\n\n{text}", encoding="utf-8")
        saved.append(path)

    return saved
