from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tiktoken


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    source_path: Path


@dataclass
class TextChunk:
    chunk_id: str
    doc_id: str
    text: str
    source_path: Path


def load_documents(corpus_dir: Path, max_documents: int | None = None) -> list[Document]:
    files = sorted(corpus_dir.glob("*.txt"))
    if max_documents is not None:
        files = files[:max_documents]

    documents: list[Document] = []
    for path in files:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        title = path.stem.replace("_", " ")
        documents.append(
            Document(
                doc_id=path.stem,
                title=title,
                text=text,
                source_path=path,
            )
        )
    return documents


def chunk_documents(
    documents: list[Document],
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    encoding_name: str = "cl100k_base",
) -> list[TextChunk]:
    encoding = tiktoken.get_encoding(encoding_name)
    chunks: list[TextChunk] = []

    for document in documents:
        tokens = encoding.encode(document.text)
        start = 0
        index = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_text = encoding.decode(tokens[start:end]).strip()
            if chunk_text:
                chunks.append(
                    TextChunk(
                        chunk_id=f"{document.doc_id}__{index}",
                        doc_id=document.doc_id,
                        text=chunk_text,
                        source_path=document.source_path,
                    )
                )
                index += 1
            if end == len(tokens):
                break
            start = max(end - chunk_overlap, start + 1)

    return chunks
