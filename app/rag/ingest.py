from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from ..config import HXH_DIR
from .chunker import chunk_text
from .embeddings import embed_texts
from .vector_store import add_documents, count_documents, reset_collection


def read_markdown_files(root: Path) -> list[tuple[str, str]]:
    items = []
    if not root.exists():
        return items
    for path in sorted(root.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            items.append((path.name, text))
    return items


def split_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_title = "Document"
    current_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("#"):
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
                current_lines = []
            current_title = line.lstrip("#").strip() or "Untitled"
        current_lines.append(line)

    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))
    return [(title, body) for title, body in sections if body]


def ingest() -> int:
    reset_collection()

    docs = read_markdown_files(HXH_DIR)
    all_ids: list[str] = []
    all_docs: list[str] = []
    all_meta: list[dict] = []

    for filename, text in docs:
        for section_title, section_text in split_sections(text):
            chunks = chunk_text(section_text)
            for i, chunk in enumerate(chunks):
                all_ids.append(f"{filename}-{i}-{uuid4().hex[:8]}")
                all_docs.append(chunk)
                all_meta.append(
                    {
                        "source": filename,
                        "chunk_index": i,
                        "topic": filename.replace(".md", ""),
                        "section": section_title,
                    }
                )

    if not all_docs:
        return 0

    vectors = embed_texts(all_docs)
    add_documents(all_ids, all_docs, vectors, all_meta)
    return len(all_docs)


def ingest_summary() -> dict[str, int]:
    file_count = len(read_markdown_files(HXH_DIR))
    chunks = ingest()
    try:
        collection_count = count_documents()
    except Exception:
        collection_count = chunks
    return {"files_loaded": file_count, "chunks_created": chunks, "collection_count": collection_count}


if __name__ == "__main__":
    summary = ingest_summary()
    print(f"Files loaded: {summary['files_loaded']}")
    print(f"Chunks created: {summary['chunks_created']}")
    print(f"Collection count: {summary['collection_count']}")


