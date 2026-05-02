"""Section-aware sliding-window chunker.

Strategy:
1. Split the document along Wikipedia section markers (== ... ==).
2. Each section becomes one or more fixed-size character windows with overlap.
3. Sections shorter than CHUNK_SIZE stay whole (no padding).
4. Each chunk carries its parent section name and an index for traceability.

Implemented from scratch (no LangChain) per project guidance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

from config import CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_CHARS

_SECTION_HEADER = re.compile(r"^\s*(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    text: str
    section: str
    index: int


def _iter_sections(text: str) -> Iterator[tuple[str, str]]:
    """Yield (section_name, section_body) pairs.

    The text before the first header is yielded as section_name="Introduction".
    """
    matches = list(_SECTION_HEADER.finditer(text))
    if not matches:
        yield "Introduction", text.strip()
        return

    intro = text[: matches[0].start()].strip()
    if intro:
        yield "Introduction", intro

    for i, match in enumerate(matches):
        name = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            yield name, body


def _window(body: str, size: int, overlap: int) -> Iterator[str]:
    if len(body) <= size:
        yield body
        return
    stride = max(1, size - overlap)
    for start in range(0, len(body), stride):
        piece = body[start : start + size]
        if piece:
            yield piece
        if start + size >= len(body):
            break


def chunk_document(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    min_chars: int = MIN_CHUNK_CHARS,
) -> list[Chunk]:
    """Split a document into section-aware overlapping chunks."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    idx = 0
    for section_name, body in _iter_sections(text):
        for piece in _window(body, chunk_size, chunk_overlap):
            piece = piece.strip()
            if len(piece) < min_chars:
                continue
            chunks.append(Chunk(text=piece, section=section_name, index=idx))
            idx += 1
    return chunks
