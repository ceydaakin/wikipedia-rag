"""Tests for the section-aware chunker."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chunking.splitter import chunk_document


def test_short_document_yields_one_chunk():
    text = "Albert Einstein was a German-born theoretical physicist, " * 3
    chunks = chunk_document(text, chunk_size=600, chunk_overlap=100)
    assert len(chunks) == 1
    assert chunks[0].section == "Introduction"


def test_section_headers_split_chunks():
    text = (
        "Intro paragraph about something interesting. " * 5
        + "\n\n== Early life ==\n"
        + "Born in 1879 in the German Empire to a middle-class family. " * 5
        + "\n\n== Career ==\n"
        + "Worked at the Swiss patent office in Bern between 1902 and 1909. " * 5
    )
    chunks = chunk_document(text, chunk_size=400, chunk_overlap=50)
    sections = {c.section for c in chunks}
    assert "Introduction" in sections
    assert "Early life" in sections
    assert "Career" in sections


def test_overlap_actually_overlaps():
    body = "abcdefghij" * 100  # 1000 chars in one section
    chunks = chunk_document("== Body ==\n" + body, chunk_size=400, chunk_overlap=100)
    assert len(chunks) >= 2
    # consecutive chunks should share their overlap region
    assert chunks[0].text[-50:] == chunks[1].text[:50]


def test_chunk_indexes_are_contiguous():
    text = "== A ==\n" + ("xyz " * 200) + "\n== B ==\n" + ("uvw " * 200)
    chunks = chunk_document(text, chunk_size=300, chunk_overlap=50)
    indexes = [c.index for c in chunks]
    assert indexes == list(range(len(chunks)))


def test_min_chars_filter_drops_tiny_chunks():
    text = "== Tiny ==\nhi.\n== Body ==\n" + ("paragraph " * 100)
    chunks = chunk_document(text, chunk_size=400, chunk_overlap=50, min_chars=80)
    assert all(len(c.text) >= 80 for c in chunks)
