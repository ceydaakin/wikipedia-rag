"""Tests for the Wikipedia text cleaner."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingest.cleaner import clean_wikipedia_text


def test_strips_inline_citation_brackets():
    text = "Einstein won the Nobel Prize[1] in 1921[2]."
    cleaned = clean_wikipedia_text(text)
    assert "[1]" not in cleaned
    assert "[2]" not in cleaned
    assert "Einstein won the Nobel Prize" in cleaned


def test_drops_trailing_references_section():
    text = (
        "Einstein was a physicist.\n\n"
        "== Career ==\nHe worked on relativity.\n\n"
        "== References ==\n[1] Some citation."
    )
    cleaned = clean_wikipedia_text(text)
    assert "Career" in cleaned
    assert "References" not in cleaned
    assert "Some citation" not in cleaned


def test_collapses_excess_blank_lines():
    text = "para one\n\n\n\n\npara two"
    cleaned = clean_wikipedia_text(text)
    assert "\n\n\n" not in cleaned
