"""Tests for the prompt builder, including multi-turn history threading.

Pure-Python — no Ollama, no Chroma, no network.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation.prompts import build_prompt
from src.store.vector_store import StoredChunk


def _chunk(idx: int, title: str, text: str) -> StoredChunk:
    return StoredChunk(
        id=f"id-{idx}",
        text=text,
        metadata={"entity_title": title, "section": "Overview"},
        distance=0.1,
    )


def test_prompt_includes_query_and_context():
    chunks = [_chunk(1, "Albert Einstein", "Theory of relativity.")]
    prompt = build_prompt("Who was Einstein?", chunks)
    assert "Who was Einstein?" in prompt
    assert "Theory of relativity" in prompt
    assert "[1] Albert Einstein" in prompt


def test_prompt_without_history_has_no_history_block():
    prompt = build_prompt("q", [_chunk(1, "T", "body")])
    assert "Previous conversation" not in prompt


def test_prompt_with_history_includes_prior_turns():
    history = [
        ("Tell me about Einstein", "He was a physicist."),
        ("And Tesla?", "An inventor."),
    ]
    prompt = build_prompt(
        "Compare them",
        [_chunk(1, "Albert Einstein", "Body")],
        history=history,
    )
    assert "Previous conversation" in prompt
    assert "Tell me about Einstein" in prompt
    assert "He was a physicist" in prompt
    assert "And Tesla?" in prompt
    # The current question still appears as Question:
    assert "Question: Compare them" in prompt


def test_history_block_appears_before_context():
    history = [("first", "answer-one")]
    prompt = build_prompt("q", [_chunk(1, "T", "body")], history=history)
    assert prompt.index("Previous conversation") < prompt.index("Context:")


def test_empty_history_is_treated_as_no_history():
    a = build_prompt("q", [_chunk(1, "T", "body")], history=[])
    b = build_prompt("q", [_chunk(1, "T", "body")], history=None)
    assert a == b
    assert "Previous conversation" not in a
