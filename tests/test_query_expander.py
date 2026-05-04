"""Offline tests for query expansion (no Ollama / no network)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation import llm
from src.retrieval import query_expander
from src.retrieval.query_expander import build_expansion_prompt, expand_query


def test_prompt_includes_query_and_keyword_marker():
    prompt = build_expansion_prompt("Where is the Eiffel Tower?")
    assert "Where is the Eiffel Tower?" in prompt
    assert "Keywords:" in prompt


def test_expand_returns_original_when_llm_unreachable(monkeypatch):
    def boom(*_args, **_kwargs):
        raise llm.LLMError("ollama down")

    monkeypatch.setattr(query_expander, "generate", boom)
    assert expand_query("What did Marie Curie discover?") == (
        "What did Marie Curie discover?"
    )


def test_expand_appends_keywords_on_success(monkeypatch):
    monkeypatch.setattr(
        query_expander,
        "generate",
        lambda *_a, **_kw: "Istanbul Hagia Sophia mosque Byzantine",
    )
    out = expand_query("Which famous place is in Turkey?")
    assert out.startswith("Which famous place is in Turkey?")
    assert "Hagia Sophia" in out


def test_expand_keeps_only_first_line(monkeypatch):
    monkeypatch.setattr(
        query_expander,
        "generate",
        lambda *_a, **_kw: "first line keywords\nsecond line should be dropped",
    )
    out = expand_query("test query")
    assert "second line" not in out
    assert "first line keywords" in out


def test_expand_returns_original_for_empty_input():
    assert expand_query("") == ""
    assert expand_query("   ") == "   "
