"""Tests for the in-process LRU response cache (no Ollama / no network)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import cache as response_cache


def setup_function(_):
    response_cache.clear()


def test_get_returns_none_for_miss():
    assert response_cache.get("never seen", 5, "llama3.2:3b") is None


def test_put_then_get_round_trips():
    response_cache.put("Who is Einstein?", 5, "llama3.2:3b", None, "answer-1")
    assert response_cache.get("Who is Einstein?", 5, "llama3.2:3b") == "answer-1"


def test_key_is_case_and_whitespace_insensitive():
    response_cache.put("  Who is Einstein?  ", 5, "llama3.2:3b", None, "answer-1")
    assert response_cache.get("WHO IS EINSTEIN?", 5, "llama3.2:3b") == "answer-1"


def test_different_top_k_separates_entries():
    response_cache.put("q", 5, "llama3.2:3b", None, "k5")
    response_cache.put("q", 8, "llama3.2:3b", None, "k8")
    assert response_cache.get("q", 5, "llama3.2:3b") == "k5"
    assert response_cache.get("q", 8, "llama3.2:3b") == "k8"


def test_different_models_separate_entries():
    response_cache.put("q", 5, "llama3.2:3b", None, "llama")
    response_cache.put("q", 5, "phi3", None, "phi")
    assert response_cache.get("q", 5, "llama3.2:3b") == "llama"
    assert response_cache.get("q", 5, "phi3") == "phi"


def test_history_changes_the_key():
    h1 = [("hi", "hello")]
    h2 = [("hi", "hi there")]
    response_cache.put("q", 5, "m", h1, "v1")
    response_cache.put("q", 5, "m", h2, "v2")
    assert response_cache.get("q", 5, "m", h1) == "v1"
    assert response_cache.get("q", 5, "m", h2) == "v2"
    assert response_cache.get("q", 5, "m", None) is None


def test_lru_evicts_oldest_when_full(monkeypatch):
    # Build a tiny LRU directly so we don't depend on RESPONSE_CACHE_SIZE.
    from src.cache import _LRU
    lru = _LRU(maxsize=2)
    lru.put("a", 1)
    lru.put("b", 2)
    lru.put("c", 3)  # should evict "a"
    assert lru.get("a") is None
    assert lru.get("b") == 2
    assert lru.get("c") == 3


def test_lru_get_marks_entry_recent():
    from src.cache import _LRU
    lru = _LRU(maxsize=2)
    lru.put("a", 1)
    lru.put("b", 2)
    lru.get("a")        # touches "a"
    lru.put("c", 3)     # should evict "b" now, not "a"
    assert lru.get("a") == 1
    assert lru.get("b") is None


def test_size_and_clear():
    response_cache.put("q1", 5, "m", None, "v")
    response_cache.put("q2", 5, "m", None, "v")
    assert response_cache.size() == 2
    response_cache.clear()
    assert response_cache.size() == 0
