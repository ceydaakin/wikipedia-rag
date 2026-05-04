"""Tiny in-process LRU cache for finalized RAG answers.

Streamlit reruns its script on every interaction but Python module state
survives, so a module-level dict here gives us free per-process caching
across reruns. Threadsafe for the rare case Streamlit uses background
threads.
"""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Any, Iterable, Optional

from config import RESPONSE_CACHE_SIZE


class _LRU:
    def __init__(self, maxsize: int):
        self._d: "OrderedDict[Any, Any]" = OrderedDict()
        self._max = max(1, maxsize)
        self._lock = Lock()

    def get(self, key: Any) -> Optional[Any]:
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
            return None

    def put(self, key: Any, value: Any) -> None:
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
            self._d[key] = value
            while len(self._d) > self._max:
                self._d.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._d.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._d)


_cache = _LRU(RESPONSE_CACHE_SIZE)


def _key(
    query: str,
    top_k: int,
    model: str,
    history: Optional[Iterable[tuple[str, str]]],
) -> tuple:
    """Normalize a cache key.

    History is included so that follow-up answers (which depend on prior
    turns) don't collide with first-turn answers. Long assistant texts are
    truncated to 200 chars so equivalent histories collapse to the same key.
    """
    h = tuple((u.strip().lower(), a[:200]) for u, a in (history or []))
    return (query.strip().lower(), int(top_k), str(model), h)


def get(
    query: str,
    top_k: int,
    model: str,
    history: Optional[Iterable[tuple[str, str]]] = None,
) -> Optional[Any]:
    return _cache.get(_key(query, top_k, model, history))


def put(
    query: str,
    top_k: int,
    model: str,
    history: Optional[Iterable[tuple[str, str]]],
    value: Any,
) -> None:
    _cache.put(_key(query, top_k, model, history), value)


def clear() -> None:
    _cache.clear()


def size() -> int:
    return len(_cache)
