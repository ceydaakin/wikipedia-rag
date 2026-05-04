"""High-level RAG pipeline glue used by both the Streamlit UI and the CLI.

Provides streaming + multi-turn history + per-call latency + LRU response
caching. Same retrieve/generate primitives — this layer just orchestrates
them and records what happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterator, Optional, Sequence

from config import DEFAULT_TOP_K, LLM_MODEL
from src import cache as response_cache
from src.generation.llm import generate, generate_stream
from src.generation.prompts import build_prompt
from src.retrieval.retriever import RetrievalResult, retrieve
from src.store.vector_store import StoredChunk

History = Sequence[tuple[str, str]]


@dataclass
class Answer:
    """Final, fully-materialised answer (used by non-streaming callers and
    by the cache)."""
    text: str
    retrieval: RetrievalResult
    retrieve_seconds: float
    generate_seconds: float
    cached: bool
    model: str


@dataclass
class StreamSession:
    """Mutable companion to the streaming iterator.

    The iterator from `stream_answer` writes into this object as tokens
    arrive: `final_text` is filled when the stream ends, and
    `generate_seconds` is recorded at the same time. Callers can check
    `cached` immediately to decide whether to render a "cached" badge.
    """
    retrieval: RetrievalResult
    model: str
    cached: bool
    retrieve_seconds: float
    generate_seconds: float = 0.0
    final_text: str = ""
    history_used: int = 0


def _trim_history(history: Optional[History], max_turns: int) -> list[tuple[str, str]]:
    if not history or max_turns <= 0:
        return []
    return list(history)[-max_turns:]


def answer_question(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    model: str = LLM_MODEL,
    history: Optional[History] = None,
    max_history_turns: int = 3,
) -> Answer:
    """Synchronous full-completion path. Used by smoke tests and the CLI."""
    history = _trim_history(history, max_history_turns)

    cached = response_cache.get(query, top_k, model, history)
    if cached is not None:
        return Answer(
            text=cached.text,
            retrieval=cached.retrieval,
            retrieve_seconds=cached.retrieve_seconds,
            generate_seconds=cached.generate_seconds,
            cached=True,
            model=model,
        )

    t0 = perf_counter()
    retrieval = retrieve(query, top_k=top_k)
    t_retrieve = perf_counter() - t0

    if not retrieval.chunks:
        ans = Answer("I don't know.", retrieval, t_retrieve, 0.0, False, model)
        response_cache.put(query, top_k, model, history, ans)
        return ans

    prompt = build_prompt(query, retrieval.chunks, history=history)
    t1 = perf_counter()
    text = generate(prompt, model=model)
    t_gen = perf_counter() - t1

    ans = Answer(text, retrieval, t_retrieve, t_gen, False, model)
    response_cache.put(query, top_k, model, history, ans)
    return ans


def stream_answer(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    model: str = LLM_MODEL,
    history: Optional[History] = None,
    max_history_turns: int = 3,
) -> tuple[StreamSession, Iterator[str]]:
    """Streaming path. Returns (session, iterator) — the iterator yields
    text pieces and writes the final state back into `session` so the UI
    can read latencies and the final text after the stream completes.

    On cache hit the iterator yields the cached answer in a single chunk
    so the UI codepath is identical (still calls `for piece in stream`).
    """
    history = _trim_history(history, max_history_turns)

    cached = response_cache.get(query, top_k, model, history)
    if cached is not None:
        session = StreamSession(
            retrieval=cached.retrieval,
            model=model,
            cached=True,
            retrieve_seconds=cached.retrieve_seconds,
            generate_seconds=cached.generate_seconds,
            final_text=cached.text,
            history_used=len(history),
        )

        def _replay() -> Iterator[str]:
            yield cached.text

        return session, _replay()

    t0 = perf_counter()
    retrieval = retrieve(query, top_k=top_k)
    t_retrieve = perf_counter() - t0

    session = StreamSession(
        retrieval=retrieval,
        model=model,
        cached=False,
        retrieve_seconds=t_retrieve,
        history_used=len(history),
    )

    if not retrieval.chunks:
        text = "I don't know."
        session.final_text = text
        ans = Answer(text, retrieval, t_retrieve, 0.0, False, model)
        response_cache.put(query, top_k, model, history, ans)

        def _no_ctx() -> Iterator[str]:
            yield text

        return session, _no_ctx()

    prompt = build_prompt(query, retrieval.chunks, history=history)

    def _stream() -> Iterator[str]:
        t1 = perf_counter()
        pieces: list[str] = []
        for piece in generate_stream(prompt, model=model):
            pieces.append(piece)
            yield piece
        full = "".join(pieces).strip()
        session.generate_seconds = perf_counter() - t1
        session.final_text = full
        ans = Answer(
            text=full or "I don't know.",
            retrieval=retrieval,
            retrieve_seconds=t_retrieve,
            generate_seconds=session.generate_seconds,
            cached=False,
            model=model,
        )
        response_cache.put(query, top_k, model, history, ans)

    return session, _stream()


def stream_with_existing_retrieval(
    query: str,
    retrieval: RetrievalResult,
    model: str,
    history: Optional[History] = None,
    max_history_turns: int = 3,
) -> tuple[StreamSession, Iterator[str]]:
    """Run only the generation half against a pre-computed retrieval.

    Used by "compare two models" so both models see the *same* retrieved
    context — keeps the comparison fair (any difference is from the model,
    not from a different vector hit).
    """
    history = _trim_history(history, max_history_turns)

    cached = response_cache.get(query, len(retrieval.chunks), model, history)
    if cached is not None:
        session = StreamSession(
            retrieval=retrieval,
            model=model,
            cached=True,
            retrieve_seconds=0.0,
            generate_seconds=cached.generate_seconds,
            final_text=cached.text,
            history_used=len(history),
        )

        def _replay() -> Iterator[str]:
            yield cached.text

        return session, _replay()

    session = StreamSession(
        retrieval=retrieval,
        model=model,
        cached=False,
        retrieve_seconds=0.0,
        history_used=len(history),
    )

    if not retrieval.chunks:
        session.final_text = "I don't know."

        def _no_ctx() -> Iterator[str]:
            yield session.final_text

        return session, _no_ctx()

    prompt = build_prompt(query, retrieval.chunks, history=history)

    def _stream() -> Iterator[str]:
        t1 = perf_counter()
        pieces: list[str] = []
        for piece in generate_stream(prompt, model=model):
            pieces.append(piece)
            yield piece
        full = "".join(pieces).strip()
        session.generate_seconds = perf_counter() - t1
        session.final_text = full
        ans = Answer(
            text=full or "I don't know.",
            retrieval=retrieval,
            retrieve_seconds=0.0,
            generate_seconds=session.generate_seconds,
            cached=False,
            model=model,
        )
        response_cache.put(query, len(retrieval.chunks), model, history, ans)

    return session, _stream()
