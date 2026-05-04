"""Command-line chat interface.

Run: python -m src.ui.cli
Commands:
    :reset      clear chat history (vector store stays)
    :context    toggle showing retrieved chunks
    :history    toggle threading prior turns into the prompt
    :cache      show cache stats / clear with `:cache clear`
    :model X    switch the active LLM (e.g. `:model phi3`)
    :quit       exit
"""

from __future__ import annotations

import sys

from config import LLM_MODEL, MAX_HISTORY_TURNS
from src import cache as response_cache
from src.pipeline import stream_answer
from src.store import vector_store


def _print_context(session) -> None:
    r = session.retrieval
    print("\n--- retrieved context ---")
    print(f"query type: {r.analysis.query_type}, comparison={r.analysis.is_comparison}")
    if r.analysis.matched_entities:
        names = ", ".join(e["title"] for e in r.analysis.matched_entities)
        print(f"matched entities: {names}")
    if r.expanded_query and r.expanded_query.strip() != "":
        print(f"expanded query: {r.expanded_query[:160]}")
    for i, chunk in enumerate(r.chunks, 1):
        title = chunk.metadata.get("entity_title", "?")
        section = chunk.metadata.get("section", "")
        print(f"  [{i}] ({chunk.distance:.3f}) {title} — {section}")
        snippet = chunk.text.strip().replace("\n", " ")
        print(f"      {snippet[:160]}{'...' if len(snippet) > 160 else ''}")
    print("--- end context ---\n")


def _print_stats(session) -> None:
    parts: list[str] = []
    if session.cached:
        parts.append("⚡ cached")
    if session.retrieve_seconds > 0:
        parts.append(f"retrieve {session.retrieve_seconds*1000:.0f}ms")
    if session.generate_seconds > 0:
        parts.append(f"generate {session.generate_seconds:.2f}s")
    parts.append(f"model={session.model}")
    if session.history_used:
        parts.append(f"history={session.history_used}")
    print(f"  ({' · '.join(parts)})")


def _history_pairs(messages: list[tuple[str, str, str]], max_turns: int) -> list[tuple[str, str]]:
    return [(u, a) for u, a in messages][-max_turns:]


def main() -> None:
    if vector_store.count() == 0:
        print("Vector store is empty. Run `python scripts/ingest.py` first.", file=sys.stderr)
        sys.exit(1)

    print("Local Wikipedia RAG (CLI). :quit to exit, :context, :history, :cache, :model <name>, :reset.")
    show_context = False
    use_history = True
    active_model = LLM_MODEL
    history: list[tuple[str, str]] = []  # (user, assistant)

    while True:
        try:
            query = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not query:
            continue
        if query in (":quit", ":q", "exit"):
            return
        if query == ":reset":
            history.clear()
            print("(chat history cleared)")
            continue
        if query == ":context":
            show_context = not show_context
            print(f"context display: {'on' if show_context else 'off'}")
            continue
        if query == ":history":
            use_history = not use_history
            print(f"history threading: {'on' if use_history else 'off'}")
            continue
        if query.startswith(":cache"):
            parts = query.split()
            if len(parts) > 1 and parts[1] == "clear":
                response_cache.clear()
                print("(cache cleared)")
            else:
                print(f"cache size: {response_cache.size()}")
            continue
        if query.startswith(":model"):
            parts = query.split(maxsplit=1)
            if len(parts) == 2:
                active_model = parts[1].strip()
                print(f"active model: {active_model}")
            else:
                print(f"active model: {active_model}")
            continue

        threaded = history if use_history else []
        session, stream = stream_answer(
            query,
            model=active_model,
            history=threaded,
            max_history_turns=MAX_HISTORY_TURNS,
        )
        if show_context:
            _print_context(session)

        print("bot> ", end="", flush=True)
        full = ""
        for piece in stream:
            print(piece, end="", flush=True)
            full += piece
        print()
        _print_stats(session)

        final = (session.final_text or full or "I don't know.").strip()
        history.append((query, final))


if __name__ == "__main__":
    main()
