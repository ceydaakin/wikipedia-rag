"""Command-line chat interface.

Run: python -m src.ui.cli
Commands:
    :reset      clear chat history (vector store stays)
    :context    toggle showing retrieved chunks
    :quit       exit
"""

from __future__ import annotations

import sys

from src.pipeline import stream_answer
from src.store import vector_store


def _print_context(result) -> None:
    print("\n--- retrieved context ---")
    print(f"query type: {result.analysis.query_type}, comparison={result.analysis.is_comparison}")
    if result.analysis.matched_entities:
        names = ", ".join(e["title"] for e in result.analysis.matched_entities)
        print(f"matched entities: {names}")
    if getattr(result, "expanded_query", None):
        print(f"expanded query: {result.expanded_query[:160]}")
    for i, chunk in enumerate(result.chunks, 1):
        title = chunk.metadata.get("entity_title", "?")
        section = chunk.metadata.get("section", "")
        print(f"  [{i}] ({chunk.distance:.3f}) {title} — {section}")
        snippet = chunk.text.strip().replace("\n", " ")
        print(f"      {snippet[:160]}{'...' if len(snippet) > 160 else ''}")
    print("--- end context ---\n")


def main() -> None:
    if vector_store.count() == 0:
        print("Vector store is empty. Run `python scripts/ingest.py` first.", file=sys.stderr)
        sys.exit(1)

    print("Local Wikipedia RAG (CLI). Type :quit to exit, :context to toggle, :reset to clear.")
    show_context = False

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
            print("(chat history is per-process; use scripts/reset.py to wipe stores)")
            continue
        if query == ":context":
            show_context = not show_context
            print(f"context display: {'on' if show_context else 'off'}")
            continue

        retrieval, stream = stream_answer(query)
        if show_context:
            _print_context(retrieval)

        print("bot> ", end="", flush=True)
        for piece in stream:
            print(piece, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
