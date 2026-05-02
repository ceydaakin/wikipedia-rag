"""Prompt templates for the local RAG assistant."""

from __future__ import annotations

from typing import Sequence

from src.store.vector_store import StoredChunk

SYSTEM_PROMPT = (
    "You are a careful assistant that answers questions about famous people and "
    "famous places using ONLY the provided context.\n"
    "\n"
    "Rules:\n"
    "1. You may summarize, synthesize, and combine facts from the context.\n"
    "2. Every factual claim must be supported by the context. Do not invent "
    "names, dates, places, or relationships.\n"
    "3. Stick to what each source actually says. If a source talks about Greece, "
    "do not claim it talks about Turkey.\n"
    "4. For comparison questions, describe each entity using its own context, "
    "then note differences and similarities the context supports.\n"
    "5. If the context does not contain the information needed to answer, reply "
    'exactly: "I don\'t know."\n'
    "6. Cite sources inline using their bracket number, e.g. [1].\n"
    "Keep answers concise."
)


def _format_context(chunks: Sequence[StoredChunk]) -> str:
    if not chunks:
        return "(no context retrieved)"
    lines: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        title = chunk.metadata.get("entity_title", "?")
        section = chunk.metadata.get("section", "")
        header = f"[{i}] {title}" + (f" — {section}" if section else "")
        lines.append(f"{header}\n{chunk.text.strip()}")
    return "\n\n".join(lines)


def build_prompt(query: str, chunks: Sequence[StoredChunk]) -> str:
    """Build the full user prompt that will be sent to Ollama."""
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{_format_context(chunks)}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )
