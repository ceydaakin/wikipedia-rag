"""Resolve follow-up references in chat queries.

Short follow-ups like "tell me more about her" or "what about Tesla?" lack
the entity names that retrieval depends on (the classifier does substring
matching against catalog titles, and the embedder has nothing concrete to
match). This module asks the LLM to rewrite such queries into a standalone
form using the prior conversation, so retrieval gets a real subject to
work with.

Fail-open: any LLM error or unusable output returns the original query
unchanged so retrieval keeps working when Ollama is offline.
"""

from __future__ import annotations

import re
from typing import Sequence

from src.generation.llm import LLMError, generate
from src.retrieval.classifier import classify

_PROMPT = (
    "You rewrite follow-up questions into standalone questions for a "
    "Wikipedia retrieval system about famous people and places.\n"
    "\n"
    "Use the prior turns to fill in any missing names, places, or "
    "subjects implied by pronouns (her/him/them/it/they) or by short "
    "follow-ups like \"tell me more\", \"what about ...\", \"and you?\".\n"
    "\n"
    "Rules:\n"
    "- Output ONE line: only the rewritten question.\n"
    "- Keep it short and natural — like a Wikipedia search query.\n"
    "- If the new question is already self-contained, output it unchanged.\n"
    "- Do NOT answer the question. Do NOT add commentary or quotes.\n"
    "\n"
    "{history}\n"
    "New question: {query}\n"
    "Standalone question:"
)

MAX_TOKENS = 80
TEMPERATURE = 0.0
HISTORY_ANSWER_TRUNC = 300

_PRONOUN_CUES = {
    "her", "him", "them", "they", "their", "his", "hers", "she", "he",
    "it", "its", "this", "that", "these", "those",
}
_FOLLOWUP_PHRASES = (
    "tell me more", "what about", "and you", "go on", "more details",
    "more about", "continue", "what else",
)


def _looks_like_followup(query: str) -> bool:
    q = query.lower().strip()
    if not q:
        return False
    if any(p in q for p in _FOLLOWUP_PHRASES):
        return True
    tokens = set(re.findall(r"[a-zA-Z']+", q))
    return bool(tokens & _PRONOUN_CUES)


def _format_history(history: Sequence[tuple[str, str]]) -> str:
    lines = ["Previous conversation (most recent last):"]
    for u, a in history:
        u = u.strip()
        a = a.strip()
        if u:
            lines.append(f"User: {u}")
        if a:
            lines.append(f"Assistant: {a[:HISTORY_ANSWER_TRUNC]}")
    return "\n".join(lines)


def build_contextualization_prompt(
    query: str, history: Sequence[tuple[str, str]]
) -> str:
    """Pure function for unit tests."""
    return _PROMPT.format(history=_format_history(history), query=query.strip())


def contextualize_query(
    query: str, history: Sequence[tuple[str, str]] | None = None
) -> str:
    """Rewrite a follow-up query into standalone form using history.

    Returns the original query when:
    - History is empty
    - The query doesn't look like a follow-up (no pronouns or cue phrases)
    - The query already names a known catalog entity
    - The LLM fails or returns unusable output
    """
    if not history:
        return query
    if not _looks_like_followup(query):
        return query
    # Already self-contained — skip the LLM round-trip.
    if classify(query).matched_entities:
        return query

    try:
        raw = generate(
            build_contextualization_prompt(query, history),
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
    except LLMError:
        return query

    rewritten = raw.splitlines()[0].strip() if raw else ""
    rewritten = rewritten.strip("\"' ")
    if not rewritten or rewritten.lower() == query.strip().lower():
        return query
    # Guard against runaway expansions.
    if len(rewritten) > 4 * len(query) + 200:
        return query
    return rewritten
