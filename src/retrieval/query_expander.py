"""LLM-driven query expansion for higher retrieval recall.

Sends the user's query to the local LLM with a tight prompt that asks for
related entity names, synonyms, and Wikipedia-style keywords. The expanded
text is appended to the original query before embedding, which helps surface
chunks that share concepts but not surface words (e.g. "Turkey" -> "Hagia
Sophia", "Istanbul").

Fail-open: any error returns the original query unchanged so retrieval keeps
working when the LLM is offline.
"""

from __future__ import annotations

from src.generation.llm import LLMError, generate

EXPANSION_PROMPT = (
    "You expand search queries for a Wikipedia knowledge base about famous "
    "people and places. Given a user query, produce ONE single line of "
    "additional keywords: related entity names, locations, synonyms, and "
    "domain terms that would appear in relevant Wikipedia articles. "
    "If the query mentions a country, region, or city, include the most "
    "famous landmarks or people associated with it. "
    "Do NOT answer the question. Do NOT add commentary. "
    "Output ONLY the keywords on one line, separated by spaces.\n\n"
    "Query: {query}\n"
    "Keywords:"
)

MAX_EXPANSION_TOKENS = 80
EXPANSION_TEMPERATURE = 0.1


def build_expansion_prompt(query: str) -> str:
    """Pure function so it can be unit-tested without hitting Ollama."""
    return EXPANSION_PROMPT.format(query=query.strip())


def expand_query(query: str) -> str:
    """Return the query concatenated with LLM-generated keywords.

    Returns the original query unchanged if the LLM is unreachable or
    produces unusable output.
    """
    cleaned = query.strip()
    if not cleaned:
        return query
    try:
        raw = generate(
            build_expansion_prompt(cleaned),
            temperature=EXPANSION_TEMPERATURE,
            max_tokens=MAX_EXPANSION_TOKENS,
        )
    except LLMError:
        return query

    keywords = raw.splitlines()[0].strip() if raw else ""
    if not keywords or keywords.lower() == cleaned.lower():
        return query
    return f"{cleaned} {keywords}"
