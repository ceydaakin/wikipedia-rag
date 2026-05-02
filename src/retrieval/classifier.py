"""Rule-based query router.

Decides whether a user query is about a person, a place, or both, and
extracts which known entities are mentioned. Used to pick the right
metadata filter for vector search.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal

from src.store import catalog

QueryType = Literal["person", "place", "both"]

_PLACE_HINTS = {
    "where", "located", "city", "country", "continent", "monument",
    "landmark", "tower", "wall", "mountain", "river", "lake", "canyon",
    "ruins", "temple", "palace", "cathedral", "mosque",
}
_PERSON_HINTS = {
    "who", "born", "died", "discovered", "invented", "wrote", "painted",
    "composed", "founded", "scientist", "artist", "musician", "poet",
    "athlete", "footballer", "physicist", "actress", "actor",
}
_COMPARISON_HINTS = {"compare", "vs", "versus", "and", "or", "between"}


@dataclass(frozen=True)
class QueryAnalysis:
    query_type: QueryType
    matched_entities: list[dict]  # [{title, entity_type}, ...]
    is_comparison: bool


def _tokens(query: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z]+", query.lower()))


def _known_entities() -> list[dict]:
    """Return [{title, entity_type}] for everything in the catalog."""
    out: list[dict] = []
    for entity_type in ("person", "place"):
        for title in catalog.list_titles(entity_type):
            out.append({"title": title, "entity_type": entity_type})
    return out


def _matches_in_query(query: str, entities: Iterable[dict]) -> list[dict]:
    """Return entities whose title appears (case-insensitive substring) in the query."""
    q = query.lower()
    matches: list[dict] = []
    for ent in entities:
        title_lc = ent["title"].lower()
        if title_lc in q:
            matches.append(ent)
            continue
        # Allow last-name-only mention for single-token last names of multi-word people.
        parts = title_lc.split()
        if len(parts) > 1 and ent["entity_type"] == "person":
            last = parts[-1]
            if len(last) >= 4 and re.search(rf"\b{re.escape(last)}\b", q):
                matches.append(ent)
    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[dict] = []
    for m in matches:
        if m["title"] not in seen:
            seen.add(m["title"])
            unique.append(m)
    return unique


def classify(query: str, entities: Iterable[dict] | None = None) -> QueryAnalysis:
    """Classify a query and identify mentioned entities."""
    entity_list = list(entities) if entities is not None else _known_entities()
    matches = _matches_in_query(query, entity_list)
    types_in_matches = {m["entity_type"] for m in matches}
    tokens = _tokens(query)
    is_comparison = bool(tokens & _COMPARISON_HINTS) and len(matches) >= 2

    if {"person", "place"} <= types_in_matches:
        return QueryAnalysis("both", matches, is_comparison)
    if types_in_matches == {"person"}:
        return QueryAnalysis("person", matches, is_comparison)
    if types_in_matches == {"place"}:
        return QueryAnalysis("place", matches, is_comparison)

    # No entity-name hits -> fall back to keyword heuristics.
    person_score = len(tokens & _PERSON_HINTS)
    place_score = len(tokens & _PLACE_HINTS)
    if person_score > place_score:
        return QueryAnalysis("person", [], is_comparison)
    if place_score > person_score:
        return QueryAnalysis("place", [], is_comparison)
    return QueryAnalysis("both", [], is_comparison)
