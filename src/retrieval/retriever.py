"""Top-level retrieval API.

Combines query classification with a metadata-filtered vector search.
For comparison queries that mention multiple entities, retrieves a
small batch per entity and concatenates so each side is represented.

Optionally expands the query with LLM-generated keywords (toggled by
QUERY_EXPANSION_ENABLED in config) to improve recall for under-specified
queries like "Which famous place is located in Turkey?".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from config import (
    COMPARISON_TOP_K_PER_ENTITY,
    DEFAULT_TOP_K,
    QUERY_EXPANSION_ENABLED,
)
from src.embeddings.embedder import embed_text
from src.retrieval.classifier import QueryAnalysis, classify
from src.retrieval.contextualizer import contextualize_query
from src.retrieval.query_expander import expand_query
from src.store import vector_store
from src.store.vector_store import StoredChunk


@dataclass
class RetrievalResult:
    analysis: QueryAnalysis
    chunks: list[StoredChunk]
    expanded_query: str
    resolved_query: str = ""  # original if no contextualization happened


def _filter_for_type(query_type: str) -> Optional[dict]:
    if query_type == "person":
        return {"entity_type": "person"}
    if query_type == "place":
        return {"entity_type": "place"}
    return None


def _classify_with_expansion(query: str, expanded: str) -> QueryAnalysis:
    """Classify on the original query first; if no entities matched, retry on
    the expanded text so query expansion can surface entities (e.g. Turkey ->
    Hagia Sophia).
    """
    analysis = classify(query)
    if analysis.matched_entities or expanded == query:
        return analysis
    expanded_analysis = classify(expanded)
    if expanded_analysis.matched_entities:
        return expanded_analysis
    return analysis


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    history: Optional[Sequence[tuple[str, str]]] = None,
) -> RetrievalResult:
    """Retrieve relevant chunks for a query.

    Strategy:
    - When `history` is provided, rewrite follow-ups like "tell me more
      about her" into standalone queries before doing anything else, so
      the classifier and embedder have a real subject to work with.
    - Optionally expand the resolved query with the LLM for better recall.
    - If 2+ specific entities are mentioned (comparison), retrieve
      COMPARISON_TOP_K_PER_ENTITY chunks per named entity.
    - Otherwise, retrieve top_k chunks filtered by entity_type when known.
    """
    resolved = contextualize_query(query, history) if history else query
    expanded = expand_query(resolved) if QUERY_EXPANSION_ENABLED else resolved
    analysis = _classify_with_expansion(resolved, expanded)
    embedding = embed_text(expanded)

    def _result(chunks: list[StoredChunk]) -> RetrievalResult:
        return RetrievalResult(
            analysis=analysis,
            chunks=chunks,
            expanded_query=expanded,
            resolved_query=resolved,
        )

    # Comparison: pull a small batch per named entity so both sides are covered.
    if analysis.is_comparison and len(analysis.matched_entities) >= 2:
        bag: list[StoredChunk] = []
        seen: set[str] = set()
        for entity in analysis.matched_entities:
            results = vector_store.query(
                embedding=embedding,
                top_k=COMPARISON_TOP_K_PER_ENTITY,
                where={"entity_title": entity["title"]},
            )
            for chunk in results:
                if chunk.id not in seen:
                    seen.add(chunk.id)
                    bag.append(chunk)
        bag.sort(key=lambda c: c.distance)
        return _result(bag)

    # Single named entity: restrict search to that entity's chunks.
    if len(analysis.matched_entities) == 1:
        entity = analysis.matched_entities[0]
        chunks = vector_store.query(
            embedding=embedding,
            top_k=top_k,
            where={"entity_title": entity["title"]},
        )
        return _result(chunks)

    # Multiple named entities but no comparison keyword: union them.
    if len(analysis.matched_entities) > 1:
        titles = [e["title"] for e in analysis.matched_entities]
        chunks = vector_store.query(
            embedding=embedding,
            top_k=top_k,
            where={"entity_title": {"$in": titles}},
        )
        return _result(chunks)

    # No specific entity matched: fall back to type-level filter.
    where = _filter_for_type(analysis.query_type)
    chunks = vector_store.query(embedding=embedding, top_k=top_k, where=where)
    return _result(chunks)
