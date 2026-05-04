"""Chroma persistent vector store wrapper.

Single collection with metadata filtering (Option B) so comparison queries
across people and places can be handled by one search call.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, Sequence

# Silence Chroma 0.4.24's anonymous-usage telemetry. The `Settings(...)` flag
# below disables most calls, but a known bug in 0.4.24 still emits
# "Failed to send telemetry event ..." log lines because the bundled posthog
# call signature is incompatible with posthog>=3.0. Setting the env var before
# importing chromadb prevents the calls; lowering the logger to CRITICAL
# hides the leftover error log if any slip through. Telemetry is fire-and-
# forget so this never affects stored data.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

import chromadb  # noqa: E402  (must come after the env var above)
from chromadb.config import Settings  # noqa: E402

from config import CHROMA_COLLECTION, CHROMA_DIR  # noqa: E402


@dataclass(frozen=True)
class StoredChunk:
    id: str
    text: str
    metadata: dict
    distance: float


def _client() -> chromadb.api.ClientAPI:
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def _collection():
    return _client().get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(
    ids: Sequence[str],
    documents: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    metadatas: Sequence[dict],
) -> None:
    if not ids:
        return
    _collection().upsert(
        ids=list(ids),
        documents=list(documents),
        embeddings=[list(e) for e in embeddings],
        metadatas=list(metadatas),
    )


def delete_by_title(title: str) -> None:
    _collection().delete(where={"entity_title": title})


def query(
    embedding: Sequence[float],
    top_k: int,
    where: Optional[dict] = None,
) -> list[StoredChunk]:
    res = _collection().query(
        query_embeddings=[list(embedding)],
        n_results=top_k,
        where=where or None,
    )
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    distances = (res.get("distances") or [[]])[0]
    return [
        StoredChunk(id=i, text=d, metadata=m or {}, distance=float(dist))
        for i, d, m, dist in zip(ids, docs, metas, distances)
    ]


def count() -> int:
    return _collection().count()


def reset() -> None:
    """Drop the collection entirely. Used by scripts/reset.py."""
    client = _client()
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass
    client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
