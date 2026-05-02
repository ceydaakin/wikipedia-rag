"""End-to-end ingestion: fetch -> clean -> chunk -> embed -> store.

Usage:
    python scripts/ingest.py            # use cache, skip already-embedded
    python scripts/ingest.py --refresh  # re-fetch Wikipedia + re-embed all
    python scripts/ingest.py --only "Albert Einstein"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import ENTITIES_FILE
from src.chunking.splitter import chunk_document
from src.embeddings.embedder import embed_texts
from src.ingest.wikipedia import fetch_page
from src.store import catalog, vector_store


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text).strip("_").lower()


def _load_entities() -> list[dict]:
    data = json.loads(ENTITIES_FILE.read_text(encoding="utf-8"))
    entities: list[dict] = []
    for kind, key in (("person", "people"), ("place", "places")):
        for entry in data[key]:
            entities.append(
                {
                    "title": entry["title"],
                    "wikipedia_title": entry.get("wikipedia", entry["title"]),
                    "entity_type": kind,
                }
            )
    return entities


def _ingest_one(entity: dict, refresh: bool) -> tuple[int, float]:
    """Return (num_chunks, elapsed_seconds)."""
    started = time.time()

    page = fetch_page(
        title=entity["title"],
        entity_type=entity["entity_type"],
        wikipedia_title=entity["wikipedia_title"],
        refresh=refresh,
    )
    catalog.upsert_document(
        title=page.title,
        entity_type=page.entity_type,
        wikipedia_title=page.wikipedia_title,
        url=page.url,
        raw_text=page.raw_text,
    )

    chunks = chunk_document(page.raw_text)
    if not chunks:
        return 0, time.time() - started

    vector_store.delete_by_title(page.title)

    texts = [c.text for c in chunks]
    vectors = embed_texts(texts)

    ids = [f"{_slug(page.title)}__{c.index}" for c in chunks]
    metadatas = [
        {
            "entity_title": page.title,
            "entity_type": page.entity_type,
            "section": c.section,
            "url": page.url,
        }
        for c in chunks
    ]
    vector_store.upsert_chunks(ids=ids, documents=texts, embeddings=vectors, metadatas=metadatas)

    return len(chunks), time.time() - started


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Wikipedia entities into the local RAG store.")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch Wikipedia ignoring cache")
    parser.add_argument("--only", action="append", default=[], help="Restrict to titles (repeatable)")
    args = parser.parse_args()

    entities = _load_entities()
    if args.only:
        wanted = {t.lower() for t in args.only}
        entities = [e for e in entities if e["title"].lower() in wanted]
        if not entities:
            print(f"No entities matched --only {args.only}", file=sys.stderr)
            sys.exit(2)

    print(f"Ingesting {len(entities)} entities (refresh={args.refresh})...")
    total_chunks = 0
    total_time = 0.0
    for i, entity in enumerate(entities, 1):
        label = f"[{i}/{len(entities)}] {entity['entity_type']:5s} | {entity['title']}"
        try:
            n, elapsed = _ingest_one(entity, refresh=args.refresh)
            total_chunks += n
            total_time += elapsed
            print(f"{label}: {n} chunks in {elapsed:.1f}s")
        except Exception as exc:
            print(f"{label}: FAILED -- {exc}", file=sys.stderr)

    print(f"\nDone. {total_chunks} chunks across {len(entities)} entities in {total_time:.1f}s.")
    print(f"Vector store now contains {vector_store.count()} chunks total.")


if __name__ == "__main__":
    main()
