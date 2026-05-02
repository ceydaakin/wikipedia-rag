"""Fetch Wikipedia pages with on-disk caching."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import wikipediaapi

from config import RAW_DIR, WIKIPEDIA_LANGUAGE, WIKIPEDIA_USER_AGENT
from src.ingest.cleaner import clean_wikipedia_text


@dataclass(frozen=True)
class FetchedPage:
    title: str
    entity_type: str
    wikipedia_title: str
    url: str
    raw_text: str


def _slug(title: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in title).strip("_").lower()


def _cache_path(entity_type: str, title: str) -> Path:
    return RAW_DIR / f"{entity_type}__{_slug(title)}.json"


def _client() -> wikipediaapi.Wikipedia:
    return wikipediaapi.Wikipedia(
        user_agent=WIKIPEDIA_USER_AGENT,
        language=WIKIPEDIA_LANGUAGE,
        extract_format=wikipediaapi.ExtractFormat.WIKI,
    )


def fetch_page(
    title: str,
    entity_type: str,
    wikipedia_title: Optional[str] = None,
    refresh: bool = False,
) -> FetchedPage:
    """Fetch a single Wikipedia page, using cache unless refresh=True."""
    wikipedia_title = wikipedia_title or title
    cache = _cache_path(entity_type, title)

    if cache.exists() and not refresh:
        data = json.loads(cache.read_text(encoding="utf-8"))
        return FetchedPage(**data)

    page = _client().page(wikipedia_title)
    if not page.exists():
        raise ValueError(f"Wikipedia page not found: {wikipedia_title}")

    cleaned = clean_wikipedia_text(page.text)
    if not cleaned:
        raise ValueError(f"Wikipedia page is empty after cleaning: {wikipedia_title}")

    fetched = FetchedPage(
        title=title,
        entity_type=entity_type,
        wikipedia_title=wikipedia_title,
        url=page.fullurl,
        raw_text=cleaned,
    )

    cache.write_text(
        json.dumps(fetched.__dict__, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return fetched
