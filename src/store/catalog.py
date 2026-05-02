"""SQLite catalog of fetched Wikipedia documents.

Stores raw text + metadata so the system has a complete on-disk record
independent of the vector store. Useful for re-chunking without re-fetching
and for displaying full source documents in the UI.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

from config import CATALOG_DB

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    title          TEXT PRIMARY KEY,
    entity_type    TEXT NOT NULL,
    wikipedia_title TEXT NOT NULL,
    url            TEXT NOT NULL,
    raw_text       TEXT NOT NULL,
    fetched_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_entity_type ON documents(entity_type);
"""


@contextmanager
def _connect(db_path: Path = CATALOG_DB) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def upsert_document(
    title: str,
    entity_type: str,
    wikipedia_title: str,
    url: str,
    raw_text: str,
) -> None:
    fetched_at = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO documents(title, entity_type, wikipedia_title, url, raw_text, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(title) DO UPDATE SET
                entity_type=excluded.entity_type,
                wikipedia_title=excluded.wikipedia_title,
                url=excluded.url,
                raw_text=excluded.raw_text,
                fetched_at=excluded.fetched_at
            """,
            (title, entity_type, wikipedia_title, url, raw_text, fetched_at),
        )


def list_titles(entity_type: Optional[str] = None) -> list[str]:
    with _connect() as conn:
        if entity_type:
            cursor = conn.execute(
                "SELECT title FROM documents WHERE entity_type = ? ORDER BY title",
                (entity_type,),
            )
        else:
            cursor = conn.execute("SELECT title FROM documents ORDER BY title")
        return [row["title"] for row in cursor.fetchall()]


def get_document(title: str) -> Optional[dict]:
    with _connect() as conn:
        cursor = conn.execute(
            "SELECT title, entity_type, wikipedia_title, url, raw_text, fetched_at "
            "FROM documents WHERE title = ?",
            (title,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def all_documents(entity_type: Optional[str] = None) -> Iterable[dict]:
    with _connect() as conn:
        if entity_type:
            cursor = conn.execute(
                "SELECT title, entity_type, wikipedia_title, url, raw_text, fetched_at "
                "FROM documents WHERE entity_type = ? ORDER BY title",
                (entity_type,),
            )
        else:
            cursor = conn.execute(
                "SELECT title, entity_type, wikipedia_title, url, raw_text, fetched_at "
                "FROM documents ORDER BY title"
            )
        return [dict(row) for row in cursor.fetchall()]


def reset() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM documents")
