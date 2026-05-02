"""Wipe the SQLite catalog and Chroma vector store. Cached Wikipedia text in
data/raw/ is left in place so re-ingestion is fast.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.store import catalog, vector_store


def main() -> None:
    catalog.reset()
    vector_store.reset()
    print("Catalog and vector store cleared. Cached pages in data/raw/ kept.")


if __name__ == "__main__":
    main()
