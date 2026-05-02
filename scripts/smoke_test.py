"""Run the example queries end-to-end and print answers + retrieval info.

Used to verify the full pipeline. Not part of the regular test suite
(requires Ollama running).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import answer_question

QUERIES = [
    ("person",  "Who was Albert Einstein and what is he known for?"),
    ("person",  "What did Marie Curie discover?"),
    ("person",  "Why is Nikola Tesla famous?"),
    ("person",  "What is Frida Kahlo known for?"),
    ("place",   "Where is the Eiffel Tower located?"),
    ("place",   "Why is the Great Wall of China important?"),
    ("place",   "What was the Colosseum used for?"),
    ("place",   "Where is Mount Everest?"),
    ("mixed",   "Which famous place is located in Turkey?"),
    ("mixed",   "Which person is associated with electricity?"),
    ("compare", "Compare Albert Einstein and Nikola Tesla"),
    ("compare", "Compare the Eiffel Tower and the Statue of Liberty"),
    ("fail",    "Who is the president of Mars?"),
    ("fail",    "Tell me about a random unknown person John Doe"),
]


def main() -> None:
    for tag, query in QUERIES:
        print("=" * 78)
        print(f"[{tag}] {query}")
        started = time.time()
        result = answer_question(query, top_k=5)
        elapsed = time.time() - started
        print(
            f"  query_type={result.retrieval.analysis.query_type} "
            f"comparison={result.retrieval.analysis.is_comparison} "
            f"matched={[e['title'] for e in result.retrieval.analysis.matched_entities]}"
        )
        print(f"  retrieved {len(result.retrieval.chunks)} chunks "
              f"(top distance {result.retrieval.chunks[0].distance:.3f})"
              if result.retrieval.chunks else "  retrieved 0 chunks")
        print(f"  answer ({elapsed:.1f}s): {result.text}")
        print()


if __name__ == "__main__":
    main()
