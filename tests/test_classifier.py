"""Tests for the rule-based query classifier (no DB / no network)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.classifier import classify

ENTITIES = [
    {"title": "Albert Einstein", "entity_type": "person"},
    {"title": "Nikola Tesla", "entity_type": "person"},
    {"title": "Marie Curie", "entity_type": "person"},
    {"title": "Eiffel Tower", "entity_type": "place"},
    {"title": "Hagia Sophia", "entity_type": "place"},
    {"title": "Statue of Liberty", "entity_type": "place"},
]


def test_person_question_with_known_name():
    result = classify("Who was Albert Einstein?", entities=ENTITIES)
    assert result.query_type == "person"
    assert any(e["title"] == "Albert Einstein" for e in result.matched_entities)


def test_place_question_with_known_landmark():
    result = classify("Where is the Eiffel Tower located?", entities=ENTITIES)
    assert result.query_type == "place"


def test_comparison_across_two_people():
    result = classify("Compare Albert Einstein and Nikola Tesla", entities=ENTITIES)
    assert result.query_type == "person"
    assert result.is_comparison is True
    assert len(result.matched_entities) == 2


def test_mixed_query_routes_to_both():
    result = classify(
        "Compare the Eiffel Tower and Albert Einstein", entities=ENTITIES
    )
    assert result.query_type == "both"
    assert result.is_comparison is True


def test_keyword_fallback_when_no_entity_matched():
    result = classify("Where is this famous mountain located?", entities=ENTITIES)
    assert result.query_type == "place"


def test_unknown_entity_no_keyword_signal():
    result = classify("Tell me about John Doe", entities=ENTITIES)
    assert result.query_type in ("person", "both")


def test_last_name_only_match():
    result = classify("What did Curie discover?", entities=ENTITIES)
    assert any(e["title"] == "Marie Curie" for e in result.matched_entities)
