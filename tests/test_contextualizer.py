"""Tests for the follow-up query contextualizer."""

from __future__ import annotations

from unittest.mock import patch

from src.retrieval.contextualizer import (
    _looks_like_followup,
    build_contextualization_prompt,
    contextualize_query,
)


class TestLooksLikeFollowup:
    def test_empty_string_is_not_a_followup(self) -> None:
        assert _looks_like_followup("") is False

    def test_pronoun_triggers_followup(self) -> None:
        assert _looks_like_followup("Tell me more about her") is True
        assert _looks_like_followup("What did he invent?") is True
        assert _looks_like_followup("Where is it?") is True

    def test_phrase_cue_triggers_followup(self) -> None:
        assert _looks_like_followup("tell me more") is True
        assert _looks_like_followup("What about Tesla?") is True
        assert _looks_like_followup("go on") is True

    def test_self_contained_query_is_not_a_followup(self) -> None:
        assert _looks_like_followup("What did Marie Curie discover?") is False
        assert _looks_like_followup("Where is the Eiffel Tower?") is False


class TestBuildContextualizationPrompt:
    def test_includes_history_and_query(self) -> None:
        history = [("Who was Marie Curie?", "A physicist who discovered radium.")]
        prompt = build_contextualization_prompt("Tell me more about her", history)
        assert "Marie Curie" in prompt
        assert "Tell me more about her" in prompt
        assert "Standalone question:" in prompt

    def test_truncates_long_assistant_turns(self) -> None:
        long_answer = "x" * 1000
        history = [("Who was Curie?", long_answer)]
        prompt = build_contextualization_prompt("more about her", history)
        # Should not contain the full 1000 x's
        assert "x" * 1000 not in prompt


class TestContextualizeShortCircuits:
    def test_no_history_returns_original(self) -> None:
        assert contextualize_query("tell me more about her", history=None) == (
            "tell me more about her"
        )
        assert contextualize_query("tell me more about her", history=[]) == (
            "tell me more about her"
        )

    def test_self_contained_query_returns_original_without_llm_call(self) -> None:
        history = [("a", "b")]
        with patch("src.retrieval.contextualizer.generate") as mocked:
            result = contextualize_query("Where is the Eiffel Tower?", history)
            mocked.assert_not_called()
        assert result == "Where is the Eiffel Tower?"

    def test_query_with_known_entity_skips_llm(self) -> None:
        # Even with pronouns, if the query already names an entity that the
        # classifier can match, don't waste an LLM round-trip.
        history = [("Who was she?", "Marie Curie was a scientist.")]
        fake_match = [{"title": "Marie Curie", "entity_type": "person"}]
        with patch("src.retrieval.contextualizer.generate") as mocked, patch(
            "src.retrieval.contextualizer.classify"
        ) as mocked_classify:
            mocked_classify.return_value.matched_entities = fake_match
            result = contextualize_query("Tell me more about her, Marie Curie", history)
            mocked.assert_not_called()
        assert result == "Tell me more about her, Marie Curie"

    def test_llm_failure_returns_original(self) -> None:
        from src.generation.llm import LLMError

        history = [("Who was Curie?", "A physicist.")]
        with patch("src.retrieval.contextualizer.generate", side_effect=LLMError("down")), \
             patch("src.retrieval.contextualizer.classify") as mocked_classify:
            mocked_classify.return_value.matched_entities = []
            result = contextualize_query("tell me more about her", history)
        assert result == "tell me more about her"

    def test_empty_llm_output_returns_original(self) -> None:
        history = [("Who was Curie?", "A physicist.")]
        with patch("src.retrieval.contextualizer.generate", return_value="   "), \
             patch("src.retrieval.contextualizer.classify") as mocked_classify:
            mocked_classify.return_value.matched_entities = []
            result = contextualize_query("tell me more about her", history)
        assert result == "tell me more about her"

    def test_runaway_expansion_returns_original(self) -> None:
        history = [("Who was Curie?", "A physicist.")]
        with patch(
            "src.retrieval.contextualizer.generate",
            return_value="x" * 5000,
        ), patch("src.retrieval.contextualizer.classify") as mocked_classify:
            mocked_classify.return_value.matched_entities = []
            result = contextualize_query("tell me more about her", history)
        assert result == "tell me more about her"

    def test_successful_rewrite_is_returned(self) -> None:
        history = [("Who was Curie?", "Marie Curie was a physicist.")]
        with patch(
            "src.retrieval.contextualizer.generate",
            return_value="Tell me more about Marie Curie's life",
        ), patch("src.retrieval.contextualizer.classify") as mocked_classify:
            mocked_classify.return_value.matched_entities = []
            result = contextualize_query("tell me more about her", history)
        assert result == "Tell me more about Marie Curie's life"

    def test_strips_quotes_from_llm_output(self) -> None:
        history = [("Who was Curie?", "A physicist.")]
        with patch(
            "src.retrieval.contextualizer.generate",
            return_value='"Tell me more about Marie Curie"',
        ), patch("src.retrieval.contextualizer.classify") as mocked_classify:
            mocked_classify.return_value.matched_entities = []
            result = contextualize_query("tell me more about her", history)
        assert result == "Tell me more about Marie Curie"
