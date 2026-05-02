"""High-level RAG pipeline glue used by both the Streamlit UI and the CLI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from src.generation.llm import generate, generate_stream
from src.generation.prompts import build_prompt
from src.retrieval.retriever import RetrievalResult, retrieve


@dataclass
class Answer:
    text: str
    retrieval: RetrievalResult


def answer_question(query: str, top_k: int = 5) -> Answer:
    retrieval = retrieve(query, top_k=top_k)
    if not retrieval.chunks:
        return Answer(text="I don't know.", retrieval=retrieval)
    prompt = build_prompt(query, retrieval.chunks)
    text = generate(prompt)
    return Answer(text=text, retrieval=retrieval)


def stream_answer(query: str, top_k: int = 5) -> tuple[RetrievalResult, Iterator[str]]:
    retrieval = retrieve(query, top_k=top_k)
    if not retrieval.chunks:
        def _no_ctx() -> Iterator[str]:
            yield "I don't know."
        return retrieval, _no_ctx()
    prompt = build_prompt(query, retrieval.chunks)
    return retrieval, generate_stream(prompt)
