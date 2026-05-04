"""Streamlit chat UI.

Run from project root:
    streamlit run src/ui/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from config import DEFAULT_TOP_K, LLM_MODEL
from src.pipeline import stream_answer
from src.store import catalog, vector_store

st.set_page_config(page_title="Local Wikipedia RAG", page_icon="🧠", layout="wide")
st.title("Local Wikipedia RAG Assistant")
st.caption(f"Local model: `{LLM_MODEL}` · everything runs on your machine.")

with st.sidebar:
    st.subheader("Settings")
    top_k = st.slider("Top-K chunks", min_value=2, max_value=10, value=DEFAULT_TOP_K)
    show_context = st.checkbox("Show retrieved context", value=True)

    st.divider()
    st.subheader("Index status")
    chunk_count = vector_store.count()
    st.metric("Chunks indexed", chunk_count)
    people = catalog.list_titles("person")
    places = catalog.list_titles("place")
    st.metric("People", len(people))
    st.metric("Places", len(places))
    if chunk_count == 0:
        st.warning("Vector store is empty. Run `python scripts/ingest.py` first.")

    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if show_context and message.get("context"):
            with st.expander("Retrieved context"):
                for i, chunk in enumerate(message["context"], 1):
                    title = chunk["entity_title"]
                    section = chunk.get("section", "")
                    st.markdown(f"**[{i}] {title}** — {section}  *(distance {chunk['distance']:.3f})*")
                    st.write(chunk["text"])

prompt = st.chat_input("Ask about a famous person or place...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        retrieval, stream = stream_answer(prompt, top_k=top_k)

        if show_context:
            with st.expander(
                f"Retrieved {len(retrieval.chunks)} chunks "
                f"(query type: {retrieval.analysis.query_type}, "
                f"comparison: {retrieval.analysis.is_comparison})"
            ):
                if retrieval.expanded_query and retrieval.expanded_query != prompt:
                    st.caption(f"**Expanded query:** {retrieval.expanded_query}")
                for i, chunk in enumerate(retrieval.chunks, 1):
                    title = chunk.metadata.get("entity_title", "?")
                    section = chunk.metadata.get("section", "")
                    st.markdown(
                        f"**[{i}] {title}** — {section}  *(distance {chunk.distance:.3f})*"
                    )
                    st.write(chunk.text)

        placeholder = st.empty()
        full_text = ""
        for piece in stream:
            full_text += piece
            placeholder.markdown(full_text + "▌")
        placeholder.markdown(full_text or "I don't know.")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_text or "I don't know.",
            "context": [
                {
                    "entity_title": c.metadata.get("entity_title", "?"),
                    "section": c.metadata.get("section", ""),
                    "distance": c.distance,
                    "text": c.text,
                }
                for c in retrieval.chunks
            ],
        }
    )
