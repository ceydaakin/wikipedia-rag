"""Streamlit chat UI.

Run from project root:
    streamlit run src/ui/streamlit_app.py
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from config import DEFAULT_TOP_K, LLM_MODEL, MAX_HISTORY_TURNS, MODEL_OPTIONS
from src import cache as response_cache
from src.pipeline import stream_answer, stream_with_existing_retrieval
from src.retrieval.retriever import retrieve as do_retrieve
from src.store import catalog, vector_store

# ─── Page setup ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Wikipedia RAG",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Design tokens + custom CSS ───────────────────────────────────────────────

CSS = """
<style>
:root {
    --primary: #4F46E5;
    --primary-soft: #EEF2FF;
    --person: #8B5CF6;
    --person-soft: #F5F3FF;
    --place:  #0EA5E9;
    --place-soft: #F0F9FF;
    --ink: #0F172A;
    --muted: #64748B;
    --border: #E2E8F0;
    --surface: #FFFFFF;
    --bg: #F8FAFC;
    --success: #10B981;
    --success-soft: #ECFDF5;
    --warn: #F59E0B;
    --warn-soft: #FFFBEB;
}

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 4rem;
    max-width: 1100px;
}

.hero {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 60%, #EC4899 100%);
    border-radius: 16px;
    padding: 22px 28px;
    color: white;
    margin-bottom: 18px;
    box-shadow: 0 8px 24px -12px rgba(79, 70, 229, 0.45);
}
.hero h1 { margin: 0 0 4px 0; font-size: 26px; font-weight: 700;
           letter-spacing: -0.02em; color: white; }
.hero p  { margin: 0; opacity: 0.92; font-size: 14px; }
.hero .meta { display: inline-flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; }
.hero .pill {
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.25);
    color: white;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 500;
    backdrop-filter: blur(6px);
}

section[data-testid="stSidebar"] {
    background: var(--bg);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] h3 {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-top: 18px;
    margin-bottom: 6px;
    font-weight: 600;
}
.status-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 8px;
}
.status-card .label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
    margin-bottom: 2px;
}
.status-card .value {
    font-size: 22px; font-weight: 700; color: var(--ink); line-height: 1.1;
}
.status-card.ready { border-left: 3px solid var(--success); }
.status-card.empty { border-left: 3px solid var(--warn); }

.welcome {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 22px 24px;
    margin-bottom: 18px;
}
.welcome h3 { margin: 0 0 4px 0; font-size: 17px; color: var(--ink); font-weight: 600; }
.welcome p  { margin: 0 0 14px 0; color: var(--muted); font-size: 13.5px; }

.meta-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 4px 0 10px 0; }
.tag {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 3px 9px; border-radius: 999px;
    font-size: 11.5px; font-weight: 500;
    border: 1px solid var(--border);
    background: var(--surface); color: var(--muted);
}
.tag.primary  { background: var(--primary-soft); color: var(--primary); border-color: transparent; }
.tag.person   { background: var(--person-soft);  color: var(--person);  border-color: transparent; }
.tag.place    { background: var(--place-soft);   color: var(--place);   border-color: transparent; }
.tag.success  { background: var(--success-soft); color: var(--success); border-color: transparent; }
.tag.cached   { background: #FEF3C7; color: #B45309; border-color: transparent; }
.tag.mono     { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; }

.callout {
    background: var(--primary-soft);
    border-left: 3px solid var(--primary);
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0 14px 0;
    font-size: 13px; color: var(--ink);
}
.callout .label {
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.06em; color: var(--primary); margin-bottom: 3px;
}

.source-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 8px;
}
.source-card .head {
    display: flex; align-items: center; justify-content: space-between;
    gap: 8px; margin-bottom: 6px;
}
.source-card .title-row {
    display: flex; align-items: center; gap: 8px;
    font-size: 13.5px; font-weight: 600; color: var(--ink);
}
.source-card .idx {
    display: inline-flex; align-items: center; justify-content: center;
    width: 22px; height: 22px; border-radius: 6px;
    background: var(--primary-soft); color: var(--primary);
    font-size: 11.5px; font-weight: 700;
}
.source-card .section { color: var(--muted); font-size: 12px; font-weight: 400; }
.source-card .distance {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 11.5px; color: var(--muted);
}
.source-card .body { color: #1E293B; font-size: 13.5px; line-height: 1.55; }

.model-col-header {
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 6px;
    font-weight: 600; color: var(--ink);
}
.model-col-header .swatch {
    width: 10px; height: 10px; border-radius: 50%;
    background: var(--primary);
}

[data-testid="stChatMessage"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 14px 16px;
}
[data-testid="stChatInput"] textarea { border-radius: 12px !important; }

footer { display: none; }
[data-testid="stStatusWidget"] { display: none; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _safe(text: str) -> str:
    return html.escape(text or "")


def _entity_class(entity_type: str) -> str:
    if entity_type == "person":
        return "person"
    if entity_type == "place":
        return "place"
    return "primary"


def render_meta_row(retrieval) -> None:
    qt = retrieval.analysis.query_type
    parts = [
        f'<span class="tag primary">📚 {len(retrieval.chunks)} chunks</span>',
        f'<span class="tag {_entity_class(qt)}">🎯 {qt}</span>',
    ]
    if retrieval.analysis.is_comparison:
        parts.append('<span class="tag">⚖️ comparison</span>')
    if retrieval.analysis.matched_entities:
        names = ", ".join(e["title"] for e in retrieval.analysis.matched_entities)
        parts.append(f'<span class="tag">🔗 {_safe(names)}</span>')
    st.markdown(f'<div class="meta-row">{"".join(parts)}</div>', unsafe_allow_html=True)


def render_latency_row(session) -> None:
    parts: list[str] = []
    if session.cached:
        parts.append('<span class="tag cached">⚡ cached</span>')
    if session.retrieve_seconds > 0:
        parts.append(
            f'<span class="tag mono">retrieve {session.retrieve_seconds*1000:.0f} ms</span>'
        )
    if session.generate_seconds > 0:
        parts.append(
            f'<span class="tag mono">generate {session.generate_seconds:.2f} s</span>'
        )
    parts.append(
        f'<span class="tag">🧠 {_safe(session.model)}</span>'
    )
    if session.history_used:
        parts.append(
            f'<span class="tag">🧵 {session.history_used} prior turn{"s" if session.history_used != 1 else ""}</span>'
        )
    st.markdown(f'<div class="meta-row">{"".join(parts)}</div>', unsafe_allow_html=True)


def render_expanded(query: str, expanded: str) -> None:
    if not expanded or expanded == query:
        return
    extra = expanded[len(query):].strip() if expanded.startswith(query) else expanded
    if not extra:
        return
    st.markdown(
        f'<div class="callout"><div class="label">Query expansion</div>{_safe(extra)}</div>',
        unsafe_allow_html=True,
    )


def render_sources(items) -> None:
    for i, c in enumerate(items, 1):
        ent_class = _entity_class(c.get("entity_type", ""))
        section = c.get("section") or "—"
        st.markdown(
            f"""
            <div class="source-card">
              <div class="head">
                <div class="title-row">
                  <span class="idx">{i}</span>
                  <span class="tag {ent_class}">{_safe(c['entity_title'])}</span>
                  <span class="section">· {_safe(section)}</span>
                </div>
                <span class="distance">d={c['distance']:.3f}</span>
              </div>
              <div class="body">{_safe(c['text'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def get_history_pairs(messages: list[dict], max_turns: int) -> list[tuple[str, str]]:
    """Convert Streamlit's flat message list into (user, assistant) pairs
    for the prompt builder. Only completed turns count; an unmatched user
    message at the end is dropped (it's about to become the new query)."""
    pairs: list[tuple[str, str]] = []
    pending_user: str | None = None
    for m in messages:
        if m["role"] == "user":
            pending_user = m["content"]
        elif m["role"] == "assistant" and pending_user is not None:
            pairs.append((pending_user, m["content"]))
            pending_user = None
    return pairs[-max_turns:]


# ─── Session state ────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Model")
    model_options = list(MODEL_OPTIONS)
    default_idx = model_options.index(LLM_MODEL) if LLM_MODEL in model_options else 0
    compare_models = st.toggle("Compare two models", value=False, help=(
        "Run two local models on the same retrieved context, side by side. "
        "Both models must already be `ollama pull`'ed."
    ))
    if compare_models:
        model_a = st.selectbox("Model A", model_options, index=default_idx, key="ma")
        idx_b = (default_idx + 1) % len(model_options)
        model_b = st.selectbox("Model B", model_options, index=idx_b, key="mb")
        active_model = model_a
    else:
        active_model = st.selectbox("Active model", model_options, index=default_idx)
        model_a = active_model
        model_b = None

    st.markdown("### Retrieval")
    top_k = st.slider("Top-K chunks", 2, 10, DEFAULT_TOP_K)
    show_context = st.toggle("Show retrieved context", value=True)
    use_history = st.toggle(
        "Use chat history",
        value=True,
        help=f"Thread the last {MAX_HISTORY_TURNS} turns into the prompt so "
             "follow-ups (\"compare them\") resolve correctly.",
    )

    st.markdown("### Index status")
    chunk_count = vector_store.count()
    people = catalog.list_titles("person")
    places = catalog.list_titles("place")
    state_class = "ready" if chunk_count > 0 else "empty"
    state_text = "Ready" if chunk_count > 0 else "Empty"
    st.markdown(
        f"""
        <div class="status-card {state_class}">
          <div class="label">{state_text}</div>
          <div class="value">{chunk_count:,}</div>
          <div class="label" style="margin-top:2px;">chunks indexed</div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
          <div class="status-card"><div class="label">People</div><div class="value">{len(people)}</div></div>
          <div class="status-card"><div class="label">Places</div><div class="value">{len(places)}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if chunk_count == 0:
        st.warning("Run `python scripts/ingest.py` to populate the store.")

    st.markdown("### Cache")
    cache_n = response_cache.size()
    st.caption(f"{cache_n} answer{'s' if cache_n != 1 else ''} cached this session")
    if st.button("Clear response cache", use_container_width=True):
        response_cache.clear()
        st.rerun()

    st.markdown("### Session")
    if st.button("🗑️  Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption(f"Active: `{active_model}`")


# ─── Hero header ──────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <div class="hero">
      <h1>🧭 Wikipedia RAG Assistant</h1>
      <p>Ask anything about 20 famous people and 20 famous places. Fully local — embeddings, vector store, and LLM all run on your machine.</p>
      <div class="meta">
        <span class="pill">⚡ Streaming</span>
        <span class="pill">🔍 Query expansion</span>
        <span class="pill">🧵 Multi-turn memory</span>
        <span class="pill">💾 Response cache</span>
        <span class="pill">🧠 {_safe(active_model)}{(" vs " + _safe(model_b)) if compare_models and model_b else ""}</span>
        <span class="pill">📦 {chunk_count:,} chunks</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─── Welcome / example chips ──────────────────────────────────────────────────

EXAMPLES = [
    "What did Marie Curie discover?",
    "Where is the Eiffel Tower located?",
    "Compare Einstein and Tesla",
    "Why is Nikola Tesla famous?",
    "What is Machu Picchu?",
    "Compare Lionel Messi and Cristiano Ronaldo",
    "Which famous place is located in Turkey?",
    "What was the Colosseum used for?",
]

if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome">
          <h3>👋 Try one of these</h3>
          <p>Or type your own question in the box at the bottom.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.pending_query = ex
            st.rerun()


# ─── Chat history rendering ───────────────────────────────────────────────────

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if show_context and message.get("context"):
            with st.expander(f"📚 Sources ({len(message['context'])})"):
                if message.get("expanded_query"):
                    render_expanded(message.get("query", ""), message["expanded_query"])
                render_sources(message["context"])


# ─── Input + answer ───────────────────────────────────────────────────────────

prompt = st.chat_input("Ask about a famous person or place…")
if not prompt and st.session_state.pending_query:
    prompt = st.session_state.pending_query
    st.session_state.pending_query = None


def _consume_stream(session, stream, placeholder) -> str:
    full = ""
    for piece in stream:
        full += piece
        placeholder.markdown(full + "▌")
    final = (session.final_text or full or "I don't know.").strip() or "I don't know."
    placeholder.markdown(final)
    return final


def _persist_assistant_turn(query: str, session, final_text: str) -> None:
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": final_text,
            "query": query,
            "expanded_query": session.retrieval.expanded_query,
            "context": [
                {
                    "entity_title": c.metadata.get("entity_title", "?"),
                    "entity_type": c.metadata.get("entity_type", ""),
                    "section": c.metadata.get("section", ""),
                    "distance": c.distance,
                    "text": c.text,
                }
                for c in session.retrieval.chunks
            ],
        }
    )


if prompt:
    history = (
        get_history_pairs(st.session_state.messages, MAX_HISTORY_TURNS)
        if use_history
        else []
    )

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if compare_models and model_b:
        # Single shared retrieval, two parallel generations.
        retrieval = do_retrieve(prompt, top_k=top_k)

        with st.chat_message("assistant"):
            render_meta_row(retrieval)
            if show_context:
                with st.expander(f"📚 Sources ({len(retrieval.chunks)})"):
                    render_expanded(prompt, retrieval.expanded_query)
                    render_sources([
                        {
                            "entity_title": c.metadata.get("entity_title", "?"),
                            "entity_type": c.metadata.get("entity_type", ""),
                            "section": c.metadata.get("section", ""),
                            "distance": c.distance,
                            "text": c.text,
                        }
                        for c in retrieval.chunks
                    ])

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown(
                    f'<div class="model-col-header"><span class="swatch" style="background:#4F46E5"></span>{_safe(model_a)}</div>',
                    unsafe_allow_html=True,
                )
                session_a, stream_a = stream_with_existing_retrieval(
                    prompt, retrieval, model=model_a, history=history,
                    max_history_turns=MAX_HISTORY_TURNS,
                )
                placeholder_a = st.empty()
                final_a = _consume_stream(session_a, stream_a, placeholder_a)
                render_latency_row(session_a)

            with col_b:
                st.markdown(
                    f'<div class="model-col-header"><span class="swatch" style="background:#EC4899"></span>{_safe(model_b)}</div>',
                    unsafe_allow_html=True,
                )
                session_b, stream_b = stream_with_existing_retrieval(
                    prompt, retrieval, model=model_b, history=history,
                    max_history_turns=MAX_HISTORY_TURNS,
                )
                placeholder_b = st.empty()
                final_b = _consume_stream(session_b, stream_b, placeholder_b)
                render_latency_row(session_b)

        # Persist the Model A answer as the canonical assistant turn so
        # history threading stays coherent. Append B as a sibling note.
        combined = (
            f"**{model_a}:** {final_a}\n\n---\n\n**{model_b}:** {final_b}"
        )
        _persist_assistant_turn(prompt, session_a, combined)

    else:
        with st.chat_message("assistant"):
            session, stream = stream_answer(
                prompt,
                top_k=top_k,
                model=active_model,
                history=history,
                max_history_turns=MAX_HISTORY_TURNS,
            )
            render_meta_row(session.retrieval)
            if show_context:
                with st.expander(f"📚 Sources ({len(session.retrieval.chunks)})"):
                    render_expanded(prompt, session.retrieval.expanded_query)
                    render_sources([
                        {
                            "entity_title": c.metadata.get("entity_title", "?"),
                            "entity_type": c.metadata.get("entity_type", ""),
                            "section": c.metadata.get("section", ""),
                            "distance": c.distance,
                            "text": c.text,
                        }
                        for c in session.retrieval.chunks
                    ])

            placeholder = st.empty()
            final_text = _consume_stream(session, stream, placeholder)
            render_latency_row(session)

        _persist_assistant_turn(prompt, session, final_text)
