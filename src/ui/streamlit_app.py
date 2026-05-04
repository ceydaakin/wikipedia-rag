"""Streamlit chat UI — ChatGPT-style.

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
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Design tokens + custom CSS (ChatGPT-inspired) ───────────────────────────

CSS = """
<style>
:root {
    --bg: #FFFFFF;
    --sidebar-bg: #F9F9F9;
    --user-bubble: #F4F4F4;
    --ink: #0D0D0D;
    --ink-soft: #353740;
    --muted: #6E6E80;
    --border: #E5E5E5;
    --avatar-bot: #19C37D;
    --accent: #10A37F;
    --accent-hover: #0E8C6E;
    --warn: #F59E0B;
    --tag-bg: #F4F4F4;
}

html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Söhne",
                 "Helvetica Neue", Helvetica, Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 8rem;
    max-width: 760px;
}

/* ─── Sidebar ─── */
section[data-testid="stSidebar"] {
    background: var(--sidebar-bg);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
section[data-testid="stSidebar"] h3 {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
    margin-top: 22px;
    margin-bottom: 8px;
    font-weight: 600;
}
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    font-size: 12.5px;
    color: var(--muted);
}

/* ─── Welcome ─── */
.welcome-wrap {
    text-align: center;
    margin: 12vh 0 26px 0;
}
.welcome-wrap h1 {
    font-size: 30px;
    font-weight: 600;
    color: var(--ink);
    letter-spacing: -0.02em;
    margin: 0 0 4px 0;
}
.welcome-wrap p {
    color: var(--muted);
    font-size: 14px;
    margin: 0;
}

/* example prompt cards (rendered via st.button -- styled below) */
div[data-testid="column"] .stButton > button {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    color: var(--ink-soft);
    font-size: 13.5px;
    font-weight: 400;
    padding: 14px 16px;
    text-align: left;
    line-height: 1.4;
    width: 100%;
    transition: all .12s ease;
    box-shadow: none;
    height: auto;
    min-height: 64px;
    white-space: normal;
}
div[data-testid="column"] .stButton > button:hover {
    border-color: #C9C9D1;
    background: #FAFAFA;
}
div[data-testid="column"] .stButton > button:active,
div[data-testid="column"] .stButton > button:focus {
    border-color: var(--accent);
    color: var(--ink);
    box-shadow: none !important;
}

/* sidebar buttons stay normal (less aggressive) */
section[data-testid="stSidebar"] .stButton > button {
    min-height: 36px;
    font-size: 13px;
    padding: 8px 12px;
}

/* ─── Chat messages ─── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 14px 0 !important;
    box-shadow: none !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    font-size: 15.5px;
    color: var(--ink);
    line-height: 1.65;
}

/* user message bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    display: flex !important;
    justify-content: flex-end !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) > div:last-child {
    background: var(--user-bubble);
    border-radius: 18px;
    padding: 10px 16px !important;
    max-width: 75%;
    color: var(--ink);
    font-size: 15px;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="chatAvatarIcon-user"] {
    display: none !important;
}

/* assistant avatar (green circle, like ChatGPT) */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="chatAvatarIcon-assistant"] {
    background: var(--avatar-bot) !important;
    color: white !important;
    width: 28px !important;
    height: 28px !important;
    border-radius: 50% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 14px !important;
}

/* ─── Tags / meta rows ─── */
.meta-row {
    display: flex; flex-wrap: wrap; gap: 6px;
    margin: 2px 0 10px 0;
}
.tag {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 2px 9px;
    border-radius: 999px;
    font-size: 11.5px;
    font-weight: 400;
    background: var(--tag-bg);
    color: var(--muted);
    border: none;
}
.tag.cached { background: #FEF3C7; color: #92400E; }
.tag.mono   { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
              font-size: 11px; }

/* ─── Sources expander ─── */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    box-shadow: none !important;
    background: var(--bg) !important;
}
[data-testid="stExpander"] summary {
    font-size: 13px !important;
    color: var(--muted) !important;
    font-weight: 500 !important;
}

.callout {
    background: #F7F7F8;
    border-left: 2px solid var(--accent);
    border-radius: 6px;
    padding: 8px 12px;
    margin: 4px 0 12px 0;
    font-size: 12.5px;
    color: var(--ink-soft);
}
.callout .label {
    font-size: 10.5px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--accent); margin-bottom: 2px;
}

.source-card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 6px;
}
.source-card .head {
    display: flex; align-items: center; justify-content: space-between;
    gap: 8px; margin-bottom: 4px;
}
.source-card .title-row {
    display: flex; align-items: center; gap: 8px;
    font-size: 13px; font-weight: 500; color: var(--ink);
}
.source-card .idx {
    display: inline-flex; align-items: center; justify-content: center;
    width: 18px; height: 18px; border-radius: 5px;
    background: #F4F4F4; color: var(--muted);
    font-size: 10.5px; font-weight: 600;
}
.source-card .section { color: var(--muted); font-size: 12px; }
.source-card .distance {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 11px; color: var(--muted);
}
.source-card .body {
    color: var(--ink-soft); font-size: 13px; line-height: 1.55;
    margin-top: 4px;
}

/* compare-mode column header */
.model-col-header {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 0;
    font-weight: 600; color: var(--ink); font-size: 13.5px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 8px;
}
.model-col-header .swatch {
    width: 8px; height: 8px; border-radius: 50%;
}

/* ─── Input bar (ChatGPT-style pill) ─── */
[data-testid="stChatInput"] {
    background: var(--bg) !important;
    padding-top: 16px !important;
    padding-bottom: 24px !important;
}
[data-testid="stChatInput"] > div {
    border: 1px solid var(--border) !important;
    border-radius: 26px !important;
    background: var(--bg) !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04) !important;
    padding: 4px !important;
}
[data-testid="stChatInput"] textarea {
    border: none !important;
    background: transparent !important;
    font-size: 15px !important;
    padding: 12px 14px !important;
    box-shadow: none !important;
    border-radius: 22px !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #B4B4C4 !important; }
[data-testid="stChatInput"] button {
    background: var(--ink) !important;
    border-radius: 50% !important;
    width: 32px !important;
    height: 32px !important;
    color: white !important;
}
[data-testid="stChatInput"] button:hover {
    background: var(--ink-soft) !important;
}
[data-testid="stChatInput"] button:disabled {
    background: var(--border) !important;
}

/* footer cleanup */
footer { display: none; }
[data-testid="stStatusWidget"] { display: none; }

/* hide page anchor links */
h1 a, h2 a, h3 a { display: none !important; }

/* status card (sidebar) */
.status-card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 12px;
    margin-bottom: 6px;
}
.status-card .label {
    font-size: 10.5px; text-transform: uppercase;
    letter-spacing: 0.06em; color: var(--muted);
}
.status-card .value {
    font-size: 18px; font-weight: 600; color: var(--ink);
    line-height: 1.2;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _safe(text: str) -> str:
    return html.escape(text or "")


def render_meta_row(retrieval) -> None:
    qt = retrieval.analysis.query_type
    parts = [
        f'<span class="tag">{len(retrieval.chunks)} sources</span>',
        f'<span class="tag">{qt}</span>',
    ]
    if retrieval.analysis.is_comparison:
        parts.append('<span class="tag">comparison</span>')
    if retrieval.analysis.matched_entities:
        names = ", ".join(e["title"] for e in retrieval.analysis.matched_entities)
        parts.append(f'<span class="tag">{_safe(names)}</span>')
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
    parts.append(f'<span class="tag">{_safe(session.model)}</span>')
    if session.history_used:
        parts.append(
            f'<span class="tag">{session.history_used} prior turn'
            f'{"s" if session.history_used != 1 else ""}</span>'
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
        section = c.get("section") or "—"
        st.markdown(
            f"""
            <div class="source-card">
              <div class="head">
                <div class="title-row">
                  <span class="idx">{i}</span>
                  <span>{_safe(c['entity_title'])}</span>
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
    st.markdown("### Wikipedia RAG")
    st.caption("Local · Ollama · Chroma")

    st.markdown("### Model")
    model_options = list(MODEL_OPTIONS)
    default_idx = model_options.index(LLM_MODEL) if LLM_MODEL in model_options else 0
    compare_models = st.toggle("Compare two models", value=False, help=(
        "Run two local models on the same retrieved context, side by side. "
        "Both models must be `ollama pull`'ed."
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
    show_context = st.toggle("Show sources", value=True)
    use_history = st.toggle("Use chat history", value=True)

    st.markdown("### Index")
    chunk_count = vector_store.count()
    people = catalog.list_titles("person")
    places = catalog.list_titles("place")
    st.markdown(
        f"""
        <div class="status-card">
          <div class="label">Chunks</div>
          <div class="value">{chunk_count:,}</div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
          <div class="status-card"><div class="label">People</div><div class="value">{len(people)}</div></div>
          <div class="status-card"><div class="label">Places</div><div class="value">{len(places)}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if chunk_count == 0:
        st.warning("Run `python scripts/ingest.py` to populate the store.")

    st.markdown("### Cache")
    st.caption(f"{response_cache.size()} cached")
    if st.button("Clear cache", use_container_width=True):
        response_cache.clear()
        st.rerun()

    if st.button("New chat", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.rerun()


# ─── Welcome / example prompts ────────────────────────────────────────────────

EXAMPLES = [
    ("What did Marie Curie discover?", "Find a fact"),
    ("Compare Einstein and Tesla", "Compare two people"),
    ("Where is the Eiffel Tower located?", "Locate a place"),
    ("Which famous place is located in Turkey?", "Reasoning"),
]

if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-wrap">
          <h1>How can I help?</h1>
          <p>Ask about 20 famous people and 20 famous places — fully local.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, (text, label) in enumerate(EXAMPLES):
        if cols[i % 2].button(text, key=f"ex_{i}", use_container_width=True):
            st.session_state.pending_query = text
            st.rerun()


# ─── Chat history rendering ───────────────────────────────────────────────────

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if (
            message["role"] == "assistant"
            and show_context
            and message.get("context")
        ):
            with st.expander(f"Sources · {len(message['context'])}"):
                if message.get("expanded_query"):
                    render_expanded(message.get("query", ""), message["expanded_query"])
                render_sources(message["context"])


# ─── Input + answer ───────────────────────────────────────────────────────────

prompt = st.chat_input("Message Wikipedia RAG…")
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
        retrieval = do_retrieve(prompt, top_k=top_k)

        with st.chat_message("assistant", avatar="🧭"):
            render_meta_row(retrieval)
            if show_context:
                with st.expander(f"Sources · {len(retrieval.chunks)}"):
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
                    f'<div class="model-col-header"><span class="swatch" style="background:#10A37F"></span>{_safe(model_a)}</div>',
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
                    f'<div class="model-col-header"><span class="swatch" style="background:#7C3AED"></span>{_safe(model_b)}</div>',
                    unsafe_allow_html=True,
                )
                session_b, stream_b = stream_with_existing_retrieval(
                    prompt, retrieval, model=model_b, history=history,
                    max_history_turns=MAX_HISTORY_TURNS,
                )
                placeholder_b = st.empty()
                final_b = _consume_stream(session_b, stream_b, placeholder_b)
                render_latency_row(session_b)

        combined = (
            f"**{model_a}**\n\n{final_a}\n\n---\n\n**{model_b}**\n\n{final_b}"
        )
        _persist_assistant_turn(prompt, session_a, combined)

    else:
        with st.chat_message("assistant", avatar="🧭"):
            session, stream = stream_answer(
                prompt,
                top_k=top_k,
                model=active_model,
                history=history,
                max_history_turns=MAX_HISTORY_TURNS,
            )
            render_meta_row(session.retrieval)

            placeholder = st.empty()
            final_text = _consume_stream(session, stream, placeholder)
            render_latency_row(session)

            if show_context:
                with st.expander(f"Sources · {len(session.retrieval.chunks)}"):
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

        _persist_assistant_turn(prompt, session, final_text)
