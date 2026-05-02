# 5-Minute Demo Script

Total budget: **5:00**. Aim to land at 4:50 to leave breathing room.

## 0:00 – 0:30 · System overview (30s)

> "This is a fully-local Wikipedia RAG assistant. It answers questions
> about 20 famous people and 20 famous places using only resources that
> run on this laptop — Ollama for the LLM and embeddings, Chroma for the
> vector store, SQLite for the document catalog."

Show the file tree on screen. Highlight `src/`, `scripts/`, the three
markdown deliverables (`README.md`, `Product_prd.md`, `recommendation.md`).

## 0:30 – 1:30 · Ingestion (60s)

Run:
```bash
python scripts/ingest.py --only "Albert Einstein"
```

Show the log: fetch → clean → chunk → embed → store. Then show the
already-populated full ingestion summary line ("40 entities, ~1200
chunks, 60s").

Explain in voice-over while it runs:
- "Each entity is fetched once and cached on disk."
- "We split on Wikipedia section headers, then sliding window inside
  each section."
- "Embeddings come from `nomic-embed-text` running locally."

## 1:30 – 3:30 · Live Q&A (120s)

Switch to the Streamlit UI. Run these in order, each ~15s:

1. **Person:** *"Why is Nikola Tesla famous?"* — show streaming + context expander.
2. **Place:** *"Where is the Eiffel Tower located?"*
3. **Mixed:** *"Which famous place is located in Turkey?"* — point out it surfaces Hagia Sophia and Cappadocia.
4. **Comparison:** *"Compare Albert Einstein and Nikola Tesla"* — open the context expander to show chunks from both pages.
5. **Failure case:** *"Who is the president of Mars?"* — system replies `I don't know.`

## 3:30 – 4:15 · Tech decisions & tradeoffs (45s)

Talk over the `docs/design_choices.md` file:
- "Single vector store with metadata filter — needed for comparisons."
- "Rule-based classifier — fast and explainable, fine for 40 entities."
- "Section-aware chunker written from scratch — preserves topical
  coherence at retrieval time."
- "We chose `llama3.2:3b` for speed; the prompt + low temperature
  prevent hallucination."

## 4:15 – 4:50 · Improvements (35s)

Talk over `recommendation.md`:
- "For production: hybrid search + cross-encoder rerank are the biggest
  retrieval-quality wins."
- "Add a golden eval set so quality is measurable in CI."
- "Move to a managed vector DB once corpus exceeds 100k chunks."
- "Cache `(query → answer)` by hash — RAG queries cache really well."

## 4:50 – 5:00 · Wrap (10s)

> "All code is on GitHub at [link]. Thanks for watching."

## Pre-recording checklist

- [ ] Ollama daemon running (`ollama list` shows both models)
- [ ] Vector store populated (`python scripts/ingest.py` finished)
- [ ] Streamlit launched on `localhost:8501`
- [ ] Browser zoomed to 110% so text is readable on video
- [ ] Terminal in a clean state, no clutter
- [ ] Wifi off (proves "fully local")
- [ ] Desktop notifications muted
