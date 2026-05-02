# BLG483E HW3 — Local Wikipedia RAG Assistant: Implementation Plan

## 1. Goal Recap

Build a fully-local ChatGPT-style assistant that answers questions about 20+ famous people and 20+ famous places using:
- Local Wikipedia ingestion
- Local embeddings (no external API)
- Local vector store (Chroma)
- Local LLM (Ollama: llama3.2:3b / phi3 / mistral)
- Local chat UI (Streamlit + CLI fallback)

Evaluation focuses on: functionality, retrieval quality, architectural sensibility, tradeoff understanding, demo clarity.

---

## 2. Architecture Overview

```
                ┌─────────────────────┐
                │  Wikipedia API      │
                │  (one-time fetch)   │
                └──────────┬──────────┘
                           │
                           ▼
      ┌────────────────────────────────────────┐
      │  Ingestion Pipeline (scripts/ingest.py)│
      │  fetch → clean → chunk → embed → store │
      └──────────┬─────────────────────────────┘
                 │
                 ▼
      ┌──────────────────────┐    ┌────────────────────┐
      │  SQLite (raw text +  │    │  Chroma (vectors + │
      │  metadata catalog)   │◄──►│  metadata filter)  │
      └──────────────────────┘    └─────────┬──────────┘
                                            │
                                            ▼
                              ┌─────────────────────────┐
                              │  Retriever (router)     │
                              │  classify → filter →    │
                              │  vector search → rerank │
                              └─────────┬───────────────┘
                                        │
                                        ▼
                              ┌─────────────────────────┐
                              │  Generator (Ollama)     │
                              │  prompt + context →     │
                              │  grounded answer        │
                              └─────────┬───────────────┘
                                        │
                                        ▼
                              ┌─────────────────────────┐
                              │  Streamlit UI / CLI     │
                              └─────────────────────────┘
```

### Vector store decision: **Option B (single store with metadata)**

- Why: comparison queries ("Compare Einstein and Tesla", "Which place is in Turkey?") require unified search.
- Filtering by `entity_type` metadata is cheap and Chroma supports `where={"entity_type": "person"}` natively.
- Single store keeps embeddings consistent and eliminates duplicated index code.

---

## 3. Entity List (40 total — meets the required minimum + room)

### People (20)
Required: Albert Einstein, Marie Curie, Leonardo da Vinci, William Shakespeare, Ada Lovelace, Nikola Tesla, Lionel Messi, Cristiano Ronaldo, Taylor Swift, Frida Kahlo
Additional: Isaac Newton, Charles Darwin, Mahatma Gandhi, Cleopatra, Steve Jobs, Mustafa Kemal Atatürk, Beyoncé, Pablo Picasso, Vincent van Gogh, Stephen Hawking

### Places (20)
Required: Eiffel Tower, Great Wall of China, Taj Mahal, Grand Canyon, Machu Picchu, Colosseum, Hagia Sophia, Statue of Liberty, Pyramids of Giza, Mount Everest
Additional: Stonehenge, Petra, Acropolis of Athens, Sydney Opera House, Big Ben, Christ the Redeemer, Niagara Falls, Angkor Wat, Mount Fuji, Cappadocia

> Adding Atatürk + Cappadocia + Hagia Sophia covers the "Which famous place is located in Turkey" mixed query nicely.

---

## 4. Implementation Phases

### Phase 1 — Project scaffolding (30 min)
- [x] Folder structure (`src/`, `scripts/`, `data/`, `docs/`, `tests/`)
- [ ] `requirements.txt` with pinned versions
- [ ] `.gitignore` (data/, .venv/, __pycache__/, *.sqlite, chroma_db/)
- [ ] `config.py` — single source of truth for paths, model names, chunk sizes

### Phase 2 — Ingestion (2 hr)
- [ ] `src/ingest/wikipedia.py` — fetch via `wikipedia-api` (or stdlib `urllib` against REST API to honor "language native" preference)
- [ ] Clean: strip references `[1]`, infobox templates, navigation; keep section structure
- [ ] Persist raw text + metadata to SQLite (`docs` table: id, title, entity_type, url, fetched_at, raw_text)
- [ ] `scripts/ingest.py` — CLI entry: `python scripts/ingest.py --refresh`
- [ ] Idempotent: skip already-fetched entities unless `--refresh`

### Phase 3 — Chunking (1 hr)
- [ ] `src/chunking/splitter.py` — pure-Python, no LangChain
- [ ] Strategy: **section-aware sliding window**
  - Primary split on `==` Wikipedia section headers
  - Within sections, fixed-size 600-char chunks with 100-char overlap
  - Each chunk carries: `entity_title`, `entity_type`, `section`, `chunk_idx`
- [ ] Why this strategy: section boundaries preserve semantic coherence; overlap prevents losing context at boundaries; 600 chars ≈ 150 tokens fits well within embed model + leaves room for many chunks in LLM context.

### Phase 4 — Embed & Store (1 hr)
- [ ] `src/embeddings/embedder.py` — wrapper around Ollama `nomic-embed-text` via HTTP (`http://localhost:11434/api/embeddings`)
- [ ] Batch embedding (avoid one-call-per-chunk) — async or thread pool
- [ ] `src/store/vector_store.py` — Chroma persistent client, single collection `wikipedia_rag`
- [ ] Document the design choice (Option B) in `docs/design_choices.md`

### Phase 5 — Retrieval (2 hr)
- [ ] `src/retrieval/classifier.py` — rule-based query type detection:
  1. Match query against known entity titles (exact + fuzzy)
  2. Keyword heuristics: "where", "located", "city", "country" → place-leaning
  3. Keyword heuristics: "who", "born", "discovered", "invented" → person-leaning
  4. Multiple matches → `both` (comparison query)
  5. Default → `both`
- [ ] `src/retrieval/retriever.py` — top-k vector search with metadata filter
  - For comparison queries, retrieve k chunks per entity then merge
- [ ] Optional: simple BM25-style keyword rerank on retrieved candidates

### Phase 6 — Generation (1 hr)
- [ ] `src/generation/llm.py` — Ollama HTTP client (no `ollama` library wrapper)
- [ ] Prompt template:
  ```
  You answer ONLY using the provided context. If the answer is not in the context, say "I don't know."
  Context:
  {numbered chunks}
  Question: {query}
  Answer:
  ```
- [ ] Streaming response support
- [ ] Return `(answer, sources)` tuple

### Phase 7 — Chat Interface (1.5 hr)
- [ ] `src/ui/streamlit_app.py` — primary UI
  - Chat-style message list
  - Expander showing retrieved chunks with similarity scores
  - "Reset" button (clear chat, optionally rebuild index)
  - Sidebar: model selector, top-k slider, show-context toggle
- [ ] `src/ui/cli.py` — CLI fallback for instructors who don't want to install Streamlit
  - `python -m src.ui.cli`
  - Commands: `:reset`, `:context on/off`, `:quit`

### Phase 8 — Documentation (1 hr)
- [ ] `README.md` — install + run instructions, prerequisite Ollama models, example queries
- [ ] `Product_prd.md` — PRD describing what an AI agent would need to build this from scratch
- [ ] `recommendation.md` — production deployment recommendations (managed vector DB, observability, eval pipeline, guardrails, scaling)
- [ ] `docs/design_choices.md` — chunk size, vector store option, retrieval strategy rationale

### Phase 9 — Testing & polish (1 hr)
- [ ] Manual test against all example queries from the PDF (people / places / mixed / failure)
- [ ] Verify "I don't know" path works for "Who is the president of Mars"
- [ ] Latency check: end-to-end < 10s on M1/M2 with llama3.2:3b
- [ ] Clean up logs, add progress bars to ingestion

### Phase 10 — Demo video (45 min)
- [ ] 5-min Loom walkthrough script (see `docs/demo_script.md`)
  1. System overview (60s)
  2. Live ingestion (60s)
  3. Q&A on people / places / comparison / failure (120s)
  4. Tech decisions + tradeoffs (45s)
  5. Improvements (15s)

**Total estimated effort: ~11 hours**

---

## 5. Tech Stack (Locked Choices)

| Layer        | Choice                              | Why                                                |
|--------------|-------------------------------------|----------------------------------------------------|
| Language     | Python 3.11+                        | Native ML ecosystem, instructor-friendly           |
| LLM          | Ollama + `llama3.2:3b`              | Fast on consumer hardware, good reasoning          |
| Embeddings   | Ollama + `nomic-embed-text`         | 768-dim, runs locally, same Ollama runtime         |
| Vector DB    | Chroma (persistent, file-backed)    | Zero-config, metadata filtering, SQLite under hood |
| Catalog DB   | SQLite (stdlib `sqlite3`)           | Required by spec, no extra dependency              |
| Wikipedia    | `wikipedia-api` library             | Stable; fallback to `urllib` + REST if needed      |
| UI           | Streamlit (primary) + CLI fallback  | Spec-recommended, fast to build                    |
| HTTP client  | stdlib `urllib` for Ollama calls    | Honors "prefer language native" guidance           |

> **"Language native" interpretation**: chunking, query classification, and Ollama HTTP calls are written from scratch. Embeddings/LLM/vector-DB are domain-specific infrastructure where building from scratch would be inappropriate scope creep.

---

## 6. Risk Register

| Risk                                  | Mitigation                                              |
|---------------------------------------|---------------------------------------------------------|
| Ollama not installed on grader's box  | Bold instructions in README, `make setup` script        |
| Wikipedia API rate limits             | Cache raw HTML/text on first fetch, ship cached data    |
| Embedding model download time         | README warns; `scripts/setup.sh` pulls models upfront   |
| Hallucination on adversarial queries  | Strict prompt + low temperature (0.1) + "I don't know"  |
| Streamlit not installed               | Provide CLI fallback                                    |
| First-run latency > 30s               | Pre-warm Ollama by running a dummy generation at boot   |

---

## 7. File Tree (target)

```
blg483e-hw3-wikipedia-rag/
├── README.md                       # install + run + examples
├── Product_prd.md                  # PRD for AI to rebuild this
├── recommendation.md               # production deployment notes
├── PLAN.md                         # this file
├── requirements.txt
├── .gitignore
├── config.py                       # paths, model names, chunk params
├── scripts/
│   ├── setup.sh                    # ollama pull + pip install
│   ├── ingest.py                   # CLI: fetch + chunk + embed + store
│   └── reset.py                    # wipe vector store + sqlite
├── src/
│   ├── __init__.py
│   ├── ingest/
│   │   ├── wikipedia.py
│   │   └── cleaner.py
│   ├── chunking/
│   │   └── splitter.py
│   ├── embeddings/
│   │   └── embedder.py             # Ollama nomic-embed-text
│   ├── store/
│   │   ├── vector_store.py         # Chroma wrapper
│   │   └── catalog.py              # SQLite raw-text store
│   ├── retrieval/
│   │   ├── classifier.py           # query type router
│   │   └── retriever.py
│   ├── generation/
│   │   ├── llm.py                  # Ollama generate
│   │   └── prompts.py
│   └── ui/
│       ├── streamlit_app.py
│       └── cli.py
├── data/
│   ├── entities.json               # canonical list of people/places
│   ├── raw/                        # cached Wikipedia text
│   └── processed/                  # chunked output (debug)
├── docs/
│   ├── design_choices.md
│   ├── demo_script.md
│   └── architecture.md
└── tests/
    ├── test_chunking.py
    ├── test_classifier.py
    └── test_retrieval.py
```

---

## 8. Definition of Done

- [ ] `python scripts/ingest.py` runs end-to-end on a clean machine (after `setup.sh`)
- [ ] `streamlit run src/ui/streamlit_app.py` launches the chat UI
- [ ] All 14+ example queries from the PDF produce sensible answers
- [ ] Failure-case queries return "I don't know" rather than hallucinating
- [ ] README is sufficient for a first-time user — no extra Slack messages needed
- [ ] PRD, recommendation.md, README, and demo video link are committed
- [ ] Repository pushed to public GitHub
- [ ] 5-minute Loom video uploaded and linked in README

---

## 9. Next Action

Confirm this plan, then I'll start with Phase 1 scaffolding (`requirements.txt`, `.gitignore`, `config.py`, `data/entities.json`) so the foundation is in place before any ingestion code.
