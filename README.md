# Local Wikipedia RAG Assistant

A fully-local ChatGPT-style assistant that answers questions about 20 famous
people and 20 famous places using a retrieval-augmented generation (RAG)
pipeline. **No external LLM API is used.** Embeddings, the vector store, and
the language model all run on your machine.

## How it works

```
Wikipedia ──► scripts/ingest.py ──► chunks ──► embeddings ──► Chroma
                                                                  │
   user query ──► expand ──► classify ──► search ◄────────────────┘
                                            │
                                            ▼
                                    Ollama LLM (streamed)
                                            │
                                            ▼
                            Streamlit chat UI / CLI
```

## Quick start

Four commands, in order, from a fresh clone. Everything else in this README
is just detail on these steps.

```bash
# 1. Clone and enter the project
git clone <your-repo-url> wikipedia-rag && cd wikipedia-rag

# 2. One-shot setup: venv + Python deps + pulls both Ollama models
bash scripts/setup.sh

# 3. Ingest the 40 Wikipedia pages and build the vector store (~5–10 min, one time)
source .venv/bin/activate
python scripts/ingest.py

# 4. Launch the chat UI
streamlit run src/ui/streamlit_app.py
```

Then open <http://localhost:8501> in your browser and ask questions like
*"Compare Einstein and Tesla"* or *"Where is the Eiffel Tower?"*.

> **Before you run anything, make sure Ollama is running** — install it from
> <https://ollama.com>, then either start the desktop app (macOS) or run
> `ollama serve` in a separate terminal.

## Stack

| Layer        | Choice                                    |
|--------------|-------------------------------------------|
| Language     | Python 3.11+                              |
| LLM          | Ollama · `llama3.2:3b` (default)          |
| Embeddings   | Ollama · `nomic-embed-text` (768-dim)     |
| Vector store | Chroma (persistent, file-backed)          |
| Catalog DB   | SQLite (stdlib `sqlite3`)                 |
| UI           | Streamlit chat app + CLI fallback         |

The chunker, query classifier, Ollama HTTP client, and prompt assembly are
written in plain Python — no LangChain.

## Prerequisites

1. **Python 3.11+** (`python3 --version`)
2. **Ollama** — install from <https://ollama.com>
   - Confirm with `ollama --version`
   - Make sure the daemon is running (`ollama serve` in a separate terminal,
     or just launch the desktop app on macOS).

## Step 1 — Setup (one time)

The Quick Start uses `scripts/setup.sh`. If you'd rather do it by hand:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

## Step 2 — Ingest the Wikipedia data (one time)

```bash
source .venv/bin/activate
python scripts/ingest.py
```

This fetches the 40 entities listed in `data/entities.json` (20 people + 20
places), cleans the text, splits it into section-aware overlapping chunks,
embeds every chunk with `nomic-embed-text`, and stores everything in:

- `chroma_db/` — Chroma vector store
- `data/catalog.sqlite` — SQLite raw-text catalog
- `data/raw/*.json` — cached Wikipedia text (so re-ingestion is instant)

Useful flags:

```bash
python scripts/ingest.py --refresh                  # re-fetch from Wikipedia
python scripts/ingest.py --only "Albert Einstein"   # ingest a single entity
```

## Step 3 — Run the chat UI

### Streamlit (recommended)

```bash
streamlit run src/ui/streamlit_app.py
```

Then open <http://localhost:8501>. The sidebar lets you change top-K, toggle
context display, see indexing status, and clear chat history.

### CLI fallback

```bash
python -m src.ui.cli
```

In-CLI commands:
- `:context` — toggle showing retrieved chunks
- `:reset` — start a fresh chat
- `:quit` — exit

## Reset the system

```bash
python scripts/reset.py     # wipes vector store + SQLite, keeps cached pages
```

## Troubleshooting

- **`Failed to reach Ollama at http://localhost:11434`** — Ollama isn't
  running. Open the Ollama desktop app or run `ollama serve` in another
  terminal, then retry.
- **Streamlit says "Vector store is empty"** — you skipped Step 2. Run
  `python scripts/ingest.py` first.
- **First answer is slow (10–20 s)** — Ollama is loading the model into
  memory. Subsequent answers are 2–6 s on Apple Silicon.
- **`ModuleNotFoundError`** — make sure the venv is active:
  `source .venv/bin/activate`.

## Example queries

Example queries the system handles:

**People**
- *Who was Albert Einstein and what is he known for?*
- *What did Marie Curie discover?*
- *Why is Nikola Tesla famous?*
- *Compare Lionel Messi and Cristiano Ronaldo*
- *What is Frida Kahlo known for?*

**Places**
- *Where is the Eiffel Tower located?*
- *Why is the Great Wall of China important?*
- *What is Machu Picchu?*
- *What was the Colosseum used for?*
- *Where is Mount Everest?*

**Mixed**
- *Which famous place is located in Turkey?*
- *Which person is associated with electricity?*
- *Compare Albert Einstein and Nikola Tesla*
- *Compare the Eiffel Tower and the Statue of Liberty*

**Failure cases (system answers "I don't know.")**
- *Who is the president of Mars?*
- *Tell me about a random unknown person John Doe*

## How retrieval works

1. **Expand** — the query is sent to the local LLM with a tight prompt that
   asks for related entity names, locations, and Wikipedia-style keywords.
   The keywords are appended to the original query before embedding, which
   surfaces chunks that share concepts but not surface words (e.g. *"Which
   famous place is in Turkey?"* → adds `Hagia Sophia Istanbul`). Toggled by
   `QUERY_EXPANSION_ENABLED` in `config.py`. Fail-open: if the LLM is offline
   the original query is used unchanged.
2. **Classify** — a rule-based router looks for known entity titles, last-name
   mentions, and person/place keyword hints to decide whether the query is
   about a `person`, a `place`, or `both`. Runs on the original query first
   and falls back to the expanded text if no entities matched, so expansion
   can also rescue routing.
3. **Filter + search** — the expanded query is embedded, then Chroma is
   searched with a metadata filter that matches the routing decision.
4. **Comparison shortcut** — when 2+ specific entities are mentioned with a
   comparison keyword (`compare`, `vs`, `versus`), each entity gets its own
   small batch of chunks, then the merged set is sorted by similarity. This
   guarantees both sides are represented in context.

Generation is **streamed token-by-token** through `src/generation/llm.py`
(`generate_stream`) and rendered live in both the Streamlit UI and the CLI,
so answers start appearing immediately instead of after the full completion.

The full prompt instructs the LLM to use only the retrieved context and to
say `I don't know.` when it can't find an answer there.

## Project layout

```
.
├── PLAN.md                    # original implementation plan
├── README.md                  # this file
├── Product_prd.md             # PRD (rebuild-from-scratch spec)
├── recommendation.md          # production-deployment recommendations
├── config.py                  # paths, model names, chunk params
├── requirements.txt
├── scripts/
│   ├── setup.sh               # venv + deps + ollama pull
│   ├── ingest.py              # fetch -> chunk -> embed -> store
│   └── reset.py
├── src/
│   ├── ingest/                # Wikipedia fetch + cleaner
│   ├── chunking/              # section-aware sliding window
│   ├── embeddings/            # Ollama nomic-embed-text wrapper
│   ├── store/                 # SQLite catalog + Chroma vector store
│   ├── retrieval/             # query classifier + retriever
│   ├── generation/            # Ollama generate + prompts
│   ├── pipeline.py            # high-level glue
│   └── ui/                    # streamlit_app.py + cli.py
├── docs/
│   ├── design_choices.md
│   ├── demo_script.md
│   └── architecture.md
├── data/
│   └── entities.json
└── tests/                     # offline unit tests (chunking, cleaner, classifier)
```

## Tests

```bash
python -m pytest tests/ -v
```

Tests cover the cleaner, chunker, query classifier, and query expander
without requiring Ollama or network access. **20 tests, all passing.**

### End-to-end smoke test

Once Ollama is running and ingestion is complete:

```bash
python scripts/smoke_test.py
```

Runs all 14 example queries (people / places / mixed / comparison /
failure cases) and prints answers with retrieval debug info.

## Notes & limitations

- First generation after starting Ollama is slower while the model is loaded
  into memory. Subsequent calls are fast (typically 2–6 seconds on Apple
  Silicon for `llama3.2:3b`).
- The Wikipedia-API library is the only place we don't roll our own; it
  handles MediaWiki content extraction and section parsing in a way that
  isn't worth re-implementing at this scope.
- See `recommendation.md` for what would change to make this production-ready.
